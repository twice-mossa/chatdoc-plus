# rag/core.py
import os
from typing import List, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from config import EMBEDDING_DIR, INDEX_DIR, USE_RERANKER, RERANK_FETCH_K, RERANK_TOP_K, RERANK_MODEL, RETRIEVAL_MODE, RRF_K, USE_MMR, MMR_LAMBDA, MMR_CANDIDATES
from .hybrid import BM25Store, rrf_fuse


class BGEEmb(Embeddings):
    def __init__(self, local_dir: str = EMBEDDING_DIR):
        self.model = SentenceTransformer(local_dir)

    def embed_documents(self, texts):
        return [self.model.encode(t, normalize_embeddings=True).tolist() for t in texts]

    def embed_query(self, q):
        return self.model.encode(q, normalize_embeddings=True).tolist()


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _reranker = CrossEncoder(RERANK_MODEL, device=device)
        except Exception:
            _reranker = None
    return _reranker


def build_or_update_index(pdf_path: str, index_dir: str = INDEX_DIR):
    docs = PyMuPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    emb = BGEEmb()
    try:
        db = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    except Exception:
        db = FAISS.from_documents(chunks, emb)
    db.save_local(index_dir)

    # build/update BM25 store for hybrid retrieval
    try:
        bm25 = BM25Store(index_dir).load()
        bm25.add_documents(chunks)
        bm25.save()
    except Exception:
        pass
    return db


def has_index(index_dir: str = INDEX_DIR):
    return os.path.exists(os.path.join(index_dir, "index.faiss"))


def load_index(index_dir: str = INDEX_DIR):
    return FAISS.load_local(index_dir, BGEEmb(), allow_dangerous_deserialization=True)


def _similarity_search(query: str, fetch_k: int, index_dir: str) -> List[Document]:
    db = load_index(index_dir)
    return db.similarity_search(query, k=fetch_k)


def _rerank_docs(query: str, docs: List[Document], top_k: int) -> List[Document]:
    reranker = _get_reranker()
    if reranker is None or not docs:
        return docs[:top_k]
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    return [d for d, _ in ranked[:top_k]]


def _mmr_select(query: str, docs: List[Document], emb: BGEEmb, mmr_lambda: float, out_k: int) -> List[Document]:
    if not docs:
        return []
    # compute embeddings
    doc_texts = [d.page_content for d in docs]
    doc_vecs = [emb.model.encode(t, normalize_embeddings=True) for t in doc_texts]
    import numpy as np

    query_vec = emb.model.encode(query, normalize_embeddings=True)
    selected = []
    candidates = list(range(len(docs)))
    while candidates and len(selected) < min(out_k, len(docs)):
        best_idx = None
        best_score = -1e9
        for idx in candidates:
            sim_to_query = float(np.dot(query_vec, doc_vecs[idx]))
            if not selected:
                mmr_score = sim_to_query
            else:
                sim_to_selected = max(float(np.dot(doc_vecs[idx], doc_vecs[j])) for j in selected)
                mmr_score = mmr_lambda * sim_to_query - (1 - mmr_lambda) * sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)
        candidates.remove(best_idx)
    return [docs[i] for i in selected]


def retrieve(
    query: str,
    k: int = 4,
    index_dir: str = INDEX_DIR,
    fetch_k: int = RERANK_FETCH_K,
    use_reranker: bool = USE_RERANKER,
    mode: str = RETRIEVAL_MODE,
    use_mmr: bool = USE_MMR,
) -> List[Document]:
    if not has_index(index_dir):
        return []
    emb = BGEEmb()
    if mode == "dense":
        initial = _similarity_search(query, fetch_k=fetch_k, index_dir=index_dir)
    elif mode == "bm25":
        bm25 = BM25Store(index_dir).load()
        initial = bm25.search(query, top_n=fetch_k)
    else:  # hybrid
        dense_docs = _similarity_search(query, fetch_k=fetch_k, index_dir=index_dir)
        bm25_docs = BM25Store(index_dir).load().search(query, top_n=fetch_k)
        initial = rrf_fuse([dense_docs, bm25_docs], k=RRF_K, top_n=fetch_k)

    if use_mmr:
        initial = _mmr_select(query, initial, emb=emb, mmr_lambda=MMR_LAMBDA, out_k=min(fetch_k, MMR_CANDIDATES))

    if use_reranker:
        return _rerank_docs(query, initial, top_k=k)
    return initial[:k]


def build_context_and_citations(docs: List[Document]) -> Tuple[str, str]:
    if not docs:
        return "", ""
    context = []
    cites = []
    for i, d in enumerate(docs):
        context.append(f"[{i+1}] {d.page_content}")
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page")
        if page is not None:
            cites.append(f"[{i+1}] {os.path.basename(src)} p.{page}")
        else:
            cites.append(f"[{i+1}] {os.path.basename(src)}")
    return "\n\n".join(context), "\n".join(cites)
