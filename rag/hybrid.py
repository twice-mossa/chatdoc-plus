import os
import pickle
import re
from typing import List, Tuple

from langchain.schema import Document
from rank_bm25 import BM25Okapi


def _default_tokenize(text: str) -> List[str]:
    try:
        import jieba

        return [t.strip() for t in jieba.lcut(text) if t.strip()]
    except Exception:
        return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class BM25Store:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.path = os.path.join(index_dir, "bm25_store.pkl")
        self.corpus_tokens: List[List[str]] = []
        self.doc_texts: List[str] = []
        self.doc_metas: List[dict] = []
        self.bm25: BM25Okapi | None = None

    def load(self):
        if not os.path.exists(self.path):
            return self
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.corpus_tokens = data.get("corpus_tokens", [])
        self.doc_texts = data.get("doc_texts", [])
        self.doc_metas = data.get("doc_metas", [])
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        return self

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "corpus_tokens": self.corpus_tokens,
                    "doc_texts": self.doc_texts,
                    "doc_metas": self.doc_metas,
                },
                f,
            )

    def add_documents(self, docs: List[Document]):
        for d in docs:
            self.doc_texts.append(d.page_content)
            self.doc_metas.append(d.metadata)
            self.corpus_tokens.append(_default_tokenize(d.page_content))
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_n: int) -> List[Document]:
        if not self.bm25:
            return []
        tokens = _default_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)[:top_n]
        results = []
        for idx, sc in ranked:
            md = dict(self.doc_metas[idx]) if idx < len(self.doc_metas) else {}
            md["bm25_score"] = float(sc)
            results.append(Document(page_content=self.doc_texts[idx], metadata=md))
        return results


def rrf_fuse(lists: List[List[Document]], k: int = 60, top_n: int = 20) -> List[Document]:
    def _key(d: Document) -> str:
        src = d.metadata.get("source") or d.metadata.get("file_path") or ""
        page = d.metadata.get("page")
        return f"{os.path.basename(src)}::{page}::{hash(d.page_content)}"

    score_map = {}
    obj_map = {}
    for docs in lists:
        for rank, d in enumerate(docs, start=1):
            key = _key(d)
            obj_map[key] = d
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank)

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [obj_map[k] for k, _ in ranked]


