# rag/core.py
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

LOCAL_BGE = "/root/chatdoc-plus/models/embeddings/bge-small-zh/AI-ModelScope/bge-small-zh-v1.5"

class BGEEmb(Embeddings):
    def __init__(self, local_dir: str = LOCAL_BGE):
        # 仅用本地，不再访问外网
        self.model = SentenceTransformer(local_dir)

    def embed_documents(self, texts):
        return [self.model.encode(t, normalize_embeddings=True).tolist() for t in texts]

    def embed_query(self, q):
        return self.model.encode(q, normalize_embeddings=True).tolist()

def build_or_update_index(pdf_path, index_dir="index/faiss_index"):
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
    return db

def has_index(index_dir="index/faiss_index"):
    return os.path.exists(os.path.join(index_dir, "index.faiss"))

def load_index(index_dir="index/faiss_index"):
    return FAISS.load_local(index_dir, BGEEmb(), allow_dangerous_deserialization=True)

def retrieve(query, k=4, index_dir="index/faiss_index"):
    if not has_index(index_dir):
        return []  # 没有索引就返回空结果，交给上层走纯LLM
    db = load_index(index_dir)
    return db.similarity_search(query, k=k)
