# ~/chatdoc-plus/rag/core.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMB_MODEL = "BAAI/bge-small-zh"

def build_or_load_db(pdf_paths, index_dir="../index/faiss_index"):
    # 先尝试加载已有索引
    try:
        emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
        return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
    except Exception:
        pass

    # 构建新索引
    docs = []
    for p in pdf_paths:
        loader = PyMuPDFLoader(p)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = FAISS.from_documents(chunks, emb)
    db.save_local(index_dir)
    return db
