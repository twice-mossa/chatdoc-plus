import os


# Model and embedding directories can be overridden by environment variables
MODEL_DIR = os.getenv(
    "CHATDOC_MODEL_DIR",
    "/root/chatdoc-plus/models/hf/Qwen1.5-7B-Chat/qwen/Qwen1.5-7B-Chat",
)

EMBEDDING_DIR = os.getenv(
    "CHATDOC_EMBEDDING_DIR",
    "/root/chatdoc-plus/models/embeddings/bge-small-zh/AI-ModelScope/bge-small-zh-v1.5",
)

INDEX_DIR = os.getenv("CHATDOC_INDEX_DIR", "index/faiss_index")

# Reranker settings
USE_RERANKER = os.getenv("CHATDOC_USE_RERANKER", "1") == "1"
RERANK_FETCH_K = int(os.getenv("CHATDOC_RERANK_FETCH_K", "20"))
RERANK_TOP_K = int(os.getenv("CHATDOC_RERANK_TOP_K", "4"))
RERANK_MODEL = os.getenv("CHATDOC_RERANK_MODEL", "BAAI/bge-reranker-large")

# Chat settings
MAX_HISTORY_TURNS = int(os.getenv("CHATDOC_MAX_HISTORY_TURNS", "4"))
MAX_NEW_TOKENS = int(os.getenv("CHATDOC_MAX_NEW_TOKENS", "300"))

# Retrieval settings
RETRIEVAL_MODE = os.getenv("CHATDOC_RETRIEVAL_MODE", "dense")  # dense | bm25 | hybrid
RRF_K = int(os.getenv("CHATDOC_RRF_K", "60"))
USE_MMR = os.getenv("CHATDOC_USE_MMR", "1") == "1"
MMR_LAMBDA = float(os.getenv("CHATDOC_MMR_LAMBDA", "0.5"))
MMR_CANDIDATES = int(os.getenv("CHATDOC_MMR_CANDIDATES", "20"))

# Packing settings
MAX_CONTEXT_TOKENS = int(os.getenv("CHATDOC_MAX_CONTEXT_TOKENS", "1200"))
DEDUP_SIM_THRESHOLD = float(os.getenv("CHATDOC_DEDUP_SIM_THRESHOLD", "0.85"))

# Query handling
REWRITE_ENABLED = os.getenv("CHATDOC_REWRITE_ENABLED", "1") == "1"
ALLOW_NO_RETRIEVAL_ANSWER = os.getenv("CHATDOC_ALLOW_NO_RETRIEVAL_ANSWER", "1") == "1"

# Graph policy
GRAPH_POLICY = os.getenv("CHATDOC_GRAPH_POLICY", "graph")  # plain | graph
VERIFY_MIN_CITES = int(os.getenv("CHATDOC_VERIFY_MIN_CITES", "2"))
VERIFY_MIN_AVG_SCORE = float(os.getenv("CHATDOC_VERIFY_MIN_AVG_SCORE", "0.2"))
SECOND_FETCH_K = int(os.getenv("CHATDOC_SECOND_FETCH_K", "8"))

