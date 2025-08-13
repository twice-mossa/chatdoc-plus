from typing import List, Tuple
import numpy as np
from langchain.schema import Document
from .llm_qwen import get_llm


def _estimate_tokens(text: str) -> int:
    # 粗略估算：中文按字数、英文按词数近似
    return max(1, int(len(text) * 0.6))


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def pack_documents(
    query: str,
    docs: List[Document],
    max_context_tokens: int,
    dedup_sim_threshold: float = 0.85,
) -> Tuple[List[Document], float, int]:
    """
    1) 以置信度排序：优先使用带 rerank/bm25_score 等高分片段
    2) 基于向量相似度去重（阈值）
    3) 遵守 token 预算，尽量纳入互补片段

    返回：选中的文档、平均置信度、引用条数
    """
    if not docs:
        return [], 0.0, 0

    # 评分：优先 rerank_score > bm25_score > dense 相似度近似（无法直接取时默认为 0）
    def doc_score(d: Document) -> float:
        md = d.metadata or {}
        return float(md.get("score") or md.get("rerank_score") or md.get("bm25_score") or 0.0)

    sorted_docs = sorted(docs, key=doc_score, reverse=True)

    # 向量化缓存
    from .core import BGEEmb

    emb_model = BGEEmb().model
    doc_vecs = [emb_model.encode(d.page_content, normalize_embeddings=True) for d in sorted_docs]

    selected: List[Document] = []
    selected_vecs: List[np.ndarray] = []
    used_tokens = 0

    for d, v in zip(sorted_docs, doc_vecs):
        if selected_vecs:
            sim_max = max(_cos_sim(v, sv) for sv in selected_vecs)
            if sim_max >= dedup_sim_threshold:
                continue
        est = _estimate_tokens(d.page_content)
        if used_tokens + est > max_context_tokens:
            continue
        selected.append(d)
        selected_vecs.append(v)
        used_tokens += est

    avg_score = float(np.mean([doc_score(d) for d in selected])) if selected else 0.0
    return selected, avg_score, len(selected)


def estimate_tokens(text: str) -> int:
    return _estimate_tokens(text)


