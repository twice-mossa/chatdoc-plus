from typing import List, Tuple
from dataclasses import dataclass
from langchain.schema import Document

from .query import rewrite_with_history
from .core import retrieve, build_context_and_citations, BGEEmb
from .packing import pack_documents
from .llm_qwen import stream_generate_with_cot
from .telemetry import span, write_telemetry
from .config import (
    MAX_CONTEXT_TOKENS,
    DEDUP_SIM_THRESHOLD,
    VERIFY_MIN_CITES,
    VERIFY_MIN_AVG_SCORE,
    SECOND_FETCH_K,
)


@dataclass
class GraphOutput:
    answer: str
    docs: List[Document]
    cites_md: str
    avg_score: float
    cite_num: int


def run_graph(message: str, history: List[Tuple[str, str]]):
    trace = {"query": message}
    # node: rewrite
    with span("rewrite", trace):
        rewritten = rewrite_with_history(message, history)

    # node: retrieve (hybrid by config) 1st
    with span("retrieve1", trace):
        docs = retrieve(rewritten, k=4)

    # node: pack
    with span("pack", trace):
        packed_docs, avg_score, cite_num = pack_documents(
            message, docs, max_context_tokens=MAX_CONTEXT_TOKENS, dedup_sim_threshold=DEDUP_SIM_THRESHOLD
        )
    context, cites_md = build_context_and_citations(packed_docs)

    # verify: if too few cites or low score, try another fetch with larger k
    if cite_num < VERIFY_MIN_CITES or avg_score < VERIFY_MIN_AVG_SCORE:
        with span("retrieve2", trace):
            docs2 = retrieve(rewritten, k=SECOND_FETCH_K)
        if docs2:
            packed_docs2, avg_score2, cite_num2 = pack_documents(
                message, docs2, max_context_tokens=MAX_CONTEXT_TOKENS, dedup_sim_threshold=DEDUP_SIM_THRESHOLD
            )
            if cite_num2 > cite_num or avg_score2 > avg_score:
                packed_docs, avg_score, cite_num = packed_docs2, avg_score2, cite_num2
                context, cites_md = build_context_and_citations(packed_docs)

    # node: generate (stream)
    prompt = (
        "你是严谨的中文金融助手，必须依据提供的【检索片段】回答；"
        "若片段中没有相关信息，明确说明“根据材料无法确定”。"
        "回答要点化，并在末尾用 [1][2]… 标注引用片段编号。\n\n"
        f"【检索片段】\n{context}\n\n【问题】{message}\n【回答】"
    )

    partial = ""
    with span("generate", trace):
        for chunk in stream_generate_with_cot(prompt):
        partial = chunk
        yield GraphOutput(partial, packed_docs, cites_md, avg_score, cite_num)
    trace["cite_num"] = cite_num
    trace["avg_score"] = round(avg_score, 4)
    write_telemetry(trace)


