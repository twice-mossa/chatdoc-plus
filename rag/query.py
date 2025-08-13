from typing import List, Tuple
import re


SMALL_TALK_PATTERNS = [
    r"^hi$|^hello$|^你好$|^嗨$|在吗|你是谁|介绍一下你|谢谢|再见",
    r"^讲个(笑话|段子)",
]

MATH_PATTERNS = [
    r"^[\d\s\+\-\*/\(\)\.]+$",
    r"(加|减|乘|除|百分之|年化|复利)",
]


def is_small_talk(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(p, q) for p in SMALL_TALK_PATTERNS)


def is_simple_math(query: str) -> bool:
    q = query.strip().lower()
    return any(re.search(p, q) for p in MATH_PATTERNS)


def rewrite_with_history(query: str, history: List[Tuple[str, str]]) -> str:
    """把多轮合成 standalone 问题（轻量：规则+拼接）"""
    if not history:
        return query
    last_turns = history[-3:]
    context = " ".join([f"用户:{q} 助手:{a}" for q, a in last_turns])
    rewritten = f"基于此前对话：{context}。当前问题：{query}。请将问题独立化，仅保留核心实体与条件。"
    # 轻量实现：这里直接返回拼接版；若接大模型可在此调用以生成更优 rewrite
    return rewritten


