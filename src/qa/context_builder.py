"""
上下文构建器：检索 chunk → 拼装上下文 → Token 控制

对于 A 榜有 doc_ids 的题目：
    对每个 doc_id 检索 Top-k chunk → 合并上下文

内置 Token 预算控制，超出时自动降级（减少 k、缩短文本）。

依赖纯 BM25 检索器（无 Embedding）。
"""

import logging
from typing import List, Tuple

from src.indexing.retriever import BM25Retriever, build_query
from src.qa.prompt_templates import format_context

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 5               # 每个 doc 取 top-k chunk
MAX_PROMPT_TOKENS = 6000        # 单题的 prompt 总 token 上限
MAX_CHUNK_CHARS = 600           # 单个 chunk 的最大字符数


def approx_tokens(text: str) -> int:
    """估算 token 数（与 chunker 一致）"""
    import re
    cn = len(re.findall(r'[\u4e00-\u9fff]', text))
    other = len(text) - cn
    return int(cn * 1.5 + other / 4) + 10


def build_context_for_question(
    retriever: BM25Retriever,
    question: dict,
    top_k: int = DEFAULT_TOP_K,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> Tuple[str, dict]:
    """
    为单道题构建检索上下文。

    流程：
    1. 用题干+选项构建 query
    2. 从 doc_ids 逐文档 BM25 检索
    3. 每个文档取 Top-k chunk
    4. 合并后估算 token，超限则降级

    Args:
        retriever: BM25Retriever 实例
        question: 标准化题目 dict
        top_k: 每文档初始返回 chunk 数
        max_prompt_tokens: 单题 prompt 总 token 上限

    Returns:
        (context_text, stats)
    """
    query = build_query(question)
    doc_ids = question.get("doc_ids", [])
    domain = question.get("domain", "")

    stats = {
        "domain": domain,
        "doc_ids": doc_ids,
        "query": query[:100],
        "total_chunks": 0,
        "total_chars": 0,
        "dropped": False,
    }

    if not doc_ids:
        logger.warning(f"题目 {question['qid']} 无 doc_ids")
        return "【未指定相关文档】", stats

    # 逐文档检索
    raw_chunks: List[dict] = []
    for did in doc_ids:
        try:
            results = retriever.retrieve(
                query=query,
                domain=domain,
                doc_ids=[did],
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(f"检索文档 {did} 失败: {e}")
            continue

        for r in results:
            raw_chunks.append({
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "domain": r.domain,
                "text": r.text,
                "score": r.score,
                "rank": r.rank,
            })

    stats["total_chunks"] = len(raw_chunks)

    if not raw_chunks:
        return "【未检索到相关文档片段】", stats

    # 估算 token
    total_text = " ".join(c["text"] for c in raw_chunks)
    stats["total_chars"] = len(total_text)
    estimated_tokens = approx_tokens(total_text)

    if estimated_tokens > max_prompt_tokens:
        logger.info(
            f"上下文超限: {estimated_tokens} tokens > {max_prompt_tokens}, 准备降级..."
        )
        raw_chunks = _compress_context(raw_chunks)
        stats["dropped"] = True
        stats["total_chunks_after_drop"] = len(raw_chunks)

    context_text = format_context(raw_chunks)
    stats["final_chars"] = len(context_text)

    return context_text, stats


def _compress_context(chunks: List[dict]) -> List[dict]:
    """按分数排序取前 20 个 chunk"""
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_chunks[:20]