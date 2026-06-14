"""
索引模块：纯 BM25 倒排索引 + 元数据过滤

提供以下核心组件：
- BM25Retriever: 基于 bm25s + metadata 过滤的纯检索器
- build_query: 构建检索查询串
- build_index: 索引构建器

使用流程：
    1. 安装依赖：uv sync
    2. 构建 BM25 索引：python src/indexing/build_index.py
    3. 在线检索：retriever.retrieve(query="...", domain="regulatory", doc_ids=[...])
"""

from src.indexing.retriever import (
    BM25Retriever,
    RetrievedChunk,
    build_query,
)

__all__ = [
    "BM25Retriever",
    "RetrievedChunk",
    "build_query",
]