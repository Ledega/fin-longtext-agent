"""
纯 BM25 检索器（无 Embedding）

基于 bm25s + metadata 过滤，支持：
- A 榜：domain + doc_ids 硬过滤 → BM25 检索 → Top-K
- B 榜：domain 过滤 → BM25 粗召回 → 文档级 re-rank → 候选 doc 内精搜

每条 chunk 带完整 metadata：domain, doc_id, chunk_type, section_path, clause_no
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 路径
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BM25_DIR = PROJECT_ROOT / "data" / "indices" / "bm25"


# ──────────────────────────────────────────────
# 检索结果结构
# ──────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """单条检索结果"""
    chunk_id: str
    doc_id: str
    domain: str
    text: str
    page_no: int = 0
    section_path: str = "[]"
    clause_no: str = ""
    chunk_type: str = "paragraph"
    score: float = 0.0
    rank: int = 0
    approx_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "domain": self.domain,
            "text": self.text[:500],
            "page_no": self.page_no,
            "section_path": self.section_path,
            "clause_no": self.clause_no,
            "chunk_type": self.chunk_type,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


# ──────────────────────────────────────────────
# 检索器
# ──────────────────────────────────────────────

def _load_index(domain: Optional[str] = None):
    """加载 BM25 索引和元数据"""
    import bm25s

    index_name = domain if domain else "all"
    index_dir = BM25_DIR / index_name

    if not index_dir.exists():
        raise FileNotFoundError(
            f"BM25 索引不存在: {index_dir}\n"
            f"请先运行: python src/indexing/build_index.py"
        )

    index = bm25s.BM25.load(str(index_dir), load_corpus=False)

    metadata_path = index_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


class BM25Retriever:
    """
    纯 BM25 检索器。

    每道题检索时：
    1. domain 硬过滤（从 metadata 中筛选）
    2. A 榜：doc_id in question.doc_ids 白名单
    3. 用题干 + 选项拼成 query，BM25 检索
    4. 返回带 metadata 的 Top-K 结果
    """

    def __init__(self):
        self._index = None
        self._metadata: List[dict] = []
        self._domain: Optional[str] = None

    def load(self, domain: Optional[str] = None) -> None:
        """加载指定领域的索引。domain=None 加载全库。"""
        self._domain = domain
        self._index, self._metadata = _load_index(domain)
        logger.info(f"检索器就绪 (domain={domain or 'all'}, size={len(self._metadata)})")

    def _filter_ids(self, domain: Optional[str] = None, doc_ids: Optional[List[str]] = None) -> List[int]:
        """
        根据 domain 和 doc_ids 过滤出合法索引位置。

        支持 doc_id 前缀不匹配自动修正：
        - 题目中 doc_ids 可能是裸名（如 'text01'），DB 中可能是 'fc_text01'
        - 先用精确匹配，不中则尝试前缀匹配
        """
        candidates = list(range(len(self._metadata)))

        # domain 过滤
        if domain:
            candidates = [i for i in candidates if self._metadata[i]["domain"] == domain]

        # doc_ids 白名单（A 榜）
        if doc_ids:
            doc_set = set(doc_ids)
            # 精确匹配
            exact = [i for i in candidates if self._metadata[i]["doc_id"] in doc_set]
            if exact:
                return exact
            # 精确不中，尝试后缀匹配（题目写 'text01'，DB 可能是 'fc_text01'）
            fuzzy = [
                i for i in candidates
                if any(self._metadata[i]["doc_id"].endswith(did) for did in doc_ids)
            ]
            if fuzzy:
                return fuzzy
            # 还不行就空
            return []

        return candidates

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[RetrievedChunk]:
        """
        统一检索接口。

        Args:
            query: 查询文本（拼接后的题干+选项）
            domain: 领域过滤
            doc_ids: A 榜文档 ID 白名单
            top_k: 返回 chunk 数

        Returns:
            RetrievedChunk 列表
        """
        # 1. 过滤出合法候选
        valid_ids = self._filter_ids(domain, doc_ids)
        if not valid_ids:
            logger.warning(f"过滤后无候选 (domain={domain}, doc_ids={doc_ids})")
            return []

        # 2. BM25 检索（先从全索引检索更多候选，再过滤）
        from src.indexing.build_bm25 import tokenize
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        results, scores = self._index.retrieve(
            [query_tokens],
            k=min(top_k * 3, len(self._metadata)),
        )

        # results shape: (1, k)
        hit_indices = results[0]
        hit_scores = scores[0]

        # 3. 过滤 + 组装
        valid_set = set(valid_ids)
        gathered = []
        for idx, score in zip(hit_indices, hit_scores):
            if idx < 0:
                continue
            doc_idx = int(idx)
            if doc_idx not in valid_set:
                continue

            meta = self._metadata[doc_idx]
            gathered.append(RetrievedChunk(
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                domain=meta["domain"],
                text=meta["text"],
                page_no=meta.get("page_no", 0),
                section_path=meta.get("section_path", "[]"),
                clause_no=meta.get("clause_no", ""),
                chunk_type=meta.get("chunk_type", "paragraph"),
                score=float(score),
                rank=len(gathered),
                approx_tokens=meta.get("approx_tokens", 0),
            ))

            if len(gathered) >= top_k:
                break

        return gathered

    def retrieve_with_doc_rerank(
        self,
        query: str,
        domain: str,
        top_k: int = 20,
        doc_rerank_top_n: int = 5,
    ) -> List[RetrievedChunk]:
        """
        B 榜专用：先粗召回到文档级，再在候选文档内精搜。

        Args:
            query: 查询文本
            domain: 领域标识（B 榜必传）
            top_k: 最终返回 chunk 数
            doc_rerank_top_n: 文档级 re-rank 后选择的文档数

        Returns:
            RetrievedChunk 列表
        """
        # 1. 粗召回（不限制 doc_ids）
        all_results = self.retrieve(
            query, domain=domain, doc_ids=None, top_k=top_k * 3,
        )

        # 2. 文档级 re-rank（按最高分 chunk 排序）
        doc_scores: Dict[str, float] = {}
        for r in all_results:
            did = r.doc_id
            doc_scores[did] = max(doc_scores.get(did, 0.0), r.score)

        selected_docs = sorted(doc_scores, key=doc_scores.get, reverse=True)[:doc_rerank_top_n]
        logger.info(f"文档级 re-rank 选中文档: {selected_docs}")

        if not selected_docs:
            return all_results[:top_k]

        # 3. 在候选文档内精搜
        return self.retrieve(
            query, domain=domain, doc_ids=selected_docs, top_k=top_k,
        )


# ──────────────────────────────────────────────
# 便捷函数（单次检索用）
# ──────────────────────────────────────────────

def build_query(question: dict) -> str:
    """
    构建检索查询串。

    用题干 + 选项拼合。如有明确实体（公司名、年份、条款号），
    通过重复提及提升 BM25 权重。
    """
    parts = [question.get("question", "")]
    options = question.get("options", {})
    for key in sorted(options.keys()):
        parts.append(options[key])
    return " ".join(parts)