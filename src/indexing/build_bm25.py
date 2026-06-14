"""
BM25 索引构建器

构建中文 BM25 关键词索引，用于混合检索中的精确匹配通道。

功能：
1. 从 SQLite 加载 chunks，用 jieba 分词
2. 构建 BM25 索引（全库 + 每领域一份）
3. 支持自定义金融词典（提高条款编号、指标名等的分词准确率）
4. 持久化为 pickle 文件

使用：
    python src/indexing/build_bm25.py
"""

import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

logger = logging.getLogger(__name__)

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3

# 内联 load_chunks_from_sqlite（原 build_faiss 已删除）
def load_chunks_from_sqlite(db_path: str, domain: Optional[str] = None) -> List[dict]:
    conn = sqlite3.connect(db_path)
    if domain:
        rows = conn.execute(
            """SELECT chunk_id, doc_id, domain, page_no, section_path,
                      clause_no, chunk_type, text, char_len, approx_tokens
               FROM chunks WHERE domain = ? ORDER BY rowid""",
            (domain,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT chunk_id, doc_id, domain, page_no, section_path,
                      clause_no, chunk_type, text, char_len, approx_tokens
               FROM chunks ORDER BY rowid"""
        ).fetchall()
    conn.close()
    return [{
        "chunk_id": r[0], "doc_id": r[1], "domain": r[2],
        "page_no": r[3], "section_path": r[4], "clause_no": r[5],
        "chunk_type": r[6], "text": r[7], "char_len": r[8],
        "approx_tokens": r[9],
    } for r in rows]

DOMAINS = [
    "insurance", "regulatory", "financial_contracts",
    "financial_reports", "research",
]

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────

INDICES_DIR = PROJECT_ROOT / "data" / "indices"
BM25_DIR = INDICES_DIR / "bm25"
FINANCE_DICT_PATH = Path(__file__).parent / "finance_dict.txt"


# ──────────────────────────────────────────────
# 中文分词（含金融词典）
# ──────────────────────────────────────────────

_finance_dict_loaded = False


def ensure_finance_dict():
    """确保 jieba 加载了金融自定义词典"""
    global _finance_dict_loaded
    if _finance_dict_loaded:
        return

    import jieba

    # 内置常用金融词条
    builtin_words = [
        # 条款编号模式
        "第一条 第二条 第三条 第四条 第五条 第六条 第七条 第八条 第九条 第十条",
        "第十一条 第十二条 第十三条 第十四条 第十五条 第十六条 第十七条 第十八条 第十九条 第二十条",
        "第二十一条 第二十二条 第二十三条 第二十四条 第二十五条 第二十六条 第二十七条 第二十八条 第二十九条",
        "第一节 第二节 第三节 第四节 第五节",
        "第一章 第二章 第三章 第四章 第五章 第六章 第七章 第八章 第九章 第十章",
        # 金融指标
        "营业收入 营业成本 净利润 毛利润 毛利率 净利率",
        "总资产 总负债 净资产 资产负债率 流动比率 速动比率",
        "经营活动产生的现金流量净额 投资活动产生的现金流量净额 筹资活动产生的现金流量净额",
        "每股收益 稀释每股收益 加权平均净资产收益率 基本每股收益",
        "研发投入 研发费用 研发投入占营业收入的比例",
        "归属于上市公司股东的净利润 归属于上市公司股东的扣除非经常性损益的净利润",
        # 保险术语
        "保险责任 责任免除 保险金额 保险费 保险期间 犹豫期 等待期",
        "身故保险金 全残保险金 生存保险金 满期保险金 年金",
        "投保人 被保险人 受益人 保险人",
        "退保 现金价值 保单贷款 自动垫交 减额交清",
        # 监管法规
        "上市公司 证券法 公司法 监管 证监会 交易所",
        "信息披露 关联交易 对外担保 募集资金 股东会 董事会 监事会",
        "独立董事 审计委员会 薪酬委员会 提名委员会",
    ]
    for phrase in builtin_words:
        for word in phrase.split():
            jieba.add_word(word, freq=100, tag="n")

    # 外部自定义词典（如果存在）
    if FINANCE_DICT_PATH.exists():
        jieba.load_userdict(str(FINANCE_DICT_PATH))
        logger.info(f"已加载金融自定义词典: {FINANCE_DICT_PATH}")

    _finance_dict_loaded = True


def tokenize(text: str) -> List[str]:
    """
    对文本进行中文分词。

    对条款编号、指标名等保留完整词条。

    Args:
        text: 输入文本

    Returns:
        词条列表
    """
    ensure_finance_dict()
    import jieba

    # jieba 分词
    words = jieba.lcut(text)

    # 过滤过短的词和纯标点
    filtered = []
    for w in words:
        w = w.strip()
        if len(w) < 2 and not w.isdigit():
            continue
        # 跳过纯标点
        if all(not c.isalnum() and not '\u4e00' <= c <= '\u9fff' for c in w):
            continue
        filtered.append(w)

    return filtered


# ──────────────────────────────────────────────
# BM25 实现
# ──────────────────────────────────────────────

class BM25Index:
    """
    简化版 BM25 索引（不依赖 rank_bm25 库）。

    使用 BM25-OKAPI 公式，适合中文场景。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1          # 饱和度参数
        self.b = b            # 长度归一化参数
        self.corpus_size = 0  # 文档数
        self.avgdl = 0.0      # 平均文档长度
        self.doc_freqs: List[Counter] = []   # 每篇文档的词频
        self.idf: Dict[str, float] = {}      # 词 -> IDF 值
        self.doc_len: List[int] = []         # 每篇文档的长度
        self.chunk_ids: List[str] = []       # 与文档一一对应的 chunk_id
        self.metadata: List[dict] = []       # 可选元数据

    def fit(self, tokenized_corpus: List[List[str]], chunk_ids: List[str],
            metadata: Optional[List[dict]] = None) -> None:
        """
        从分词语料构建 BM25 索引。

        Args:
            tokenized_corpus: 分词后的文档列表
            chunk_ids: 与文档一一对应的 chunk_id
            metadata: 可选的元数据列表
        """
        self.corpus_size = len(tokenized_corpus)
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / max(self.corpus_size, 1)
        self.chunk_ids = chunk_ids
        self.metadata = metadata or [{} for _ in range(self.corpus_size)]

        # 计算 IDF
        df: Dict[str, int] = {}
        for doc in tokenized_corpus:
            seen = set()
            for word in doc:
                if word not in seen:
                    seen.add(word)
                    df[word] = df.get(word, 0) + 1

        n = self.corpus_size
        self.idf = {
            word: (n - freq + 0.5) / (freq + 0.5) + 1.0
            for word, freq in df.items()
        }

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        计算单篇文档的 BM25 得分。

        Args:
            query_tokens: 分词后的查询
            doc_idx: 文档索引

        Returns:
            BM25 得分
        """
        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]

        for token in query_tokens:
            if token not in self.idf:
                continue
            tf = doc_freq.get(token, 0)
            if tf == 0:
                continue

            idf = self.idf[token]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 30) -> List[dict]:
        """
        检索 top_k 结果。

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            [{"chunk_id": str, "score": float, "rank": int, ...}, ...]
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(self.corpus_size):
            s = self.score(query_tokens, i)
            if s > 0:
                scores.append((i, s))

        # 按得分排序
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        results = []
        for rank, (idx, score) in enumerate(top_scores):
            meta = self.metadata[idx]
            result = {
                "chunk_id": self.chunk_ids[idx],
                "score": score,
                "rank": rank,
                "doc_id": meta.get("doc_id", ""),
                "domain": meta.get("domain", ""),
            }
            results.append(result)

        return results

    def save(self, path: str) -> None:
        """持久化到 pickle"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "k1": self.k1,
                "b": self.b,
                "corpus_size": self.corpus_size,
                "avgdl": self.avgdl,
                "doc_freqs": self.doc_freqs,
                "idf": self.idf,
                "doc_len": self.doc_len,
                "chunk_ids": self.chunk_ids,
                "metadata": self.metadata,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 索引已保存: {path} (size={self.corpus_size})")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """从 pickle 加载"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 索引文件不存在: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls(k1=data["k1"], b=data["b"])
        index.corpus_size = data["corpus_size"]
        index.avgdl = data["avgdl"]
        index.doc_freqs = data["doc_freqs"]
        index.idf = data["idf"]
        index.doc_len = data["doc_len"]
        index.chunk_ids = data["chunk_ids"]
        index.metadata = data["metadata"]
        logger.info(f"BM25 索引已加载: {path} (size={index.corpus_size})")
        return index


# ──────────────────────────────────────────────
# 构建 BM25 索引
# ──────────────────────────────────────────────

def build_bm25_index(
    chunks: List[dict],
    index_name: str = "bm25_index",
) -> BM25Index:
    """
    从 chunk 列表构建 BM25 索引。

    Args:
        chunks: chunk dict 列表
        index_name: 索引名称（用于日志）

    Returns:
        BM25Index 实例
    """
    logger.info(f"[{index_name}] 开始分词 {len(chunks)} 个 chunk...")

    tokenized_corpus = []
    chunk_ids = []
    metadata = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        tokens = tokenize(text)
        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk.get("chunk_id", f"unknown_{i}"))
        metadata.append({
            "doc_id": chunk.get("doc_id", ""),
            "domain": chunk.get("domain", ""),
            "page_no": chunk.get("page_no"),
            "chunk_type": chunk.get("chunk_type", ""),
        })

        if (i + 1) % 2000 == 0:
            logger.info(f"[{index_name}] 已分词 {i + 1}/{len(chunks)}")

    logger.info(f"[{index_name}] 构建 BM25 索引...")
    bm25 = BM25Index()
    bm25.fit(tokenized_corpus, chunk_ids, metadata)
    logger.info(f"[{index_name}] BM25 索引构建完成: {bm25.corpus_size} 篇文档")

    return bm25


def build_all_bm25_indexes(db_path: str, force: bool = False) -> None:
    """
    构建全库 + 各领域的 BM25 索引。

    Args:
        db_path: SQLite 数据库路径
        force: 强制重新构建
    """
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 全库索引
    all_path = BM25_DIR / "all.pkl"
    if force or not all_path.exists():
        chunks = load_chunks_from_sqlite(db_path)
        bm25_all = build_bm25_index(chunks, "all")
        bm25_all.save(str(all_path))
    else:
        logger.info("[all] BM25 索引已存在，跳过")

    # 2. 各领域索引
    for domain in DOMAINS:
        domain_path = BM25_DIR / f"{domain}.pkl"
        if not force and domain_path.exists():
            logger.info(f"[{domain}] BM25 索引已存在，跳过")
            continue

        chunks = load_chunks_from_sqlite(db_path, domain)
        bm25_domain = build_bm25_index(chunks, domain)
        bm25_domain.save(str(domain_path))


def main():
    parser = argparse.ArgumentParser(description="构建 BM25 索引")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "fin_longtext.db"),
                        help="SQLite 数据库路径")
    parser.add_argument("--force", action="store_true",
                        help="强制重新构建所有索引")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    build_all_bm25_indexes(args.db, force=args.force)

    # 快速验证
    all_path = BM25_DIR / "all.pkl"
    if all_path.exists():
        bm25 = BM25Index.load(str(all_path))
        results = bm25.search("营业收入 增长率", top_k=5)
        print(f"\n验证 BM25 检索 '营业收入 增长率':")
        for r in results:
            print(f"  rank={r['rank']}, chunk={r['chunk_id']}, score={r['score']:.4f}")


if __name__ == "__main__":
    main()