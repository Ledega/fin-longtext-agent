"""
BM25 索引构建器（纯 BM25，无 Embedding）

使用 bm25s 库构建倒排索引，每个 chunk 附带完整元数据用于过滤。

构建策略：
- 全库一个索引 + 按 domain 分 5 个领域索引（共 6 个索引文件）
- 每条记录带 metadata：domain, doc_id, chunk_type, section_path, clause_no, text
- 用 jieba + 金融词典做中文分词

使用：
    python src/indexing/build_index.py            # 全量构建
    python src/indexing/build_index.py --force     # 强制重建
    python src/indexing/build_index.py --domain insurance  # 只建单个领域
"""

import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Optional

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.build_bm25 import tokenize, ensure_finance_dict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────

INDICES_DIR = PROJECT_ROOT / "data" / "indices"
BM25_DIR = INDICES_DIR / "bm25"
DB_PATH = PROJECT_ROOT / "data" / "fin_longtext.db"

DOMAINS = [
    "insurance",
    "regulatory",
    "financial_contracts",
    "financial_reports",
    "research",
]


# ──────────────────────────────────────────────
# Chunk 加载
# ──────────────────────────────────────────────

def load_chunks(db_path: str, domain: Optional[str] = None) -> List[dict]:
    """从 SQLite 加载 chunks"""
    import sqlite3

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

    return [
        {
            "chunk_id": r[0], "doc_id": r[1], "domain": r[2],
            "page_no": r[3], "section_path": r[4], "clause_no": r[5],
            "chunk_type": r[6], "text": r[7], "char_len": r[8],
            "approx_tokens": r[9],
        }
        for r in rows
    ]


# ──────────────────────────────────────────────
# BM25 索引构建（使用 bm25s）
# ──────────────────────────────────────────────

def build_index(chunks: List[dict], index_name: str = "index") -> dict:
    """
    构建 BM25 索引并保存。

    Args:
        chunks: chunk dict 列表
        index_name: 索引名称（用于日志）

    Returns:
        {"chunks": int, "index_path": str, "metadata_path": str}
    """
    import bm25s

    texts = [c["text"] for c in chunks]
    logger.info(f"[{index_name}] 分词 {len(texts)} 个 chunk...")

    # 分词
    tokenized_corpus = [tokenize(text) for text in texts]
    logger.info(f"[{index_name}] 分词完成，构建 BM25 索引...")

    # 构建索引
    index = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    index.index(tokenized_corpus)

    # 保存索引
    index_dir = BM25_DIR / index_name
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save(str(index_dir))
    index_path = index_dir / "index.pkl"
    logger.info(f"[{index_name}] BM25 索引已保存: {index_dir} (size={len(chunks)})")

    # 保存元数据
    metadata = []
    for c in chunks:
        metadata.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "domain": c["domain"],
            "page_no": c.get("page_no"),
            "section_path": c.get("section_path", "[]"),
            "clause_no": c.get("clause_no"),
            "chunk_type": c.get("chunk_type", "paragraph"),
            "text": c["text"],
            "approx_tokens": c.get("approx_tokens", 0),
        })

    metadata_path = index_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    logger.info(f"[{index_name}] 元数据已保存: {metadata_path} (size={len(metadata)})")

    return {
        "chunks": len(chunks),
        "index_path": str(index_dir),
        "metadata_path": str(metadata_path),
    }


def load_index(domain: Optional[str] = None):
    """
    加载 BM25 索引和元数据。

    Args:
        domain: 领域标识。None 表示全库。

    Returns:
        (bm25s.BM25, List[dict])
    """
    import bm25s

    index_name = domain if domain else "all"
    index_dir = BM25_DIR / index_name

    index = bm25s.BM25.load(str(index_dir), load_corpus=False)

    metadata_path = index_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"BM25 索引已加载: {index_dir} (size={len(metadata)})")
    return index, metadata


# ──────────────────────────────────────────────
# 主构建流程
# ──────────────────────────────────────────────

def build_all(db_path: str, force: bool = False, domain: Optional[str] = None) -> None:
    """构建所有索引"""
    # 预热分词
    ensure_finance_dict()

    targets = [domain] if domain else DOMAINS + [None]  # None = 全库

    results = []
    for dom in targets:
        index_name = dom if dom else "all"
        index_dir = BM25_DIR / index_name

        # 检查是否已存在
        if not force and (index_dir / "metadata.json").exists():
            with open(index_dir / "metadata.json", "r") as f:
                meta = json.load(f)
            logger.info(f"[{index_name}] 索引已存在，跳过构建 (size={len(meta)})")
            results.append({"domain": index_name, "chunks": len(meta)})
            continue

        chunks = load_chunks(db_path, domain=dom)
        if not chunks:
            logger.warning(f"[{index_name}] 无 chunk，跳过")
            continue

        result = build_index(chunks, index_name=index_name)
        results.append({"domain": index_name, "chunks": result["chunks"]})

    # 输出汇总
    print(f"\n{'='*60}")
    print("BM25 索引构建汇总:")
    total = 0
    for r in results:
        print(f"  ✓ {r['domain']:<22}  {r['chunks']:>6} chunks")
        total += r["chunks"]
    print(f"  {'='*30}")
    print(f"  总计                {total:>6} chunks")
    print(f"  索引目录: {BM25_DIR}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BM25 索引构建器")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite 数据库路径")
    parser.add_argument("--force", action="store_true", help="强制重建所有索引")
    parser.add_argument("--domain", choices=DOMAINS, default=None, help="仅构建单个领域")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    build_all(args.db, force=args.force, domain=args.domain)


if __name__ == "__main__":
    main()