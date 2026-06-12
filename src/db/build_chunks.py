"""
读取 docs 表中注册的文档，逐一解析文本、分块，结果写入：
1) SQLite chunks 表
2) chunks.jsonl 文件
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import List

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import get_config
from src.db.schema import init_db
from src.db.chunker import chunk_document_from_file


# JSONL 持久化路径
JSONL_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"


def load_docs_from_db(conn: sqlite3.Connection) -> List[dict]:
    """从 docs 表加载所有文档记录"""
    rows = conn.execute(
        "SELECT doc_id, domain, file_path, source_type FROM docs ORDER BY domain, doc_id"
    ).fetchall()
    return [
        {
            "doc_id": row[0],
            "domain": row[1],
            "file_path": row[2],
            "source_type": row[3],
        }
        for row in rows
    ]


def process_doc(
    conn: sqlite3.Connection,
    doc: dict,
    cfg,
    jsonl_writer,
    total_stats: dict,
) -> int:
    """
    处理单个文档：解析 → 分块 → 写入 SQLite + JSONL

    Returns:
        int: 该文档产生的 chunk 数
    """
    doc_id = doc["doc_id"]
    domain = doc["domain"]
    rel_path = doc["file_path"]
    file_path = cfg.data_root / rel_path

    if not file_path.exists():
        print(f"  [跳过] 文件不存在: {file_path}")
        return 0

    print(f"  处理: {doc_id} ({file_path.name})")

    try:
        chunks = chunk_document_from_file(file_path, doc_id, domain)
    except Exception as e:
        print(f"  [错误] 解析/分块失败: {e}")
        return 0

    if not chunks:
        print(f"  [警告] 无有效 chunk")
        return 0

    # 写入 SQLite
    for chunk in chunks:
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, domain, page_no, section_path, clause_no,
                chunk_type, text, char_len, approx_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["domain"],
                chunk["page_no"],
                chunk["section_path"],
                chunk["clause_no"],
                chunk["chunk_type"],
                chunk["text"],
                chunk["char_len"],
                chunk["approx_tokens"],
            ),
        )

    # 写入 JSONL
    for chunk in chunks:
        jsonl_writer.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 更新统计
    total_stats["chunks"] += len(chunks)
    total_stats["tokens"] += sum(c["approx_tokens"] for c in chunks)
    total_stats["docs_succeeded"] += 1

    print(f"    → {len(chunks)} chunks, {chunks[-1]['approx_tokens'] if chunks else 0} approx_tokens (last)")

    return len(chunks)


def main():
    cfg = get_config()

    # 数据库路径
    db_path = PROJECT_ROOT / "data" / "fin_longtext.db"
    if not db_path.exists():
        print(f"[错误] 数据库不存在: {db_path}")
        print("请先运行 python src/db/build_docs.py")
        return

    # 确保 JSONL 输出目录存在
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 连接数据库
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON;")

    # 确保 chunks 表存在
    init_db(str(db_path))

    # 清空旧 chunks 数据（重新生成时）
    conn.execute("DELETE FROM chunks")
    conn.commit()

    # 加载所有文档
    docs = load_docs_from_db(conn)
    print(f"共加载 {len(docs)} 篇文档，开始分块处理...\n")

    total_stats = {
        "docs": len(docs),
        "docs_succeeded": 0,
        "chunks": 0,
        "tokens": 0,
    }

    # 打开 JSONL 文件
    with open(JSONL_PATH, "w", encoding="utf-8") as jsonl_f:
        for i, doc in enumerate(docs, 1):
            process_doc(conn, doc, cfg, jsonl_f, total_stats)

            # 每处理 10 篇文档提交一次，避免事务过大
            if i % 10 == 0:
                conn.commit()

        # 最后再提交一次
        conn.commit()

    # 输出统计
    print(f"\n{'='*60}")
    print(f"分块完成！统计汇总:")
    print(f"  总文档数:         {total_stats['docs']}")
    print(f"  成功处理:         {total_stats['docs_succeeded']}")
    print(f"  总 chunk 数:      {total_stats['chunks']}")
    print(f"  总预估 token 数:  {total_stats['tokens']}")
    print(f"  平均 chunk/文档:  {total_stats['chunks'] / max(total_stats['docs_succeeded'], 1):.1f}")

    # 验证
    db_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"\n验证:")
    print(f"  SQLite chunks 表行数: {db_count}")

    # JSONL 行数
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        jsonl_count = sum(1 for _ in f)
    print(f"  JSONL 文件行数:       {jsonl_count}")

    # 各领域分布
    print(f"\n各领域 chunk 分布:")
    domain_rows = conn.execute(
        "SELECT domain, COUNT(*) as cnt FROM chunks GROUP BY domain ORDER BY cnt DESC"
    ).fetchall()
    for domain, cnt in domain_rows:
        print(f"  {domain:<22}  {cnt:>6} chunks")

    conn.close()
    print(f"\nJSONL 文件: {JSONL_PATH}")


if __name__ == "__main__":
    main()