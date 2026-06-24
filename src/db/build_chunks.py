"""
读取 docs 表中注册的文档，逐一解析文本、分块，结果写入：
1) SQLite chunks 表
2) chunks.jsonl 文件

⚠️ 已离线完成，当前不需要运行此脚本。
   保留代码仅作逻辑参考。
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
# ⚠️ chunk_document_from_file 已离线完成并注释，此处不再导入
# from src.db.chunker import chunk_document_from_file


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


if __name__ == "__main__":
    # ╔══════════════════════════════════════════════════════════╗
    # ║  离线分块处理已完成，不自动运行                        ║
    # ║  如确需重新分块，取消下方注释即可                     ║
    # ╚══════════════════════════════════════════════════════════╝
    pass
