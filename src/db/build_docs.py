"""
扫描五大领域的原始文档，提取元数据并写入 docs 表。

⚠️ 已离线完成，当前不需要运行此脚本。
   保留代码仅作逻辑参考。
"""

import sys
import sqlite3
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import get_config
from src.db.schema import init_db

# 领域 → doc_id 前缀映射
DOMAIN_PREFIX = {
    "financial_contracts": "fc_",
    "financial_reports": "",
    "insurance": "ins_",
    "regulatory": "",
    "research": "",
}


def derive_doc_id(file_path: Path, domain: str) -> str:
    """
    根据文件路径和领域推导 doc_id（遵循 db_schema 规范）
    """
    stem = file_path.stem
    prefix = DOMAIN_PREFIX.get(domain, "")
    if prefix:
        if not stem.startswith(prefix):
            return f"{prefix}{stem}"
    return stem


def get_pdf_page_count(file_path: Path) -> int:
    """获取 PDF 页数"""
    try:
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        with open(file_path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            return len(list(doc.get_pages()))
    except Exception:
        return 0


def get_source_type(file_path: Path) -> str:
    """判断源文件类型"""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext == ".html":
        return "html"
    elif ext == ".txt":
        return "txt"
    return ext.lstrip(".")


def is_parent_stub(file_path: Path, domain: str) -> bool:
    """判断是否是 regulatory 的 parent stub 文件"""
    if domain != "regulatory":
        return False
    return file_path.suffix.lower() == ".html"


def get_parent_doc_id(file_path: Path, domain: str) -> str | None:
    """
    获取 parent_doc_id
    """
    if domain != "regulatory":
        return None
    parts = file_path.parts
    if "attachments" in parts:
        stem = file_path.stem
        idx = stem.find("_att")
        if idx != -1:
            return stem[:idx]
    return None


def scan_and_insert(conn: sqlite3.Connection) -> int:
    """扫描所有领域的文档并插入 docs 表，返回插入数量"""
    cfg = get_config()
    inserted = 0

    for domain in cfg.get_all_domains():
        print(f"  扫描领域: {domain}")
        doc_paths = cfg.get_domain_doc_paths(domain)

        for file_path in doc_paths:
            if not file_path.exists():
                continue

            doc_id = derive_doc_id(file_path, domain)
            source_type = get_source_type(file_path)

            rel_path = file_path.relative_to(cfg.data_root)
            title = file_path.stem
            pages = get_pdf_page_count(file_path) if source_type == "pdf" else 0
            parent_doc_id = get_parent_doc_id(file_path, domain)
            split = "A"

            try:
                conn.execute(
                    """INSERT OR REPLACE INTO docs
                       (doc_id, domain, split, title, file_path, source_type, parent_doc_id, pages)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc_id, domain, split, title, str(rel_path), source_type, parent_doc_id, pages),
                )
                inserted += 1
            except Exception as e:
                print(f"    [错误] doc_id={doc_id}: {e}")

    conn.commit()
    return inserted


if __name__ == "__main__":
    # ╔══════════════════════════════════════════════════════════╗
    # ║  离线文档扫描已完成，不自动运行                        ║
    # ║  如确需重新扫描，取消下方注释即可                     ║
    # ╚══════════════════════════════════════════════════════════╝
    pass
    # cfg = get_config()
    # db_path = PROJECT_ROOT / "data" / "fin_longtext.db"
    # db_path.parent.mkdir(parents=True, exist_ok=True)
    # init_db(str(db_path))
    # conn = sqlite3.connect(str(db_path))
    # conn.execute("PRAGMA foreign_keys=ON;")
    # print("开始扫描文档并插入 docs 表...")
    # count = scan_and_insert(conn)
    # print(f"完成！共插入/更新 {count} 条文档记录。")
    # rows = conn.execute("SELECT doc_id, domain, file_path, pages FROM docs ORDER BY domain, doc_id").fetchall()
    # print("\nDocs 表内容预览:")
    # for row in rows:
    #     print(f"  {row[0]:<30} {row[1]:<22} {str(row[2] or ''):<6} {row[3]}")
    # conn.close()
