"""
扫描五大领域的原始文档，提取元数据并写入 docs 表。
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

    规则：
      - financial_contracts: 文件名 stem 前加 'fc_'，如 text01 → fc_text01
      - insurance: 文件名 stem 前加 'ins_', 如 1 → ins_1
      - financial_reports: 直接用文件名 stem，如 annual_byd_2024_report
      - regulatory html: 直接用文件名 stem，如 csrc_0001, csrc_0001_att1
      - research: 直接用文件名 stem，如 pack2_text01
    """
    stem = file_path.stem
    prefix = DOMAIN_PREFIX.get(domain, "")
    if prefix:
        # 避免重复加前缀
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
    """判断是否是 regulatory 的 parent stub 文件（html 目录下的主文档）"""
    if domain != "regulatory":
        return False
    return file_path.suffix.lower() == ".html"


def get_parent_doc_id(file_path: Path, domain: str) -> str | None:
    """
    获取 parent_doc_id

    regulatory attachments 目录下的文件，其 parent 是 html 目录中同 id 的文件
    例如 attachments/csrc_0001_att1.pdf → parent = 'csrc_0001'
    """
    if domain != "regulatory":
        return None
    parts = file_path.parts
    if "attachments" in parts:
        stem = file_path.stem
        # csrc_0001_att1 → csrc_0001
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

            # 获取相对路径（相对于 data_root）
            rel_path = file_path.relative_to(cfg.data_root)

            # 提取 title（用文件名，去掉扩展名）
            title = file_path.stem

            # 页数（仅 PDF）
            pages = get_pdf_page_count(file_path) if source_type == "pdf" else 0

            # parent_doc_id
            parent_doc_id = get_parent_doc_id(file_path, domain)

            # 判断 split（A 榜或 B 榜）— 目前全量都是 A 榜数据
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


def main():
    cfg = get_config()

    # SQLite 数据库路径（放在项目根目录）
    db_path = PROJECT_ROOT / "data" / "fin_longtext.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 初始化数据库
    init_db(str(db_path))

    # 连接并插入
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON;")

    print("开始扫描文档并插入 docs 表...")
    count = scan_and_insert(conn)
    print(f"完成！共插入/更新 {count} 条文档记录。")

    # 验证
    rows = conn.execute("SELECT doc_id, domain, file_path, pages FROM docs ORDER BY domain, doc_id").fetchall()
    print("\nDocs 表内容预览:")
    print(f"  {'doc_id':<30} {'domain':<22} {'pages':<6} {'file_path'}")
    print(f"  {'-'*30} {'-'*22} {'-'*6} {'-'*40}")
    for row in rows:
        print(f"  {row[0]:<30} {row[1]:<22} {str(row[2] or ''):<6} {row[3]}")

    conn.close()


if __name__ == "__main__":
    main()