"""
SQLite 建表 DDL：docs（文档主表）+ chunks（切分数据表）
"""

# 文档主表
CREATE_DOCS_TABLE = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id        TEXT PRIMARY KEY,      -- 如 'fc_text01', 'annual_byd_2024_report'
    domain        TEXT NOT NULL,         -- 'financial_reports', 'financial_contracts', 'insurance', 'regulatory', 'research'
    split         TEXT,                  -- 'A' / 'B'，可选
    title         TEXT,                  -- 人类可读标题
    file_path     TEXT NOT NULL,         -- 相对路径，如 'raw/financial_contracts/text01.pdf'
    source_type   TEXT NOT NULL,         -- 'pdf' / 'html'
    parent_doc_id TEXT,                  -- 对监管 attachments 的父文档
    pages         INT,                   -- 页数
    created_at    TEXT DEFAULT (datetime('now'))
);
"""

# Chunk 表
CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,      -- 全局唯一，如 'fc_text01_p{page}_c{idx}'
    doc_id        TEXT NOT NULL REFERENCES docs(doc_id),
    domain        TEXT NOT NULL,
    page_no       INT,
    section_path  TEXT,                  -- JSON 数组字符串，如 '["第四节 财务报表","合并利润表"]'
    clause_no     TEXT,                  -- '第四十七条' 等；非条款类可为 NULL
    chunk_type    TEXT,                  -- 'paragraph'/'clause'/'table'/'list'/'header'
    text          TEXT NOT NULL,
    char_len      INT,
    approx_tokens INT,
    created_at    TEXT DEFAULT (datetime('now'))
);
"""

# 索引
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(domain);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_id, chunk_type);",
]


def get_all_ddl() -> list[str]:
    """获取所有 DDL 语句"""
    return [CREATE_DOCS_TABLE, CREATE_CHUNKS_TABLE, *CREATE_INDEXES]


def init_db(db_path: str) -> None:
    """初始化数据库，创建所有表"""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    for ddl in get_all_ddl():
        conn.execute(ddl)
    conn.commit()
    conn.close()
    print(f"数据库初始化完成: {db_path}")