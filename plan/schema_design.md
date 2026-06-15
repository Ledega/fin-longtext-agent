# 📐 fin-longtext-agent Schema 设计文档

> 版本: 0.1.0  
> 说明: 记录当前系统所有已落地的数据结构、索引格式、检索结果结构以及题目/答案格式。  
> 标注 `[已实现]` 为当前已落地的结构，`[计划中]` 为设计阶段/待实现。

---

## 1. SQLite 数据库（`data/fin_longtext.db`）

### 1.1 docs 表 — 文档主表 `[已实现]`

```sql
CREATE TABLE IF NOT EXISTS docs (
    doc_id        TEXT PRIMARY KEY,      -- 文档唯一标识符
    domain        TEXT NOT NULL,         -- 所属领域
    split         TEXT,                  -- 榜单分组: 'A' / 'B'，可选
    title         TEXT,                  -- 人类可读标题（文件名去除扩展名）
    file_path     TEXT NOT NULL,         -- 相对 data_root 的文件路径
    source_type   TEXT NOT NULL,         -- 源文件类型: 'pdf' / 'html' / 'txt'
    parent_doc_id TEXT,                  -- 监管附件场景: 指向主文档 doc_id
    pages         INT,                   -- PDF 页数（非 PDF 为 0）
    created_at    TEXT DEFAULT (datetime('now'))
);
```

#### 字段注解

| 字段 | 类型 | 约束 | 业务含义 | 示例值 |
|------|------|------|----------|--------|
| `doc_id` | TEXT | PK | **全局唯一文档 ID**，供题目 `doc_ids` 引用 | `fc_text01`, `byd_2024_annual`, `csrc_0001` |
| `domain` | TEXT | NOT NULL | **五大领域标识**，枚举值见下方 | `financial_contracts` |
| `split` | TEXT | 可选 | 所属榜单，当前全量 A 榜 | `A` |
| `title` | TEXT | 可选 | 人类可读展示名，取自文件名 stem | `text01`, `byd_2024_annual` |
| `file_path` | TEXT | NOT NULL | **关键定位字段**，相对于 `data_root` 的相对路径 | `raw/financial_contracts/text01.pdf` |
| `source_type` | TEXT | NOT NULL | 源格式，决定解析器选择 | `pdf`, `html`, `txt` |
| `parent_doc_id` | TEXT | 可为 NULL | 监管法规附件关联: `csrc_0001_att1` → `csrc_0001` | `csrc_0001` |
| `pages` | INT | 可为 NULL | 文档篇幅参考 | `42` |

#### domain 枚举

| 枚举值 | 领域名称 | 文档数 | doc_id 前缀 |
|--------|----------|--------|-------------|
| `insurance` | 保险条款 | 16 | `ins_` |
| `regulatory` | 监管法规 | 26 | (无前缀) |
| `financial_contracts` | 金融合同 | 14 | `fc_` |
| `financial_reports` | 财务报表 | 10 | (无前缀) |
| `research` | 行业研报 | 20 | (无前缀) |

#### doc_id 命名规则（`build_docs.py` 中 `derive_doc_id`）

| 领域 | 规则 | 示例 |
|------|------|------|
| financial_contracts | 文件名 stem 前加 `fc_` | `text01` → `fc_text01` |
| insurance | 文件名 stem 前加 `ins_` | `1` → `ins_1` |
| financial_reports | 直接用文件名 stem | `byd_2024_annual` → `byd_2024_annual` |
| regulatory html | 直接用文件名 stem | `csrc_0001` → `csrc_0001` |
| regulatory attachments | 直接用文件名 stem | `csrc_0001_att1` → `csrc_0001_att1` |
| research | 直接用文件名 stem | `pack2_text01` → `pack2_text01` |

---

### 1.2 chunks 表 — 文档分块表 `[已实现]`

```sql
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,      -- 全局唯一 chunk 标识符
    doc_id        TEXT NOT NULL REFERENCES docs(doc_id),  -- 所属文档
    domain        TEXT NOT NULL,         -- 所属领域（冗余字段，加速过滤）
    page_no       INT,                   -- 来源页码（从 1 开始）
    section_path  TEXT,                  -- 章节路径，JSON 数组字符串
    clause_no     TEXT,                  -- 条款号
    chunk_type    TEXT,                  -- chunk 内容类型
    text          TEXT NOT NULL,         -- chunk 文本内容
    char_len      INT,                   -- 字符数
    approx_tokens INT,                   -- 预估 token 数
    created_at    TEXT DEFAULT (datetime('now'))
);
```

#### 字段注解

| 字段 | 类型 | 约束 | 业务含义 | 示例值 |
|------|------|------|----------|--------|
| `chunk_id` | TEXT | PK | **全局唯一**，格式 `{doc_id}_p{page}_c{idx}` | `fc_text01_p1_c0` |
| `doc_id` | TEXT | FK→docs | 所属文档，用于 JOIN 和 doc_ids 过滤 | `fc_text01` |
| `domain` | TEXT | NOT NULL | **冗余字段**，避免 JOIN 即可 domain 过滤 | `financial_contracts` |
| `page_no` | INT | 可为 NULL | 页码，HTML 文档固定为 1 | `1`, `5` |
| `section_path` | TEXT | 可为 NULL | **JSON 数组字符串**，记录全文路径层级 | `["第四节 财务报表","合并利润表"]` |
| `clause_no` | TEXT | 可为 NULL | **条款号**，非条款类文档为 NULL | `第四十七条` |
| `chunk_type` | TEXT | 可为 NULL | **内容类型枚举**，详见下表 | `paragraph` |
| `text` | TEXT | NOT NULL | 分块文本，最大 800 字符（超长会拆分+overlap） | `"第十条 公司...` |
| `char_len` | INT | 可为 NULL | `len(text)`，用于统计 | `756` |
| `approx_tokens` | INT | 可为 NULL | `中文×1.5 + 英文/4 + 10` 估算 | `1145` |

#### chunk_type 枚举

| 枚举值 | 含义 | 判定逻辑 |
|--------|------|----------|
| `paragraph` | 普通段落 | 默认类型 |
| `clause` | 条款内容 | 匹配 `第X条` 或 `（X）` 开头 |
| `header` | 章节标题 | ≤2 行、无标点、<80 字符 |
| `table` | 表格数据 | ≥2 行含 `|` 且占比 > 30% |
| `list` | 列表项 | ≥2 行含数字/编号 marker 且占比 > 40% |

#### 辅助索引

```sql
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);        -- 按文档检索
CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(domain);     -- 按领域过滤
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_id, chunk_type); -- 按文档+类型过滤
```

---

### 1.3 领域特化表（计划中）`[计划中]`

以下表结构来自 `plan/db_schema.md`，当前**尚未实现**，仅作为 Schema 设计参考：

#### financial_report_metrics — 财报指标表

```sql
CREATE TABLE financial_report_metrics (
    id              SERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES docs(doc_id),
    page_no         INT,
    table_id        TEXT,
    report_type     TEXT,          -- 'profit_and_loss' / 'balance_sheet' / 'cash_flow'
    section_path    TEXT[],
    period          TEXT,          -- '2024-12-31' 或 '2024'
    metric_name     TEXT,          -- 原始行名：营业收入
    metric_norm     TEXT,          -- 归一化名：revenue
    unit            TEXT,          -- 元/万元/亿元
    raw_value       TEXT,
    value_num       DOUBLE PRECISION
);
```

#### insurance_rules — 保险规则表

```sql
CREATE TABLE insurance_rules (
    id              SERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES docs(doc_id),
    page_no         INT,
    clause_no       TEXT,
    section_path    TEXT[],
    item_name       TEXT,          -- 计划A，身故责任
    condition       TEXT,          -- 触发条件
    coverage_type   TEXT,          -- death_benefit / surrender_value ...
    formula_text    TEXT,
    amount_num      DOUBLE PRECISION,
    currency        TEXT
);
```

---

## 2. BM25 索引结构

### 2.1 目录布局 `[已实现]`

```
data/indices/bm25/
├── all/                        # 全库索引（86 份文档的所有 chunk）
│   ├── index.pkl              # bm25s 序列化索引
│   └── metadata.json           # chunk 元数据列表（与 index 位置一一对应）
├── insurance/                  # 保险条款领域
│   ├── index.pkl
│   └── metadata.json
├── regulatory/                 # 监管法规领域
├── financial_contracts/        # 金融合同领域
├── financial_reports/          # 财务报表领域
└── research/                   # 行业研报领域
```

### 2.2 metadata.json 格式 `[已实现]`

每条记录对应一个 chunk，数组顺序与 BM25 内部 corpus 顺序一致：

```json
{
  "chunk_id": "fc_text01_p1_c0",
  "doc_id": "fc_text01",
  "domain": "financial_contracts",
  "page_no": 1,
  "section_path": "[\"第一条\"]",
  "clause_no": "第一条",
  "chunk_type": "clause",
  "text": "为维护公司、股东及债权人的合法权益...",
  "approx_tokens": 142
}
```

#### 字段注解

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | str | 与 SQLite chunks.chunk_id 完全一致 |
| `doc_id` | str | 用于 `doc_ids` 白名单过滤 |
| `domain` | str | 用于 `domain` 领域过滤 |
| `page_no` | int \| null | 来源页码 |
| `section_path` | str | 章节路径（JSON 字符串），用于章节级过滤 |
| `clause_no` | str \| null | 条款号，用于条款级精确匹配 |
| `chunk_type` | str | 内容类型枚举 |
| `text` | str | 完整 chunk 文本，用于拼接上下文 |
| `approx_tokens` | int | token 估算值 |

### 2.3 index.pkl — bm25s 序列化格式 `[已实现]`

基于 `bm25s` 库，使用 **BM25-OKAPI (lucene)** 公式：

```python
index = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
index.index(tokenized_corpus)            # 构建倒排索引
index.save(str(index_dir))               # 序列化到 index.pkl
index = bm25s.BM25.load(str(index_dir))  # 反序列化加载
```

- 无 Embedding：只存倒排列表 + 文档长度 + IDF
- `k1=1.5`：词频饱和度
- `b=0.75`：文档长度归一化

### 2.4 自实现 BM25Index 类（备用方案）`[已实现，但未在线使用]`

位于 `src/indexing/build_bm25.py`，使用自实现的 `BM25Index` 类（非 bm25s 库）：

```python
class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.corpus_size = 0    # 文档数
        self.avgdl = 0.0        # 平均文档长度
        self.doc_freqs = []     # List[Counter]，每篇的词频
        self.idf = {}           # 词 → IDF
        self.doc_len = []       # 每篇文档长度
        self.chunk_ids = []     # 对应 chunk_id
        self.metadata = []      # 可选元数据
```

> 注意：当前 `retriever.py` 使用 `bm25s` 库索引，`build_bm25.py` 中的自实现 `BM25Index` **未接入在线推理管线**。两套 BM25 实现存在功能重叠，建议统一。

---

## 3. 检索结果结构

### 3.1 RetrievedChunk（检索输出）`[已实现]`

位于 `src/indexing/retriever.py`，dataclass 定义：

```python
@dataclass
class RetrievedChunk:
    chunk_id: str        # 全局唯一 chunk ID
    doc_id: str          # 所属文档
    domain: str          # 所属领域
    text: str            # chunk 文本
    page_no: int = 0     # 页码
    section_path: str = "[]"  # JSON 章节路径
    clause_no: str = ""       # 条款号
    chunk_type: str = "paragraph"  # 内容类型
    score: float = 0.0         # BM25 检索得分
    rank: int = 0              # 排名 (0-based)
    approx_tokens: int = 0     # 预估 token 数
```

#### to_dict() 序列化格式

```json
{
  "chunk_id": "fc_text01_p1_c0",
  "doc_id": "fc_text01",
  "domain": "financial_contracts",
  "text": "为维护公司、股东及债权人的合法权益...",
  "page_no": 1,
  "section_path": "[\"第一条\"]",
  "clause_no": "第一条",
  "chunk_type": "clause",
  "score": 12.3456,
  "rank": 0
}
```

### 3.2 build_query() 输出 `[已实现]`

位于 `src/indexing/retriever.py`：

```python
def build_query(question: dict) -> str:
    """拼接 题干 + 排序后的选项文本"""
```

**示例输入**：
```json
{
  "qid": "fc_a_001",
  "question": "根据募集说明书，下列哪些发行人权力受到限制？",
  "options": {"A": "质押资产", "B": "对外担保", "C": "分红", "D": "增发"}
}
```

**输出 query**：
```
根据募集说明书，下列哪些发行人权力受到限制？ 质押资产 对外担保 分红 增发
```

> ⚠️ 当前 query 仅简单拼接，**未做实体 boosting、关键词抽取或同义词扩展**，这是 P0 优化项。

---

## 4. 题目与答案结构

### 4.1 标准化题目 JSON（question_loader 产出）`[已实现]`

经过 `question_loader.py` 的 `normalize_question()` 标准化后：

```json
{
  "qid": "fc_a_001",                          // 题目 ID，如 "fc_a_001"
  "domain": "financial_contracts",            // 领域标识
  "split": "A",                               // 榜单: "A" / "B"
  "question": "根据募集说明书...",            // 题干文本
  "options": {                                // 选项字典
    "A": "质押资产",
    "B": "对外担保",
    "C": "分红",
    "D": "增发"
  },
  "answer_format": "multi",                   // 题型: "mcq" / "multi" / "tf"
  "type": "债券条款分析",                     // 原始题型描述
  "doc_ids": ["fc_text01", "fc_text02"]       // A 榜文档 ID 列表（B 榜可能为空）
}
```

#### answer_format 枚举

| 值 | 中文 | 输出格式 | 示例 |
|----|------|----------|------|
| `mcq` | 单选题 | 单个大写字母 | `A` |
| `multi` | 多选题 | 多个字母，排序去重 | `ACD` |
| `tf` | 判断题 | `A`(正确) 或 `B`(错误) | `A` |

### 4.2 中间结果结构 `[已实现]`

`process_single_question()` 返回的每道题结果：

```python
{
  "qid": "fc_a_001",
  "answer": "ACD",                    # 最终答案字符串
  "answer_format": "multi",           # 题型
  "prompt_tokens": 48500,             # 本次处理累加的 prompt tokens
  "completion_tokens": 120,           # 本次处理累加的 completion tokens
  "total_tokens": 48620               # 总计
}
```

### 4.3 answer.csv 输出格式 `[已实现]`

```
qid,answer,prompt_tokens,completion_tokens,total_tokens
summary,,3627557,629,3628186
fc_a_001,ACD,48500,120,48620
fc_a_002,B,32000,80,32080
...
```

| 列 | 类型 | 说明 |
|----|------|------|
| `qid` | str | 题目 ID，首行固定为 `summary` |
| `answer` | str | 答案字符串（summary 行为空） |
| `prompt_tokens` | int | 所有 prompt 输入 token 累加 |
| `completion_tokens` | int | 所有模型输出 token 累加 |
| `total_tokens` | int | 上述两者之和 |

---

## 5. 数据流转全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        离线预处理管线                                │
│                                                                     │
│  原始文档 (PDF/HTML/TXT)                                            │
│       │                                                             │
│       ▼ build_docs.py                                               │
│   docs 表 ────────────── 写入 ────── SQLite fin_longtext.db         │
│       │                                                             │
│       ▼ build_chunks.py + chunker.py                                │
│   chunks 表 ──────────── 写入 ────── SQLite + chunks.jsonl          │
│       │                                                             │
│       ▼ build_index.py (bm25s)                                      │
│   metadata.json ──────── 写入 ────── data/indices/bm25/{domain}/    │
│   index.pkl                                                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 运行时加载
┌─────────────────────────────────────────────────────────────────────┐
│                        在线问答管线                                  │
│                                                                     │
│   题目 JSON ──→ question_loader  ──→ 标准化题目 dict                │
│                                              │                      │
│   BM25 索引 ──→ BM25Retriever.load()  ──→ index + metadata          │
│                                              │                      │
│   标准化题目 ──→ build_context_for_question() ──→ context_text       │
│   + index+meta       │                                              │
│                      ├─ build_query(): 题干+选项 → query            │
│                      ├─ retriever.retrieve(): domain/doc_ids 过滤   │
│                      │    → BM25 检索 → top_k RetrievedChunk        │
│                      ├─ format_context(): 按 doc_id 分组 → 文本     │
│                      └─ 超限则 _compress_context: 截断 top-20       │
│                                                                     │
│   context_text ──→ build_prompt() ──→ 完整 prompt 字符串            │
│   + 标准化题目                                                      │
│                                                                     │
│   prompt ──→ client.call() ──→ raw_output + token_usage             │
│                                                                     │
│   raw_output ──→ process_answer() ──→ final_answer                  │
│                   ├─ extract_answer_from_text(): 末行正则           │
│                   ├─ _normalize_answer(): 按题型规范化              │
│                   ├─ 非法 → retry_with_fix(): 降温重试              │
│                   └─ 仍非法 → _get_fallback(): 回退                 │
│                                                                     │
│   final_answer ──→ batch_validate_results() ──→ write_answer_csv()  │
│   + token ↕                        └──→ answer.csv (summary + 题)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 评测指标公式

```
Accuracy            = CorrectAns / TotalQuestions
TokenBudget         = 5,000,000
TokenScore          = max(0, min(1, (5,000,000 - TotalTokens) / 5,000,000))
FinalScore          = 100 × Accuracy × (0.7 + 0.3 × TokenScore)
```

**排序规则**：FinalScore → Accuracy → TotalTokens(越小越好) → 提交时间

---

## 附录：文件引用映射

| 模块 | 文件 | 涉及 Schema |
|------|------|-------------|
| 数据库初始化 | `src/db/schema.py` | docs 表 + chunks 表 DDL |
| 文档入库 | `src/db/build_docs.py` | docs 表写入 |
| 分块入库 | `src/db/build_chunks.py` | chunks 表写入 + JSONL |
| 分块核心 | `src/db/chunker.py` | chunks 表字段生成 |
| BM25 构建(bm25s) | `src/indexing/build_index.py` | metadata.json + index.pkl |
| BM25 构建(自实现) | `src/indexing/build_bm25.py` | BM25Index pickle |
| 检索器 | `src/indexing/retriever.py` | RetrievedChunk + build_query |
| 上下文构建 | `src/qa/context_builder.py` | context_text 格式 |
| Prompt 模板 | `src/qa/prompt_templates.py` | format_context 格式 |
| 答案后处理 | `src/qa/post_processor.py` | 答案提取规范 |
| CSV 输出 | `src/qa/csv_writer.py` | answer.csv 格式 |
| 题目加载 | `src/qa/question_loader.py` | 标准化题目 JSON |
| Qwen 客户端 | `src/qa/qwen_client.py` | token_usage 格式 |