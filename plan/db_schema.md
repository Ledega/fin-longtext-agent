核心思路是：  
- 用一张“文档主表”统一管理五大领域和文件路径。  
- 用一张“chunk 表”承接所有切分后的文本块，并持久化。  
- 视领域再挂几张“领域特化表”（财报指标表、保险责任表等）。  
- 快速查询 chunk 靠：向量索引 + 过滤字段（domain/doc_id/chunk_type）+ 主键索引。  

下面按你给的目录结构具体说。 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/46346423/0c0c7dd5-f857-48e8-b511-dfcd03f05635/tree.txt?AWSAccessKeyId=ASIA2F3EMEYE7P7FUG7C&Signature=o18fLKK1C0KSmapFs411AYi8hG0%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEEgaCXVzLWVhc3QtMSJIMEYCIQDylJ%2BrHpc2lWwtE2BSc%2FRby0l04Nz0POl2n%2F9tk%2BFkrgIhAOj3U10d3jQqmYAVk4Yw%2FeqxYv15yPZp3mUeHDwsdCpHKvMECBAQARoMNjk5NzUzMzA5NzA1IgwJ6rBJsKf4GLtDvwUq0AQWcCiSPhpmTpckcStQSl1t%2Fa4uf%2BiR1I9rhZQ8OMKoi%2BsXj7PD54OaShLS2EdOvVmCoSVaMyH%2FR6vjp83DHaF7uSTRNaS2YwzOKjx4%2Baz5cF2u%2B6NcdOsxFTnBZzgGu6VW2rs2DQ1%2B79fMuOv98mGp144cB13gp81rYCKEGDevAdyjUZyT441UaJ7QSxssAN9R9r2vCHIGTX1%2FX6vADUuf%2BtjI6GyeRFFF8qsfHOIUYoOYC288H12TmBqILcu2SVC2uy9wKKT6VFA6ZXcj8CyzR5Z7VDX1FzXtE%2FteAgCUa%2BNEZO1TzD%2B64Zhg3EX%2FowkhbTsbd6h2wL5XKTz8y5qOKEifVy4zPHC%2FnoYVsQn%2FVc4189wSVJYuW4ejh%2FcKdiqiP5h0h5QGHF69kzfx2QnbbbrDS%2FxQ4tTLLtPRUaNhM2ecvmkqLUjLY3vLj3Mm5q55Mk%2F2QeG73twhu6HeDn4lrY%2F3ZeASvNuWxCuOWgimof9PZ5xJlZ%2FTOB%2BK%2F3iQtIm12ldU9pvAFnw8PLBOgcfvVcKfvJsMnxm5sZdyllboD%2FgAPONquFTRlXyEMimKWSGSW%2Bx9uQOJVtauDXfCREtxqdGnflKqrQjojqTsISSk7QqDhDI9tvOQPDlr9o%2F7v9TzHUtknOIf2UciS%2BDNVNm84I1SBrFdKU0XqLb7Gol0e%2Bf02nCtUpeHh3sZ%2BoR094N1V7UjQDKIuONDZgIwhXjogNBiWmDen59nb6%2BmaT6CjcLbBn7%2FYeo3XGJHDVi4%2FZhV6p%2FSh76e8gtlWNan39YLMIHfrtEGOpcBbgtaPipkH%2Bxoyg5LGg2DjIPQfNNrxruOsKTsjQIXS%2BRI6vqDe7RbySAwQiPEB4o3qjKSKpnq7AqVlanODIqIa%2BQPVlQyheaJD0HW%2FSulvM9OZmfxJWBg74J6T5g6e%2F3dmgAEV8kaBh0SpMcZ8CLwR7QLlNCbQ27rWLcWn6SDuDZgZkj8dJdh9aZA1H2%2Fixv4Z4FksNxnOg%3D%3D&Expires=1781251412)

***

## 1. 先回答三个问题

1）怎么建表、主次关系？  
- 主表：`docs`（文档级元数据）。  
- 核心子表：`chunks`（所有文本块）。  
- 领域子表（可选）：`financial_report_metrics`、`insurance_rules`、`contract_terms` 等，针对强结构信息。  

3）如何快速查到对应 chunk？  
- 在线检索：向量库按 `domain`/`doc_id` 过滤 + embedding 检索。  
- 溯源/定位：用 `chunk_id` 或 `(doc_id, page_no, local_idx)` 主键在 `chunks` 中 O(1) 查。

***

## 2. 文档主表：docs（最高优先级）

根据你当前目录，主表可以长这样（关系型或 CSV 都行）：

```sql
CREATE TABLE docs (
  doc_id        TEXT PRIMARY KEY,      -- 如 'annual_byd_2024_report', 'csrc_0001', 'csrc_0001_att1'
  domain        TEXT NOT NULL,         -- 'financial_reports', 'financial_contracts', 'insurance', 'regulatory', 'research'
  split         TEXT,                  -- 'A' / 'B'，可选
  title         TEXT,                  -- 人类可读标题
  file_path     TEXT NOT NULL,         -- 如 'financial_reports/annual_byd_2024_report.PDF'
  source_type   TEXT NOT NULL,         -- 'pdf' / 'html'
  parent_doc_id TEXT,                  -- 对监管 attachments: 比如 'csrc_0001_att1' 的 parent 是 'csrc_0001'
  pages         INT                    -- 页数，可选
);
```

- `financial_contracts/text01.pdf` → `doc_id = 'fc_text01'`, `domain = 'financial_contracts'`。 
- `financial_reports/annual_byd_2024_report.PDF` → `doc_id = 'annual_byd_2024_report'`, `domain = 'financial_reports'`。 
- `insurance/1.pdf` → `doc_id = 'ins_1'`, `domain = 'insurance'`。 
- 监管：  
  - `regulatory/html/csrc_0001.html` → `doc_id = 'csrc_0001'`, `domain = 'regulatory'`。 
  - `regulatory/attachments/csrc_0001_att1.pdf` → `doc_id = 'csrc_0001_att1'`, `parent_doc_id = 'csrc_0001'`。 

这张表是“一切的根”：题目里的 `doc_ids` 就要和这里对得上，后续任何检索都先按 `domain` 和 `doc_id` 过滤。

***

## 3. chunk 表：chunks（第二优先级，RAG 主数据）

统一放五个领域所有 chunk，不拆库，这样向量索引和 BM25 都只对一张表建。推荐 schema：

```sql
CREATE TABLE chunks (
  chunk_id      TEXT PRIMARY KEY,      -- 全局唯一，如 '{doc_id}_p{page}_c{idx}'
  doc_id        TEXT NOT NULL REFERENCES docs(doc_id),
  domain        TEXT NOT NULL,
  page_no       INT,
  section_path  TEXT[],                -- ['第四节 财务报表','合并利润表']；regulatory 可是 ['第一章','总则']
  clause_no     TEXT,                  -- '第四十七条' 等；非条款类可为 NULL
  chunk_type    TEXT,                  -- 'header'/'paragraph'/'clause'/'table'/'list'...
  text          TEXT NOT NULL,
  char_len      INT,
  approx_tokens INT
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE INDEX idx_chunks_domain ON chunks(domain);
CREATE INDEX idx_chunks_doc_type ON chunks(doc_id, chunk_type);
```

- 切分策略前面讲过：按段落/条款为基本单位，限制长度，必要时拆分 + overlap。  
- `section_path`/`clause_no` 对保险/监管/合同非常有用，后续可以做“只检索某章/某条”的过滤。

**持久化方式**：  
- 要么就真建 DB 表（Postgres + JSONB/pgvector），  
- 要么用 `chunks.jsonl` 承载同样字段，再由索引工具读进去建向量库。核心是：**不要只存在内存或只存在向量库里**，否则重跑/调参非常痛苦。

***

## 4. 领域特化表（第三优先级，可按需做）

这些是“锦上添花”，不是第一版必须全部搞定，但财报和保险值得优先做。

### 4.1 财报指标表：financial_report_metrics

```sql
CREATE TABLE financial_report_metrics (
  id              SERIAL PRIMARY KEY,
  doc_id          TEXT NOT NULL REFERENCES docs(doc_id),
  page_no         INT,
  table_id        TEXT,
  report_type     TEXT,          -- 'profit_and_loss'/'balance_sheet'/'cash_flow'
  section_path    TEXT[],
  period          TEXT,          -- '2024-12-31' 或 '2024'
  metric_name     TEXT,          -- 原行名：营业收入
  metric_norm     TEXT,          -- 归一名：revenue
  unit            TEXT,          -- 元/万元/亿元
  raw_value       TEXT,
  value_num       DOUBLE PRECISION
);
CREATE INDEX idx_frm_doc_metric ON financial_report_metrics(doc_id, metric_norm, period);
```

财报题时，先在这张表查 `doc_id + metric_norm + period` 拿数值，再回 `chunks` 查附近说明段落做语义确认。 [github](https://github.com/lazyaccountant/FinTable)

### 4.2 保险/合同条款表：insurance_rules / contract_terms

类似前面说的 `contracts_ins_tables.csv`，可以拆为两张或一张：

```sql
CREATE TABLE insurance_rules (
  id              SERIAL PRIMARY KEY,
  doc_id          TEXT NOT NULL REFERENCES docs(doc_id),
  page_no         INT,
  clause_no       TEXT,
  section_path    TEXT[],
  item_name       TEXT,      -- 计划A，身故责任
  condition       TEXT,      -- 触发条件
  coverage_type   TEXT,      -- death_benefit / surrender_value ...
  formula_text    TEXT,
  amount_num      DOUBLE PRECISION,
  currency        TEXT
);
```

这些表主要在 Step2 用来提高准确率，不必一开始就全面覆盖，但 schema 可以先定死。  

***

## 5. chunks 与索引的关系：怎么“快速查到对应 chunk”

### 5.1 RAG 检索路径

1）**在线检索用向量库 + filter**  
- 对 `chunks.text` 做 embedding，存到向量库（pgvector/Milvus/FAISS 等），只存：`embedding, chunk_id, doc_id, domain, chunk_type`。 [learn.microsoft](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-enrichment-phase)
- 查询时：  
  - 已知 domain：加 `domain = 'regulatory'` 过滤；  
  - A 榜已知 doc_ids：加 `doc_id in (...)` 过滤，只在这些文档内搜；  
  - B 榜无 doc_ids：先用题干在文档级做粗召回，再在候选 `doc_id` 列表内检索 chunk。  

2）**拿到 chunk_id → 回表**  
- 向量库返回一堆 `(chunk_id, score)`，你再到 `chunks` 表里用 PK 查 `text, section_path, page_no, clause_no`，拼成上下文给 Qwen。 [community.openai](https://community.openai.com/t/source-document-chunk-identification-and-highlighting-for-rag-usecase/883302)

### 5.2 手动定位 / 溯源

- 你在日志里记录“这道题最终用到了哪些 chunk_id”，要追查时可以 `SELECT * FROM chunks WHERE chunk_id = ...`，直接定位到页码和章节。  
- 如果后面要做“证据高亮”，也可以按 `doc_id + page_no` 再结合 chunk 内的 offset 定位到原 PDF/HTML。 [community.openai](https://community.openai.com/t/source-document-chunk-identification-and-highlighting-for-rag-usecase/883302)

***

## 6. 主次关系总结成一句话

- **第一优先级**：`docs`（统一文档 ID 和领域），`chunks`（所有文本块 + 元数据 + 持久化）。  
- **第二优先级**：向量索引 + BM25 索引，围绕 `chunks` 搭检索。  
- **第三优先级**：领域特化表，用于结构化数值和规则（财报指标、保险责任、合同条款），提高准确率和可解释性。  

按这个结构，你后面无论是改 chunk 策略、换向量库、还是加新领域，都不会推翻已有存量，只是在现有 schema 上增量迭代。