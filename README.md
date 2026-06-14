# fin-longtext-agent

基于 **纯 BM25 倒排索引 + Qwen 系列模型** 的金融长文本 RAG 问答系统，覆盖保险条款、监管法规、金融合同、财务报表、行业研报五大领域。

## 架构

```
原始文档 (PDF/HTML)
    │
    ▼ src/db/
  文本抽取 + 段落切片 (800 char chunks) → SQLite docs + chunks 表
    │
    ▼ src/indexing/
  jieba 中文分词 + BM25 倒排索引 → data/indices/bm25/
    │
    ▼ src/qa/
  题目加载 → BM25 检索 (domain + doc_id 硬过滤) → 上下文拼接
    │
    ▼ Qwen API (langchain-openai 兼容模式)
  Prompt 模板 (mcq / multi / tf) → 思维链推理 → 答案提取
    │
    ▼
  post_processor (排序去重、降温重试) → answer.csv
```

## 目录结构

```
fin-longtext-agent/
├── main.py                         # 主入口：一键运行完整管线
├── config/settings.yaml            # 数据集路径、模型参数
├── pyproject.toml                  # 依赖管理
│
├── src/
│   ├── config_loader.py            # 配置加载 + .env 自动读取
│   ├── db/
│   │   ├── schema.py               # SQLite DDL (docs + chunks 表)
│   │   ├── build_docs.py           # 扫描文档 → docs 表
│   │   ├── build_chunks.py         # 切片 → chunks 表 + chunks.jsonl
│   │   └── chunker.py              # 分块核心逻辑 (按段落/条款)
│   ├── indexing/
│   │   ├── build_index.py          # BM25 索引构建 (bm25s)
│   │   ├── retriever.py            # BM25 检索器 (domain/doc_id 过滤)
│   │   └── finance_dict.txt        # jieba 金融自定义词典
│   └── qa/
│       ├── question_loader.py      # 题目 JSON 加载与标准化
│       ├── prompt_templates.py     # 3 套 Prompt 模板 (mcq/multi/tf)
│       ├── qwen_client.py          # Qwen API 封装 (ChatOpenAI 兼容)
│       ├── context_builder.py      # 检索上下文拼装 + Token 控制
│       ├── post_processor.py       # 答案提取 + 降温重试 + fallback
│       └── csv_writer.py           # answer.csv 输出
│
├── data/
│   ├── fin_longtext.db             # SQLite 数据库
│   ├── chunks.jsonl                # Chunk 持久化
│   └── indices/bm25/               # BM25 索引文件
│
├── dataset/                        # 原始数据集 (不提交)
└── plan/plan.md                    # 解题计划
```

## 快速开始

### 1. 安装依赖

```bash
# Python >= 3.12
pip install uv
uv sync
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY
```

### 3. 准备数据

确保 `dataset/public_dataset_upload/raw/` 下有五个领域的原始文档，运行：

```bash
python src/db/build_docs.py        # 扫描文档 → docs 表
python src/db/build_chunks.py      # 切片 → chunks 表
python src/indexing/build_index.py  # 构建 BM25 索引
```

### 4. 运行问答管线

```bash
# 全量运行
python main.py --split A

# 单领域测试
python main.py --split A --domain regulatory

# Dry run (不调 API，验证流程)
python main.py --split A --dry-run
```

输出 `answer.csv`，格式：

| qid | answer | prompt_tokens | completion_tokens | total_tokens |
|-----|--------|---------------|-------------------|--------------|
| summary | | 204286 | 176390 | 380676 |
| fc_a_001 | ABD | 2503 | 761 | 3264 |
| ... | ... | ... | ... | ... |

## 技术特点

- **纯 BM25 索引**：不依赖 embedding 模型
- **中文分词优化**：jieba + 120+ 金融术语自定义词典
- **端到端答案处理**：正则提取末行 → 排序去重 → 非法时降温重试
- **Token 统计完整**：每道题和 summary 行记录 prompt/completion/total tokens
- **A/B 榜双路径**：A 榜用 doc_ids 硬过滤，B 榜文档级 re-rank