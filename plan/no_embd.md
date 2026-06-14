可以搞定，不至于挂。思路就是从「向量 RAG」切到「纯稀疏检索 RAG」：BM25 / TF‑IDF + 结构化索引 + LLM 辅助检索。整体架构不变，只是“检索层”换实现。

下面按你现有的 pipeline 改造，说两个层面：怎么替代 embedding、以及对这个赛题要做哪些加强。

***

## 1. 总体思路：从 dense RAG 改成 lexical RAG

- 不再算 embedding，不再用向量库。  
- 检索全靠：倒排索引 + BM25 / TF‑IDF 打分，外加结构化字段过滤（domain、doc_id、section、clause_no）。 
- 依然保留：  
  - 文档主表 docs（doc_id / domain / path）；  
  - chunks 表（chunk_id / doc_id / text / section_path / clause_no / chunk_type）；  
  - 领域特化表（财报指标表、保险责任表等）。  

对金融长文档来说，很多实验本来就发现：BM25 在数字、术语密集文本上不比 dense 差，甚至更稳。

***

## 2. 检索层替代方案：BM25 / TF‑IDF + 领域过滤

### 索引构建

- 对 `chunks.text` 建 **倒排索引 + BM25**，可以选：  
  - Elasticsearch / OpenSearch；  
  - 或 Python 里 `rank_bm25` / `bm25s`（量不很大也够用）。
- 每条 doc 存 metadata：`domain, doc_id, chunk_type, section_path, clause_no`，支持 filter。  

### Query 组成

每道题检索时：

- 先用题目 JSON 字段做硬过滤：  
  - `domain = question.domain`；  
  - A 榜：`doc_id in question.doc_ids`；  
  - B 榜：先不限制 doc_id，只按 domain 搜，后面再做文档级 re-rank。  
- Query 文本：  
  - 用「题干 + 选项」拼一个查询串；  
  - 如有明确实体（公司名、年份、条款号），可以单独作为 must / boost 关键字。  

BM25 非常适合这种“术语+数字+条款号”的精确匹配场景。
***

## 3. 没有 embedding 怎么提高命中率？

关键靠三个招：

### 3.1 结构化字段 + 手动 query rewrite

- 领域/章节过滤：例如财报只搜 `section_path` 含“财务报表 / 合并利润表”的 chunk；监管只搜包含“第四十七条”“特别决议”这类关键词的条款段。  
- 增加同义词/别名词表：  
  - “营业收入”/“主营业务收入”；  
  - “股东大会特别决议”/“特别决议”。  
- 题目解析：手写一点规则从题干里抓公司名、年份、金额阈值、条款号，拼到 query 里做 boosting。  

### 3.2 LLM 辅助生成关键词（不算 embedding）

- 虽然不能用 embedding，但可以用 Qwen 帮你做「Query → 关键词」：  
  - 输入题干，要求输出一串中英文关键词、数字、条款号，用逗号分隔；  
  - 用这串 keywords 当 BM25 查询串，多语言/多形态同时覆盖。 
- Qwen-Agent 官方 RAG 文档里就有类似用法：SplitQueryThenGenKeyword + BM25。你可以自己手写一个轻量版。 

### 3.3 选项级检索

- 对每个选项单独构造 query，做一遍 BM25：  
  - 例如选项 A 里提到“资产负债率超过 70%”“担保金额占净资产 6.67%”，就拿这些关键词去搜条款；  
  - 返回的 top-k chunk 作为选项 A 的证据候选。  
- 最终把「题干 + 各选项对应的若干证据段」一并丢给 Qwen 做逐项判断。  

这能弥补“没有语义相似度，长段描述可能匹配不到”的一部分损失。 [unstructured](https://unstructured.io/blog/rethinking-rag-without-embeddings)

***

## 4. B 榜无 doc_ids：没有 embedding 怎么做文档级召回？

可以做两层 BM25：

1. **文档级索引**  
   - 给每个 doc 建一个「文档简介索引」，例如：  
     - 截取文档开头几段 + 章节标题拼成一个短文本；  
     - 对每个附件（regulatory/attachments）也建一条。  
   - 对题干做 BM25，在这些文档简介里召回 Top-N doc_id。  

2. **文档内 chunk 级检索**  
   - 确定候选 doc_ids 后，在对应文档的 chunk 子集上再跑 BM25，拿 top-k chunk 作为证据。  

3. **再加规则/LLM 帮判断 doc 是否相关**  
   - 可以让 Qwen 读“题干 + 文档标题/简介”，输出：这份 doc 是否高度相关（是/否 + 理由），作为 re-rank 信号。  

这本质是“纯 lexical、多级索引”的召回，不用任何 embedding， 而对金融文档这种“关键词+数字密集”的场景，效果其实不差。 [youtube](https://www.youtube.com/watch?v=4de5RVMcneU)

***

## 5. 你现有设计需要改哪些点？

- 保留：  
  - docs 表、chunks 表、财报指标/保险规则这些结构化表的 schema，不用动；  
  - Agent 流程：检索 → 聚证据 → Qwen CoT → 严格抽取选项字母。  
- 替换：  
  - 把“构建向量 + 向量检索”那一段全部换成 “BM25/倒排检索 + 字段过滤 + 选项级查询”； [app.daily](https://app.daily.dev/posts/beyond-vector-databases-rag-architectures-without-embeddings-q61cmk522)
  - 不再需要 embedding 持久化，只需要倒排索引/ES 索引持久化（速度很快）。  
- 新增：  
  - 一个“题干/选项 → 关键词串”的小模块（可以是手写规则 + LLM 辅助）；  
  - 一个文档级索引用于 B 榜 doc 召回。  

***

## 6. 心态层面

从 dense 换 lexical 不代表“回到石器时代”，尤其是金融这种高度结构化、术语固定的长文档；不少基准实验里 BM25 在这类场景甚至优于通用 dense 模型。 [chatpaper](https://chatpaper.com/paper/264067)

你真正要卷的变成：  
- **索引怎么切**（chunk 粒度、section/clause 元数据）；  
- **查询怎么写**（题干理解、关键词抽取、选项级检索）；  
- **证据怎么拼给 Qwen**（多文档、多段落联合推理）。  

如果你愿意，我可以帮你把“embedding 版 Step1 pipeline”改写成“BM25 版 Step1 pipeline”的伪代码框架，直接映射到你现在的项目结构。