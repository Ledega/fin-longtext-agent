阅读plan\db_schema.md，忽略其中的领域特化表。
需要chunk的文档已经放在dataset\public_dataset_upload\raw下，按照五大domain分了五个文件夹，每个文件夹中存放了该domain下的所有文档，pdf格式。
根据db_schema.md中的schema，编写代码，创建文档主表docs的sqlite表；以及chunk 表chunks，按照jsonl的格式持久化到本地。
先按自然段（空行、缩进）或条款号分段，每一段作为初级 chunk。chunk 上限在800 字，超过就拆成多个子 chunk，并在相邻 chunk 之间加一点 overlap，比例在chunk的15%左右。
不要做向量化，切分完持久化到本地即可。

---

这个项目是为了@/task/afac_4.md 这个赛题的 当前 我们已经按照@/plan/plan.md 将dataset/public_dataset_upload/raw下的文档切成了 chunks 持久化到了本地的@/data/chunks.jsonl ，并且创建文档主表docs的sqlite表。接下来，请你准备将这些 chunks 向量化，存入向量数据库。不需要实现代码，给出方案即可。可参考@/plan/plan.md 中的“2. 建索引：BM25 + 向量 + 领域过滤“

---

问题的输入在dataset/public_dataset_upload/questions/group_a下，按照 5 大领域进行了分类，分为 5 个 json，各个字段的属性在@task/afac_4.md里都有，需要提交的 answer.csv 的格式和要求在里面也有。接下来，请你设计方案， 正确解析题目输入，从题目 JSON 获取 doc_ids，从这些文档中检索与题干关键词（公司名、年份、指标、金额等）最相关的若干 chunk（例如每个 doc 取 Top-k=5）。 把若干 chunk 组成上下文，连同题干和选项一起作为 prompt 喂给 Qwen，使用一个通用 QA 模板。 在 prompt 中明确：先在“思考区域”一步步分析，但最终只在最后一行输出选项字母（单选 1 个字母，多选若干排序好的字母，判断题 A/B）。可以考虑，针对三类题型设计三个 prompt 模板，但共用一条原则：思维链在中间，最后单独一行输出答案字母；在系统层用正则只抽取最后一行写入 answer.csv。在 post-processing 中做：校验是否只包含合法字母（A–D）；多选题对字母排序、去重；若不合法可以触发一次“降温重试”或回退策略。
同时记录每次 API 调用的 prompt_tokens、completion_tokens，并汇总出 summary 行中的 total_tokens，以满足评测对 Token 统计的要求。随后输出符合要求格式的answer.csv。先给出方案，不用编码。

---

请使用如下的方式调用 qwen api。chat_model = ChatOpenAI(
    model="qwen3.7-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
) 修改代码 不要运行

---

这道题目不支持任何embedding，所以取消 embedding 的使用，改为对 chunks.text 建 倒排索引 + BM25，可以选Elasticsearch / OpenSearch；或 Python 里 rank_bm25 / bm25s（量不很大也够用）。每条 doc 存 metadata：domain, doc_id, chunk_type, section_path, clause_no，支持 filter。每道题检索时：先用题目 JSON 字段做硬过滤：domain = question.domain；A 榜：doc_id in question.doc_ids；B 榜：先不限制 doc_id，只按 domain 搜，后面再做文档级 re-rank。Query 的文本：用「题干 + 选项」拼一个查询串；如有明确实体（公司名、年份、条款号），可以单独作为 must / boost 关键字。请以此方案重构代码，不要跑测试。