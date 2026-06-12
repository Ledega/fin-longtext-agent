阅读plan\db_schema.md，忽略其中的领域特化表。
需要chunk的文档已经放在dataset\public_dataset_upload\raw下，按照五大domain分了五个文件夹，每个文件夹中存放了该domain下的所有文档，pdf格式。
根据db_schema.md中的schema，编写代码，创建文档主表docs的sqlite表；以及chunk 表chunks，按照jsonl的格式持久化到本地。
先按自然段（空行、缩进）或条款号分段，每一段作为初级 chunk。chunk 上限在800 字，超过就拆成多个子 chunk，并在相邻 chunk 之间加一点 overlap，比例在chunk的15%左右。
不要做向量化，切分完持久化到本地即可。