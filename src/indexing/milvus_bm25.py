"""
Milvus 3.0 服务端原生 BM25 索引模块

基于 pymilvus >= 3.0.0，**纯稀疏向量**，**不使用任何 Embedding 模型**。

功能：
- create_collection()      — 建 Collection + Schema + Function + Index
- insert_chunks()          — 遍历 data/chunks/*.jsonl → batch insert
- search_bm25()            — BM25 检索入口
- search_bm25_multi_doc()  — 支持多 doc_id + domain 同时过滤（专门给 workflow.py）
- drop_collection()        — 删除 Collection

服务端 BM25 原理：
    Milvus 3.0 FunctionType.BM25 在服务端自动对 text 字段做中文分词
    → 生成 BM25 稀疏向量存入 sparse_bm25 字段，客户端无需传向量。

红线：本文件**绝对不含 FloatVector（稠密向量）**相关字段或逻辑。
"""

import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from pymilvus import (
    MilvusClient,
    DataType,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

# 默认值（可通过 config/settings.yaml 覆盖）
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_COLLECTION_NAME = "fin_longtext_chunks"
DEFAULT_BATCH_SIZE = 1000

# Collection Schema 定义
COLLECTION_NAME = "fin_longtext_chunks"

# 字段名常量
FIELD_ID = "id"                # PK, auto_id
FIELD_DOC_ID = "doc_id"       # 文档 ID
FIELD_DOMAIN = "domain"       # 领域
FIELD_TEXT = "text"           # 文本（中文 analyzer + enable_match）
FIELD_SPARSE = "sparse_bm25"  # 服务端 BM25 稀疏向量


# ──────────────────────────────────────────────
# ChunkRow（与 workflow.py 对齐的结构）
# ──────────────────────────────────────────────

@dataclass
class ChunkRow:
    """
    单条 BM25 检索结果，与 src/qa/workflow.py 的 ChunkRow 完全对齐。
    用于 search_bm25_multi_doc() 的返回值。
    """
    id: int = 0
    chunk_id: str = ""
    doc_id: str = ""
    domain: str = ""
    text: str = ""
    heading_path: List[str] = field(default_factory=list)
    chunk_type: str = "paragraph"
    distance: float = 0.0


# ──────────────────────────────────────────────
# Collection Schema
# ──────────────────────────────────────────────

def get_collection_schema() -> dict:
    """
    获取 Collection Schema 定义。
    用于 MilvusClient.create_collection()。
    """
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )

    # id — 主键（auto_id）
    schema.add_field(
        field_name=FIELD_ID,
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    # doc_id — 文档 ID（用于过滤/溯源）
    schema.add_field(
        field_name=FIELD_DOC_ID,
        datatype=DataType.VARCHAR,
        max_length=256,
    )

    # domain — 领域（用于过滤）
    schema.add_field(
        field_name=FIELD_DOMAIN,
        datatype=DataType.VARCHAR,
        max_length=64,
    )

    # text — 待索引文本（中文 analyzer + enable_match）
    schema.add_field(
        field_name=FIELD_TEXT,
        datatype=DataType.VARCHAR,
        max_length=65535,
        enable_match=True,           # 支持 TEXT_MATCH 表达式过滤
        analyzer_params={"type": "chinese"},  # 中文分析器
    )

    # sparse_bm25 — 服务端 BM25 稀疏向量（由 Function 自动填充）
    schema.add_field(
        field_name=FIELD_SPARSE,
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )

    # Function: text → sparse_bm25 服务端自动转换
    schema.add_function(
        function_name="bm25_func",
        function_type=FunctionType.BM25,
        input_field_names=[FIELD_TEXT],
        output_field_names=[FIELD_SPARSE],
    )

    return schema


def get_index_params() -> dict:
    """
    为 sparse_bm25 建立索引的参数。
    使用 SPARSE_INVERTED_INDEX + metric_type=BM25。
    """
    return {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
        "params": {},
    }


# ──────────────────────────────────────────────
# 创建 / 获取 Collection
# ──────────────────────────────────────────────

def create_collection(
    client: MilvusClient,
    collection_name: str = COLLECTION_NAME,
    overwrite: bool = False,
) -> None:
    """
    创建集合（Collection）+ Schema + Function + Index。

    Args:
        client: MilvusClient 实例
        collection_name: 集合名
        overwrite: 如果已存在是否删除重建
    """
    if client.has_collection(collection_name):
        if not overwrite:
            logger.info(f"Collection '{collection_name}' 已存在，跳过创建")
            return
        logger.info(f"删除现有 Collection '{collection_name}'...")
        client.drop_collection(collection_name)

    schema = get_collection_schema()

    logger.info(f"创建 Collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    # 为 sparse_bm25 创建索引
    index_params = get_index_params()
    logger.info(
        f"创建索引: {index_params['index_type']}, "
        f"metric={index_params['metric_type']}..."
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
        field_name=FIELD_SPARSE,
    )

    # 索引加载（使索引生效）
    client.load_collection(collection_name)
    logger.info(f"Collection '{collection_name}' 就绪 ✓")


def get_or_create_collection(
    client: MilvusClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    获取已有集合引用，不存在则创建。

    Args:
        client: MilvusClient 实例
        collection_name: 集合名
    """
    if not client.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' 不存在，自动创建")
        create_collection(client, collection_name, overwrite=False)
    else:
        # 确保索引就绪
        client.load_collection(collection_name)
        logger.info(f"Collection '{collection_name}' 已就绪")


# ──────────────────────────────────────────────
# 数据导入
# ──────────────────────────────────────────────

def _load_jsonl_chunks(jsonl_path: Path) -> List[dict]:
    """加载单个 JSONL 文件的所有 chunk"""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
    return chunks


def insert_chunks(
    client: MilvusClient,
    collection_name: str = COLLECTION_NAME,
    chunks_dir: Path = CHUNKS_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    遍历 data/chunks/*.jsonl，逐领域读取 chunk 并批量插入 Milvus。

    **稀疏向量由服务端 FunctionType.BM25 自动计算，客户端只传 text。**

    Args:
        client: MilvusClient 实例
        collection_name: 目标集合名
        chunks_dir: chunks JSONL 目录
        batch_size: 每批插入条数

    Returns:
        统计信息 dict: {"total": int, "errors": int, "per_domain": {...}}
    """
    stats: Dict[str, Any] = {
        "total": 0,
        "errors": 0,
        "per_domain": {},
    }

    jsonl_files = sorted(chunks_dir.glob("*_chunks.jsonl"))
    if not jsonl_files:
        logger.warning(f"未找到任何 *_chunks.jsonl 文件: {chunks_dir}")
        return stats

    logger.info(f"找到 {len(jsonl_files)} 个 JSONL 文件")

    for jsonl_path in jsonl_files:
        # 从文件名提取领域: "insurance_chunks.jsonl" → "insurance"
        domain = jsonl_path.stem.replace("_chunks", "")

        chunks = _load_jsonl_chunks(jsonl_path)
        logger.info(f"[{domain}] 加载 {len(chunks)} chunks")

        domain_success = 0
        domain_errors = 0

        # 批量插入
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # 构造插入数据：只传 doc_id, domain, text 三列
            # sparse_bm25 由服务端 Function 自动填充
            # id 由 auto_id 自动生成
            insert_data = []
            for c in batch:
                doc_id = c.get("doc_id", "")
                text = c.get("text", "")
                if not text:
                    domain_errors += 1
                    continue

                insert_data.append({
                    FIELD_DOC_ID: doc_id,
                    FIELD_DOMAIN: domain,
                    FIELD_TEXT: text,
                })

            if not insert_data:
                continue

            try:
                result = client.insert(
                    collection_name=collection_name,
                    data=insert_data,
                )
                success_count = len(insert_data)
                domain_success += success_count
            except Exception as e:
                logger.error(f"插入失败 (batch {i//batch_size}): {e}")
                domain_errors += len(insert_data)

        stats["total"] += domain_success
        stats["errors"] += domain_errors
        stats["per_domain"][domain] = {
            "total": len(chunks),
            "inserted": domain_success,
            "errors": domain_errors,
        }

        logger.info(
            f"[{domain}] 插入完成: {domain_success}/{len(chunks)} "
            f"(errors={domain_errors})"
        )

    logger.info(f"插入汇总: total={stats['total']}, errors={stats['errors']}")
    return stats


# ──────────────────────────────────────────────
# BM25 检索（单 doc_id 版，保留向后兼容）
# ──────────────────────────────────────────────

def search_bm25(
    client: MilvusClient,
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 20,
    domain_filter: Optional[str] = None,
    doc_id_filter: Optional[str] = None,
    output_fields: Optional[List[str]] = None,
) -> List[Dict]:
    """
    BM25 检索入口（单 doc_id 版）。

    使用 AnnSearchRequest 构建 BM25 稀疏向量检索请求。
    配合 RRFRanker 做多路召回（当前仅一路 BM25，保留扩展性）。

    Args:
        client: MilvusClient 实例
        query: 检索 query 字符串（中文即可）
        collection_name: 集合名
        top_k: 返回 top-k 条
        domain_filter: 可选，按领域过滤（如 "insurance"）
        doc_id_filter: 可选，按 doc_id 过滤
        output_fields: 返回字段列表（默认 ["id", "doc_id", "domain", "text"]）

    Returns:
        [{"id": ..., "doc_id": ..., "domain": ..., "text": ..., "distance": ...}, ...]
    """
    if output_fields is None:
        output_fields = [FIELD_ID, FIELD_DOC_ID, FIELD_DOMAIN, FIELD_TEXT]

    # 构建表达式过滤
    expr_parts = []
    if domain_filter:
        expr_parts.append(f'{FIELD_DOMAIN} == "{domain_filter}"')
    if doc_id_filter:
        expr_parts.append(f'{FIELD_DOC_ID} == "{doc_id_filter}"')

    expr = None
    if expr_parts:
        expr = " and ".join(expr_parts)

    return _execute_search(client, query, collection_name, top_k, expr, output_fields)


# ──────────────────────────────────────────────
# BM25 检索（多 doc_id + domain 版，给 workflow.py）
# ──────────────────────────────────────────────

def search_bm25_multi_doc(
    client: MilvusClient,
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 20,
    domain_filter: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    output_fields: Optional[List[str]] = None,
) -> List[ChunkRow]:
    """
    BM25 检索入口（多 doc_id + domain 同时过滤）。
    专门供 workflow.py 的 AsyncBM25Search 签名使用。

    Args:
        client: MilvusClient 实例
        query: 检索 query 字符串
        collection_name: 集合名
        top_k: 返回 top-k 条
        domain_filter: 可选，按领域过滤（如 "insurance"）
        doc_ids: 可选，按 doc_id 列表过滤（A 榜多文档场景）
        output_fields: 返回字段列表

    Returns:
        List[ChunkRow] — 与 workflow.py 签名对齐
    """
    if output_fields is None:
        output_fields = [FIELD_ID, FIELD_DOC_ID, FIELD_DOMAIN, FIELD_TEXT]

    # 构建表达式过滤
    expr_parts = []
    if domain_filter:
        expr_parts.append(f'{FIELD_DOMAIN} == "{domain_filter}"')

    if doc_ids:
        # 多 doc_id 用 in 表达式
        ids_quoted = [f'"{did}"' for did in doc_ids]
        expr_parts.append(f'{FIELD_DOC_ID} in [{", ".join(ids_quoted)}]')

    expr = None
    if expr_parts:
        expr = " and ".join(expr_parts)

    raw_results = _execute_search(client, query, collection_name, top_k, expr, output_fields)

    # 转为 ChunkRow
    rows = []
    for r in raw_results:
        rows.append(ChunkRow(
            id=r.get("id", 0),
            doc_id=r.get("doc_id", ""),
            domain=r.get("domain", ""),
            text=r.get("text", ""),
            distance=r.get("distance", 0.0),
        ))
    return rows


def _execute_search(
    client: MilvusClient,
    query: str,
    collection_name: str,
    top_k: int,
    expr: Optional[str],
    output_fields: List[str],
) -> List[Dict]:
    """
    执行 BM25 检索的内部函数。

    Args:
        client: MilvusClient 实例
        query: 检索 query 字符串
        collection_name: 集合名
        top_k: 返回 top-k 条
        expr: 过滤表达式
        output_fields: 返回字段列表

    Returns:
        [{"id": ..., "doc_id": ..., "domain": ..., "text": ..., "distance": ...}, ...]
    """
    search_params = {
        "metric_type": "BM25",
    }

    req = AnnSearchRequest(
        data=[query],  # BM25 Function 直接接受原文
        anns_field=FIELD_SPARSE,
        param=search_params,
        limit=top_k,
        expr=expr,
    )

    logger.debug(f"BM25 检索: query='{query[:80]}', expr={expr}")

    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=[req],
        ranker=RRFRanker(),
        limit=top_k,
        output_fields=output_fields,
    )

    # 解析结果
    parsed = []
    for hits in results:
        for hit in hits:
            item = {
                "id": hit["id"],
                "doc_id": hit.get("entity", {}).get(FIELD_DOC_ID, ""),
                "domain": hit.get("entity", {}).get(FIELD_DOMAIN, ""),
                "text": hit.get("entity", {}).get(FIELD_TEXT, ""),
                "distance": hit.get("distance", 0.0),
            }
            parsed.append(item)

    return parsed


# ──────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────

def describe_collection(
    client: MilvusClient,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """打印 Collection 详细信息"""
    desc = client.describe_collection(collection_name)
    logger.info(f"Collection: {desc.get('collection_name')}")
    for field in desc.get("fields", []):
        logger.info(f"  Field: {field['name']}  type={field['type']}")
    for func in desc.get("functions", []):
        logger.info(
            f"  Function: {func['name']}  type={func['type']} "
            f"input={func['input_field_names']} output={func['output_field_names']}"
        )
    index_desc = client.describe_index(collection_name, FIELD_SPARSE)
    logger.info(f"  Index: {index_desc}")
    logger.info(f"  Rows: {client.query(collection_name, output_fields=['count(*)'])}")
    return desc


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Milvus 3.0 服务端原生 BM25 索引管理"
    )
    parser.add_argument(
        "--uri",
        default=DEFAULT_MILVUS_URI,
        help=f"Milvus 服务地址 (默认 {DEFAULT_MILVUS_URI})",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Collection 名 (默认 {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"批量插入大小 (默认 {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = subparsers.add_parser("create", help="创建 Collection")
    create_parser.add_argument(
        "--overwrite", action="store_true",
        help="如果已存在则删除重建",
    )

    # drop
    drop_parser = subparsers.add_parser("drop", help="删除 Collection")

    # insert
    insert_parser = subparsers.add_parser("insert", help="导入所有 chunk")

    # search
    search_parser = subparsers.add_parser("search", help="BM25 检索测试")
    search_parser.add_argument("query", help="检索文本")
    search_parser.add_argument("--topk", type=int, default=10, help="Top-K")
    search_parser.add_argument("--domain", default=None, help="领域过滤")

    # describe
    subparsers.add_parser("describe", help="描述 Collection 状态")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 连接 Milvus
    logger.info(f"连接 Milvus: {args.uri}")
    client = MilvusClient(uri=args.uri)

    if args.command == "create":
        create_collection(client, args.collection, overwrite=args.overwrite)

    elif args.command == "drop":
        if client.has_collection(args.collection):
            client.drop_collection(args.collection)
            logger.info(f"Collection '{args.collection}' 已删除")
        else:
            logger.info(f"Collection '{args.collection}' 不存在")

    elif args.command == "insert":
        # 确保 collection 存在
        get_or_create_collection(client, args.collection)
        stats = insert_chunks(
            client=client,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
        print(f"\n导入统计:")
        print(f"  成功: {stats['total']}")
        print(f"  失败: {stats['errors']}")
        for domain, dstat in stats.get("per_domain", {}).items():
            print(f"  {domain}: {dstat['inserted']}/{dstat['total']}")

    elif args.command == "search":
        results = search_bm25(
            client=client,
            query=args.query,
            collection_name=args.collection,
            top_k=args.topk,
            domain_filter=args.domain,
        )
        print(f"\nBM25 检索: '{args.query}' (top-{args.topk})")
        print(f"{'rank':<5} {'doc_id':<30} {'domain':<20} {'distance':<10} text")
        print("-" * 100)
        for i, r in enumerate(results):
            text_short = r["text"][:60].replace("\n", " ")
            print(
                f"{i+1:<5} {r['doc_id']:<30} {r['domain']:<20} "
                f"{r['distance']:<10.4f} {text_short}"
            )

    elif args.command == "describe":
        if client.has_collection(args.collection):
            desc = describe_collection(client, args.collection)
        else:
            logger.error(f"Collection '{args.collection}' 不存在")


if __name__ == "__main__":
    main()