"""
AFAC2026 - 金融长文本 Agent 主入口

用法：
    # 跑全量 A 榜（100 题）：
    python main.py --split A
    
    # 只跑某个领域（如 regulatory）：
    python main.py --split A --domain regulatory
    
    # dry run（不调 Qwen，只验证流程）：
    python main.py --split A --dry-run

    # 指定输出文件：
    python main.py --split A --output my_answer.csv
"""

import sys
import os
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import get_config, load_env_file

# 确保 .env 被加载
load_env_file()

from src.qa.question_loader import load_all_questions, get_answer_format_info, validate_options
from src.qa.qwen_client import QwenClient, QwenClientConfig
from src.qa.csv_writer import write_answer_csv, print_answer_summary
from src.qa.workflow import process_single_task
from src.indexing.milvus_bm25 import (
    MilvusClient,
    search_bm25_multi_doc,
    ChunkRow,
    DEFAULT_MILVUS_URI,
    COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 构建 Milvus BM25 搜索闭包（供 workflow 注入）
# ──────────────────────────────────────────────

def build_milvus_search_fn(milvus_client: MilvusClient):
    """
    创建一个 async 闭包，符合 workflow.py 的 AsyncBM25Search 签名。

    async def bm25_search(query, domain, doc_ids, top_k) -> List[ChunkRow]
    """
    async def bm25_search(
        query: str,
        domain: str,
        doc_ids: Optional[list] = None,
        top_k: int = 20,
    ) -> list:
        """Milvus BM25 异步检索（通过 asyncio.to_thread 避免阻塞）"""
        return await asyncio.to_thread(
            search_bm25_multi_doc,
            client=milvus_client,
            query=query,
            collection_name=COLLECTION_NAME,
            top_k=top_k,
            domain_filter=domain if domain else None,
            doc_ids=doc_ids if doc_ids else None,
        )
    return bm25_search


# ──────────────────────────────────────────────
# 单题处理（异步包装）
# ──────────────────────────────────────────────

async def process_single_question_async(
    question: dict,
    qwen_client: QwenClient,
    bm25_search_fn,
    dry_run: bool = False,
) -> dict:
    """
    异步处理单道题，包装 workflow.process_single_task 的结果。

    Args:
        question: 标准化题目 dict
        qwen_client: QwenClient 实例
        bm25_search_fn: async BM25 搜索函数
        dry_run: 如果为 True，不调用 Qwen，直接返回占位

    Returns:
        dict: {"qid", "answer", "answer_format", "prompt_tokens", "completion_tokens", "total_tokens"}
    """
    qid = question.get("qid", "?")
    fmt = question.get("answer_format", "")

    if dry_run:
        logger.info(f"[{qid}] [DRY RUN] 跳过 workflow 调用")
        return {
            "qid": qid,
            "answer": "",
            "answer_format": fmt,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    try:
        answer = await process_single_task(
            task_json=question,
            qwen_client=qwen_client,
            bm25_search=bm25_search_fn,
        )
    except Exception as e:
        logger.error(f"[{qid}] workflow 处理异常: {e}", exc_info=True)
        answer = "A"  # fallback

    # 从 QwenClient 获取本次调用的 Token 数（累计，但作为近似值）
    ts = qwen_client.get_token_summary()
    # 注意：这里返回的是全局累计 token。实际应该在 process_single_task 内部做细粒度追踪。
    # 但由于 workflow 里面是通过 call_async 自动累加的，这里用 delta 不现实。
    # 稳妥做法：取当前累计值，交给调用方在循环中计算差值。
    # 这里简化为返回累计值，调用方自己算 delta。
    pt = ts["prompt_tokens"]
    ct = ts["completion_tokens"]

    return {
        "qid": qid,
        "answer": answer,
        "answer_format": fmt,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

async def run_pipeline_async(
    split: str = "A",
    domain: Optional[str] = None,
    output_path: str = "answer.csv",
    dry_run: bool = False,
    model: str = "qwen3-plus",
    milvus_uri: str = DEFAULT_MILVUS_URI,
) -> None:
    """
    运行完整的问答管线（异步版）。

    Args:
        split: "A" 或 "B"
        domain: 可选，仅处理指定领域
        output_path: answer.csv 输出路径
        dry_run: 不调 Qwen，只验证流程
        model: Qwen 模型名
        milvus_uri: Milvus 服务地址
    """
    print(f"{'='*60}")
    print(f"金融长文本 Agent - 问答管线 (Workflow + Milvus)")
    print(f"{'='*60}")
    print(f"  Split: {split}")
    print(f"  Domain: {domain or '全部'}")
    print(f"  Dry run: {dry_run}")
    print(f"  Model: {model}")
    print(f"  Milvus: {milvus_uri}")
    print(f"  Output: {output_path}")
    print()

    # ── 1. 连接 Milvus（dry-run 模式跳过）──
    bm25_search_fn = None
    if not dry_run:
        print("连接 Milvus...")
        try:
            milvus_client = MilvusClient(uri=milvus_uri)
            # 检查 collection 是否存在
            if not milvus_client.has_collection(COLLECTION_NAME):
                print(f"[错误] Milvus Collection '{COLLECTION_NAME}' 不存在")
                print("请先运行: python src/indexing/milvus_bm25.py insert")
                return
            # 加载 collection
            milvus_client.load_collection(COLLECTION_NAME)
            print(f"  Milvus 就绪 ({COLLECTION_NAME})")
        except Exception as e:
            print(f"[错误] 连接 Milvus 失败: {e}")
            print(f"  请确认 Milvus 服务已启动: {milvus_uri}")
            return

        bm25_search_fn = build_milvus_search_fn(milvus_client)
    else:
        print("  [DRY RUN] 跳过 Milvus 连接")

    # ── 2. 加载题目 ──
    print("加载题目...")
    questions = load_all_questions(split=split, domain=domain)

    if not questions:
        print("[错误] 未加载到任何题目，请检查数据目录")
        return

    info = get_answer_format_info(questions)
    print(f"  共 {len(questions)} 道题: {info}")

    # 校验
    all_warnings = []
    for q in questions:
        all_warnings.extend(validate_options(q))
    if all_warnings:
        print(f"  [警告] {len(all_warnings)} 个校验问题:")
        for w in all_warnings[:5]:
            print(f"    {w}")

    # ── 3. 初始化 Qwen 客户端 ──
    if not dry_run:
        qwen_client = QwenClient(QwenClientConfig(model=model))

        # 检查 API Key
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("[错误] 未设置 DASHSCOPE_API_KEY 环境变量")
            print("请在 .env 文件中配置 DASHSCOPE_API_KEY")
            return
        print("  Qwen 客户端就绪")
    else:
        qwen_client = None
        print("  [DRY RUN] 跳过 Qwen 初始化")

    print()

    # ── 4. 逐题处理 ──
    results = []
    total = len(questions)
    prev_pt = 0
    prev_ct = 0

    for idx, q in enumerate(questions, 1):
        qid = q.get("qid", "?")
        domain_q = q.get("domain", "")
        fmt = q.get("answer_format", "?")

        print(f"[{idx}/{total}] {qid} [{fmt}] {domain_q}...", end=" ", flush=True)

        result = await process_single_question_async(
            question=q,
            qwen_client=qwen_client,
            bm25_search_fn=bm25_search_fn,
            dry_run=dry_run,
        )

        # 计算本次调用的 Token 增量
        if not dry_run and qwen_client:
            ts = qwen_client.get_token_summary()
            delta_pt = ts["prompt_tokens"] - prev_pt
            delta_ct = ts["completion_tokens"] - prev_ct
            result["prompt_tokens"] = delta_pt
            result["completion_tokens"] = delta_ct
            result["total_tokens"] = delta_pt + delta_ct
            prev_pt = ts["prompt_tokens"]
            prev_ct = ts["completion_tokens"]

        results.append(result)

        answer_display = result["answer"] if result["answer"] else "(空)"
        print(f"→ {answer_display}")

    print()

    # ── 5. 校验最终答案 ──
    from src.qa.post_processor import batch_validate_results
    issues = batch_validate_results(results)
    if issues:
        print(f"[警告] {len(issues)} 道题答案异常:")
        for qid in issues:
            print(f"  {qid}")
    else:
        print("✓ 所有答案格式合法")

    # ── 6. 写入 CSV ──
    if not dry_run:
        # Token 汇总
        token_summary = qwen_client.get_token_summary() if qwen_client else None
        csv_path = write_answer_csv(results, output_path, token_summary)
        print(f"\nanswer.csv 已生成: {csv_path}")

        # 打印摘要
        print_answer_summary(csv_path)

        # 评测预估
        print(f"\n=== 评测预估 ===")
        print(f"  Token 预算: 5,000,000")
        print(f"  实际消耗:  {token_summary['total_tokens']}")
        token_score = max(
            0.0,
            min(1.0, (5_000_000 - token_summary["total_tokens"]) / 5_000_000),
        )
        print(f"  TokenScore: {token_score:.4f}")
    else:
        print(f"\n[DRY RUN] 完成，未输出 answer.csv")

    # 关闭 Milvus 连接
    if not dry_run:
        milvus_client.close()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="金融长文本 Agent - 问答管线 (Workflow + Milvus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --split A                     # 跑 A 榜全部 100 题
  python main.py --split A --domain regulatory  # 只跑 regulatory 领域
  python main.py --split A --dry-run            # 不调 API，只验证流程
  python main.py --split A --output my.csv      # 指定输出文件
        """,
    )
    parser.add_argument(
        "--split", choices=["A", "B"], default="A",
        help="榜单（默认 A）",
    )
    parser.add_argument(
        "--domain", choices=[
            "insurance", "regulatory", "financial_contracts",
            "financial_reports", "research",
        ], default=None,
        help="领域过滤（可选）",
    )
    parser.add_argument(
        "--output", default="answer.csv",
        help="answer.csv 输出路径（默认 answer.csv）",
    )
    parser.add_argument(
        "--model", default=None,
        help="Qwen 模型名（默认从 .env 的 QWEN_MODEL 读取）",
    )
    parser.add_argument(
        "--milvus-uri", default=DEFAULT_MILVUS_URI,
        help=f"Milvus 服务地址（默认 {DEFAULT_MILVUS_URI}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="dry run，不调用 Qwen API",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 如果命令行未指定模型名，则留空让 QwenClientConfig 从 .env 读取
    model = args.model or ""

    # 运行异步管线
    asyncio.run(run_pipeline_async(
        split=args.split,
        domain=args.domain,
        output_path=args.output,
        dry_run=args.dry_run,
        model=model,
        milvus_uri=args.milvus_uri,
    ))


if __name__ == "__main__":
    main()