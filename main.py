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
from src.qa.csv_writer import write_answer_csv, print_answer_summary
from src.qa.workflow import (
    TaskOrchestrator,
    AsyncLLMClient,
    MilvusRetriever,
)
from src.qa.post_processor import batch_validate_results

logger = logging.getLogger(__name__)

# 默认 Milvus 地址
DEFAULT_MILVUS_URI = "http://localhost:19530"


# ──────────────────────────────────────────────
# 单题处理（异步包装）
# ──────────────────────────────────────────────

async def process_single_question_async(
    question: dict,
    orchestrator: TaskOrchestrator,
    dry_run: bool = False,
) -> dict:
    """
    异步处理单道题。

    Args:
        question: 标准化题目 dict
        orchestrator: TaskOrchestrator 实例
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
        result = await orchestrator.run_with_result(question)
        result["answer_format"] = fmt
    except Exception as e:
        logger.error(f"[{qid}] workflow 处理异常: {e}", exc_info=True)
        result = {
            "qid": qid,
            "answer": "A",
            "answer_format": fmt,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return result


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

async def run_pipeline_async(
    split: str = "A",
    domain: Optional[str] = None,
    output_path: str = "answer.csv",
    dry_run: bool = False,
    model: str = "",
    milvus_uri: str = DEFAULT_MILVUS_URI,
) -> None:
    """
    运行完整的问答管线（异步版）。

    Args:
        split: "A" 或 "B"
        domain: 可选，仅处理指定领域
        output_path: answer.csv 输出路径
        dry_run: 不调 Qwen，只验证流程
        model: Qwen 模型名（未使用，AsyncLLMClient 从 .env 读取）
        milvus_uri: Milvus 服务地址
    """
    print(f"{'='*60}")
    print(f"金融长文本 Agent - 问答管线 (TaskOrchestrator + Milvus BM25)")
    print(f"{'='*60}")
    print(f"  Split: {split}")
    print(f"  Domain: {domain or '全部'}")
    print(f"  Dry run: {dry_run}")
    print(f"  Milvus: {milvus_uri}")
    print(f"  Output: {output_path}")
    print()

    # ── 1. 初始化组件 ──
    orchestrator = None
    if not dry_run:
        print("初始化组件...")

        # 1a. LLM 客户端
        try:
            llm = AsyncLLMClient()
            print("  AsyncLLMClient 就绪")
        except Exception as e:
            print(f"[错误] 初始化 LLM 客户端失败: {e}")
            return

        # 1b. Milvus 检索器
        try:
            retriever = MilvusRetriever(uri=milvus_uri)
            print(f"  MilvusRetriever 就绪 ({milvus_uri})")
        except Exception as e:
            print(f"[错误] 连接 Milvus 失败: {e}")
            print(f"  请确认 Milvus 服务已启动: {milvus_uri}")
            return

        # 1c. 编排器
        orchestrator = TaskOrchestrator(llm, retriever)
        print("  TaskOrchestrator 就绪")
    else:
        print("  [DRY RUN] 跳过组件初始化")

    # ── 2. 加载题目 ──
    print("\n加载题目...")
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

    print()

    # ── 3. 逐题处理 ──
    results = []
    total = len(questions)

    for idx, q in enumerate(questions, 1):
        qid = q.get("qid", "?")
        domain_q = q.get("domain", "")
        fmt = q.get("answer_format", "?")

        print(f"[{idx}/{total}] {qid} [{fmt}] {domain_q}...", end=" ", flush=True)

        result = await process_single_question_async(
            question=q,
            orchestrator=orchestrator,
            dry_run=dry_run,
        )

        results.append(result)

        answer_display = result["answer"] if result["answer"] else "(空)"
        print(f"→ {answer_display}")

    print()

    # ── 4. 校验最终答案 ──
    issues = batch_validate_results(results)
    if issues:
        print(f"[警告] {len(issues)} 道题答案异常:")
        for qid in issues:
            print(f"  {qid}")
    else:
        print("✓ 所有答案格式合法")

    # ── 5. 写入 CSV ──
    if not dry_run and orchestrator:
        # Token 汇总
        token_summary = orchestrator._llm.get_token_summary()
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


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="金融长文本 Agent - 问答管线 (TaskOrchestrator + Milvus BM25)",
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

    # 运行异步管线
    asyncio.run(run_pipeline_async(
        split=args.split,
        domain=args.domain,
        output_path=args.output,
        dry_run=args.dry_run,
        model=args.model or "",
        milvus_uri=args.milvus_uri,
    ))


if __name__ == "__main__":
    main()