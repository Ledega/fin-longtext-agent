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
from src.qa.prompt_templates import build_prompt
from src.qa.qwen_client import QwenClient, QwenClientConfig
from src.qa.context_builder import build_context_for_question, DEFAULT_TOP_K, MAX_PROMPT_TOKENS
from src.qa.post_processor import process_answer, batch_validate_results
from src.qa.csv_writer import write_answer_csv, print_answer_summary
from src.indexing.retriever import BM25Retriever
from src.indexing.build_bm25 import ensure_finance_dict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 单题处理
# ──────────────────────────────────────────────

def process_single_question(
    question: dict,
    retriever: BM25Retriever,
    client: QwenClient,
    top_k: int = DEFAULT_TOP_K,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    dry_run: bool = False,
) -> dict:
    """
    处理单道题：检索 → prompt → 调用 Qwen → 后处理

    Args:
        question: 标准化题目 dict
        retriever: 混合检索器（已加载对应领域索引）
        client: Qwen 客户端
        top_k: 每文档 top-k chunk
        max_prompt_tokens: 单题 prompt 上限
        dry_run: 如果为 True，不调用 Qwen，只组装 context 并返回占位答案

    Returns:
        dict: {"qid", "answer", "answer_format", "prompt_tokens", "completion_tokens", "total_tokens"}
    """
    qid = question.get("qid", "?")
    domain = question.get("domain", "")
    fmt = question.get("answer_format", "")
    opts = question.get("options", {})
    q_text = question.get("question", "")

    # 1. 检索上下文
    logger.info(f"[{qid}] 开始检索 (domain={domain}, top_k={top_k})")
    context, ctx_stats = build_context_for_question(
        retriever, question, top_k=top_k, max_prompt_tokens=max_prompt_tokens,
    )
    logger.info(
        f"[{qid}] 上下文: {ctx_stats.get('total_chunks', 0)} chunks, "
        f"{len(context)} chars"
        + (" (已降级)" if ctx_stats.get("dropped") else "")
    )

    if dry_run:
        logger.info(f"[{qid}] [DRY RUN] 跳过 Qwen 调用")
        return {
            "qid": qid,
            "answer": "",
            "answer_format": fmt,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # 2. 调用 Qwen + 后处理
    logger.info(f"[{qid}] 调用 Qwen...")
    final_answer, pt, ct = process_answer(
        client=client,
        prompt_builder=build_prompt,
        question=question,
        context=context,
        max_retries=1,
    )
    logger.info(f"[{qid}] 答案={final_answer}, tokens=({pt}+{ct})")

    return {
        "qid": qid,
        "answer": final_answer,
        "answer_format": fmt,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run_pipeline(
    split: str = "A",
    domain: Optional[str] = None,
    output_path: str = "answer.csv",
    top_k: int = DEFAULT_TOP_K,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    dry_run: bool = False,
    model: str = "qwen3-plus",
) -> None:
    """
    运行完整的问答管线。

    Args:
        split: "A" 或 "B"
        domain: 可选，仅处理指定领域
        output_path: answer.csv 输出路径
        top_k: 每文档检索 chunk 数
        max_prompt_tokens: 单题 prompt token 上限
        dry_run: 不调 Qwen，只验证流程
        model: Qwen 模型名
    """
    print(f"{'='*60}")
    print(f"金融长文本 Agent - 问答管线")
    print(f"{'='*60}")
    print(f"  Split: {split}")
    print(f"  Domain: {domain or '全部'}")
    print(f"  Dry run: {dry_run}")
    print(f"  Model: {model}")
    print(f"  Top-k: {top_k}")
    print(f"  Max prompt tokens: {max_prompt_tokens}")
    print(f"  Output: {output_path}")
    print()

    # ── 1. 初始化 ──
    # 预热分词词典
    logger.info("预热中文金融词典...")
    try:
        ensure_finance_dict()
        logger.info("中文金融词典就绪")
    except Exception as e:
        logger.warning(f"金融词典加载失败（部分检索可能受影响）: {e}")

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

    # ── 3. 初始化检索器 ──
    print("初始化检索器...")
    retriever = BM25Retriever()

    try:
        retriever.load(domain=None)  # 全库
        print("  检索器就绪（全库 BM25 索引）")
    except FileNotFoundError as e:
        print(f"[错误] 检索器加载失败: {e}")
        print("请先运行索引构建脚本:")
        print("  python src/indexing/build_index.py")
        return

    # ── 4. 初始化 Qwen 客户端 ──
    if not dry_run:
        client = QwenClient(
            QwenClientConfig(model=model)
        )
        # 检查 API Key
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("[错误] 未设置 DASHSCOPE_API_KEY 环境变量")
            print("请在 .env 文件中配置 DASHSCOPE_API_KEY")
            return
        print("  Qwen 客户端就绪")
    else:
        client = None
        print("  [DRY RUN] 跳过 Qwen 初始化")

    print()

    # ── 5. 逐题处理 ──
    results = []
    total = len(questions)

    for idx, q in enumerate(questions, 1):
        qid = q.get("qid", "?")
        domain_q = q.get("domain", "")
        fmt = q.get("answer_format", "?")

        # 对于 A 榜，按领域切换检索器索引（更精确）
        # 但已加载全库索引也能工作（因为检索时传 domain 参数）
        # 如果需要更精细的索引隔离，这里可以按需切换

        print(f"[{idx}/{total}] {qid} [{fmt}] {domain_q}...", end=" ", flush=True)

        result = process_single_question(
            question=q,
            retriever=retriever,
            client=client,
            top_k=top_k,
            max_prompt_tokens=max_prompt_tokens,
            dry_run=dry_run,
        )
        results.append(result)

        answer_display = result["answer"] if result["answer"] else "(空)"
        print(f"→ {answer_display}")

    print()

    # ── 6. 校验最终答案 ──
    issues = batch_validate_results(results)
    if issues:
        print(f"[警告] {len(issues)} 道题答案异常:")
        for qid in issues:
            print(f"  {qid}")
    else:
        print("✓ 所有答案格式合法")

    # ── 7. 写入 CSV ──
    if not dry_run:
        # Token 汇总
        token_summary = client.get_token_summary() if client else None
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
        description="金融长文本 Agent - 问答管线",
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
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"每文档检索 chunk 数（默认 {DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_PROMPT_TOKENS,
        help=f"单题 prompt token 上限（默认 {MAX_PROMPT_TOKENS}）",
    )
    parser.add_argument(
        "--model", default=None,
        help="Qwen 模型名（默认从 .env 的 QWEN_MODEL 读取）",
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
    run_pipeline(
        split=args.split,
        domain=args.domain,
        output_path=args.output,
        top_k=args.top_k,
        max_prompt_tokens=args.max_tokens,
        dry_run=args.dry_run,
        model=model,
    )


if __name__ == "__main__":
    main()