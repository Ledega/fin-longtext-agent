"""
CSV 写入器：生成符合评测要求的 answer.csv

格式：
| qid | answer | prompt_tokens | completion_tokens | total_tokens |
|-----|--------|---------------|-------------------|--------------|
| summary | (空)   | 3627557       | 629               | 3628186      |
| fc_a_001 | AC    | 48500         | 120               | 48620        |
| ...     | ...    | ...           | ...               | ...          |

规则：
- 第一行为 summary 行
- 之后每行一道题
- 答案为空、包含非法字符、顺序不规范则记为错误（赛题规则）
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

CSV_HEADER = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]


def write_answer_csv(
    results: List[dict],
    output_path: str,
    token_summary: Optional[dict] = None,
) -> str:
    """
    生成 answer.csv。

    Args:
        results: 每道题的结果列表，每项包含：
            - qid: str
            - answer: str（最终答案，如 "A", "AC", "B"）
            - prompt_tokens: int
            - completion_tokens: int
            - total_tokens: int
            - answer_format: str（可选，用于校验）
        output_path: 输出 CSV 文件路径
        token_summary: Token 汇总（如无，则自动从 results 汇总）

    Returns:
        写入的 CSV 文件路径
    """
    # 计算汇总
    if token_summary is None:
        total_pt = sum(r.get("prompt_tokens", 0) for r in results)
        total_ct = sum(r.get("completion_tokens", 0) for r in results)
        token_summary = {
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "total_tokens": total_pt + total_ct,
        }

    # 确保目录存在
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 写表头
        writer.writerow(CSV_HEADER)

        # 第一行：summary
        writer.writerow([
            "summary",
            "",
            token_summary["prompt_tokens"],
            token_summary["completion_tokens"],
            token_summary["total_tokens"],
        ])

        # 逐题写入
        for r in results:
            answer = r.get("answer", "")
            # 确保 answer 是字符串（不含空格、逗号等非法字符）
            answer = answer.strip().upper()

            writer.writerow([
                r.get("qid", ""),
                answer,
                r.get("prompt_tokens", 0),
                r.get("completion_tokens", 0),
                r.get("total_tokens", 0),
            ])

    logger.info(
        f"answer.csv 已写入: {out_path} "
        f"({len(results)} 题, "
        f"prompt={token_summary['prompt_tokens']}, "
        f"completion={token_summary['completion_tokens']}, "
        f"total={token_summary['total_tokens']})"
    )

    return str(out_path)


def read_answer_csv(csv_path: str) -> List[dict]:
    """读取 answer.csv 并解析（用于调试和验证）"""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"answer.csv 不存在: {csv_path}")

    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "qid": row.get("qid", ""),
                "answer": row.get("answer", ""),
                "prompt_tokens": int(row.get("prompt_tokens", 0)),
                "completion_tokens": int(row.get("completion_tokens", 0)),
                "total_tokens": int(row.get("total_tokens", 0)),
            })

    return results


def print_answer_summary(csv_path: str) -> None:
    """打印 answer.csv 的摘要（用于快速验证）"""
    results = read_answer_csv(csv_path)

    summary = results[0] if results and results[0]["qid"] == "summary" else None
    questions = results[1:] if summary else results

    print(f"\n=== answer.csv 摘要 ===")
    print(f"  题目数: {len(questions)}")
    if summary:
        print(f"  Total tokens: {summary['total_tokens']}")
    print(f"\n  前 10 题:")
    for r in questions[:10]:
        print(f"    {r['qid']}: {r['answer'] if r['answer'] else '(空)'}  "
              f"(tokens={r['total_tokens']})")
    if len(questions) > 10:
        print(f"    ... 共 {len(questions)} 题")