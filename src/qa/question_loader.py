"""
题目加载器：从 dataset/public_dataset_upload/questions/group_a/ 加载并标准化
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 标准化题目结构
# ──────────────────────────────────────────────

QUESTION_FIELDS = {
    "qid": str,           # 题目 ID，如 "fc_a_001"
    "domain": str,        # 领域，如 "financial_contracts"
    "split": str,         # "A" 或 "B"
    "question": str,      # 题干
    "options": dict,      # {"A": "...", "B": "...", "C": "...", "D": "..."}
    "answer_format": str, # "mcq" / "multi" / "tf"
    "type": str,          # 题型描述，如 "多选题"
    "doc_ids": list,      # 文档 ID 列表
}

# Answer format 与中文题型的映射
ANSWER_FORMAT_MAP = {
    "mcq": "单选题",
    "multi": "多选题",
    "tf": "判断题",
    "单选题": "mcq",
    "多选题": "multi",
    "判断题": "tf",
    "单选": "mcq",
    "多选": "multi",
    "判断": "tf",
}


def normalize_answer_format(raw: str) -> str:
    """将 answer_format 统一为 'mcq'/'multi'/'tf'"""
    raw = raw.strip().lower()
    if raw in ("mcq", "multi", "tf"):
        return raw
    # 中文映射
    cn_map = {
        "单选题": "mcq", "单选": "mcq",
        "多选题": "multi", "多选": "multi",
        "判断题": "tf", "判断": "tf",
    }
    return cn_map.get(raw, raw)


def normalize_question(raw: dict) -> dict:
    """将原始题目 JSON 标准化为统一格式"""
    q = {
        "qid": raw.get("qid", ""),
        "domain": raw.get("domain", ""),
        "split": raw.get("split", "A"),
        "question": raw.get("question", "").strip(),
        "options": raw.get("options", {}),
        "answer_format": normalize_answer_format(raw.get("answer_format", "")),
        "type": raw.get("type", ""),
        "doc_ids": raw.get("doc_ids", []),
    }

    # 校验必填字段
    missing = [k for k in ("qid", "question", "options") if not q[k]]
    if missing:
        logger.warning(f"题目 {q.get('qid', '?')} 缺少必填字段: {missing}")

    return q


# ──────────────────────────────────────────────
# 题型统计 / 校验
# ──────────────────────────────────────────────

def get_answer_format_info(questions: List[dict]) -> Dict[str, int]:
    """统计各题型数量"""
    counts: Dict[str, int] = {}
    for q in questions:
        fmt = q.get("answer_format", "unknown")
        counts[fmt] = counts.get(fmt, 0) + 1
    return counts


def validate_options(q: dict) -> List[str]:
    """
    校验选项合法性。

    Returns:
        问题描述列表；空列表表示完全合法
    """
    warnings = []
    fmt = q["answer_format"]
    opts = q["options"]

    if fmt == "tf":
        # 判断题：必须恰好有 A/B 两个选项
        expected = {"A", "B"}
        actual = set(opts.keys())
        if actual != expected:
            warnings.append(f"判断题选项异常: {actual} (期望 {expected})")
        # 检查选项内容是否为"正确"/"错误"
        if opts.get("A", "").strip() not in ("正确", "对", "是"):
            warnings.append(f"判断题 A 选项不是'正确': {opts.get('A')}")

    elif fmt == "mcq":
        if len(opts) < 2:
            warnings.append(f"单选题选项不足: {len(opts)}")

    elif fmt == "multi":
        if len(opts) < 2:
            warnings.append(f"多选题选项不足: {len(opts)}")

    return warnings


# ──────────────────────────────────────────────
# 题目加载
# ──────────────────────────────────────────────

DOMAIN_FILE_MAP = {
    "financial_contracts": "financial_contracts_questions.json",
    "financial_reports": "financial_reports_questions.json",
    "insurance": "insurance_questions.json",
    "regulatory": "regulatory_questions.json",
    "research": "research_questions.json",
}


def find_question_dir() -> Optional[Path]:
    """自动探测题目 JSON 文件所在的目录"""
    candidates = [
        Path("dataset/public_dataset_upload/questions/group_a"),
        Path("data/questions"),
        Path("questions"),
    ]
    for cand in candidates:
        if (cand / "financial_contracts_questions.json").exists():
            return cand
    return None


def load_all_questions(
    split: str = "A",
    domain: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> List[dict]:
    """
    加载指定 split 的所有题目（或仅加载指定领域）。

    Args:
        split: "A" 或 "B"
        domain: 可选，领域过滤
        base_dir: 题目 JSON 目录；None 时自动探测

    Returns:
        标准化后的题目字典列表
    """
    if base_dir is None:
        base_dir = find_question_dir()
        if base_dir is None:
            raise FileNotFoundError(
                "无法自动定位题目 JSON 目录。请手动指定 base_dir。"
            )

    if domain:
        domains_to_load = [domain]
    else:
        domains_to_load = list(DOMAIN_FILE_MAP.keys())

    all_questions: List[dict] = []
    for dom in domains_to_load:
        fname = DOMAIN_FILE_MAP.get(dom)
        if not fname:
            logger.warning(f"未知领域: {dom}")
            continue

        fpath = base_dir / fname
        if not fpath.exists():
            logger.warning(f"题目文件不存在: {fpath}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            raw_qs = json.load(f)

        # 只加载指定 split 的题目
        for raw in raw_qs:
            if raw.get("split", "").upper() != split.upper():
                continue
            q = normalize_question(raw)
            all_questions.append(q)

        logger.info(f"  {dom}: {len(raw_qs)} -> 筛选后 {len([q for q in all_questions if q['domain'] == dom])}")

    logger.info(
        f"共加载 {len(all_questions)} 道题目 (split={split})"
    )

    # 统计题型
    info = get_answer_format_info(all_questions)
    logger.info(f"  题型分布: {info}")

    return all_questions


def load_question_by_qid(qid: str, split: str = "A", base_dir: Optional[Path] = None) -> Optional[dict]:
    """按 qid 加载单道题"""
    questions = load_all_questions(split=split, base_dir=base_dir)
    for q in questions:
        if q["qid"] == qid:
            return q
    logger.warning(f"未找到 qid={qid}")
    return None


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────

def main():
    """验证题目加载"""
    logging.basicConfig(level=logging.INFO)

    # 加载所有 A 榜题目
    questions = load_all_questions(split="A")
    print(f"\n共 {len(questions)} 道题")

    # 显示前 3 道题
    for q in questions[:3]:
        print(f"\n  {q['qid']} [{q['answer_format']}] {q['domain']}")
        print(f"  题干: {q['question'][:80]}...")
        print(f"  选项: {list(q['options'].values())}")
        print(f"  doc_ids: {q['doc_ids']}")

    # 显示题型分布
    info = get_answer_format_info(questions)
    print(f"\n题型分布: {info}")

    # 校验每道题
    warnings = []
    for q in questions:
        ws = validate_options(q)
        for w in ws:
            warnings.append(f"  {q['qid']}: {w}")
    if warnings:
        print(f"\n校验警告 ({len(warnings)}):")
        for w in warnings:
            print(w)
    else:
        print("\n✓ 所有题目校验通过")


if __name__ == "__main__":
    main()