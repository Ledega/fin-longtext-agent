"""
答案后处理器：从模型输出中提取答案 → 校验 → 降温重试

核心流程：
1. 正则提取最后一行的答案字母
2. 按题型做规范化（单字母 / 排序去重多字母 / A/B 判断）
3. 非法时触发降温重试
4. 第二次仍非法则 fallback
"""

import re
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# 降温重试时追加的纠正指令
RETRY_INSTRUCTION = (
    "⚠️ 注意：你刚才的输出格式不符合要求。\n"
    "请只在最后一行输出答案字母，不要输出任何其他字符（包括思考过程、标点符号等）。\n"
    "例如：A  或  AC  或  B"
)


def extract_answer_from_text(
    raw_output: str,
    answer_format: str,
) -> Optional[str]:
    """
    从模型输出文本中提取合法答案。

    策略：
    1. 取最后一行非空文本
    2. 用正则 `[A-D]+` 匹配
    3. 按 answer_format 做后处理

    Args:
        raw_output: 模型原始输出
        answer_format: "mcq" / "multi" / "tf"

    Returns:
        规范化后的答案字符串，非法则返回 None
    """
    if not raw_output or not raw_output.strip():
        logger.warning("模型输出为空")
        return None

    lines = [l.strip() for l in raw_output.split('\n') if l.strip()]
    if not lines:
        logger.warning("模型输出无非空行")
        return None

    # 取最后一行
    last_line = lines[-1]
    logger.debug(f"提取最后一行: {last_line}")

    # 正则匹配 A-D 字母
    match = re.search(r'[A-D]+', last_line.upper())
    if not match:
        logger.warning(f"最后一行无匹配字母: {last_line}")
        return None

    raw_letters = match.group()

    # 按题型后处理
    return _normalize_answer(raw_letters, answer_format)


def _normalize_answer(letters: str, answer_format: str) -> Optional[str]:
    """
    规范化答案。

    - mcq: 返回第一个字母（如 ABC → A）
    - multi: 去重排序（如 CAC → AC）
    - tf: 返回第一个字母，须为 A/B
    """
    if not letters:
        return None
    letters = letters.upper()

    if answer_format == "mcq":
        if letters[0] in ("A", "B", "C", "D"):
            return letters[0]
        return None

    elif answer_format == "multi":
        valid = {c for c in letters if c in ("A", "B", "C", "D")}
        if not valid:
            return None
        return ''.join(sorted(valid))

    elif answer_format == "tf":
        if letters[0] in ("A", "B"):
            return letters[0]
        return None

    return None


def process_answer(
    client,
    prompt_builder,
    question: dict,
    context: str,
    max_retries: int = 1,
) -> Tuple[str, int, int]:
    """
    完整的答案处理流程：
    1. 首次调用 Qwen
    2. 提取答案 → 合法则返回
    3. 不合法则降温重试
    4. 仍不合法则 fallback

    Args:
        client: QwenClient 实例
        prompt_builder: build_prompt 函数
        question: 标准化题目 dict
        context: 检索到的上下文文本
        max_retries: 最大重试次数（首次调用不计）

    Returns:
        (final_answer, prompt_tokens, completion_tokens)
        - final_answer: 最终答案（可能为 fallback 值）
        - prompt_tokens: 本次处理总的 prompt tokens
        - completion_tokens: 本次处理总的 completion tokens
    """
    qid = question.get("qid", "?")
    fmt = question.get("answer_format", "")
    opts = question.get("options", {})
    q_text = question.get("question", "")

    total_pt = 0
    total_ct = 0

    # --- 第一次调用 ---
    prompt = prompt_builder(q_text, opts, fmt, context, retry_instruction=None)
    raw, pt, ct = client.call(prompt, temperature=0.3)
    total_pt += pt
    total_ct += ct

    answer = extract_answer_from_text(raw, fmt)
    if answer is not None:
        logger.info(f"[{qid}] 首次调用成功: {answer}")
        return answer, total_pt, total_ct

    logger.warning(f"[{qid}] 首次格式非法, raw={raw[:200]}..., 准备降温重试")

    # --- 降温重试 ---
    for attempt in range(1, max_retries + 1):
        # 使用 retry_with_fix，追加纠正指令
        raw, pt, ct = client.retry_with_fix(prompt, RETRY_INSTRUCTION, temperature=0.0)
        total_pt += pt
        total_ct += ct

        answer = extract_answer_from_text(raw, fmt)
        if answer is not None:
            logger.info(f"[{qid}] 第{attempt}次重试成功: {answer}")
            return answer, total_pt, total_ct

        logger.warning(f"[{qid}] 第{attempt}次重试仍非法")

    # --- 最终 fallback ---
    fallback = _get_fallback(fmt)
    logger.error(f"[{qid}] 多次重试失败, fallback={fallback}")
    return fallback, total_pt, total_ct


def _get_fallback(answer_format: str) -> str:
    """返回默认 fallback 值"""
    if answer_format == "tf":
        return "B"  # 保守：选错误
    return "A"  # 保守：选第一个


# ──────────────────────────────────────────────
# 答案校验（写 CSV 前二次校验）
# ──────────────────────────────────────────────

def validate_answer(answer: str, answer_format: str) -> Optional[str]:
    """
    二次校验答案合法性。

    Args:
        answer: 待校验的答案字符串
        answer_format: "mcq" / "multi" / "tf"

    Returns:
        合法则返回原值；非法则返回 None
    """
    return _normalize_answer(answer, answer_format)


def batch_validate_results(results: List[dict]) -> List[str]:
    """
    批量校验所有答案，返回问题列表。

    Args:
        results: [{"qid": str, "answer": str, "answer_format": str}, ...]

    Returns:
        有问题的 qid 列表
    """
    issues = []
    for r in results:
        valid = validate_answer(r.get("answer", ""), r.get("answer_format", ""))
        if valid is None:
            issues.append(r.get("qid", "?"))
    return issues