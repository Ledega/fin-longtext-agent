"""
Prompt 模板：为 mcq / multi / tf 三类题型设计三套模板

所有模板共享同一原则：
- 思考链在中间
- 最终单独一行只输出答案字母
- 不输出任何额外文字
"""

from typing import List, Optional

# ──────────────────────────────────────────────
# 构建上下文文本
# ──────────────────────────────────────────────

def format_context(chunks: List[dict]) -> str:
    """
    将检索到的 chunk 列表格式化为上下文文本。

    Args:
        chunks: [{"doc_id": str, "rank": int, "text": str, ...}, ...]

    Returns:
        格式化的上下文字符串
    """
    if not chunks:
        return "【未检索到相关文档片段】"

    # 按 doc_id 分组
    doc_groups: dict = {}
    for c in chunks:
        did = c.get("doc_id", "unknown")
        doc_groups.setdefault(did, []).append(c)

    parts = []
    for did, dchunks in doc_groups.items():
        dchunks_sorted = sorted(dchunks, key=lambda x: x.get("rank", 0))
        chunk_texts = []
        for c in dchunks_sorted:
            text = c.get("text", "").strip()
            if text:
                # 截断过长文本
                if len(text) > 600:
                    text = text[:600] + "..."
                chunk_texts.append(f"  [片段 {c.get('rank', 0) + 1}]: {text}")
        if chunk_texts:
            parts.append(f"文档【{did}】:\n" + "\n".join(chunk_texts))

    return "\n\n".join(parts) if parts else "【未检索到相关文档片段】"


def format_options(options: dict) -> str:
    """格式化选项文本"""
    lines = []
    for key in sorted(options.keys()):
        if key in ("A", "B", "C", "D"):
            lines.append(f"{key}. {options[key]}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 系统角色设定
# ──────────────────────────────────────────────

SYSTEM_PROMPT_MCQ = """你是一位专业的金融文档分析专家。你的任务是：
1. 仔细阅读提供的文档片段
2. 严格依据文档内容，不依赖常识或外部知识
3. 逐项分析每个选项与文档片段中的对应事实
4. 最终给出正确答案"""

SYSTEM_PROMPT_MULTI = """你是一位专业的金融文档分析专家。你的任务是：
1. 仔细阅读提供的文档片段
2. 严格依据文档内容，不依赖常识或外部知识
3. 对每个选项逐一核验其正确性
4. 汇总所有正确选项，按字母顺序输出"""

SYSTEM_PROMPT_TF = """你是一位专业的金融文档分析专家。你的任务是：
1. 仔细阅读提供的文档片段
2. 严格依据文档内容判断陈述的真伪
3. 如果陈述与文档事实一致选A（正确），不一致选B（错误）"""


# ──────────────────────────────────────────────
# 构建完整 Prompt
# ──────────────────────────────────────────────

def build_mcq_prompt(
    question: str,
    options: dict,
    context: str,
    retry_instruction: Optional[str] = None,
) -> str:
    """
    构建单选题 Prompt。

    Args:
        question: 题干
        options: 选项字典 {"A": "...", "B": "...", ...}
        context: 检索到的上下文文本
        retry_instruction: 降温重试时附加的纠正指令

    Returns:
        完整的 prompt 字符串
    """
    opt_text = format_options(options)
    extra = f"\n\n{retry_instruction}" if retry_instruction else ""

    prompt = f"""以下是从相关文档中检索到的片段：

===== 文档片段 =====
{context}
=====================

【题目】
{question}

【选项】
{opt_text}

请先在<思考>区域逐项分析每个选项与文档片段的对应关系，判断哪些选项是正确的。
最后，单独一行输出答案字母（如：B）。{extra}
"""
    return prompt


def build_multi_prompt(
    question: str,
    options: dict,
    context: str,
    retry_instruction: Optional[str] = None,
) -> str:
    """
    构建多选题 Prompt。

    Args:
        question: 题干
        options: 选项字典 {"A": "...", "B": "...", ...}
        context: 检索到的上下文文本
        retry_instruction: 降温重试时附加的纠正指令

    Returns:
        完整的 prompt 字符串
    """
    opt_text = format_options(options)
    extra = f"\n\n{retry_instruction}" if retry_instruction else ""

    prompt = f"""以下是从相关文档中检索到的片段：

===== 文档片段 =====
{context}
=====================

【题目】
{question}

【选项】
{opt_text}

请先在<思考>区域逐一核验每个选项的正确性。对于每个选项，明确标注"正确"或"错误"并给出理由。
最后，汇总所有正确选项，在单独一行按字母顺序输出答案（如：ACD）。{extra}
"""
    return prompt


def build_tf_prompt(
    question: str,
    context: str,
    retry_instruction: Optional[str] = None,
) -> str:
    """
    构建判断题 Prompt。

    判断题的 question 即为陈述句，选项恒为 A=正确/B=错误。

    Args:
        question: 陈述句
        context: 检索到的上下文文本
        retry_instruction: 降温重试时附加的纠正指令

    Returns:
        完整的 prompt 字符串
    """
    extra = f"\n\n{retry_instruction}" if retry_instruction else ""

    prompt = f"""以下是从相关文档中检索到的片段：

===== 文档片段 =====
{context}
=====================

【陈述】
{question}

请先在<思考>区域分析上述陈述与文档片段是否一致，找出支持判断的关键事实。
最后，单独一行输出答案字母：
- 如果陈述与文档事实一致，输出 A（正确）
- 如果陈述与文档事实不一致，输出 B（错误）

只有 A 或 B 两个选择。{extra}
"""
    return prompt


# ──────────────────────────────────────────────
# 统一的 Prompt 构建入口
# ──────────────────────────────────────────────

def build_prompt(
    question: str,
    options: dict,
    answer_format: str,
    context: str,
    retry_instruction: Optional[str] = None,
) -> str:
    """
    根据题型自动选择模板构建 Prompt。

    Args:
        question: 题干
        options: 选项字典
        answer_format: "mcq" / "multi" / "tf"
        context: 检索到的上下文文本
        retry_instruction: 降温重试时附加的纠正指令

    Returns:
        完整的 prompt 字符串
    """
    if answer_format == "mcq":
        return build_mcq_prompt(question, options, context, retry_instruction)
    elif answer_format == "multi":
        return build_multi_prompt(question, options, context, retry_instruction)
    elif answer_format == "tf":
        return build_tf_prompt(question, context, retry_instruction)
    else:
        raise ValueError(f"未知的 answer_format: {answer_format}")


# ──────────────────────────────────────────────
# Token 估算
# ──────────────────────────────────────────────

def estimate_prompt_tokens(prompt: str) -> int:
    """
    估算 prompt 的 token 数（与 chunker 中的 approx_tokens 一致）。

    中文字符约 1.5 token，英文约 4 字符/token。
    """
    import re
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', prompt))
    other_chars = len(prompt) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4) + 10