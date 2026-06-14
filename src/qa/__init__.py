"""
问答管线模块

提供以下核心组件：
- question_loader: 题目加载与标准化
- prompt_templates: 3 套 prompt 模板（mcq/multi/tf）
- qwen_client: Qwen API 封装 + Token 累加
- context_builder: 检索 + 上下文拼装 + Token 控制
- post_processor: 答案提取 + 降温重试 + fallback
- csv_writer: answer.csv 输出
"""

from src.qa.question_loader import (
    load_all_questions,
    load_question_by_qid,
    normalize_question,
    validate_options,
    get_answer_format_info,
)
from src.qa.prompt_templates import (
    build_prompt,
    build_mcq_prompt,
    build_multi_prompt,
    build_tf_prompt,
    format_context,
    estimate_prompt_tokens,
)
from src.qa.qwen_client import QwenClient, QwenClientConfig
from src.qa.context_builder import (
    build_context_for_question,
    approx_tokens,
    DEFAULT_TOP_K,
    MAX_PROMPT_TOKENS,
)
from src.qa.post_processor import (
    process_answer,
    extract_answer_from_text,
    validate_answer,
    batch_validate_results,
)
from src.qa.csv_writer import (
    write_answer_csv,
    read_answer_csv,
    print_answer_summary,
)

__all__ = [
    # question_loader
    "load_all_questions",
    "load_question_by_qid",
    "normalize_question",
    "validate_options",
    "get_answer_format_info",
    # prompt_templates
    "build_prompt",
    "build_mcq_prompt",
    "build_multi_prompt",
    "build_tf_prompt",
    "format_context",
    "estimate_prompt_tokens",
    # qwen_client
    "QwenClient",
    "QwenClientConfig",
    # context_builder
    "build_context_for_question",
    "approx_tokens",
    "DEFAULT_TOP_K",
    "MAX_PROMPT_TOKENS",
    # post_processor
    "process_answer",
    "extract_answer_from_text",
    "validate_answer",
    "batch_validate_results",
    # csv_writer
    "write_answer_csv",
    "read_answer_csv",
    "print_answer_summary",
]