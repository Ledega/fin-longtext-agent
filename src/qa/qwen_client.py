"""
Qwen API 封装：通过 langchain_openai.ChatOpenAI 调用阿里云百炼 Qwen 系列模型

调用方式：
    chat_model = ChatOpenAI(
        model="qwen3.7-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )

功能：
- 单条调用（含重试和 Token 记录）
- 降温重试
- 全局 Token 累加器（用于 answer.csv 的 summary 行）
"""

import os
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


@dataclass
class QwenClientConfig:
    """Qwen 客户端配置"""
    model: str = ""                        # 模型名（默认从环境变量 QWEN_MODEL 读取，fallback qwen3.7-plus）
    api_key_env: str = "DASHSCOPE_API_KEY" # API Key 环境变量名
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    max_retries: int = 3
    retry_delay: float = 2.0
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 120

    def __post_init__(self):
        if not self.model:
            self.model = os.environ.get("QWEN_MODEL", "qwen3.7-plus")


DEFAULT_CONFIG = QwenClientConfig()


class QwenClient:
    """
    Qwen API 客户端，通过 langchain_openai.ChatOpenAI 调用。

    自动从环境变量读取 API Key，支持重试和 Token 统计。
    """

    def __init__(self, config: Optional[QwenClientConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._chat_model: Optional[ChatOpenAI] = None

        # 全局 Token 累加器
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def chat_model(self) -> ChatOpenAI:
        """延迟初始化 ChatOpenAI 客户端"""
        if self._chat_model is None:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"请设置环境变量 {self.config.api_key_env}，"
                    "或在 .env 文件中配置"
                )

            self._chat_model = ChatOpenAI(
                model=self.config.model,
                api_key=api_key,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._chat_model

    def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        """
        调用 Qwen 模型。

        Args:
            prompt: 输入 prompt
            temperature: 温度（覆盖默认）
            max_tokens: 最大输出 token（覆盖默认）

        Returns:
            (response_text, prompt_tokens, completion_tokens)
        """
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # 每次调用创建新的 ChatOpenAI 实例（支持每次不同 temperature）
                model = self._chat_model if (temperature is None and max_tokens is None) else ChatOpenAI(
                    model=self.config.model,
                    api_key=os.environ.get(self.config.api_key_env),
                    base_url=self.config.base_url,
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    timeout=self.config.timeout,
                    max_retries=0,  # 避免嵌套重试
                )

                response = model.invoke([HumanMessage(content=prompt)])
                text = response.content.strip()

                # 从 response_metadata 提取 token 用量
                usage = response.response_metadata.get("token_usage", {}) if hasattr(response, "response_metadata") else {}
                pt = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                ct = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct

                logger.debug(
                    f"Qwen 调用成功: tokens=({pt}+{ct}), temperature={temperature or self.config.temperature}"
                )
                return text, pt, ct

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Qwen 调用失败 (attempt {attempt}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)

        raise RuntimeError(f"Qwen 调用全部重试失败: {last_error}")

    def retry_with_fix(
        self,
        prompt: str,
        retry_instruction: str,
        temperature: float = 0.0,
    ) -> Tuple[str, int, int]:
        """
        降温重试：附加纠正指令后重新调用。

        Args:
            prompt: 原始 prompt
            retry_instruction: 追加的纠正指令
            temperature: 降温后的温度（默认 0）

        Returns:
            (response_text, prompt_tokens, completion_tokens)
        """
        fixed_prompt = prompt.rstrip() + f"\n\n{retry_instruction}"
        return self.call(fixed_prompt, temperature=temperature)

    def get_token_summary(self) -> dict:
        """获取 Token 消耗汇总"""
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def reset_token_counts(self) -> None:
        """重置 Token 计数器"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0