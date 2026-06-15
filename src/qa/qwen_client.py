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
- 双渠道容灾：主模型不可用时自动切换到备用模型
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
    model: str = ""                        # 主模型名（默认从环境变量 QWEN_MODEL 读取，fallback qwen3.7-plus）
    fallback_model: str = "qwen3.6-plus"   # 备用模型（主模型因限流/异常不可用时切换）
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

# ---------- 错误类型判定 ----------

_SERVER_ERROR_KEYWORDS = [
    "429",           # Too Many Requests — 限流
    "5xx",           # 服务端错误通用
    "500", "502", "503", "504",
    "rate limit",    # 限流
    "too many requests",
    "quota exceeded",
    "server error",
    "service unavailable",
    "internal error",
    "connection error",
    "timeout",
]


def _is_server_error(error: Exception) -> bool:
    """判断异常是否属于服务端/限流错误（应触发渠道切换）"""
    err_str = str(error).lower()
    for kw in _SERVER_ERROR_KEYWORDS:
        if kw in err_str:
            return True
    return False


# ---------- 主客户端 ----------

class QwenClient:
    """
    Qwen API 客户端，通过 langchain_openai.ChatOpenAI 调用。

    双渠道容灾策略：
      1. 先用主模型（默认 qwen3.7-plus）调用，含 max_retries 次重试
      2. 若主模型因限流/服务端异常全部失败，自动切换到备用模型（qwen3.6-plus）
      3. 备用模型同样有 max_retries 次重试
      4. 备用模型也失败才抛出异常

    自动从环境变量读取 API Key，支持重试和 Token 统计。
    """

    def __init__(self, config: Optional[QwenClientConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # 当前活跃模型名（初始为主模型）
        self._active_model: str = self.config.model
        self._fallback_used: bool = False      # 是否已触发过备用模型
        self._fallback_permanently: bool = False  # True 表示永久使用备用模型

        # 延迟初始化的 ChatOpenAI 实例（按需创建）
        self._chat_model: Optional[ChatOpenAI] = None

        # 全局 Token 累加器
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def chat_model(self) -> ChatOpenAI:
        """延迟初始化 ChatOpenAI 客户端（主模型）"""
        if self._chat_model is None:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"请设置环境变量 {self.config.api_key_env}，"
                    "或在 .env 文件中配置"
                )

            self._chat_model = ChatOpenAI(
                model=self._active_model,
                api_key=api_key,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                extra_body={"enable_thinking": False}
            )
        return self._chat_model

    def _build_chat_model(self, model: str, temperature: float, max_tokens: int, max_retries: int = 0) -> ChatOpenAI:
        """创建一个指定参数的 ChatOpenAI 实例"""
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(
                f"请设置环境变量 {self.config.api_key_env}，"
                "或在 .env 文件中配置"
            )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=self.config.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.config.timeout,
            max_retries=max_retries,
            extra_body={"enable_thinking": False}
        )

    def _do_call(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
    ) -> Tuple[str, int, int]:
        """
        对指定模型执行一次调用（含重试）。

        Returns:
            (response_text, prompt_tokens, completion_tokens)

        Raises:
            RuntimeError: 所有重试均失败
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                chat = self._build_chat_model(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=0,  # 我们自己控制重试
                )
                response = chat.invoke([HumanMessage(content=prompt)])
                text = response.content.strip()

                # 从 response_metadata 提取 token 用量
                usage = {}
                if hasattr(response, "response_metadata"):
                    usage = response.response_metadata.get("token_usage", {})
                pt = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                ct = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct

                logger.debug(
                    f"Qwen 调用成功: model={model}, tokens=({pt}+{ct}), "
                    f"temperature={temperature}"
                )
                return text, pt, ct

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Qwen 调用失败 (model={model}, attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(self.config.retry_delay * attempt)

        raise RuntimeError(
            f"Qwen 模型 {model} 全部 {max_retries} 次重试失败: {last_error}"
        )

    def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        """
        调用 Qwen 模型（含双渠道容灾）。

        策略：
          1. 使用当前活跃模型（初始为主模型 qwen3.7-plus）
          2. 若失败且错误是服务端/限流类型，切换到备用模型（qwen3.6-plus）
          3. 若已触发过 fallback 且再次失败，不再回切

        Args:
            prompt: 输入 prompt
            temperature: 温度（覆盖默认）
            max_tokens: 最大输出 token（覆盖默认）

        Returns:
            (response_text, prompt_tokens, completion_tokens)
        """
        temp = temperature if temperature is not None else self.config.temperature
        mt = max_tokens or self.config.max_tokens
        retries = self.config.max_retries

        # ---- 首次尝试：当前活跃模型 ----
        try:
            return self._do_call(
                prompt=prompt,
                model=self._active_model,
                temperature=temp,
                max_tokens=mt,
                max_retries=retries,
            )
        except RuntimeError as e:
            # 判断是否因服务端/限流异常导致
            if _is_server_error(e):
                logger.warning(
                    f"主模型 {self._active_model} 因服务端/限流异常失败，"
                    f"切换至备用模型 {self.config.fallback_model}"
                )
                self._switch_to_fallback()
            else:
                # 非服务端异常（如 API Key 无效、参数错误等），直接抛出
                raise e

        # ---- 第二次尝试：备用模型 ----
        logger.info(
            f"使用备用模型 {self._active_model} 重试..."
        )
        return self._do_call(
            prompt=prompt,
            model=self._active_model,
            temperature=temp,
            max_tokens=mt,
            max_retries=retries,
        )

    def _switch_to_fallback(self) -> None:
        """切换到备用模型，并清空缓存的 ChatOpenAI 实例"""
        if not self._fallback_used:
            self._fallback_used = True
            self._active_model = self.config.fallback_model
            self._chat_model = None  # 清空缓存，下次按需重建
            logger.info(f"模型已切换至: {self._active_model}")
        else:
            # 备用模型也失败，永久标记（但调用方会继续收到异常）
            self._fallback_permanently = True

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