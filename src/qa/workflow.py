"""
A 榜评测路由与调度框架 (Agentic RAG Workflow) — Production-Ready

组件：
  - MilvusRetriever      : 真实 Milvus BM25 稀疏向量检索（doc_id 硬隔离）
  - AsyncLLMClient       : 异步 LLM 调用封装（重试 + 超时 + 双渠道容灾）
  - IntentRouter         : 路由决策（doc_ids > 1 + 关键词 + LLM）
  - BaseStrategy + 子类   : 三路检索策略（LOOKUP / CROSS_DOC / CALCULATION）
  - AnswerFormatter      : 答案生成（single_shot + pointwise_multi NLI）
  - TaskOrchestrator     : 单题/批量编排入口

红线：
  - 不使用任何 embedding 模型 / FloatVector
  - 不使用 SQL 数据库
  - 所有数据仅来自 Milvus BM25（稀疏向量）索引中的 Chunk
"""

import asyncio
import json
import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Coroutine, List, Dict, Optional, Any, Tuple

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 常量 & 配置
# ═══════════════════════════════════════════════════

_DEFAULT_LLM_TIMEOUT = 15       # LLM 单次调用超时（秒）
_DEFAULT_BM25_TIMEOUT = 15      # BM25 检索超时（秒）
_DEFAULT_BM25_TOP_K = 20        # BM25 默认召回数
_MAX_LLM_RETRIES = 3            # LLM 最大重试次数


# ═══════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════

@dataclass
class TaskJSON:
    """标准化题目结构（来自 question_loader）"""
    qid: str
    domain: str
    split: str
    question: str
    options: Dict[str, str]
    answer_format: str                # "mcq" | "multi" | "tf"
    type: str
    doc_ids: List[str]

    @classmethod
    def from_raw(cls, raw: dict) -> "TaskJSON":
        return cls(
            qid=raw.get("qid", ""),
            domain=raw.get("domain", ""),
            split=raw.get("split", "A"),
            question=raw.get("question", ""),
            options=raw.get("options", {}),
            answer_format=raw.get("answer_format", "mcq"),
            type=raw.get("type", ""),
            doc_ids=raw.get("doc_ids", []),
        )


class RouteIntent(str, Enum):
    LOOKUP = "LOOKUP"
    CROSS_DOC = "CROSS_DOC"
    CALCULATION = "CALCULATION"


@dataclass
class RouteResult:
    intent: RouteIntent
    reason: str = ""


@dataclass
class ChunkRow:
    """单条 BM25 检索结果"""
    id: int = 0
    chunk_id: str = ""
    doc_id: str = ""
    domain: str = ""
    text: str = ""
    heading_path: List[str] = field(default_factory=list)
    chunk_type: str = "paragraph"
    distance: float = 0.0


@dataclass
class ContextDict:
    """策略产出上下文"""
    evidence: str = ""
    calculations: str = ""
    numbers: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════
# 工具类 1 — Milvus BM25 检索器
# ═══════════════════════════════════════════════════

class MilvusRetriever:
    """
    真实 Milvus BM25 稀疏向量检索封装。

    构建 AnnSearchRequest + RRFRanker，通过 hybrid_search 执行检索。
    doc_id 硬隔离：用 expr 表达式过滤，防止跨文档幻觉。
    """

    def __init__(self, uri: str = "http://localhost:19530",
                 collection_name: str = "fin_longtext_chunks"):
        self._uri = uri
        self._collection = collection_name
        self._client: Optional[MilvusClient] = None

    def _get_client(self) -> MilvusClient:
        """延迟初始化 MilvusClient"""
        if self._client is None:
            self._client = MilvusClient(uri=self._uri)
        return self._client

    async def search(
        self,
        queries: List[str],
        doc_ids: Optional[List[str]] = None,
        domain: Optional[str] = None,
        limit: int = _DEFAULT_BM25_TOP_K,
        timeout: int = _DEFAULT_BM25_TIMEOUT,
    ) -> List[ChunkRow]:
        """
        执行 BM25 稀疏向量检索。

        Args:
            queries: 检索 query 列表（通常单条，保留列表扩展性）
            doc_ids: 文档 ID 白名单（硬隔离，防止幻觉串台）
            domain:  领域过滤
            limit:   返回 top-k 条
            timeout: 超时秒数

        Returns:
            List[ChunkRow] 检索结果，空列表表示无匹配
        """
        # 构造表达式过滤条件
        expr_parts = []
        if doc_ids:
            # 用 json.dumps 安全序列化 doc_ids 列表
            ids_json = json.dumps(doc_ids, ensure_ascii=False)
            expr_parts.append(f"doc_id in {ids_json}")
        if domain:
            expr_parts.append(f'domain == "{domain}"')
        expr = " and ".join(expr_parts) if expr_parts else None

        search_params = {"metric_type": "BM25"}

        reqs = []
        for q in queries:
            if not q.strip():
                continue
            req = AnnSearchRequest(
                data=[q],
                anns_field="sparse_bm25",
                param=search_params,
                limit=limit,
                expr=expr,
            )
            reqs.append(req)

        if not reqs:
            logger.warning("MilvusRetriever: 无有效 query，跳过检索")
            return []

        try:
            client = self._get_client()
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.hybrid_search(
                        collection_name=self._collection,
                        reqs=reqs,
                        ranker=RRFRanker(),
                        limit=limit,
                        output_fields=["id", "doc_id", "domain", "text"],
                    ),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Milvus BM25 检索超时 ({timeout}s)")
            return []
        except Exception as e:
            logger.error(f"Milvus BM25 检索失败: {e}")
            return []

        # 解析结果
        rows: List[ChunkRow] = []
        seen_ids: set = set()
        for hits in results:
            for hit in hits:
                hit_id = hit.get("id")
                if hit_id in seen_ids:
                    continue
                seen_ids.add(hit_id)
                entity = hit.get("entity", {})
                rows.append(ChunkRow(
                    id=hit_id,
                    doc_id=entity.get("doc_id", ""),
                    domain=entity.get("domain", ""),
                    text=entity.get("text", ""),
                    distance=hit.get("distance", 0.0),
                ))

        return rows


# ═══════════════════════════════════════════════════
# 工具类 2 — 异步 LLM 客户端（自包含，不依赖 QwenClient）
# ═══════════════════════════════════════════════════

class AsyncLLMClient:
    """
    异步 LLM 调用封装。

    底层使用 LangChain ChatOpenAI（阿里云百炼兼容模式）。
    含重试 + 超时 + 双渠道容灾。
    """

    def __init__(self):
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._model = os.environ.get("QWEN_MODEL", "qwen3.6-27b")
        self._fallback_model = "qwen3.6-27b"
        self._base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # Token 累加器（用于 answer.csv 的 summary 行）
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0

    async def call(
        self,
        prompt: str,
        timeout: int = _DEFAULT_LLM_TIMEOUT,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        调用 LLM，返回文本。

        Args:
            prompt:  输入 prompt
            timeout: 单次超时（秒）
            temperature: 温度
            max_tokens: 最大输出 token

        Returns:
            模型输出文本；异常/超时返回空字符串
        """
        last_error = None
        active_model = self._model

        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                chat = ChatOpenAI(
                    model=active_model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    max_retries=0,
                    extra_body={"enable_thinking": False},
                )

                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: chat.invoke([HumanMessage(content=prompt)]),
                    ),
                    timeout=timeout,
                )
                # 提取并累加 token 用量
                pt, ct = 0, 0
                if hasattr(response, "response_metadata"):
                    usage = response.response_metadata.get("token_usage", {})
                    pt = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                    ct = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
                self._total_prompt_tokens += pt
                self._total_completion_tokens += ct
                return response.content.strip()

            except asyncio.TimeoutError:
                logger.warning(
                    f"LLM 调用超时 (attempt {attempt}/{_MAX_LLM_RETRIES}), "
                    f"model={active_model}"
                )
                last_error = None  # 超时继续重试
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_server = any(kw in err_str for kw in [
                    "429", "5xx", "500", "502", "503", "504",
                    "rate limit", "too many requests", "quota exceeded",
                    "server error", "service unavailable",
                    "internal error", "timeout",
                ])
                if is_server and active_model != self._fallback_model:
                    logger.warning(
                        f"模型 {active_model} 异常，切换至 {self._fallback_model}: {e}"
                    )
                    active_model = self._fallback_model
                else:
                    logger.warning(
                        f"LLM 调用失败 (attempt {attempt}/{_MAX_LLM_RETRIES}, "
                        f"model={active_model}): {e}"
                    )

            # 最后一次失败就不再等待
            if attempt < _MAX_LLM_RETRIES:
                wait = 2.0 * attempt
                await asyncio.sleep(wait)

        logger.error(f"LLM 调用全部 {_MAX_LLM_RETRIES} 次重试失败")
        return ""

    def get_token_summary(self) -> Dict[str, int]:
        """获取 Token 消耗汇总（用于 answer.csv 的 summary 行）"""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    def reset_token_counts(self) -> None:
        """重置 Token 计数器"""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0


# ═══════════════════════════════════════════════════
# 意图路由器
# ═══════════════════════════════════════════════════

class IntentRouter:
    """
    路由决策。

    规则：
      1. doc_ids > 1 → CROSS_DOC
      2. 关键词预判 → CALCULATION
      3. LLM 深度判断 → CALCULATION / LOOKUP
    """

    _CALC_KEYWORDS = [
        "增长率", "增长比例", "同比", "环比",
        "占比", "比例", "比率", "率",
        "多少", "差额", "差值", "多出", "高出",
        "超过", "低于", "高于",
        "倍数", "翻", "百分", "%",
        "加总", "合计", "总和", "之和", "之差",
        "营业收入增长率", "利润增长率",
    ]

    def __init__(self, llm: AsyncLLMClient):
        self._llm = llm

    async def route(self, task: TaskJSON) -> RouteResult:
        if len(task.doc_ids) > 1:
            return RouteResult(RouteIntent.CROSS_DOC, "len(doc_ids) > 1")

        combined = f"{task.question} {' '.join(task.options.values())}"
        if any(kw in combined for kw in self._CALC_KEYWORDS):
            return RouteResult(RouteIntent.CALCULATION, "关键词命中")

        llm_hint = await self._llm_assess(task)
        if llm_hint == "CALCULATION":
            return RouteResult(RouteIntent.CALCULATION, "LLM 判定含计算")

        return RouteResult(RouteIntent.LOOKUP, "默认 fallback")

    async def _llm_assess(self, task: TaskJSON) -> str:
        prompt = (
            f"你是一个金融题目分类器。只回答 CALCULATION 或 LOOKUP。\n\n"
            f"题干：{task.question}\n"
            f"选项：{' '.join(task.options.values())}\n\n"
            f"请判断这道题是否需要做数值计算（如增长率、占比、差额、倍数、超过/低于多少等）？"
            f"如果是 → CALCULATION；如果只是查找条款/事实/定义 → LOOKUP。"
        )
        resp = await self._llm.call(prompt, temperature=0.0, max_tokens=16)
        resp = resp.strip().upper()
        if "CALCULATION" in resp:
            return "CALCULATION"
        return "LOOKUP"


# ═══════════════════════════════════════════════════
# 检索策略基类
# ═══════════════════════════════════════════════════

class BaseStrategy(ABC):
    """策略基类，所有子类必须实现 retrieve()"""

    def __init__(self, llm: AsyncLLMClient, retriever: MilvusRetriever):
        self._llm = llm
        self._retriever = retriever

    @abstractmethod
    async def retrieve(self, task: TaskJSON) -> ContextDict:
        ...

    async def _bm25_search(
        self,
        query: str,
        domain: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = _DEFAULT_BM25_TOP_K,
    ) -> List[ChunkRow]:
        return await self._retriever.search(
            queries=[query],
            doc_ids=doc_ids,
            domain=domain,
            limit=top_k,
            timeout=_DEFAULT_BM25_TIMEOUT,
        )


# ═══════════════════════════════════════════════════
# ClauseLookupStrategy — 单文档条款/事实检索
# ═══════════════════════════════════════════════════

class ClauseLookupStrategy(BaseStrategy):
    """
    条款与事实查找策略。

    流程：
      1. LLM Query Rewrite：题干 → 提取最具区分度的名词短语（最多5个词）
      2. 用改写后的关键词 + 选项组成新 query
      3. BM25 检索（doc_id 硬隔离）
      4. 拼接证据上下文
    """

    async def retrieve(self, task: TaskJSON) -> ContextDict:
        # Step 1: LLM Query Rewrite
        query = await self._rewrite_query(task)

        # Step 2: BM25 检索
        doc_ids = task.doc_ids if task.doc_ids else None
        rows = await self._bm25_search(query, task.domain, doc_ids, top_k=20)

        # Step 3: 拼接证据
        evidence = self._format_evidence(rows)
        return ContextDict(evidence=evidence)

    async def _rewrite_query(self, task: TaskJSON) -> str:
        """用 LLM 提取最具区分度的关键词，最多 5 个词"""
        prompt = (
            f"请从以下题干和选项中提取最具区分度的核心名词短语，"
            f"用于 BM25 全文检索。去除废话（如'根据…'、'下列哪些'、'描述'等）。"
            f"最多输出 5 个词，以空格分隔。\n\n"
            f"题干：{task.question}\n"
            f"选项：{' '.join(task.options.values())}"
        )
        rewritten = await self._llm.call(prompt, temperature=0.0, max_tokens=64)
        # 如果 LLM 返回为空，回退到原始拼接
        if not rewritten.strip():
            query_parts = [task.question]
            for key in sorted(task.options.keys()):
                query_parts.append(task.options[key])
            return " ".join(query_parts)
        return rewritten.strip()

    @staticmethod
    def _format_evidence(rows: List[ChunkRow]) -> str:
        if not rows:
            return "未检索到相关事实，无法判断"

        blocks = []
        for i, r in enumerate(rows):
            text_clean = r.text[:600].replace("\n", " ")
            src = f"[{r.doc_id}]" if r.doc_id else ""
            blocks.append(f"--- 片段{i+1} {src} ---\n{text_clean}")

        return "\n\n".join(blocks)


# ═══════════════════════════════════════════════════
# CrossDocCompareStrategy — 跨文档对比（并发检索）
# ═══════════════════════════════════════════════════

class CrossDocCompareStrategy(BaseStrategy):
    """
    跨文档对比策略。

    对每个 doc_id 启动独立的并发 BM25 检索 → 打标签组装上下文。
    """

    async def retrieve(self, task: TaskJSON) -> ContextDict:
        if not task.doc_ids or len(task.doc_ids) <= 1:
            return await ClauseLookupStrategy(self._llm, self._retriever).retrieve(task)

        # 构造统一 query
        query_parts = [task.question]
        for key in sorted(task.options.keys()):
            query_parts.append(task.options[key])
        query = " ".join(query_parts)

        # 并发对每个 doc_id 检索（带超时保护）
        async def search_one(doc_id: str) -> Tuple[str, List[ChunkRow]]:
            try:
                rows = await asyncio.wait_for(
                    self._retriever.search(
                        queries=[query],
                        doc_ids=[doc_id],
                        domain=task.domain,
                        limit=10,
                        timeout=_DEFAULT_BM25_TIMEOUT,
                    ),
                    timeout=_DEFAULT_BM25_TIMEOUT + 2,
                )
                return doc_id, rows
            except Exception as e:
                logger.warning(f"跨文档检索失败 doc_id={doc_id}: {e}")
                return doc_id, []

        results = await asyncio.gather(*[search_one(d) for d in task.doc_ids],
                                       return_exceptions=True)

        blocks = []
        for item in results:
            if isinstance(item, Exception):
                continue
            doc_id, rows = item
            if not rows:
                blocks.append(f"===== 文档【{doc_id}】=====\n（无匹配片段）")
                continue
            parts = [f"===== 文档【{doc_id}】====="]
            for i, r in enumerate(rows):
                parts.append(f"片段{i+1}: {r.text[:500].replace(chr(10), ' ')}")
            blocks.append("\n".join(parts))

        return ContextDict(evidence="\n\n".join(blocks) if blocks else "未检索到相关事实，无法判断")


# ═══════════════════════════════════════════════════
# CalculationStrategy — 纯文本计算（BM25→LLM提取→Python计算）
# ═══════════════════════════════════════════════════

class CalculationStrategy(BaseStrategy):
    """
    纯文本计算策略。

    步骤：
      1. BM25 检索（优选 table 类型 chunk）
      2. LLM 提取指标名 + 并发提取数字
      3. try-except float() 解析数字，失败返回兜底文本
      4. 计算过程文本化
    """

    async def retrieve(self, task: TaskJSON) -> ContextDict:
        # ── Step 1: BM25 检索 ──
        query_parts = [task.question]
        for key in sorted(task.options.keys()):
            query_parts.append(task.options[key])
        query = " ".join(query_parts)

        doc_ids = task.doc_ids if task.doc_ids else None
        rows = await self._bm25_search(query, task.domain, doc_ids, top_k=30)

        if not rows:
            return ContextDict(
                evidence="未检索到相关事实，无法判断",
                calculations="未检索到相关事实，无法判断",
            )

        # 优选 table 类型
        table_rows = [r for r in rows if r.chunk_type == "table"]
        text_rows = [r for r in rows if r.chunk_type != "table"]
        top_rows = (table_rows[:5] + text_rows[:10])[:15]
        evidence_text = "\n".join(f"[{r.doc_id}] {r.text[:800]}" for r in top_rows)

        # ── Step 2: LLM 提取指标名 ──
        extraction_tasks = []
        for opt_key in sorted(task.options.keys()):
            opt_text = task.options[opt_key]
            prompt = (
                f"从以下选项中提取需要计算的财务指标名称（如'营业收入增长率'、'净利润'），"
                f"只返回指标名称，不要多余文字。\n选项：{opt_text}"
            )
            extraction_tasks.append(self._llm.call(prompt, temperature=0.0, max_tokens=64))

        metric_names = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # ── Step 3: 并发提取数字（带严格解析）──
        number_tasks = []
        for mn in metric_names:
            metric = str(mn).strip() if not isinstance(mn, Exception) else ""
            if not metric:
                continue
            prompt = (
                f"请从以下文档文本中提取【{metric}】的数值。"
                f"对于每个数值，标注年份/期间。"
                f"只输出纯数字，不要单位，不要标点，找不到输出 'None'。"
                f"每行一个。\n\n文本：\n{evidence_text[:3000]}"
            )
            number_tasks.append(self._llm.call(prompt, temperature=0.0, max_tokens=256))

        number_results = await asyncio.gather(*number_tasks, return_exceptions=True)

        # ── Step 4: 安全解析数字 ──
        numbers: Dict[str, Any] = {}
        for i, nr in enumerate(number_results):
            opt_key = sorted(task.options.keys())[i]
            raw = str(nr).strip() if not isinstance(nr, Exception) else "None"
            # 严格 try-float 过滤
            parsed_values = []
            for line in raw.split("\n"):
                line = line.strip()
                # 尝试提取纯数字
                try:
                    # 先试整体
                    val = float(line)
                    parsed_values.append(str(val))
                except ValueError:
                    # 检查是否包含 None
                    if line.upper() == "NONE":
                        continue
                    # 尝试从行中正则找数字
                    nums = re.findall(r'-?\d+(?:\.\d+)?', line)
                    if nums:
                        parsed_values.append(nums[0])
            if parsed_values:
                numbers[f"选项{opt_key}"] = "; ".join(parsed_values)
            else:
                numbers[f"选项{opt_key}"] = "数值提取失败"

        # ── Step 5: 计算过程文本化 ──
        calculations = self._build_calculations(task, numbers)

        evidence = "\n".join(
            f"[{r.doc_id}] {r.text[:600].replace(chr(10), ' ')}"
            for r in top_rows
        )

        return ContextDict(evidence=evidence, calculations=calculations, numbers=numbers)

    @staticmethod
    def _build_calculations(task: TaskJSON, numbers: Dict[str, Any]) -> str:
        if not numbers:
            return "（未能提取到有效数值）"
        lines = ["【数值提取结果】"]
        for key, val in numbers.items():
            lines.append(f"  {key}: {val}")
        lines.append("")
        lines.append("【题目】")
        lines.append(task.question)
        for k in sorted(task.options.keys()):
            lines.append(f"  {k}. {task.options[k]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════
# 策略注册表
# ═══════════════════════════════════════════════════

STRATEGY_REGISTRY: Dict[RouteIntent, type] = {
    RouteIntent.LOOKUP: ClauseLookupStrategy,
    RouteIntent.CROSS_DOC: CrossDocCompareStrategy,
    RouteIntent.CALCULATION: CalculationStrategy,
}


# ═══════════════════════════════════════════════════
# 智能答题器
# ═══════════════════════════════════════════════════

_RETRY_INSTRUCTION = (
    "⚠️ 注意：你刚才的输出格式不符合要求。\n"
    "请只在最后一行输出答案字母，不要输出任何其他字符（包括思考过程、标点符号等）。\n"
    "例如：A  或  AC  或  B"
)


class AnswerFormatter:
    """
    基于策略产出的上下文，生成最终答案。

    - mcq / tf: 单次 LLM + 降温重试
    - multi: NLI 模式并发判断 A/B/C/D True/False → Python 拼接
    """

    def __init__(self, llm: AsyncLLMClient):
        self._llm = llm

    async def evaluate(self, context: ContextDict, task: TaskJSON) -> str:
        if task.answer_format in ("mcq", "tf"):
            return await self._single_shot(context, task)
        elif task.answer_format == "multi":
            return await self._pointwise_multi(context, task)
        else:
            logger.warning(f"未知 answer_format: {task.answer_format}")
            return "A"

    # ── 单次调用 ──

    async def _single_shot(self, context: ContextDict, task: TaskJSON) -> str:
        parts = []
        if context.evidence:
            parts.append(f"===== 文档证据 =====\n{context.evidence}")
        if context.calculations:
            parts.append(f"===== 计算过程 =====\n{context.calculations}")
        context_text = "\n\n".join(parts) if parts else "（无上下文）"

        options_text = "\n".join(
            f"{k}. {v}" for k, v in sorted(task.options.items())
        )
        fmt_hint = {
            "mcq": "单选题：从 A、B、C、D 中选择唯一正确答案。",
            "tf": "判断题：A=正确，B=错误。",
        }.get(task.answer_format, "")

        prompt = (
            f"你是一位专业的金融文档分析专家。\n\n"
            f"基于以下文档片段和（可能的）计算过程，回答题目。\n"
            f"严格按照文档内容，不要依赖外部知识。\n\n"
            f"{context_text}\n\n"
            f"【题目】\n{task.question}\n\n"
            f"【选项】\n{options_text}\n\n"
            f"{fmt_hint}\n"
            f"先在<思考>区域分析每个选项的正确性。\n"
            f"最后，单独一行输出答案字母。"
        )

        # 第一次
        resp = await self._llm.call(prompt, temperature=0.3)
        answer = self._extract_last_letter(resp)
        if answer:
            return answer

        # 降温重试
        resp = await self._llm.call(
            prompt + f"\n\n{_RETRY_INSTRUCTION}",
            temperature=0.0,
        )
        answer = self._extract_last_letter(resp)
        if answer:
            return answer

        return "A"

    # ── Point-wise 多选 ──

    async def _pointwise_multi(self, context: ContextDict, task: TaskJSON) -> str:
        options = task.options
        context_text = context.evidence or "（无上下文）"
        if context.calculations:
            context_text += f"\n\n===== 计算过程 =====\n{context.calculations}"

        async def judge_one(opt_key: str, opt_text: str) -> Tuple[str, bool]:
            """NLI 模式：前提→假设→TRUE/FALSE"""
            prompt = (
                f"前提：{context_text[:4000]}\n\n"
                f"假设：{task.question} 因此 {opt_text}\n\n"
                f"请问前提是否充分支持了假设？请且仅输出大写的 TRUE 或 FALSE。"
            )
            try:
                resp = await asyncio.wait_for(
                    self._llm.call(prompt, temperature=0.0, max_tokens=16),
                    timeout=_DEFAULT_LLM_TIMEOUT,
                )
                # 健壮解析：剥离所有前缀空格和标点
                cleaned = re.sub(r'^[\s.,!?;:。，！？；：\']*', '', resp.strip()).upper()
                is_true = cleaned.startswith("TRUE")
                return opt_key, is_true
            except Exception:
                # 任一协程异常 → 该选项默认 False，不拖垮整题
                return opt_key, False

        tasks = [judge_one(k, v) for k, v in sorted(options.items())]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        correct_letters = []
        for r in results:
            if isinstance(r, Exception):
                continue
            opt_key, is_true = r
            if is_true:
                correct_letters.append(opt_key)

        return "".join(sorted(correct_letters)) if correct_letters else "A"

    @staticmethod
    def _extract_last_letter(text: str) -> Optional[str]:
        if not text:
            return None
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return None
        last = lines[-1]
        match = re.search(r'\b([A-D])\b', last.upper())
        if match:
            return match.group(1)
        if re.match(r'^[A-D]$', last.upper()):
            return last.upper()
        return None


# ═══════════════════════════════════════════════════
# 编排器 — TaskOrchestrator
# ═══════════════════════════════════════════════════

class TaskOrchestrator:
    """
    任务编排器：单题/批量处理入口。
    持有所有依赖，按需路由 + 策略 + 格式化。
    """

    def __init__(self, llm: AsyncLLMClient, retriever: MilvusRetriever):
        self._llm = llm
        self._retriever = retriever
        self._router = IntentRouter(llm)

    async def run(self, task_json: dict) -> str:
        """
        处理单道题。

        Args:
            task_json: 原始题目 dict（来自 question_loader）

        Returns:
            答案字符串，如 "A", "ACD"
        """
        task = TaskJSON.from_raw(task_json)

        # 路由
        route_result = await self._router.route(task)
        logger.info(f"[{task.qid}] 路由: {route_result.intent.value} ({route_result.reason})")

        # 策略
        strategy_cls = STRATEGY_REGISTRY.get(route_result.intent, ClauseLookupStrategy)
        strategy = strategy_cls(self._llm, self._retriever)
        context = await strategy.retrieve(task)
        logger.info(
            f"[{task.qid}] 上下文: evidence={len(context.evidence)}c, "
            f"calcs={len(context.calculations)}c, nums={len(context.numbers)}"
        )

        # 答案
        formatter = AnswerFormatter(self._llm)
        answer = await formatter.evaluate(context, task)
        logger.info(f"[{task.qid}] 答案: {answer}")
        return answer

    async def run_batch(self, task_jsons: List[dict]) -> List[str]:
        """
        批量处理多道题（串行执行，避免 QPS 超限）。
        """
        answers = []
        for tj in task_jsons:
            ans = await self.run(tj)
            answers.append(ans)
        return answers

    async def run_with_result(self, task_json: dict) -> Dict[str, Any]:
        """
        处理单道题并返回带 Token 消耗的结果 dict（用于 answer.csv）。

        Args:
            task_json: 原始题目 dict

        Returns:
            {"qid": str, "answer": str, "prompt_tokens": int,
             "completion_tokens": int, "total_tokens": int}
        """
        # 在每题开始时重置 LLM token计数器记录该题消耗，
        # 但 AsyncLLMClient 是全局累加，所以捕获前后的差值
        pt_before = self._llm._total_prompt_tokens
        ct_before = self._llm._total_completion_tokens

        answer = await self.run(task_json)
        qid = task_json.get("qid", "")

        pt_after = self._llm._total_prompt_tokens
        ct_after = self._llm._total_completion_tokens
        pt_diff = pt_after - pt_before
        ct_diff = ct_after - ct_before

        return {
            "qid": qid,
            "answer": answer,
            "prompt_tokens": pt_diff,
            "completion_tokens": ct_diff,
            "total_tokens": pt_diff + ct_diff,
        }

    async def run_batch_with_results(self, task_jsons: List[dict]) -> List[Dict[str, Any]]:
        """
        批量处理并返回每道题的结果 dict（含 Token 消耗）。

        Returns:
            [{"qid": ..., "answer": ..., "prompt_tokens": ..., ...}, ...]
        """
        results = []
        for tj in task_jsons:
            result = await self.run_with_result(tj)
            results.append(result)
        return results
