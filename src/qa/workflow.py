"""
A 榜评测路由与调度框架 (Agentic RAG Workflow)

采用策略模式 (Strategy Pattern) 实现三路检索+推理策略：
  - ClauseLookupStrategy    : 单文档事实检索
  - CrossDocCompareStrategy : 跨文档对比（并发检索）
  - CalculationStrategy     : 纯文本计算（BM25→LLM提取数字→Python计算）

依赖注入：
  - llm_call: async (prompt: str) → str           （对接 QwenClient）
  - bm25_search: async (query, domain, doc_ids) → [ChunkRow] （对接 Milvus BM25）

红线：
  - 不使用任何 embedding 模型
  - 不使用 SQL 数据库
  - 所有数据仅来自 Milvus BM25（稀疏向量）索引中的 Chunk
"""

import asyncio
import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Coroutine, List, Dict, Optional, Any, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 类型定义
# ═══════════════════════════════════════════════════

@dataclass
class TaskJSON:
    """标准化题目结构（来自 question_loader）"""
    qid: str
    domain: str
    split: str
    question: str
    options: Dict[str, str]           # {"A": "...", "B": "...", ...}
    answer_format: str                # "mcq" | "multi" | "tf"
    type: str                         # 原始题型（不可信）
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
    """路由意图枚举"""
    LOOKUP = "LOOKUP"                    # 条款/事实查找
    CROSS_DOC = "CROSS_DOC"              # 跨文档对比
    CALCULATION = "CALCULATION"          # 数值计算


@dataclass
class RouteResult:
    """路由决策结果"""
    intent: RouteIntent
    reason: str = ""


@dataclass
class ChunkRow:
    """单条 BM25 检索结果（与 milvus_bm25.search_bm25 输出对齐）"""
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
    """
    策略产出上下文，包含多来源的对齐文本和中间计算结果。

    fields:
      - evidence:   检索到的证据片段（自然语言拼接）
      - calculations: 计算过程的文本描述（仅 CALCULATION 策略使用）
      - numbers:     提取到的结构化数字（仅 CALCULATION 策略使用）
    """
    evidence: str = ""
    calculations: str = ""
    numbers: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════
# 依赖注入类型签名
# ═══════════════════════════════════════════════════

# LLM 调用桩：接收 prompt → 返回模型输出文本
AsyncLLMCall = Callable[[str], Coroutine[Any, Any, str]]

# BM25 检索桩：接收 query + domain + doc_ids + top_k → 返回 ChunkRow 列表
AsyncBM25Search = Callable[
    [str, str, Optional[List[str]], int],
    Coroutine[Any, Any, List[ChunkRow]],
]


# ═══════════════════════════════════════════════════
# 意图路由器
# ═══════════════════════════════════════════════════

class IntentRouter:
    """
    根据题目特征和 LLM 分析，路由到合适的策略。

    规则：
      1. doc_ids > 1 → CROSS_DOC（跨文档）
         (A 榜的 doc_ids 是可信的原始数据)
      2. LLM 分析题干 → 含"增长率/比例/占比/多少"等计算暗示 → CALCULATION
      3. 兜底 → LOOKUP
    """

    # 关键词快速判定（替代 LLM 的轻量线索）
    _CALC_KEYWORDS = [
        "增长率", "增长比例", "同比", "环比",
        "占比", "比例", "比率", "率",
        "多少", "差额", "差值", "多出", "高出",
        "超过", "低于", "高于",
        "倍数", "翻", "百分", "%",
        "加总", "合计", "总和", "之和", "之差",
        "营业收入增长率", "利润增长率",
    ]

    def __init__(self, llm_call: AsyncLLMCall):
        """
        Args:
            llm_call: async (prompt) → str，LLM 调用函数
        """
        self._llm_call = llm_call

    async def route(self, task: TaskJSON) -> RouteResult:
        """
        路由决策主入口。

        Args:
            task: 标准化题目

        Returns:
            RouteResult: 意图 + 理由
        """
        # 规则1: 多文档 → 跨文档对比
        if len(task.doc_ids) > 1:
            return RouteResult(RouteIntent.CROSS_DOC, "len(doc_ids) > 1")

        # 规则2: 轻量关键词预判（快速路径）
        combined = f"{task.question} {' '.join(task.options.values())}"
        if any(kw in combined for kw in self._CALC_KEYWORDS):
            return RouteResult(RouteIntent.CALCULATION, "关键词命中")

        # 规则2bis: LLM 深度判断
        llm_hint = await self._llm_assess(task)
        if llm_hint == "CALCULATION":
            return RouteResult(RouteIntent.CALCULATION, "LLM 判定含计算")

        # 兜底
        return RouteResult(RouteIntent.LOOKUP, "默认 fallback")

    async def _llm_assess(self, task: TaskJSON) -> str:
        """用 LLM 判断是否涉及计算"""
        prompt = (
            f"你是一个金融题目分类器。只回答 CALCULATION 或 LOOKUP。\n\n"
            f"题干：{task.question}\n"
            f"选项：{' '.join(task.options.values())}\n\n"
            f"请判断这道题是否需要做数值计算（如增长率、占比、差额、倍数、超过/低于多少等）？"
            f"如果是 → CALCULATION；如果只是查找条款/事实/定义 → LOOKUP。"
        )
        try:
            resp = await self._llm_call(prompt)
            resp = resp.strip().upper()
            if "CALCULATION" in resp:
                return "CALCULATION"
        except Exception as e:
            logger.warning(f"LLM 路由评估异常: {e}")
        return "LOOKUP"


# ═══════════════════════════════════════════════════
# 检索策略基类 + 子类
# ═══════════════════════════════════════════════════

class BaseStrategy(ABC):
    """策略基类，所有子类必须实现 retrieve()"""

    def __init__(self, llm_call: AsyncLLMCall, bm25_search: AsyncBM25Search):
        """
        Args:
            llm_call: async (prompt) → str
            bm25_search: async (query, domain, doc_ids, top_k) → List[ChunkRow]
        """
        self._llm = llm_call
        self._bm25 = bm25_search

    @abstractmethod
    async def retrieve(self, task: TaskJSON) -> ContextDict:
        """执行检索并返回上下文"""
        ...

    async def _bm25_query(
        self,
        query: str,
        domain: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[ChunkRow]:
        """简化的 BM25 检索入口"""
        try:
            return await self._bm25(query, domain, doc_ids, top_k)
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []


class ClauseLookupStrategy(BaseStrategy):
    """
    条款与事实查找策略（单文档检索）。

    用题干 + 选项拼接成 query → BM25 检索（按 doc_ids 过滤）→ 拼接证据文本。
    """

    async def retrieve(self, task: TaskJSON) -> ContextDict:
        # 构造查询文本
        query_parts = [task.question]
        for key in sorted(task.options.keys()):
            query_parts.append(task.options[key])
        query = " ".join(query_parts)

        # BM25 检索（A 榜固定 doc_ids 过滤）
        doc_ids = task.doc_ids if task.doc_ids else None
        rows = await self._bm25_query(query, task.domain, doc_ids, top_k=20)

        # 拼接证据（保留文档来源）
        evidence = self._format_evidence(rows)
        return ContextDict(evidence=evidence)

    @staticmethod
    def _format_evidence(rows: List[ChunkRow]) -> str:
        """将检索结果格式化为可读的上下文"""
        if not rows:
            return "【未检索到相关文档片段】"

        blocks = []
        for i, r in enumerate(rows):
            text_clean = r.text[:600].replace("\n", " ")
            src = f"[{r.doc_id}]" if r.doc_id else ""
            blocks.append(f"--- 片段{i+1} {src} ---\n{text_clean}")

        return "\n\n".join(blocks)


class CrossDocCompareStrategy(BaseStrategy):
    """
    跨文档对比策略。

    对每个 doc_id 启动独立的并发 BM25 检索 → 打上文档标签组装。
    """

    async def retrieve(self, task: TaskJSON) -> ContextDict:
        if not task.doc_ids or len(task.doc_ids) <= 1:
            # 退化到 LOOKUP
            return await ClauseLookupStrategy(self._llm, self._bm25).retrieve(task)

        # 构造统一 query
        query_parts = [task.question]
        for key in sorted(task.options.keys()):
            query_parts.append(task.options[key])
        query = " ".join(query_parts)

        # 并发对每个 doc_id 检索
        async def search_one(doc_id: str) -> Tuple[str, List[ChunkRow]]:
            rows = await self._bm25_query(query, task.domain, doc_ids=[doc_id], top_k=10)
            return doc_id, rows

        results = await asyncio.gather(*[search_one(d) for d in task.doc_ids])

        # 组装：每个文档标签独立区块
        blocks = []
        for doc_id, rows in results:
            if not rows:
                blocks.append(f"===== 文档【{doc_id}】=====\n（无匹配片段）")
                continue
            parts = [f"===== 文档【{doc_id}】====="]
            for i, r in enumerate(rows):
                parts.append(f"片段{i+1}: {r.text[:500].replace(chr(10), ' ')}")
            blocks.append("\n".join(parts))

        return ContextDict(evidence="\n\n".join(blocks))


class CalculationStrategy(BaseStrategy):
    """
    纯文本计算策略（无 SQL 版）。

    步骤：
      1. BM25 召回包含财务指标的 Markdown 表格 Chunk（优先 table 类型）
      2. LLM 提取数字（"提取[指标]的数值，只返回数字"）
      3. Python 内存中执行计算
      4. 将计算过程和结果包装为文本
    """

    async def retrieve(self, task: TaskJSON) -> ContextDict:
        # ── Step 1: BM25 检索（召回复盖表格和文本）──
        query_parts = [task.question]
        for key in sorted(task.options.keys()):
            query_parts.append(task.options[key])
        query = " ".join(query_parts)

        doc_ids = task.doc_ids if task.doc_ids else None
        rows = await self._bm25_query(query, task.domain, doc_ids, top_k=30)

        if not rows:
            return ContextDict(
                evidence="【未检索到相关文档片段】",
                calculations="无数据，无法计算",
            )

        # 优选 table 类型 chunk（更可能有结构化数值）
        table_rows = [r for r in rows if r.chunk_type == "table"]
        text_rows = [r for r in rows if r.chunk_type != "table"]
        # 最多取 5 个 table + 10 个 text
        top_rows = (table_rows[:5] + text_rows[:10])[:15]

        # 将所有召回文本合并，供 LLM 提取
        evidence_text = "\n".join(f"[{r.doc_id}] {r.text[:800]}" for r in top_rows)

        # ── Step 2: LLM 提取数字 ──
        # 从每个选项表述中抽取指标名，然后并发提取数字
        numbers: Dict[str, Any] = {}
        extraction_tasks = []

        for opt_key in sorted(task.options.keys()):
            opt_text = task.options[opt_key]
            # 用 LLM 提取该选项需要的指标名
            extract_prompt = (
                f"从以下选项中提取需要计算的财务指标名称（如'营业收入增长率'、'净利润'），"
                f"只返回指标名称，不要多余文字。\n选项：{opt_text}"
            )
            extraction_tasks.append(self._llm(extract_prompt))

        metric_names = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # 并发提取每个指标的数字
        number_tasks = []
        for mn in metric_names:
            metric = str(mn).strip() if not isinstance(mn, Exception) else None
            if not metric or metric == "":
                continue
            prompt = (
                f"请从以下文档文本中提取【{metric}】的数值。"
                f"对于每个数值，标注年份/期间。只返回数字和对应年份，每行一个。"
                f"如果找不到返回'找不到'。\n\n文本：\n{evidence_text[:3000]}"
            )
            number_tasks.append(self._llm(prompt))

        number_results = await asyncio.gather(*number_tasks, return_exceptions=True)

        # 解析数字提取结果
        for i, (opt_key, nr) in enumerate(zip(sorted(task.options.keys()), number_results)):
            metric = str(metric_names[i]) if i < len(metric_names) and not isinstance(metric_names[i], Exception) else ""
            val = str(nr) if not isinstance(nr, Exception) else "提取失败"
            if val and val != "找不到":
                numbers[f"选项{opt_key}({metric})"] = val

        # ── Step 3: Python 内存计算 ──
        # （这里只能做简单字符串比较，因为数字已被 LLM 提取为文本）
        calculations = self._build_calculations(task, numbers)

        # ── 证据正文（保留原始检索结果）──
        evidence = "\n".join(
            f"[{r.doc_id}] {r.text[:600].replace(chr(10), ' ')}"
            for r in top_rows
        )

        return ContextDict(
            evidence=evidence,
            calculations=calculations,
            numbers=numbers,
        )

    @staticmethod
    def _build_calculations(task: TaskJSON, numbers: Dict[str, str]) -> str:
        """基于提取到的数字构建计算过程文本"""
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
# 策略注册表（工厂）
# ═══════════════════════════════════════════════════

STRATEGY_REGISTRY: Dict[RouteIntent, type] = {
    RouteIntent.LOOKUP: ClauseLookupStrategy,
    RouteIntent.CROSS_DOC: CrossDocCompareStrategy,
    RouteIntent.CALCULATION: CalculationStrategy,
}


# ═══════════════════════════════════════════════════
# 智能答题器
# ═══════════════════════════════════════════════════

class AnswerFormatter:
    """
    基于策略产出的上下文，执行最终答案生成。

    分题型：
      - mcq / tf: 单次 LLM 调用，强制输出唯一大写字母
      - multi: Point-wise 并发判决（A/B/C/D 分别判断 True/False），由 Python 拼接
    """

    def __init__(self, llm_call: AsyncLLMCall):
        self._llm = llm_call

    async def evaluate(
        self,
        context: ContextDict,
        task: TaskJSON,
    ) -> str:
        """
        根据上下文生成最终答案。

        Args:
            context: 策略产出的上下文
            task: 原题信息

        Returns:
            答案字符串，如 "A", "AC", "B"
        """
        if task.answer_format in ("mcq", "tf"):
            return await self._single_shot(context, task)
        elif task.answer_format == "multi":
            return await self._pointwise_multi(context, task)
        else:
            logger.warning(f"未知 answer_format: {task.answer_format}")
            return "A"  # 安全 fallback

    # ── 单次调用（单选题 / 判断题）──

    async def _single_shot(self, context: ContextDict, task: TaskJSON) -> str:
        """单次 LLM 调用，综合上下文输出唯一答案字母"""

        # 组装上下文（证据 + 计算过程）
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

        try:
            resp = await self._llm(prompt)
            answer = self._extract_last_letter(resp)
            if answer:
                return answer
        except Exception as e:
            logger.error(f"_single_shot LLM 调用失败: {e}")

        return "A"  # fallback

    # ── Point-wise 多选（多选题）──

    async def _pointwise_multi(self, context: ContextDict, task: TaskJSON) -> str:
        """
        对 A/B/C/D 四个选项并发判断 True/False，由 Python 收集后拼接。

        每路 prompt：让 LLM 只输出 True 或 False。
        """
        options = task.options
        context_text = context.evidence or "（无上下文）"
        if context.calculations:
            context_text += f"\n\n===== 计算过程 =====\n{context.calculations}"

        async def judge_one(opt_key: str, opt_text: str) -> Tuple[str, bool]:
            prompt = (
                f"基于以下文档片段，判断选项是否正确。\n\n"
                f"{context_text[:4000]}\n\n"
                f"【题干】{task.question}\n"
                f"【选项 {opt_key}】{opt_text}\n\n"
                f"这个选项是正确的吗？只回答 True 或 False。"
            )
            try:
                resp = await self._llm(prompt)
                return opt_key, "TRUE" in resp.strip().upper()
            except Exception:
                return opt_key, False

        # 并发 4 路判断
        tasks = [judge_one(k, v) for k, v in sorted(options.items())]
        results = await asyncio.gather(*tasks)

        # 收集被判定为 True 的选项字母，按字母顺序拼接
        correct_letters = [k for k, v in results if v]
        return "".join(sorted(correct_letters)) if correct_letters else "A"

    @staticmethod
    def _extract_last_letter(text: str) -> Optional[str]:
        """
        从 LLM 输出中提取最后一行的单个大写字母（A/B/C/D）。

        Returns:
            提取到的字母，或 None
        """
        if not text:
            return None
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return None
        last = lines[-1]
        match = re.search(r'\b([A-D])\b', last.upper())
        if match:
            return match.group(1)
        # 如果最后一行是纯字母且合法
        if re.match(r'^[A-D]$', last.upper()):
            return last.upper()
        return None


# ═══════════════════════════════════════════════════
# LLM 调用工厂：构建与 QwenClient 一致的 ChatOpenAI 异步包装
# ═══════════════════════════════════════════════════

def build_llm_call() -> AsyncLLMCall:
    """
    构建一个 async (prompt: str) → str 的 LLM 调用函数。

    底层使用 LangChain 的 ChatOpenAI（与 qwen_client.py 一致的初始化方式）：
      - 模型名从环境变量 QWEN_MODEL 读取（.env 中配置）
      - API Key 从环境变量 DASHSCOPE_API_KEY 读取
      - base_url 固定为阿里云百炼 OpenAI 兼容端点
      - temperature=0.3, max_tokens=2048, timeout=120, enable_thinking=False
      - 重试 3 次

    Returns:
        async (prompt: str) → str 的异步可调用对象

    用法：
        llm_call = build_llm_call()
        answer = await llm_call("请回答问题...")
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    model = os.environ.get("QWEN_MODEL", "qwen3.6-27b")

    fallback_model = "qwen3.6-27b"
    active_model = model

    async def llm_call(prompt: str) -> str:
        nonlocal active_model
        last_error = None
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(1, max_retries + 1):
            try:
                chat = ChatOpenAI(
                    model=active_model,
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=120,
                    max_retries=0,
                    extra_body={"enable_thinking": False},
                )
                response = chat.invoke([HumanMessage(content=prompt)])
                return response.content.strip()
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_server = any(kw in err_str for kw in [
                    "429", "5xx", "500", "502", "503", "504",
                    "rate limit", "too many requests", "quota exceeded",
                    "server error", "service unavailable",
                    "internal error", "timeout",
                ])
                # 如果是服务端错误且尚未切换模型，尝试备用模型
                if is_server and active_model != fallback_model:
                    logger.warning(
                        f"模型 {active_model} 异常，切换至 {fallback_model}: {e}"
                    )
                    active_model = fallback_model
                elif attempt < max_retries:
                    logger.warning(
                        f"LLM 调用失败 (attempt {attempt}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(retry_delay * attempt)

        raise RuntimeError(f"LLM 调用全部重试失败: {last_error}")

    return llm_call


# ═══════════════════════════════════════════════════
# 题级编排入口：逐题处理
# ═══════════════════════════════════════════════════

async def process_single_task(
    task_json: dict,
    llm_call: AsyncLLMCall,
    bm25_search: AsyncBM25Search,
) -> str:
    """
    处理单道题：路由 → 策略检索 → 答案生成。

    Args:
        task_json: 原始题目 dict（来自 question_loader）
        llm_call: async (prompt) → str
        bm25_search: async (query, domain, doc_ids, top_k) → List[ChunkRow]

    Returns:
        答案字符串，如 "A", "ACD"
    """
    task = TaskJSON.from_raw(task_json)

    # ── 路由 ──
    router = IntentRouter(llm_call)
    route_result = await router.route(task)
    logger.info(f"[{task.qid}] 路由: {route_result.intent.value} ({route_result.reason})")

    # ── 策略检索 ──
    strategy_cls = STRATEGY_REGISTRY.get(route_result.intent)
    if strategy_cls is None:
        logger.warning(f"[{task.qid}] 无对应策略，回退 LOOKUP")
        strategy_cls = ClauseLookupStrategy

    strategy = strategy_cls(llm_call, bm25_search)
    context = await strategy.retrieve(task)
    logger.info(f"[{task.qid}] 上下文: evidence={len(context.evidence)}c, "
                f"calcs={len(context.calculations)}c, nums={len(context.numbers)}")

    # ── 答案生成 ──
    formatter = AnswerFormatter(llm_call)
    answer = await formatter.evaluate(context, task)
    logger.info(f"[{task.qid}] 答案: {answer}")

    return answer