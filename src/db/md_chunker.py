"""
基于 MarkdownHeaderTextSplitter 对 mineru 离线解析的 Markdown 文档进行分块。

使用方法：
    python src/db/md_chunker.py                          # 全部分块
    python src/db/md_chunker.py --domain insurance       # 只分某个领域

处理流程：
    1. 清洗（通用 + regulatory 专属）
    2. inject_financial_headings() — 将金融文档中的多级序号转为 Markdown 标题语法
    3. MarkdownHeaderTextSplitter — 按层级标题切分，产出带 heading_path 的 chunk
    4. 对每个 chunk：提取内部 <table> 为独立 table chunk；正文 text 做二级拆分
    5. heading_path 扁平化拼接到 chunk text 开头
"""

import re
import json
import logging
import argparse
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MINERU_DIR = PROJECT_ROOT / "data" / "mineru_output"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

CHUNK_MAX_CHARS = 1000  # 每个 chunk 最大字符数

DOMAINS = [
    "insurance",
    "regulatory",
    "financial_contracts",
    "financial_reports",
    "research",
]

# ──────────────────────────────────────────────
# HTML Table → Markdown 解析器
# ──────────────────────────────────────────────

class TableToMarkdownParser(HTMLParser):
    """将 HTML <table> 转为 Markdown 管道表格"""

    def __init__(self):
        super().__init__()
        self._rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell = ""
        self._in_td = False
        self._in_th = False
        self._in_table = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._rows = []
        elif tag in ("td", "th"):
            self._current_cell = ""
            if tag == "td":
                self._in_td = True
            else:
                self._in_th = True
        elif tag == "tr":
            self._current_row = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table":
            self._in_table = False
        elif tag in ("td", "th"):
            self._current_row.append(self._current_cell.strip())
            self._current_cell = ""
            self._in_td = False
            self._in_th = False
        elif tag == "tr":
            if self._current_row:
                self._rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data):
        if self._in_td or self._in_th:
            self._current_cell += data

    def get_markdown(self) -> str:
        """将解析到的表格转为 Markdown 管道表格"""
        if not self._rows:
            return ""

        lines = []
        for i, row in enumerate(self._rows):
            # 过滤纯空白行
            cleaned = [c.replace("\n", " ").strip() for c in row]
            lines.append("| " + " | ".join(cleaned) + " |")
            if i == 0:
                # 表头分隔行
                lines.append("| " + " | ".join(["---"] * len(row)) + " |")
        return "\n".join(lines)


def html_table_to_markdown(html_str: str) -> str:
    """将 HTML 表格标签转换为 Markdown 管道表格"""
    parser = TableToMarkdownParser()
    parser.feed(html_str)
    return parser.get_markdown()


# ──────────────────────────────────────────────
# Markdown 清洗
# ──────────────────────────────────────────────


def clean_regulatory_markdown(text: str) -> str:
    """
    清洗 regulatory/html 中的网页导航噪音。
    策略：从出现第一个 '##' 标题行的位置开始保留。
    """
    lines = text.split("\n")
    # 找到第一个 ## 或 # 标题行
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("# "):
            start_idx = i
            break

    if start_idx == 0:
        # 如果未找到标题，用简单规则：丢弃前 20 行
        start_idx = 20

    return "\n".join(lines[start_idx:])


def clean_generic_markdown(text: str) -> str:
    """通用清洗：
    - 移除图片链接 ![](images/...)
    - 压缩连续空行为单个空行
    """
    # 移除图片链接
    text = re.sub(r'!\[.*?\]\(images/[^)]+\)', '', text)
    # 移除纯图片链接行（可能占一整行）
    text = re.sub(r'^!\[.*?\]\([^)]+\)\s*$', '', text, flags=re.MULTILINE)
    # 压缩连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ──────────────────────────────────────────────
# 金融文档多级序号 → Markdown 标题注入
# ──────────────────────────────────────────────

def inject_financial_headings(md_text: str) -> str:
    """
    将金融文档中典型的多级序号强行转换为 Markdown 标题语法（# / ## / ### / ####）。

    转换规则（优先级从高到低，写在前面先匹配）：
      一级(#)  ：第X章 / 第X节 / 第X篇
      二级(##) ：一、 / 二、 / （中文数字顿号）
      三级(###)：（一）/ (一) / 一） / 等括号序号
      四级(####)：1. / 1. / 1) / ① / 等数字序号
      四级(####)：第一条 / 第十条 / 等条款号

    注意：
      - 每条正则均以 ^ 开头，保证只匹配行首独立成行的标题
      - 不覆盖 mineru 已自带的正规 Markdown 标题（行首已是 # 的不重复处理）
      - 处理后的标题依然保留原文标题文本（含序号原文）
    """
    # ── 一级：章 / 节 / 篇 ──
    md_text = re.sub(
        r'^(第[一二三四五六七八九十百千]+[章节篇]\s+.*)$',
        r'# \1', md_text, flags=re.MULTILINE
    )

    # ── 二级：一、/ 二、/ 三、 ... ──
    md_text = re.sub(
        r'^([一二三四五六七八九十]+[、，]\s*.*)$',
        r'## \1', md_text, flags=re.MULTILINE
    )

    # ── 三级：（一）/ (一) / 一） ──
    md_text = re.sub(
        r'^([（\(][一二三四五六七八九十]+[）\)]\s*.*)$',
        r'### \1', md_text, flags=re.MULTILINE
    )

    # ── 四级：1. / 1．/ 1) / 数字序号 ──
    #    注意：排除小数点（如 1.25 这种数字），要求数字后跟 . 、．、) 等
    md_text = re.sub(
        r'^(\d+[\.、．）\)]\s+.*)$',
        r'#### \1', md_text, flags=re.MULTILINE
    )

    # ── 四级：第X条 / 第X款 / 第X项 ──
    #    法规/保险中最高频的结构
    md_text = re.sub(
        r'^(第[一二三四五六七八九十百千零〇]+[条条款款项]\s+.*)$',
        r'#### \1', md_text, flags=re.MULTILINE
    )

    # ── 四级：① / ② / (1) / (2) ──
    md_text = re.sub(
        r'^([①-⑩]|[(（]\d+[)）]\s*.*)$',
        r'#### \1', md_text, flags=re.MULTILINE
    )

    # ── 二级补充：“首先,” “其次,” “最后,” ──
    md_text = re.sub(
        r'^((首先|其次|再次|最后|此外|总之)[、，:：]\s*.*)$',
        r'## \1', md_text, flags=re.MULTILINE
    )

    return md_text


# ──────────────────────────────────────────────
# 二级拆分（超长 chunk）
# ──────────────────────────────────────────────

def split_long_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """在句子边界拆分超长文本"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # 优先在句子边界拆分
    sentences = re.split(r'(?<=[。！？；\n])', text)
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            if current:
                chunks.append(current.strip())
            # 单句超长时硬切
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunk = sentence[i:i + max_chars].strip()
                    if chunk:
                        chunks.append(chunk)
                current = ""
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


# ──────────────────────────────────────────────
# 文件发现
# ──────────────────────────────────────────────

def resolve_md_paths(base_dir: Path = MINERU_DIR) -> dict:
    """
    扫描 mineru_output 目录，按领域返回 [(doc_id, md_path), ...]。

    Returns:
        Dict[str, List[Tuple[str, Path]]]
        e.g. {"insurance": [("ins_1", Path(...)), ...], ...}
    """
    result: dict = {}

    # ── insurance ──
    ins_dir = base_dir / "insurance"
    ins_list = []
    if ins_dir.exists():
        for sub_dir in sorted(ins_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            # 尝试 auto 子目录
            md_files = list((sub_dir / "auto").glob("*.md"))
            if not md_files:
                md_files = list((sub_dir / "hybrid_auto").glob("*.md"))
            if md_files:
                doc_id = f"ins_{sub_dir.name}"
                ins_list.append((doc_id, md_files[0]))
    result["insurance"] = ins_list

    # ── financial_contracts ──
    fc_dir = base_dir / "financial_contracts"
    fc_list = []
    if fc_dir.exists():
        for sub_dir in sorted(fc_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            # 优先 hybrid_auto，其次 auto
            md_files = list((sub_dir / "hybrid_auto").glob("*.md"))
            if not md_files:
                md_files = list((sub_dir / "auto").glob("*.md"))
            if md_files:
                doc_id = f"fc_{sub_dir.name}"
                fc_list.append((doc_id, md_files[0]))
    result["financial_contracts"] = fc_list

    # ── financial_reports ──
    fr_dir = base_dir / "financial_reports"
    fr_list = []
    if fr_dir.exists():
        for sub_dir in sorted(fr_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            doc_id = sub_dir.name  # 目录名即 doc_id
            # auto 子目录
            md_files = list((sub_dir / "auto").glob("*.md"))
            if md_files:
                fr_list.append((doc_id, md_files[0]))
    result["financial_reports"] = fr_list

    # ── regulatory ──
    reg_list = []
    reg_base = base_dir / "regulatory"

    # html 目录（扁平，doc_id = 文件名 stem）
    html_dir = reg_base / "html"
    if html_dir.exists():
        for md_file in sorted(html_dir.glob("*.md")):
            doc_id = md_file.stem
            reg_list.append((doc_id, md_file))

    # txt 目录（扁平，doc_id = 文件名 stem）
    txt_dir = reg_base / "txt"
    if txt_dir.exists():
        for md_file in sorted(txt_dir.glob("*.md")):
            doc_id = md_file.stem
            reg_list.append((doc_id, md_file))

    # attachments 目录（子目录结构，doc_id = 子目录名）
    att_dir = reg_base / "attachments"
    if att_dir.exists():
        for sub_dir in sorted(att_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            doc_id = sub_dir.name
            md_files = list((sub_dir / "hybrid_auto").glob("*.md"))
            if md_files:
                reg_list.append((doc_id, md_files[0]))

    result["regulatory"] = reg_list

    # ── research ──
    res_dir = base_dir / "research"
    res_list = []
    if res_dir.exists():
        for sub_dir in sorted(res_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            md_files = list((sub_dir / "hybrid_auto").glob("*.md"))
            if md_files:
                doc_id = sub_dir.name
                res_list.append((doc_id, md_files[0]))
    result["research"] = res_list

    return result


# ──────────────────────────────────────────────
# 核心分块
# ──────────────────────────────────────────────

def split_by_headers(md_text: str) -> List[dict]:
    """
    使用 MarkdownHeaderTextSplitter 按层级标题切分。

    Returns:
        [{"heading_path": ["h1", "h2"], "text": "...", "char_len": N}, ...]
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )

    docs = splitter.split_text(md_text)
    result = []
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        # 提取 heading_path（过滤空值）
        metadata = doc.metadata
        heading_path = []
        for level in ("h1", "h2", "h3", "h4"):
            val = metadata.get(level, "").strip()
            if val:
                heading_path.append(val)
        result.append({
            "heading_path": heading_path,
            "text": text,
            "char_len": len(text),
        })

    return result


def _format_chunk_text(heading_path: List[str], body: str) -> str:
    """
    将 heading_path 扁平化为 " > " 分隔的字符串，拼在 chunk text 开头。

    Returns:
        "{h1} > {h2} > {h3}\n{body}"  或  "{body}"（当 heading_path 为空时）
    """
    if not heading_path:
        return body
    prefix = " > ".join(heading_path)
    return f"{prefix}\n{body}"


def _extract_tables_from_chunk_text(chunk_text: str) -> List[str]:
    """
    从单个 chunk 文本中提取所有 HTML <table>，转为 Markdown 表格。

    Returns:
        [markdown_table_str, ...]
    """
    pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
    tables = []
    for match in pattern.finditer(chunk_text):
        md_table = html_table_to_markdown(match.group())
        if md_table.strip():
            tables.append(md_table)
    return tables


def _remove_tables_from_text(chunk_text: str) -> str:
    """移除 chunk 文本中的 HTML <table> 标签"""
    return re.sub(r'<table[^>]*>.*?</table>', '', chunk_text, flags=re.DOTALL | re.IGNORECASE).strip()


def chunk_document(
    doc_id: str,
    domain: str,
    md_text: str,
    global_idx_offset: int = 0,
) -> Tuple[List[dict], int]:
    """
    对单个文档的 markdown 文本进行完整分块。

    新流程：
      1. 清洗（通用 + regulatory 专属）
      2. inject_financial_headings() — 金融序号 → Markdown 标题
      3. MarkdownHeaderTextSplitter 按标题切分 → 产出带 heading_path 的原始 chunk
      4. 对每个原始 chunk：
         a. 检测内部是否含 <table> → 提取为独立 table chunk（继承 heading_path）
         b. 正文 text 做二级拆分（<=1000 字符）
         c. heading_path 扁平化拼接在 text 开头
      5. 统一写入

    Args:
        doc_id: 文档 ID
        domain: 领域
        md_text: 原始 markdown 文本
        global_idx_offset: 全局 chunk 序号偏移

    Returns:
        (chunks_list, new_offset)
    """
    chunks = []
    chunk_idx = global_idx_offset

    # Step 1: 清洗
    if domain == "regulatory":
        if doc_id.startswith("strict_v3_"):
            md_text = clean_generic_markdown(md_text)
        elif doc_id.startswith("csrc_"):
            md_text = clean_regulatory_markdown(md_text)
            md_text = clean_generic_markdown(md_text) if md_text else md_text
        else:
            md_text = clean_generic_markdown(md_text)
    else:
        md_text = clean_generic_markdown(md_text)

    if not md_text:
        return [], global_idx_offset

    # Step 2: 金融序号 → Markdown 标题
    md_text = inject_financial_headings(md_text)

    # Step 3: MarkdownHeaderTextSplitter 按标题切分
    header_chunks = split_by_headers(md_text)

    # Step 4: 逐 chunk 处理（提 table + 二级拆分 + heading_path 拼接）
    for hc in header_chunks:
        raw_text = hc["text"]
        if not raw_text.strip():
            continue

        heading_path = hc["heading_path"]

        # Step 4a: 从 chunk 文本中提取 HTML table
        table_mds = _extract_tables_from_chunk_text(raw_text)

        # Step 4b: 从 chunk 文本中移除 table，只留正文
        body_text = _remove_tables_from_text(raw_text)

        # Step 4c: 正文二级拆分 → paragraph chunks
        if body_text:
            sub_texts = split_long_text(body_text, CHUNK_MAX_CHARS)
            for st in sub_texts:
                if not st.strip():
                    continue
                chunk_text = _format_chunk_text(heading_path, st)
                chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_idx:04d}",
                    "doc_id": doc_id,
                    "domain": domain,
                    "heading_path": heading_path,
                    "chunk_type": "paragraph",
                    "text": chunk_text,
                    "char_len": len(chunk_text),
                })
                chunk_idx += 1

        # Step 4d: 每个 table 作为独立 chunk（继承当前 heading_path，不截断）
        for tbl_md in table_mds:
            chunk_text = _format_chunk_text(heading_path, tbl_md)
            chunks.append({
                "chunk_id": f"{doc_id}_c{chunk_idx:04d}",
                "doc_id": doc_id,
                "domain": domain,
                "heading_path": heading_path,
                "chunk_type": "table",
                "text": chunk_text,
                "char_len": len(chunk_text),
            })
            chunk_idx += 1

    return chunks, chunk_idx


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def chunk_domain(
    domain: str,
    doc_list: List[Tuple[str, Path]],
    output_dir: Path,
) -> List[dict]:
    """
    对某个领域的所有文档进行分块并持久化。

    Args:
        domain: 领域标识
        doc_list: [(doc_id, md_path), ...]
        output_dir: 输出目录

    Returns:
        该领域的 chunk 列表（也用于打印统计）
    """
    all_chunks = []
    global_idx = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"领域: {domain} ({len(doc_list)} 文档)")

    for doc_id, md_path in doc_list:
        if not md_path.exists():
            logger.warning(f"  [跳过] 文件不存在: {md_path}")
            continue

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            logger.warning(f"  [跳过] 读取失败: {md_path}: {e}")
            continue

        if not md_text.strip():
            logger.warning(f"  [跳过] 空文件: {md_path}")
            continue

        chunks, global_idx = chunk_document(
            doc_id=doc_id,
            domain=domain,
            md_text=md_text,
            global_idx_offset=global_idx,
        )
        all_chunks.extend(chunks)
        logger.info(
            f"  {doc_id}: {len(chunks)} chunks "
            f"(table={sum(1 for c in chunks if c['chunk_type']=='table')})"
        )

    # 持久化
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{domain}_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"  写入: {out_path} ({len(all_chunks)} chunks)")

    return all_chunks


def run_all(
    base_dir: Path = MINERU_DIR,
    output_dir: Path = CHUNKS_DIR,
    domain_filter: Optional[str] = None,
) -> None:
    """主入口：发现文件 → 逐领域分块 → 持久化"""
    md_paths = resolve_md_paths(base_dir)

    total_stats = {}

    for domain in DOMAINS:
        if domain_filter and domain != domain_filter:
            continue
        doc_list = md_paths.get(domain, [])
        if not doc_list:
            logger.info(f"[{domain}] 无文档，跳过")
            continue

        chunks = chunk_domain(domain, doc_list, output_dir)
        total_stats[domain] = {
            "docs": len(doc_list),
            "chunks": len(chunks),
            "tables": sum(1 for c in chunks if c["chunk_type"] == "table"),
        }

    # 汇总输出
    print(f"\n{'='*60}")
    print("分块汇总:")
    print(f"{'领域':<22} {'文档':>5} {'Chunks':>8} {'Tables':>6}")
    print(f"{'-'*22} {'-'*5} {'-'*8} {'-'*6}")
    total_docs = 0
    total_chunks = 0
    total_tables = 0
    for domain, stats in total_stats.items():
        print(
            f"{domain:<22} {stats['docs']:>5} {stats['chunks']:>8} {stats['tables']:>6}"
        )
        total_docs += stats["docs"]
        total_chunks += stats["chunks"]
        total_tables += stats["tables"]
    print(f"{'='*22} {'='*5} {'='*8} {'='*6}")
    print(f"{'总计':<22} {total_docs:>5} {total_chunks:>8} {total_tables:>6}")
    print(f"\n输出目录: {output_dir.resolve()}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="对 mineru 离线解析的 Markdown 文档进行分块"
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        default=None,
        help="仅处理指定领域（默认全量）",
    )
    parser.add_argument(
        "--base-dir",
        default=str(MINERU_DIR),
        help="mineru_output 根目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CHUNKS_DIR),
        help="chunk JSONL 输出目录",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_all(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output_dir),
        domain_filter=args.domain,
    )


if __name__ == "__main__":
    main()