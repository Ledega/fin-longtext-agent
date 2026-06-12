"""
文本分块核心逻辑

分块策略：
1. 按自然段（空行、缩进）或条款号（如'第X条'、'第四十七条'）作为初级 chunk
2. 每个 chunk 上限 800 字符
3. 超过上限的 chunk，拆分为多个子 chunk，相邻 chunk 间 15% overlap
4. 提取 section_path、clause_no、chunk_type 等元数据
"""

import re
import math
import json
from pathlib import Path
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
CHUNK_MAX_CHARS = 800          # 每 chunk 上限字符数
OVERLAP_RATIO = 0.15           # overlap 比例
SECTION_HEADING_PATTERN = re.compile(
    r'^[第第第第第第第第第第第第第第]?\s*[一二三四五六七八九十百千]+[、．\.]?\s*\S+|'
    r'^第[一二三四五六七八九十百千]+[章节节节节节节节节]\s*\S+|'
    r'^第[0-9一二三四五六七八九十百千]+[条章]\s*\S+|'
    r'^[（(][一二三四五六七八九十百千]+[)）]',
    re.MULTILINE
)
CLAUSE_PATTERN = re.compile(
    r'^(第[一二三四五六七八九十百千零〇]+[条条条条条条条条条条])\s',
    re.MULTILINE
)
ITEM_PATTERN = re.compile(
    r'^[（(][一二三四五六七八九十百千零〇]+[）)]\s*',
    re.MULTILINE
)


def approx_tokens(text: str) -> int:
    """估算 token 数：中文字符约 1.5，英文约 4 字符/token"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4) + 10


# ──────────────────────────────────────────────
# 文本提取
# ──────────────────────────────────────────────

def extract_text_from_pdf(file_path: Path) -> List[Tuple[int, str]]:
    """
    从 PDF 提取文本，按页返回 [(page_no, text), ...]

    使用 pdfminer.six 作为 PDF 解析引擎，兼容性更好。

    Returns:
        List[Tuple[int, str]] 每页的 (页码, 文本)，页码从 1 开始
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTChar, LAParams

    laparams = LAParams(
        line_margin=0.5,
        word_margin=0.1,
        char_margin=2.0,
        boxes_flow=0.5,
        detect_vertical=False,
    )

    pages: List[Tuple[int, str]] = []
    for page_no, page_layout in enumerate(extract_pages(str(file_path), laparams=laparams), start=1):
        text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text += element.get_text()
        text = text.strip()
        if text:
            pages.append((page_no, text))
    return pages


def extract_text_from_html(file_path: Path) -> List[Tuple[int, str]]:
    """
    从 HTML 提取正文文本（监管法规领域使用）

    Returns:
        List[Tuple[int, str]] 单页的 [(1, text), ...]
    """
    from bs4 import BeautifulSoup

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # 移除 script, style 等标签
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # 提取正文
    body = soup.find("body")
    text = body.get_text(separator="\n") if body else soup.get_text(separator="\n")
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return [(1, text)]


# ──────────────────────────────────────────────
# 段落分割
# ──────────────────────────────────────────────

def split_into_paragraphs(text: str) -> List[str]:
    """
    将文本按自然段分割。

    段落边界判断：
    1. 连续两个及以上换行符
    2. 段落开头缩进（行首有空格/制表符）
    3. 条款号开头（如'第X条'）
    """
    # 先 normalize 换行
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 按双换行分割
    raw_paragraphs = re.split(r'\n\s*\n', text)

    paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # 尝试按条款号再次分割（适用于保险条款、监管法规）
        sub_paras = re.split(r'(?=第[一二三四五六七八九十百千零〇]+[条])', para)
        for sub in sub_paras:
            sub = sub.strip()
            if sub:
                paragraphs.append(sub)

    return paragraphs


# ──────────────────────────────────────────────
# 识别段落类型和元数据
# ──────────────────────────────────────────────

def identify_chunk_type(text: str) -> str:
    """识别 chunk 类型"""
    text_stripped = text.strip()
    if not text_stripped:
        return "paragraph"

    # 标题检测：行数少 + 无标点
    lines = [l.strip() for l in text_stripped.split('\n') if l.strip()]
    if len(lines) <= 2:
        has_punct = any(c in text_stripped for c in '，。；：？！、')
        if not has_punct and len(text_stripped) < 80:
            return "header"

    # 条款检测
    if CLAUSE_PATTERN.match(text_stripped) or ITEM_PATTERN.match(text_stripped):
        return "clause"

    # 表格检测：包含较多竖线或空格分隔的对齐数据
    lines_with_pipe = [l for l in lines if '|' in l]
    if len(lines_with_pipe) >= 2 and len(lines_with_pipe) / max(len(lines), 1) > 0.3:
        return "table"

    # 列表检测
    list_markers = 0
    for line in lines:
        if re.match(r'^[\d一二三四五六七八九十]+[、．\.\s]', line):
            list_markers += 1
        elif line.startswith('-') or line.startswith('*') or line.startswith('·'):
            list_markers += 1
    if list_markers >= 2 and list_markers / max(len(lines), 1) > 0.4:
        return "list"

    return "paragraph"


def extract_clause_no(text: str) -> Optional[str]:
    """提取条款号，如'第四十七条'"""
    m = CLAUSE_PATTERN.match(text.strip())
    if m:
        return m.group(1)
    return None


def extract_section_path(text: str, prev_sections: List[str]) -> List[str]:
    """
    从文本中提取章节路径。

    检测文本是否是章节标题，如果是则更新 prev_sections。
    """
    sections = list(prev_sections)
    text_stripped = text.strip()

    # 章节标题模式：第一章、第一节、第四条等
    patterns = [
        r'^第[一二三四五六七八九十百千]+[章篇]',
        r'^第[一二三四五六七八九十百千]+[节节节]',
        r'^第[一二三四五六七八九十百千]+[部分]',
        r'^[（(][一二三四五六七八九十]+[)）]',
    ]

    for pattern in patterns:
        m = re.match(pattern, text_stripped)
        if m:
            heading = text_stripped[:50]  # 取前 50 字作为章节名
            sections.append(heading)
            break

    return sections


# ──────────────────────────────────────────────
# 超长段落拆分（带 overlap）
# ──────────────────────────────────────────────

def split_oversized_paragraph(text: str, max_chars: int = CHUNK_MAX_CHARS,
                              overlap_ratio: float = OVERLAP_RATIO) -> List[str]:
    """
    将超长段落拆分为多个子段落，在相邻 chunk 间加 overlap。

    overlap 量 = int(max_chars * overlap_ratio)，约 120 字符。
    """
    if len(text) <= max_chars:
        return [text]

    overlap_chars = int(max_chars * overlap_ratio)
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
            # 如果单句就超长，硬切
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars - overlap_chars):
                    chunk = sentence[i:i + max_chars].strip()
                    if chunk:
                        chunks.append(chunk)
                current = ""
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    # 如果只有一个 chunk，直接返回
    if len(chunks) <= 1:
        return chunks if chunks else [text]

    # 应用 overlap
    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
        else:
            # 从前一个 chunk 末尾取 overlap_chars 作为前缀
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-overlap_chars:] if len(prev_chunk) > overlap_chars else prev_chunk
            result.append(overlap_text + chunk)

    return result if result else [text]


# ──────────────────────────────────────────────
# 主分块函数
# ──────────────────────────────────────────────

def chunk_document(
    doc_id: str,
    domain: str,
    page_texts: List[Tuple[int, str]],
) -> List[dict]:
    """
    将一个文档的所有页文本进行分块。

    Args:
        doc_id: 文档 ID
        domain: 领域标识
        page_texts: [(page_no, text), ...] 按页提取的文本

    Returns:
        List[dict] 每个 dict 包含 chunk 的所有字段
    """
    chunks = []
    global_chunk_idx = 0
    prev_section_path: List[str] = []

    for page_no, text in page_texts:
        # 按段落分割
        paragraphs = split_into_paragraphs(text)

        for para in paragraphs:
            if not para.strip():
                continue

            # 识别类型和元数据
            chunk_type = identify_chunk_type(para)
            clause_no = extract_clause_no(para)
            section_path = extract_section_path(para, prev_section_path)

            # 更新 prev_section_path（只在遇到新章节标题时更新）
            if section_path != prev_section_path:
                prev_section_path = section_path

            # 超长段落拆分
            sub_paras = split_oversized_paragraph(para)

            for sub_para in sub_paras:
                if not sub_para.strip():
                    continue

                char_len = len(sub_para)
                tokens = approx_tokens(sub_para)

                chunk = {
                    "chunk_id": f"{doc_id}_p{page_no}_c{global_chunk_idx}",
                    "doc_id": doc_id,
                    "domain": domain,
                    "page_no": page_no,
                    "section_path": json.dumps(section_path, ensure_ascii=False),
                    "clause_no": clause_no,
                    "chunk_type": chunk_type,
                    "text": sub_para,
                    "char_len": char_len,
                    "approx_tokens": tokens,
                }
                chunks.append(chunk)
                global_chunk_idx += 1

    return chunks


def chunk_document_from_file(file_path: Path, doc_id: str, domain: str) -> List[dict]:
    """
    从文件读取并分块。

    Args:
        file_path: 文档文件路径
        doc_id: 文档 ID
        domain: 领域标识

    Returns:
        List[dict] chunk 列表
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        page_texts = extract_text_from_pdf(file_path)
    elif ext == ".html":
        page_texts = extract_text_from_html(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        page_texts = [(1, text)]
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    return chunk_document(doc_id, domain, page_texts)