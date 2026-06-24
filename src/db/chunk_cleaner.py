"""
ChunkCleaner — 金融长文档 Chunk 数据清洗模块

用途：在将 chunk 导入 BM25 索引前，剔除混入的垃圾数据：
  - 极短文本（< 10 字符）
  - 全符号/乱码（有效字符密度 < 30%）
  - 水印高频词重复（top1 词频 > 50%，或唯一字符类型过少）
  - table 类型旁路（不误删表格）

用法：
    from src.db.chunk_cleaner import ChunkCleaner
    cleaner = ChunkCleaner()
    cleaned = cleaner.clean_chunks(raw_chunks)
"""

import re
import json
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ChunkCleaner:
    """
    Chunk 数据清洗器。
    对输入的 chunk 列表逐条判定，返回过滤后的干净 chunk 列表。
    """

    # ── 可调阈值 ──
    MIN_TEXT_LENGTH: int = 10                # 极短文本拦截阈值
    MIN_CHAR_DENSITY: float = 0.30           # 有效字符密度下限
    MAX_TOP_WORD_RATIO: float = 0.50         # 最高频词占比上限
    MIN_UNIQUE_CHARS: int = 15               # 唯一字符种类下限（长文本）

    def __init__(self, min_text_length: Optional[int] = None,
                 min_char_density: Optional[float] = None,
                 max_top_word_ratio: Optional[float] = None,
                 min_unique_chars: Optional[int] = None):
        """
        Args:
            min_text_length: 极短文本拦截阈值（默认 10）
            min_char_density: 有效字符密度下限（默认 0.30）
            max_top_word_ratio: 最高频词占比上限（默认 0.50）
            min_unique_chars: 唯一字符种类下限（默认 15）
        """
        if min_text_length is not None:
            self.MIN_TEXT_LENGTH = min_text_length
        if min_char_density is not None:
            self.MIN_CHAR_DENSITY = min_char_density
        if max_top_word_ratio is not None:
            self.MAX_TOP_WORD_RATIO = max_top_word_ratio
        if min_unique_chars is not None:
            self.MIN_UNIQUE_CHARS = min_unique_chars

        # 编译正则提高效率
        self._valid_char_pattern = re.compile(r'[\u4e00-\u9fff\w]')
        self._word_split_pattern = re.compile(r'[\s，。、；：？！,.;:!?（）()【】\[\]{}""''\u3000]+')

    # ──────────────────────────────────────────────
    # 对外接口
    # ──────────────────────────────────────────────

    def clean_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        批量清洗 chunk 列表。

        Args:
            chunks: 输入 chunk 列表，每项含 "chunk_id", "chunk_type", "text" 等

        Returns:
            过滤后的干净 chunk 列表（保持原顺序）
        """
        kept = []
        dropped = 0
        for chunk in chunks:
            if self._is_valid(chunk):
                kept.append(chunk)
            else:
                dropped += 1

        if dropped > 0:
            logger.info(f"ChunkCleaner: {dropped}/{len(chunks)} chunks 已丢弃")

        return kept

    # ──────────────────────────────────────────────
    # 单条判定
    # ──────────────────────────────────────────────

    def _is_valid(self, chunk: dict) -> bool:
        """
        判定单条 chunk 是否有效（True = 保留，False = 丢弃）。

        规则：
          1. table 类型旁路 → 直接放行
          2. paragraph 且 text < MIN_TEXT_LENGTH → 丢弃
          3. 有效字符密度 < MIN_CHAR_DENSITY → 丢弃
          4a. 高频词占比 > MAX_TOP_WORD_RATIO → 丢弃（水印）
          4b. 长文本唯一字符种类 < MIN_UNIQUE_CHARS → 丢弃（水印）
        """
        try:
            chunk_type = chunk.get("chunk_type", "paragraph")
            text = chunk.get("text", "")

            # Rule 1: table 类型旁路
            if chunk_type == "table":
                return True

            if not isinstance(text, str):
                return False

            text = text.strip()
            if not text:
                return False

            # Rule 2: 极短文本拦截
            if len(text) < self.MIN_TEXT_LENGTH:
                return False

            # Rule 3: 有效字符密度检查
            valid_count = len(self._valid_char_pattern.findall(text))
            density = valid_count / len(text)
            if density < self.MIN_CHAR_DENSITY:
                return False

            # Rule 4: 水印/高频重复检查
            if not self._check_watermark(text):
                return False

            return True

        except Exception:
            # 任何异常不干扰流程，安全丢弃
            logger.warning(f"ChunkCleaner 判定异常，丢弃: {chunk.get('chunk_id', '?')}", exc_info=True)
            return False

    # ──────────────────────────────────────────────
    # 有效字符密度检查（规则3）
    # ──────────────────────────────────────────────

    @staticmethod
    def _count_valid_chars(text: str) -> int:
        """统计有效字符：中文汉字 + 英文字母 + 数字"""
        return len(re.findall(r'[\u4e00-\u9fff\w]', text))

    # ──────────────────────────────────────────────
    # 水印/高频重复拦截（规则4）
    # ──────────────────────────────────────────────

    def _check_watermark(self, text: str) -> bool:
        """
        水印检测。
        返回 True = 无问题，False = 判定为水印垃圾。
        """
        text_len = len(text)

        # Rule 4b: 长文本唯一字符种类过少
        if text_len > 100:
            unique_chars = len(set(text))
            if unique_chars < self.MIN_UNIQUE_CHARS:
                return False

        # Rule 4a: 高频词占比检查（仅对长度 > 50 的文本）
        if text_len > 50:
            top_ratio = self._compute_top_word_ratio(text)
            if top_ratio > self.MAX_TOP_WORD_RATIO:
                return False

        return True

    def _compute_top_word_ratio(self, text: str) -> float:
        """
        计算最高频词占比。

        先用空格/标点粗切分，对长度 >= 2 的词统计词频。
        返回 top1 词频 / 总词数。
        遇到异常返回 0（安全值）。
        """
        try:
            # 用正则粗切分（按空格和标点）
            raw_tokens = self._word_split_pattern.split(text)
            # 过滤过短的词
            tokens = [t.strip() for t in raw_tokens if len(t.strip()) >= 2]

            if not tokens:
                return 0.0

            total = len(tokens)
            top_count = Counter(tokens).most_common(1)[0][1]
            return top_count / total

        except Exception:
            return 0.0


# ──────────────────────────────────────────────
# 命令行主流程：遍历 data/chunks/*.jsonl → 清洗 → 覆写
# ──────────────────────────────────────────────

def clean_all_chunks(chunks_dir: str = "data/chunks") -> dict:
    """
    遍历 chunks 目录下所有 *_chunks.jsonl，逐文件清洗并覆写。

    Returns:
        {"domain": {"before": N, "after": M, "dropped": N-M}, ...}
    """
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks 目录不存在: {chunks_path}")

    cleaner = ChunkCleaner()
    stats = {}

    for jsonl_path in sorted(chunks_path.glob("*_chunks.jsonl")):
        domain = jsonl_path.stem.replace("_chunks", "")
        logger.info(f"\n[{domain}] 加载 {jsonl_path.name}...")

        # 读取
        with open(jsonl_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]

        before = len(chunks)

        # 清洗
        cleaned = cleaner.clean_chunks(chunks)
        after = len(cleaned)

        # 覆写
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for chunk in cleaned:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        stats[domain] = {
            "before": before,
            "after": after,
            "dropped": before - after,
        }

        logger.info(f"  {domain}: {before} → {after} (丢弃 {before-after})")

    return stats


def main():
    """CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChunkCleaner — 清除 chunks 中的垃圾数据"
    )
    parser.add_argument(
        "--dir",
        default="data/chunks",
        help="chunks JSONL 目录（默认 data/chunks）",
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

    stats = clean_all_chunks(args.dir)

    print(f"\n{'='*60}")
    print("Chunk 清洗汇总:")
    print(f"{'领域':<22} {'清洗前':>8} {'清洗后':>8} {'丢弃':>6}")
    print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*6}")
    total_before = 0
    total_after = 0
    for domain, s in stats.items():
        print(f"{domain:<22} {s['before']:>8} {s['after']:>8} {s['dropped']:>6}")
        total_before += s["before"]
        total_after += s["after"]
    print(f"{'='*22} {'='*8} {'='*8} {'='*6}")
    print(f"{'总计':<22} {total_before:>8} {total_after:>8} {total_before-total_after:>6}")
    print(f"\n输出: {args.dir}（已覆写）")


if __name__ == "__main__":
    main()