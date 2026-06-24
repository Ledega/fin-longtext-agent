"""验证分块结果：检查结构完整性和数据质量"""
import json
from pathlib import Path

chunks_dir = Path("data/chunks")

print("=" * 70)
print(f"{'文件':<50} {'Chunks':>8}")
print("-" * 70)

total = 0
for f in sorted(chunks_dir.glob("*.jsonl")):
    with open(f, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    total += len(lines)
    print(f"{f.name:<50} {len(lines):>8}")

print("-" * 70)
print(f"{'总计':<50} {total:>8}")
print()

# 验证各领域 doc_id 正确性
print("=" * 70)
print("doc_id 与 raw 数据集对齐验证:")
print("-" * 70)

# insurance
with open(chunks_dir / "insurance_chunks.jsonl", encoding="utf-8") as f:
    c = json.loads(f.readline())
    print(f"  insurance 示例:  {c['doc_id']}  heading_path={c['heading_path']}")

# financial_contracts
with open(chunks_dir / "financial_contracts_chunks.jsonl", encoding="utf-8") as f:
    c = json.loads(f.readline())
    print(f"  contracts 示例:  {c['doc_id']}  heading_path={c['heading_path']}")

# financial_reports
with open(chunks_dir / "financial_reports_chunks.jsonl", encoding="utf-8") as f:
    c = json.loads(f.readline())
    print(f"  reports 示例:    {c['doc_id']}  heading_path={c['heading_path']}")

# regulatory (check distinct doc_ids)
doc_ids = set()
with open(chunks_dir / "regulatory_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["doc_id"].startswith("strict_v3_"):
            if "strict_v3_008" in c["doc_id"]:
                print(f"  regulatory txt:   {c['doc_id'][:50]}...  heading_path={c['heading_path']}")
                break

# research
with open(chunks_dir / "research_chunks.jsonl", encoding="utf-8") as f:
    c = json.loads(f.readline())
    print(f"  research 示例:   {c['doc_id']}  heading_path={c['heading_path']}")

# 验证 table chunk 不截断
print()
print("=" * 70)
print("Table Chunk 完整性验证:")
print("-" * 70)
with open(chunks_dir / "insurance_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["chunk_type"] == "table" and c["char_len"] > 50:
            print(f"  doc_id={c['doc_id']}  char_len={c['char_len']}")
            print(f"  heading_path={c['heading_path']}")
            print(f"  text preview: {c['text'][:100]}...")
            break

# 验证 heading_path 不为空且分层正确
print()
print("=" * 70)
print("heading_path 层级深度分布:")
print("-" * 70)
from collections import Counter
depth_counter = Counter()
with open(chunks_dir / "financial_reports_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        depth_counter[len(c["heading_path"])] += 1
for depth, count in sorted(depth_counter.items()):
    print(f"  深度 {depth}: {count} chunks")