"""
导入 data/chunks/*.jsonl 到本地 Milvus Lite（Windows 兼容版）

策略：
  1. monkey-patch os.rename → os.replace（绕过 Windows 文件锁）
  2. 单线程、单批次 3000 条
  3. 完成后手动 collection.flush()
"""

import os
import json
import time
import shutil
from pathlib import Path

# ═══════════════════════════════════════════════════
# Step 0: monkey-patch os.rename → os.replace
# ═══════════════════════════════════════════════════
# milvus_lite/storage/manifest.py:125 用了 os.rename(tmp, target)
# 但 Windows 上 rename 不能覆盖已有文件，抛 FileExistsError。
# 用 os.replace() 替代，它在 Windows 上会先删除目标再重命名。
import milvus_lite.storage.manifest as _manifest_module
_original_save = _manifest_module.Manifest.save


def _patched_save(self):
    """将 os.rename 替换为 os.replace 的 monkey-patch 版本"""
    import json
    import shutil

    os.makedirs(self._data_dir, exist_ok=True)

    new_version = self._version + 1
    payload = self._to_payload()
    payload["version"] = new_version

    target_path = os.path.join(self._data_dir, "manifest.json")
    prev_path = os.path.join(self._data_dir, "manifest.json.prev")
    tmp_path = os.path.join(self._data_dir, "manifest.json.tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        if os.path.exists(target_path):
            try:
                shutil.copy2(target_path, prev_path)
            except OSError as e:
                pass

        # ← 关键修复：用 os.replace 替代 os.rename
        os.replace(tmp_path, target_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    self._version = new_version
    try:
        dir_fd = os.open(self._data_dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


_manifest_module.Manifest.save = _patched_save

# ═══════════════════════════════════════════════════
# Step 1: 导入依赖 & 建表
# ═══════════════════════════════════════════════════

from pymilvus import MilvusClient, DataType, FunctionType, Function
from pymilvus.milvus_client.index import IndexParams

CHUNKS_DIR = Path("data/chunks")
DB_PATH = "milvus_demo.db"
COLLECTION = "fin_longtext_chunks"
BATCH_SIZE = 3000  # 每批 3000 条，减少 flush 次数

# 删除旧 DB
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH, ignore_errors=True)
    time.sleep(0.5)

client = MilvusClient(uri=DB_PATH)

schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field('id', DataType.INT64, is_primary=True, auto_id=True)
schema.add_field('doc_id', DataType.VARCHAR, max_length=256)
schema.add_field('domain', DataType.VARCHAR, max_length=64)
schema.add_field('text', DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True, analyzer_params={'type': 'jieba'})
schema.add_field('sparse_bm25', DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(name='bm25_func', function_type=FunctionType.BM25, input_field_names=['text'], output_field_names=['sparse_bm25']))
client.create_collection(collection_name=COLLECTION, schema=schema)

ip = IndexParams()
ip.add_index(field_name='sparse_bm25', index_name='sparse_idx', index_type='SPARSE_INVERTED_INDEX', metric_type='BM25')
client.create_index(collection_name=COLLECTION, index_params=ip)
client.load_collection(COLLECTION)
print(f"Collection '{COLLECTION}' 创建完成")

# ═══════════════════════════════════════════════════
# Step 2: 单线程批量导入
# ═══════════════════════════════════════════════════

total = 0
for f in sorted(CHUNKS_DIR.glob("*_chunks.jsonl")):
    domain = f.stem.replace("_chunks", "")
    with open(f, "r", encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    print(f"[{domain}] {len(lines)} chunks — 开始导入...", end=" ", flush=True)

    # 单线程循环，每批 3000 条
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i:i + BATCH_SIZE]
        data = []
        for c in batch:
            text = c["text"]
            if len(text) > 65535:
                text = text[:65535]
            data.append({"doc_id": c["doc_id"], "domain": domain, "text": text})

        # 单次 insert，不并发
        client.insert(collection_name=COLLECTION, data=data)

    # 每个文件刷一次
    client.flush(collection_name=COLLECTION)
    total += len(lines)
    print(f"完成")

print(f"\n=== 导入完成 ===")
print(f"总计: {total} chunks")
print(f"DB: {DB_PATH}")

# 验证
count = client.query(collection_name=COLLECTION, output_fields=["count(*)"])
print(f"Milvus 行数: {count}")