import json

with open('data/chunks.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 3:
            ch = json.loads(line)
            print(f'--- Chunk {i} ---')
            print(f'chunk_id: {ch["chunk_id"]}')
            print(f'doc_id: {ch["doc_id"]}')
            print(f'chunk_type: {ch["chunk_type"]}')
            print(f'char_len: {ch["char_len"]}')
            print(f'approx_tokens: {ch["approx_tokens"]}')
            print(f'clause_no: {ch["clause_no"]}')
            print(f'section_path: {ch["section_path"]}')
            print(f'text[:200]: {ch["text"][:200]}')
            print()

print('--- Stats ---')
sizes = []
with open('data/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        ch = json.loads(line)
        sizes.append(ch['char_len'])

import statistics
print(f'chunk count: {len(sizes)}')
print(f'min size: {min(sizes)}')
print(f'max size: {max(sizes)}')
print(f'avg size: {statistics.mean(sizes):.1f}')
print(f'median size: {statistics.median(sizes):.1f}')
over_800 = sum(1 for s in sizes if s > 800)
over_900 = sum(1 for s in sizes if s > 900)
print(f'chunks > 800: {over_800} ({over_800/len(sizes)*100:.1f}%)')
print(f'chunks > 900: {over_900} ({over_900/len(sizes)*100:.1f}%)')