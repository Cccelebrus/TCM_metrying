# -*- coding: utf-8 -*-
"""
2025-11-29
在原 build_triple_ids.py 基础上：
✅ 保留所有原有逻辑
✅ 仅在 triples 全局去重后，新增一步：
   删除可被 symptom → CPM → CHP 路径解释的 symptom → CHP 直连边
"""

import os
import torch
from tqdm import tqdm
from collections import defaultdict, deque
import random

# -------------------- 配置 --------------------
BASE_DIR = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/"

EMB_FILES = {
    "train": os.path.join(BASE_DIR, "tcm_embedding/train.pth"),
    "val":   os.path.join(BASE_DIR, "tcm_embedding/val.pth"),
    "test":  os.path.join(BASE_DIR, "tcm_embedding/test.pth"),
}

ENTITY_FILE  = os.path.join(BASE_DIR, "entity_identifiers.txt")
REL_FILE     = os.path.join(BASE_DIR, "relation_list.txt")
TRIPLES_FILE = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/triples_cpm.tsv"

# -------------------- 读取实体/关系 --------------------
def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

entities  = read_list(ENTITY_FILE)
relations = read_list(REL_FILE)

entity2id   = {e: idx for idx, e in enumerate(entities)}
relation2id = {r: idx for idx, r in enumerate(relations)}

id2entity   = {idx: e for e, idx in entity2id.items()}
id2relation = {idx: r for r, idx in relation2id.items()}

num_entities_total = len(entities)

# -------------------- 读取 triples（中文） --------------------
all_triples = []

with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        h, r, t = line.strip().split("\t")
        all_triples.append((h, r, t))

print(f"原始三元组数量: {len(all_triples)}")

# -------------------- Step 1: 全局去重 --------------------
unique_triples = list(set(all_triples))
print(f"去重后三元组数量: {len(unique_triples)}")
print(f"重复数量: {len(all_triples) - len(unique_triples)}")

# -------------------- Step 2: 构建 symptom → CPM → CHP 可达关系 --------------------
symptom_to_cpm = defaultdict(set)
cpm_to_chp     = defaultdict(set)

for h, r, t in unique_triples:
    if r == "symptom_to_cpm":
        symptom_to_cpm[h].add(t)
    elif r == "cpm_to_chp":
        cpm_to_chp[h].add(t)

# symptom → CHP（经由 CPM）
symptom_to_chp_via_cpm = defaultdict(set)

for s, cpms in symptom_to_cpm.items():
    for cpm in cpms:
        for chp in cpm_to_chp.get(cpm, []):
            symptom_to_chp_via_cpm[s].add(chp)

# -------------------- Step 3: 删除冗余的 symptom → CHP 直连 --------------------
filtered_triples = []
removed_cnt = 0

for h, r, t in unique_triples:
    if r == "symptom_to_chp":
        if t in symptom_to_chp_via_cpm.get(h, set()):
            removed_cnt += 1
            continue  # 删除该直连
    filtered_triples.append((h, r, t))

print(f"删除的 symptom→CHP 冗余直连数量: {removed_cnt}")
print(f"最终保留三元组数量: {len(filtered_triples)}")

# -------------------- Step 4: 构建 triples_by_head（ID） --------------------
triples_by_head = defaultdict(list)
triples_dedup_ids = []

for h, r, t in filtered_triples:
    if h in entity2id and r in relation2id and t in entity2id:
        hid = entity2id[h]
        rid = relation2id[r]
        tid = entity2id[t]

        triples_by_head[hid].append((rid, tid))
        triples_dedup_ids.append((hid, rid, tid))

total_edges = sum(len(v) for v in triples_by_head.values())
print(f"triples_by_head 总边数（ID）: {total_edges}")
print(f"涉及 head 实体数量: {len(triples_by_head)}")

# -------------------- 保存 ID 版三元组 --------------------
DEDUP_ID_FILE = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/triples_dedup_ids.tsv"
with open(DEDUP_ID_FILE, "w", encoding="utf-8") as fw:
    for hid, rid, tid in triples_dedup_ids:
        fw.write(f"{hid}\t{rid}\t{tid}\n")

print(f"已保存 ID 三元组: {DEDUP_ID_FILE}")

# -------------------- 保存中文可读版 --------------------
DEDUP_ZH_FILE = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/triples_dedup_zh.tsv"
with open(DEDUP_ZH_FILE, "w", encoding="utf-8") as fw:
    for hid, rid, tid in triples_dedup_ids:
        fw.write(
            f"{id2entity[hid]}\t{id2relation[rid]}\t{id2entity[tid]}\n"
        )

print(f"已保存中文三元组: {DEDUP_ZH_FILE}")

# ============================================================
# 下面所有内容：BFS / bridge / build_ids
# ✅ 与你原始文件完全一致
# ============================================================

# （此处开始，直接原样粘你现有的 collect_subgraph / find_bridge_nodes / build_ids / main）
# —— 不再重复贴，避免你误以为我动了逻辑
