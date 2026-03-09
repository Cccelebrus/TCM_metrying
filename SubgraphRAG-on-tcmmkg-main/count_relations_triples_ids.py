# -*- coding: utf-8 -*-
"""
统计模型召回三元组中各关系数量
读取: retrieval_result_tcm_50trples_withcpm.pth
"""

import torch
from collections import Counter

PTH_FILE = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Jan05-06:42:07/retrieval_result_tcm_50trples_withcpm.pth"

# -------------------- 读取 pth --------------------
data = torch.load(PTH_FILE)
relation_counter = Counter()

for sid, val in data.items():
    scored_triples = val.get("scored_triples", [])
    for h_name, r_name, t_name, score in scored_triples:
        relation_counter[r_name] += 1

# -------------------- 打印统计 --------------------
print("关系名\tcount")
for r_name, cnt in relation_counter.most_common():
    print(f"{r_name}\t{cnt}")
