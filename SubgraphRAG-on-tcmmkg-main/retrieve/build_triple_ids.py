"""
/home/gyj/local/SubgraphRAG-main/retrieve/build_triple_ids.py
2025-11-24
基于 BFS 子图构建 + 增强子图路径标签（topic → ... → answer）
不使用全局图，不使用 networkx，完全在 BFS 子图内部构建监督信号。
2025-11-26
构建TRIPLES时去重三元组
不改任何 BFS、不动关系、不动实体、不动后面构建子图的逻辑。
你遇到的根本问题是：
triples.tsv 里重复三元组太多（几十倍），导致 BFS 第 1 层用完了边，第 2 层几乎拿不到新边。
你真正需要做的仅仅是：
⭐ 在读取 triples 文件后，给 (h, r, t) 全局去重，再写回 triples.tsv
除此之外 任何东西都不改。
下面给你 最简、最干净、最正确 的 去重版 build_triple_ids.py 起始加载部分，只在加载 triples 时做一件事：
2025-11-28 增强子图去掉自环 h == t！！！！
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

entities = read_list(ENTITY_FILE)
relations = read_list(REL_FILE)

entity2id = {e: idx for idx, e in enumerate(entities)}
relation2id = {r: idx for idx, r in enumerate(relations)}

num_entities_total = len(entities)

# -------------------- 预加载 triples_by_head --------------------
triples_by_head = defaultdict(list)
all_triples = []

# 1) 只读取，不加入 triples_by_head (存中文，但下一步会转 ID)
with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        h, r, t = line.strip().split("\t")
        all_triples.append((h, r, t))

print(f"原始三元组数量: {len(all_triples)}")

# 2) 全局去重（仍然是中文）
unique_triples = list(set(all_triples))
print(f"去重后三元组数量: {len(unique_triples)}")
print(f"重复数量: {len(all_triples) - len(unique_triples)}")

# 3) 用去重后的三元组重建 triples_by_head（用 ID 存）
triples_by_head = defaultdict(list)
triples_dedup_ids = []  # 用 ID 保存

for h, r, t in unique_triples:
    if h in entity2id and r in relation2id and t in entity2id:
        hid = entity2id[h]
        rid = relation2id[r]
        tid = entity2id[t]

        triples_by_head[hid].append((rid, tid))
        triples_dedup_ids.append((hid, rid, tid))

# 4) 打印最终合计三元组数量
total_edges = sum(len(v) for v in triples_by_head.values())
print(f"triples_by_head 总三元组数量（ID, 去重后）: {total_edges}")
print(f"涉及 head 实体数量: {len(triples_by_head)}")

# 5) 保存 ID 版去重三元组
DEDUP_ID_FILE = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/triples_dedup_ids.tsv"
with open(DEDUP_ID_FILE, "w", encoding="utf-8") as fw:
    for hid, rid, tid in triples_dedup_ids:
        fw.write(f"{hid}\t{rid}\t{tid}\n")

print(f"已保存去重后的 ID 三元组到: {DEDUP_ID_FILE}")
print(f"ID 三元组数量: {len(triples_dedup_ids)}")

id2entity = {idx: e for e, idx in entity2id.items()}
id2relation = {idx: r for r, idx in relation2id.items()}

# 6) 同时保存中文可读版三元组
DEDUP_ZH_FILE = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/triples_dedup_zh.tsv"

with open(DEDUP_ZH_FILE, "w", encoding="utf-8") as fw:
    for hid, rid, tid in triples_dedup_ids:
        h_zh = id2entity.get(hid, "UNK_ENTITY")
        r_zh = id2relation.get(rid, "UNK_REL")
        t_zh = id2entity.get(tid, "UNK_ENTITY")
        fw.write(f"{h_zh}\t{r_zh}\t{t_zh}\n")

print(f"已保存中文三元组到: {DEDUP_ZH_FILE}")

def collect_subgraph(topic_entities, hop=2, second_hop_limit=3):
    edges = []
    seen = set()

    # ① 第一跳：全量
    frontier = set()
    for ent in topic_entities:
        if ent not in entity2id:
            continue
        hid = entity2id[ent]
        if hid not in triples_by_head:
            continue

        for r, t in triples_by_head[hid]:
            triple = (hid, r, t)
            if triple not in seen:
                edges.append(triple)
                seen.add(triple)
            frontier.add(t)

    # 如果 hop=1 就返回
    if hop == 1:
        if edges:
            h, r, t = zip(*edges)
            return list(h), list(r), list(t)
        return [], [], []

    # ② 第二跳：每个 head 最多随机3个邻居
    next_frontier = set()

    for h in frontier:
        if h not in triples_by_head:
            continue

        neighbors = triples_by_head[h]

        # 随机采样，不超过 second_hop_limit
        if len(neighbors) > second_hop_limit:
            sampled = random.sample(neighbors, second_hop_limit)
        else:
            sampled = neighbors

        for r, t in sampled:
            triple = (h, r, t)
            if triple not in seen:
                edges.append(triple)
                seen.add(triple)
            next_frontier.add(t)

    if edges:
        h_ids, r_ids, t_ids = zip(*edges)
        return list(h_ids), list(r_ids), list(t_ids)

    return [], [], []

# -------------------- 在 BFS 子图内部构建增强子图标签 --------------------
def find_bridge_nodes(h_ids, t_ids, q_ids, a_ids):
    """
    在 BFS 子图内部寻找桥梁节点（topic -> ... -> answer）
    要求：
      - 只在子图内部构建，不访问全局图
      - 禁止自环 (h == t)
      - 桥梁节点必须能从问题节点到达至少一个答案节点
      - 保留指向症状/证候的节点，但不包括自环
    """
    from collections import defaultdict, deque

    # 子图节点集合
    nodes_in_subgraph = set(h_ids) | set(t_ids) | set(q_ids)
    a_ids_in_sub = [a for a in a_ids if a in nodes_in_subgraph]
    if not a_ids_in_sub:
        return set()

    # 构建正向邻接表（去掉自环）
    g_fwd = defaultdict(list)
    for h, t in zip(h_ids, t_ids):
        if h != t:  # 去掉自环
            g_fwd[h].append(t)

    # BFS 遍历，从 topic 出发，只保留能最终到达答案的节点
    bridges = set()
    visited_nodes = set()
    queue = deque(q_ids)

    while queue:
        cur = queue.popleft()
        if cur in visited_nodes:
            continue
        visited_nodes.add(cur)

        # 如果当前节点能到达答案，加入桥梁节点
        if cur in a_ids_in_sub:
            bridges.add(cur)

        # 遍历邻居，只走能到达答案的节点
        for nxt in g_fwd.get(cur, []):
            if nxt not in visited_nodes:
                # 简单剪枝：只走到答案节点或能最终到达答案的节点
                # 可以在这里用 reach_to_answer 集合进一步优化
                queue.append(nxt)

    # 最后保留 topic + answer + 中间桥梁节点
    return bridges | set(q_ids) | set(a_ids_in_sub)

# -------------------- 核心构建逻辑 --------------------
def build_ids(emb_file):
    print(f"\n=========== 处理 {emb_file} ===========")

    emb_dict = torch.load(emb_file)
    new_emb_dict = {}

    for sid, val in tqdm(emb_dict.items(), desc="Samples"):

        topic_entities = val.get("topic_entities", [])
        answers        = val.get("answers", [])

        # -------- q_entity_id_list --------
        q_ids = [entity2id[e] for e in topic_entities if e in entity2id]
        val["q_entity_id_list"] = q_ids

        # -------- a_entity_id_list --------
        a_ids = [entity2id[a] for a in answers if a in entity2id]
        val["a_entity_id_list"] = a_ids

        # -------- BFS 子图 --------
        h_ids, r_ids, t_ids = collect_subgraph(topic_entities, hop=2, second_hop_limit=3)
        # print(len(h_ids), len(r_ids), len(t_ids))
        val["h_id_list"] = h_ids
        val["r_id_list"] = r_ids
        val["t_id_list"] = t_ids

        # -------- 增强子图：找 bridge nodes（只在子图内部） --------
        # h_ids, r_ids, t_ids 已由 collect_subgraph 得到（长度 N）
        # q_ids, a_ids 已转为 id
        bridge_nodes = find_bridge_nodes(h_ids, t_ids, q_ids, a_ids)
        # print(len(bridge_nodes))

        # good nodes = question + answer (仅保留子图内的答案) + bridges
        # 注意：answers 中可能有不在子图的实体，我们不把它们加入 good_nodes
        nodes_in_subgraph = set(q_ids) | set(a_ids) | set(bridge_nodes)
        a_ids_in_sub = [a for a in a_ids if a in nodes_in_subgraph]
        good_nodes = set(q_ids) | set(a_ids_in_sub) | set(bridge_nodes)

        # -------- target_triple_probs（长度 = BFS子图三元组数 N）--------
        N = len(h_ids)
        target = torch.zeros(N, dtype=torch.float32)

        # 筛选增强子图三元组：
        # 只保留：
        # 1) head 在 good_nodes
        # 2) tail 在 good_nodes 或答案集合 a_ids_in_sub
        # 3) 去掉自环 h == t
        for i, (h, t) in enumerate(zip(h_ids, t_ids)):
            if h != t and h in good_nodes and t in good_nodes.union(a_ids_in_sub):
                target[i] = 1.0

        val["target_triple_probs"] = target

        # -------- topic_entity_one_hot --------
        # # shape = [num_entities_total, 1]
        # topic_one_hot = torch.zeros(num_entities_total, 1)
        # topic_one_hot[q_ids] = 1.0
        # -------- topic_entity_one_hot --------
        # shape = [num_entities_total, 2]
        topic_one_hot = torch.zeros(num_entities_total, 2).float()

        # 非 query 节点 -> [1,0]
        topic_one_hot[:, 0] = 1.0

        # query 节点 -> [0,1]
        if len(q_ids) > 0:
            topic_one_hot[q_ids, 0] = 0.0
            topic_one_hot[q_ids, 1] = 1.0

        val["topic_entity_one_hot"] = topic_one_hot

        # -------- 当前没有 non text entities --------
        val["non_text_entity_list"] = []

        new_emb_dict[sid] = val

    # -------- 保存 -------
    new_file = emb_file.replace(".pth", "_with_ids.pth")
    torch.save(new_emb_dict, new_file)

    print(f"Saved: {new_file}")
    print(f"Samples: {len(new_emb_dict)}")


# -------------------- 执行 --------------------
for split, path in EMB_FILES.items():
    if os.path.exists(path):
        build_ids(path)
    else:
        print(f"⚠️ File not found: {path}")
