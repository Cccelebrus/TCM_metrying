import os
import torch
import pandas as pd
from collections import defaultdict

# -------------------- 配置 --------------------
BASE_DIR = "/home/gyj/local/SubgraphRAG-main/retrieve/data_files/"
EMB_FILES = {
    'train': os.path.join(BASE_DIR, 'tcm_embedding/train.pth'),
    'val': os.path.join(BASE_DIR, 'tcm_embedding/val.pth'),
    'test': os.path.join(BASE_DIR, 'tcm_embedding/test.pth')
}
ENTITY_FILE = os.path.join(BASE_DIR, 'entity_identifiers.txt')
REL_FILE = os.path.join(BASE_DIR, 'relation_list.txt')
TRIPLES_FILE = os.path.join(BASE_DIR, 'triples.tsv')

# -------------------- 读取映射 --------------------
def read_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

entities = read_list(ENTITY_FILE)
relations = read_list(REL_FILE)

print(f"实体数量: {len(entities)}, 关系数量: {len(relations)}")

# -------------------- 读取 triples --------------------
triples = []
triples_by_head = defaultdict(list)
with open(TRIPLES_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        h, r, t = line.strip().split("\t")
        triples.append((h, r, t))
        triples_by_head[h].append((r, t))

print(f"triples 总数: {len(triples)}")

# -------------------- 检查 embeddings --------------------
def check_embeddings(file_path):
    emb_dict = torch.load(file_path)
    print(f"\n检查 {file_path}, 样本数量: {len(emb_dict)}")
    
    issues_len = 0
    issues_triples = 0

    for eid, val in emb_dict.items():
        q_emb = val['q_emb']
        entity_embs = val['entity_embs']
        relation_embs = val['relation_embs']

        # 检查长度一致性
        if len(entity_embs) != len(relation_embs):
            print(f"❌ {eid} 的 entity_embs 与 relation_embs 长度不一致: "
                  f"{len(entity_embs)} vs {len(relation_embs)}")
            issues_len += 1
        
        # 检查 topic_entity + relation 是否至少能匹配一条 triples
        for idx, rel in enumerate(relation_embs):
            # 这里 rel 可能是 embedding tensor，我们假设有对应 relation list 作为顺序映射
            # 如果没有实际 mapping，可以跳过或只打印长度检查
            # 下面只做简单检查：确保 entity_embs 对应的 topic_entity 在 triples_by_head 中有对应关系
            # 假设 entity_embs 的顺序就是 topic_entity 顺序
            topic_entity = val.get('topic_entities', None)
            if topic_entity:
                for h in topic_entity:
                    rel_name = rel  # 这里如果有关系名称列表，用它替换 rel
                    found = False
                    for r, t in triples_by_head.get(h, []):
                        if r == rel_name:
                            found = True
                            break
                    if not found:
                        print(f"⚠️ {eid} 的 topic_entity {h} 与关系 {rel_name} 无匹配 triples")
                        issues_triples += 1
                        break  # 每条样本只报一次

    if issues_len == 0:
        print(f"✅ 所有样本 entity_embs 与 relation_embs 长度一致")
    else:
        print(f"⚠️ 共发现 {issues_len} 条样本长度不一致")

    if issues_triples == 0:
        print(f"✅ 所有样本 topic_entity + relation 至少匹配一条 triples")
    else:
        print(f"⚠️ 共发现 {issues_triples} 条样本 topic_entity + relation 无匹配 triples")

# -------------------- 执行检查 --------------------
for split, path in EMB_FILES.items():
    if os.path.exists(path):
        check_embeddings(path)
    else:
        print(f"⚠️ {path} 不存在")
