# -*- coding: utf-8 -*-
"""
生成三元组（含 CPM 节点）
1️⃣ 融合 symptom KG + herb KG
2️⃣ 融合处方：symptom -> herb
3️⃣ 融合 CPM：symptom -> CPM, CPM -> CHP
4️⃣ 使用 fused_herb_nodes.tsv 做别名映射
5️⃣ 输出 triples_cpm.tsv / entity_name2id.pkl / entity_mapping.xlsx
"""

import pandas as pd
import pickle
from tqdm import tqdm

# ============================
# 加载 KG 映射和边
# ============================
def load_mapping(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, idx = line.strip().split('\t')
            mapping[int(idx)] = name
    return mapping

def load_edges(path):
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split()
            edges.append((int(h), int(r), int(t)))
    return edges

def load_prescriptions(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            left, right = line.strip().split('\t')
            sym_set = [int(x) for x in left.split()]
            herb_set = [int(x) for x in right.split()]
            pairs.append((sym_set, herb_set))
    return pairs

# ============================
# CHP 别名映射
# ============================
def load_alias_to_mainname(fused_file):
    df = pd.read_csv(fused_file, sep="\t", dtype=str).fillna("")
    alias2main = {}
    for _, row in df.iterrows():
        main = row["Chinese_herbal_pieces"].strip()
        if not main:
            continue
        alias2main[main] = main
        for s in row.get("Chinese_synonyms", "").replace("|", " ").split():
            s = s.strip()
            if s:
                alias2main[s] = main
        if "CHP_ID" in row:
            chp_id = row["CHP_ID"].strip()
            if chp_id:
                alias2main[chp_id] = main
    return alias2main

# ============================
# CPM_ID → 中文名 映射
# ============================
def load_cpm_id2name(cpm_mapping_file):
    df = pd.read_csv(cpm_mapping_file, sep="\t", dtype=str).fillna("")
    cpm_id2name = {}
    for _, row in df.iterrows():
        cpm_id = row["CPM_ID"].strip()
        cpm_name = row["Chinese_patent_medicine"].strip()
        if cpm_id and cpm_name:
            cpm_id2name[cpm_id] = cpm_name
    return cpm_id2name

# ============================
# 主函数
# ============================
def generate_triples_with_cpm(
        sym_dir,
        herb_dir,
        pres_path,
        fused_file,
        presc_cpm_file,
        cpm_mapping_file,
        output_tsv
):
    sym_entities = load_mapping(f"{sym_dir}/entitymapping.txt")
    herb_entities = load_mapping(f"{herb_dir}/entitymapping.txt")
    sym_relations = load_mapping(f"{sym_dir}/relationmapping.txt")
    herb_relations = load_mapping(f"{herb_dir}/relationmapping.txt")
    sym_edges = load_edges(f"{sym_dir}/kg_final_one_hop.txt")
    herb_edges = load_edges(f"{herb_dir}/kg_final_one_hop.txt")
    prescriptions = load_prescriptions(pres_path)

    alias2main = load_alias_to_mainname(fused_file)
    presc_df = pd.read_csv(presc_cpm_file)
    cpm_id2name = load_cpm_id2name(cpm_mapping_file)

    triples = []
    entity_name2id = {}
    entity_types = {}
    current_index = 0

    # === 1️⃣ 处方 symptom -> herb ===
    for i, (sym_set, herb_set) in enumerate(prescriptions):
        for sid in sym_set:
            s_name = sym_entities[sid]
            if s_name not in entity_name2id:
                entity_name2id[s_name] = current_index
                entity_types[s_name] = "symptom" if sid < 360 else "other"
                current_index += 1

            for hid in herb_set:
                h_raw = herb_entities[hid]
                h_name = alias2main.get(h_raw, h_raw)
                if h_name not in entity_name2id:
                    entity_name2id[h_name] = current_index
                    entity_types[h_name] = "herb"
                    current_index += 1
                triples.append((s_name, "被治疗", h_name))
                triples.append((h_name, "治疗", s_name))

        # === 2️⃣ CPM 节点 ===
        if i >= len(presc_df):
            continue

        presc_row = presc_df.iloc[i]
        cpm_id = presc_row["max_CPM"]
        if pd.isna(cpm_id):
            continue

        cpm_id = str(cpm_id).strip()
        cpm_name = cpm_id2name.get(cpm_id, cpm_id)  # 🔑 用中文名

        if cpm_name not in entity_name2id:
            entity_name2id[cpm_name] = current_index
            entity_types[cpm_name] = "cpm"
            current_index += 1

        # symptom -> CPM
        for sid in sym_set:
            s_name = sym_entities[sid]
            triples.append((s_name, "关联CPM", cpm_name))
            triples.append((cpm_name, "关联症状", s_name))

        # CPM -> CHP
        chp_list_str = str(presc_row["CHPs_in_CPM"])
        chp_list = [alias2main.get(h.strip()) for h in chp_list_str.split(",")]
        chp_list = [h for h in chp_list if h]

        for h_name in chp_list:
            if h_name not in entity_name2id:
                entity_name2id[h_name] = current_index
                entity_types[h_name] = "herb"
                current_index += 1
            triples.append((cpm_name, "包含", h_name))
            triples.append((h_name, "属于", cpm_name))

    # === 3️⃣ 症状 KG ===
    for sid1, rid, sid2 in sym_edges:
        rel = sym_relations.get(rid, f"sym_rel_{rid}")
        rel_rev = rel + "_逆"
        h = sym_entities[sid1]
        t = sym_entities[sid2]
        triples.append((h, rel, t))
        triples.append((t, rel_rev, h))
        for n, sid in [(h, sid1), (t, sid2)]:
            if n not in entity_name2id:
                entity_name2id[n] = current_index
                entity_types[n] = "symptom" if sid < 360 else "other"
                current_index += 1

    # === 4️⃣ 草药 KG ===
    for hid1, rid, hid2 in herb_edges:
        rel = herb_relations.get(rid, f"herb_rel_{rid}")
        rel_rev = rel + "_逆"
        h = herb_entities[hid1]
        t = herb_entities[hid2]
        triples.append((h, rel, t))
        triples.append((t, rel_rev, h))
        for n, hid in [(h, hid1), (t, hid2)]:
            if n not in entity_name2id:
                entity_name2id[n] = current_index
                entity_types[n] = "herb" if 360 <= hid <= 1112 else "other"
                current_index += 1

    # === 保存输出 ===
    with open(output_tsv, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

    with open("/home/gyj/local/SubgraphRAG-on-tcmmkg-main/entity_name2id.pkl", "wb") as f:
        pickle.dump(entity_name2id, f)

    df = pd.DataFrame({
        "node_id": list(entity_name2id.keys()),
        "index": list(entity_name2id.values()),
        "type": [entity_types[n] for n in entity_name2id]
    })
    df.to_excel("/home/gyj/local/SubgraphRAG-on-tcmmkg-main/entity_mapping.xlsx", index=False)

    with open("/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/entity_identifiers.txt", "w", encoding="utf-8") as f:
        for n, _ in sorted(entity_name2id.items(), key=lambda x: x[1]):
            f.write(f"{n}\n")

    with open("/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/relation_list.txt", "w", encoding="utf-8") as f:
        for r in sorted(set(r for _, r, _ in triples)):
            f.write(f"{r}\n")

    print(f"✅ 三元组数：{len(triples)}")
    print(f"✅ 实体数：{len(entity_name2id)}")

    return triples, entity_name2id, entity_types


if __name__ == "__main__":
    generate_triples_with_cpm(
        sym_dir="/home/gyj/local/SubgraphRAG-on-tcmmkg-main/KGAT_sym_kg",
        herb_dir="/home/gyj/local/SubgraphRAG-on-tcmmkg-main/KGAT_herb_kg",
        pres_path="/home/gyj/local/SubgraphRAG-on-tcmmkg-main/KGAT_herb_kg/train.txt",
        fused_file="/home/gyj/local/on_tcmmkg/fused_herb_nodes.tsv",
        presc_cpm_file="/home/gyj/local/on_tcmmkg/prescription_CPM_max_coverage.csv",
        cpm_mapping_file="/home/gyj/local/on_tcmmkg/tcmmkg/D2_Chinese_patent_medicine.tsv",
        output_tsv="/home/gyj/local/SubgraphRAG-on-tcmmkg-main/triples_cpm.tsv"
    )
