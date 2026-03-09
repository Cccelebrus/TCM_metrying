##2025-11-28晚。把target_triple_probs转为中文，便于后面给llm看下理想状态下的推荐效果。
import torch
import json


ENTITY_FILE = "/home/gyj/local/SubgraphRAG-main/retrieve/data_files/entity_identifiers.txt"
RELATION_FILE = "/home/gyj/local/SubgraphRAG-main/retrieve/data_files/relation_list.txt"
VAL_PTH = "/home/gyj/local/SubgraphRAG-main/retrieve/data_files/tcm_embedding/test_with_ids.pth"

OUTPUT_JSONL = "test_gt_triples.jsonl"


# ------------------ 加载映射表 ------------------ #
def load_entity_map(path):
    return {idx: line.strip() for idx, line in enumerate(open(path, "r", encoding="utf-8"))}


def load_relation_map(path):
    return {idx: line.strip() for idx, line in enumerate(open(path, "r", encoding="utf-8"))}


# ------------------ 单样本转换 ------------------ #
def convert_sample_to_text(sample, id2entity, id2rel):
    target = sample["target_triple_probs"]
    print("target length:", len(target))  # 打印 target 长度

    # tensor 可能有浮点 0.0 / 1.0
    gt_indices = (target == 1.0).nonzero(as_tuple=True)[0].tolist()
    print("gt_indices length:", len(gt_indices))  # 打印 ground-truth 三元组数

    h_list = sample["h_id_list"]
    r_list = sample["r_id_list"]
    t_list = sample["t_id_list"]

    triples_text = []
    for idx in gt_indices:
        h = id2entity.get(h_list[idx], f"[UNK_{h_list[idx]}]")
        r = id2rel.get(r_list[idx], f"[UNK_{r_list[idx]}]")
        t = id2entity.get(t_list[idx], f"[UNK_{t_list[idx]}]")
        triples_text.append([h, r, t])  # 用列表形式更干净

    return triples_text


# ------------------ 主程序 ------------------ #
if __name__ == "__main__":
    print("Loading mapping files...")
    id2entity = load_entity_map(ENTITY_FILE)
    id2rel = load_relation_map(RELATION_FILE)

    print("Loading val_with_ids.pth...")
    data = torch.load(VAL_PTH, map_location="cpu")

    print(f"Processing {len(data)} samples...")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for sample_id, sample in data.items():
            gt_triples = convert_sample_to_text(sample, id2entity, id2rel)

            record = {
                "sample_id": sample_id,
                "ground_truth": gt_triples
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done. Saved to", OUTPUT_JSONL)
