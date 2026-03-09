'''
tcm_emb.py就是补充了原来/home/gyj/local/SubgraphRAG-main/retrieve/tcm_compute_entity_relation_embeddings.py只有全量实体/关系向量的那一步：

它针对每条 QA 样本（train/val/test），把问题文本 question、topic entity 列表、relation 列表送进你的中文 text encoder (GTELargeZH)

返回向量后，按 SubgraphRAG 原论文要求的结构保存为 train.pth / val.pth / test.pth：
'''
# file: retrieve/tcm_emb.py
import os
import torch
from tqdm import tqdm
import json

from src.model.text_encoders.tcm_gte_large_zh import GTELargeZH
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# --------------------- 配置 ---------------------
BASE_DIR = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files"
DATASET_DIR = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/tcm"  # train/val/test JSONL 在这个目录下
ENTITY_FILE = f"{BASE_DIR}/entity_identifiers.txt"
REL_FILE = f"{BASE_DIR}/relation_list.txt"
EMB_DIR = f"{BASE_DIR}/tcm_embedding"

TRAIN_FILE = os.path.join(DATASET_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATASET_DIR, "val.jsonl")
TEST_FILE = os.path.join(DATASET_DIR, "test.jsonl")

TRAIN_EMB = os.path.join(EMB_DIR, "train.pth")
VAL_EMB = os.path.join(EMB_DIR, "val.pth")
TEST_EMB = os.path.join(EMB_DIR, "test.pth")

BATCH_SIZE = 64

# --------------------- 工具函数 ---------------------
def read_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_dict(path, d):
    torch.save(d, path)
    print(f"Saved embeddings to {path}")

def chunked(iterable, n):
    """分块迭代"""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

# --------------------- 数据集读取（已修改） ---------------------
def read_jsonl(file_path):
    """
    自动保证每条样本存在 relations 字段；
    如果不存在，则填充 "被治疗"
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            obj = json.loads(line.strip())

            question = obj.get("question", "")
            topic_entities = obj.get("topic_entities", [])
            relation_list = ["被治疗","治疗","主治","主治_逆","功能","功能_逆","关联CPM","关联症状"]
            answers_list = obj.get("answers", [])
            data.append((f"id_{idx}", question, topic_entities, relation_list, answers_list))

    return data


# --------------------- 生成 embeddings ---------------------
def get_emb(dataset, encoder, save_path, all_entities, all_relations):
    """
    dataset: 每条样本 (id, question, topic_entities, relation_list, answers_list)
    all_entities: list[str] 全量实体，用于生成全量 entity embedding
    """
    emb_dict = {}

    # 先生成全量实体 embedding，保证每条样本 entity_embs 一致
    all_entity_embs = encoder.embed(all_entities)  # [num_entities, emb_dim]
    all_relation_embs = encoder.embed(all_relations)
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=f"Embedding {save_path}"):
        batch = dataset[i:i+BATCH_SIZE]
        ids, q_texts, entity_lists, relation_lists, answers_list = zip(*batch)

        # question embedding
        q_embs = encoder.embed(list(q_texts))


        for idx, eid in enumerate(ids):
            emb_dict[eid] = {
                'q_emb': q_embs[idx],
                'question': q_texts[idx],        # ✅ 新增原始文本
                'entity_embs': all_entity_embs,        # 改为全量实体 embedding
                'relation_embs': all_relation_embs,       # 仍然全量 relation embedding
                'topic_entities': entity_lists[idx],  # 保留 topic_entities
                'answers': answers_list[idx],  
            }

    save_dict(save_path, emb_dict)

# --------------------- 主函数 ---------------------
def main():
    os.makedirs(EMB_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化中文 text encoder
    model_path = "/mnt/local2/gyj/text2vec-large-chinese"
    encoder = GTELargeZH(device=device, model_path=model_path, normalize=True)

    # 读取实体和关系
    entities = read_list(ENTITY_FILE)
    relations = read_list(REL_FILE)

    # 读取数据集（自动补 relations）
    train_set = read_jsonl(TRAIN_FILE)
    val_set = read_jsonl(VAL_FILE)
    test_set = read_jsonl(TEST_FILE)

    # 生成 embeddings
    get_emb(train_set, encoder, TRAIN_EMB, entities,relations)
    get_emb(val_set, encoder, VAL_EMB, entities,relations)
    get_emb(test_set, encoder, TEST_EMB, entities,relations)

if __name__ == "__main__":
    main()
