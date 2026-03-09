# -*- coding: utf-8 -*-
"""
tcm_emb.py
embedding 阶段加入【中国药科大学语义增强后的节点 embedding】
CHP / CPM / symptom embedding 对齐 entity_identifiers.txt (需要先跑/home/gyj/local/on_tcmmkg/embeddings/)
"""

import os
import json
import torch
from tqdm import tqdm

from src.model.text_encoders.tcm_gte_large_zh import GTELargeZH
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class HFTextEncoder:
    def __init__(
        self,
        model_path="/home/gyj/local2/text2vec-large-chinese",
        device="cuda:0",
        max_length=64,
        out_dim=None,  # 新增：投影维度
    ):
        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.out_dim = out_dim
        if out_dim is not None:
            # 添加线性投影层
            self.proj = torch.nn.Linear(self.model.config.hidden_size, out_dim).to(self.device)
        else:
            self.proj = None

    @torch.no_grad()
    def encode(self, texts, batch_size=64, normalize=True):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            # mean pooling
            emb = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)

            if self.proj is not None:
                emb = self.proj(emb)

            if normalize:
                emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb)

        return torch.cat(all_embeddings, dim=0)


import pandas as pd

# ===================== 配置 =====================
BASE_DIR = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files"
DATASET_DIR = f"{BASE_DIR}/tcm"

ENTITY_FILE = f"{BASE_DIR}/entity_identifiers.txt"
REL_FILE    = f"{BASE_DIR}/relation_list.txt"

# === 你已有的三类 embedding（真实存在） ===
CHP_EMB_PTH      = "/home/gyj/local/on_tcmmkg/embeddings/chp_embeddings.pt"
CPM_EMB_PTH      = "/home/gyj/local/on_tcmmkg/embeddings/cpm_embeddings.pt"
SYMPTOM_EMB_PTH  = "/home/gyj/local/on_tcmmkg/embeddings/symptom_embeddings.pt"

EMB_DIR = f"{BASE_DIR}/tcm_embedding"

TRAIN_FILE = os.path.join(DATASET_DIR, "train.jsonl")
VAL_FILE   = os.path.join(DATASET_DIR, "val.jsonl")
TEST_FILE  = os.path.join(DATASET_DIR, "test.jsonl")

TRAIN_EMB = os.path.join(EMB_DIR, "train.pth")
VAL_EMB   = os.path.join(EMB_DIR, "val.pth")
TEST_EMB  = os.path.join(EMB_DIR, "test.pth")

BATCH_SIZE = 64

# ===================== 工具函数 =====================
def read_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_dict(path, obj):
    torch.save(obj, path)
    print(f"[✓] Saved embeddings to {path}")

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line.strip())
            data.append((
                f"id_{idx}",
                obj.get("question", ""),
                obj.get("topic_entities", []),
                obj.get("answers", []),
            ))
    return data

# ===================== 构建实体 embedding =====================
def build_entity_embedding(entity_ids, device):
    """
    根据 entity_identifiers.txt 顺序
    拼接 CHP / CPM / symptom embedding
    对缺失节点用 HFTextEncoder 正常生成 embedding
    返回: Tensor [num_entities, dim]
    """

    print("🚀 加载已有 CHP/CPM/Symptom embedding...")
    chp = torch.load(CHP_EMB_PTH, map_location=device)
    cpm = torch.load(CPM_EMB_PTH, map_location=device)
    sym = torch.load(SYMPTOM_EMB_PTH, map_location=device)

    chp_map = dict(zip(chp["names"], chp["embeddings"].to(device)))
    cpm_map = dict(zip(cpm["names"], cpm["embeddings"].to(device)))
    sym_map = dict(zip(sym["ids"], sym["embeddings"].to(device)))

    dim = chp["embeddings"].shape[1]
    all_embs = []

    found_chp, found_cpm, found_sym = 0, 0, 0
    missing_nodes = []

    # HFTextEncoder 用于缺失节点
    hf_encoder = HFTextEncoder(
        model_path="/home/gyj/local2/text2vec-large-chinese",
        device=device,
        max_length=64,
        out_dim=dim
    )

    for eid in tqdm(entity_ids, desc="对齐实体 embedding"):
        if eid in chp_map:
            all_embs.append(chp_map[eid])
            found_chp += 1
        elif eid in cpm_map:
            all_embs.append(cpm_map[eid])
            found_cpm += 1
        elif eid in sym_map:
            all_embs.append(sym_map[eid])
            found_sym += 1
        else:
            missing_nodes.append(eid)
            # 正常编码
            emb = hf_encoder.encode([eid], batch_size=1, normalize=True)[0]
            all_embs.append(emb)

    print(f"✅ CHP 匹配数量: {found_chp}")
    print(f"✅ CPM 匹配数量: {found_cpm}")
    print(f"✅ Symptom 匹配数量: {found_sym}")
    print(f"⚠️ 缺失节点数量（已自动编码）: {len(missing_nodes)}")
    if missing_nodes:
        print("示例缺失节点:", missing_nodes[:10])

    return torch.stack(all_embs, dim=0)

# ===================== Embedding 生成 =====================
def get_emb(dataset, encoder, save_path, entity_embs, relations):
    emb_dict = {}
    relation_embs = encoder.embed(relations)

    for i in tqdm(range(0, len(dataset), BATCH_SIZE),
                  desc=f"Embedding {os.path.basename(save_path)}"):
        batch = dataset[i:i + BATCH_SIZE]
        ids, questions, topic_entities, answers = zip(*batch)

        q_embs = encoder.embed(list(questions))

        for j, eid in enumerate(ids):
            emb_dict[eid] = {
                "q_emb": q_embs[j],
                "question": questions[j],
                "entity_embs": entity_embs,       # ✅ 对齐后的整表
                "relation_embs": relation_embs,   # ✅ relation_list.txt
                "topic_entities": topic_entities[j],
                "answers": answers[j],
            }

    save_dict(save_path, emb_dict)

# ===================== 主函数 =====================
def main():
    os.makedirs(EMB_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = GTELargeZH(
        device=device,
        model_path="/mnt/local2/gyj/text2vec-large-chinese",
        normalize=True,
    )

    entities  = read_list(ENTITY_FILE)
    relations = read_list(REL_FILE)

    print(f"实体总数: {len(entities)}")
    print(f"关系总数: {len(relations)}")

    # 构建【最终对齐的实体 embedding】
    entity_embs = build_entity_embedding(entities, device)
    print("entity_embs shape:", entity_embs.shape)

    train_set = read_jsonl(TRAIN_FILE)
    val_set   = read_jsonl(VAL_FILE)
    test_set  = read_jsonl(TEST_FILE)

    get_emb(train_set, encoder, TRAIN_EMB, entity_embs, relations)
    get_emb(val_set,   encoder, VAL_EMB,   entity_embs, relations)
    get_emb(test_set,  encoder, TEST_EMB,  entity_embs, relations)

if __name__ == "__main__":
    main()
