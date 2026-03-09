'''
读取预训练中文语言模型，把词语转为向量embedding，保存为：
"tcm_emb/entity_embeddings.pth"
"tcm_emb/relation_embeddings.pth"
'''
# file: retrieve/compute_entity_relation_embeddings.py 
import os
import torch
from tqdm import tqdm

# change import path if needed
from src.model.text_encoders.tcm_gte_large_zh import GTELargeZH

BASE_DIR = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files"
TRIPLE_FILE = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/triples_cpm.tsv" ###fnew
ENT_FILE = f"{BASE_DIR}/entity_identifiers.txt"
REL_FILE = f"{BASE_DIR}/relation_list.txt"
EMB_DIR = f"{BASE_DIR}/tcm_embedding"
ENTITY_EMB_PATH = f"{EMB_DIR}/entity_embeddings.pth"
RELATION_EMB_PATH = f"{EMB_DIR}/relation_embeddings.pth"

def read_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_dict(path, d):
    torch.save(d, path)
    print("Saved", path)

def chunked(iterable, n):
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

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 你的模型位置
    model_path = "/mnt/local2/gyj/text2vec-large-chinese"
    encoder = GTELargeZH(device=device, model_path=model_path, normalize=True)

    # ========== 如果 relation_list.txt 不存在，自动从 triples.tsv 构造 ==========
    if not os.path.exists(REL_FILE):
        if os.path.exists(TRIPLE_FILE):
            rels = set()
            with open(TRIPLE_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        rels.add(parts[1])

            with open(REL_FILE, 'w', encoding='utf-8') as out:
                for r in sorted(rels):
                    out.write(r + "\n")
            print("Built relation_list.txt from triples.tsv")
        else:
            raise FileNotFoundError(
                f"No relation_list.txt and no triples.tsv found in {BASE_DIR}"
            )

    # ========= 加载实体 & 关系 =============
    ents = read_list(ENT_FILE)
    rels = read_list(REL_FILE)

    # ========= 批量编码实体 =========
    entity_emb = {}
    batch_size = 64
    for chunk in tqdm(list(chunked(ents, batch_size)), desc="entities"):
        embs = encoder.embed(chunk)
        for name, emb in zip(chunk, embs):
            entity_emb[name] = emb.numpy()

    # ========= 批量编码关系 =========
    relation_emb = {}
    for chunk in tqdm(list(chunked(rels, batch_size)), desc="relations"):
        embs = encoder.embed(chunk)
        for name, emb in zip(chunk, embs):
            relation_emb[name] = emb.numpy()

    # ========= 写入 emb =========
    os.makedirs(EMB_DIR, exist_ok=True)
    save_dict(ENTITY_EMB_PATH, entity_emb)
    save_dict(RELATION_EMB_PATH, relation_emb)
