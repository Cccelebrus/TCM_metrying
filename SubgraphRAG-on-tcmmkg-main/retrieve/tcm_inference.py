# tcm_inference.py
import os
import torch
from tqdm import tqdm

from src.dataset.tcm_retriever import TCMRetrieverDataset, collate_fn_tcm
from src.model.tcm_retriever2 import Retriever
from src.setup import set_seed, prepare_sample


def load_entity_relation_map(entity_path, relation_path):
    with open(entity_path, "r", encoding="utf-8") as f:
        entity_list = [line.strip() for line in f if line.strip()]
    with open(relation_path, "r", encoding="utf-8") as f:
        relation_list = [line.strip() for line in f if line.strip()]
    return entity_list, relation_list


@torch.no_grad()
def main(args):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # ------------------------ 加载实体和关系 ------------------------
    entity_list, relation_list = load_entity_relation_map(
        "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/entity_identifiers.txt",
        "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/relation_list.txt"
    )

    print("Loading checkpoint:", args.path)
    cpt = torch.load(args.path, map_location="cpu")
    config = cpt["config"]

    set_seed(config["env"]["seed"])
    torch.set_num_threads(config["env"]["num_threads"])

    # ------------------------ 读取测试集 ------------------------
    test_set = TCMRetrieverDataset("/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/tcm_embedding/test_with_ids.pth")

    emb_size = test_set[0]["q_emb"].shape[-1]
    model = Retriever(emb_size, **config["retriever"]).to(device)
    model.load_state_dict(cpt["model_state_dict"])
    model.eval()

    pred_dict = {}

    for i, raw_sample in enumerate(tqdm(test_set)):
        sample = collate_fn_tcm([raw_sample])
        (
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb,
            entity_embs, num_non_text_entities, relation_embs,
            topic_entity_one_hot, target_triple_probs, a_entity_id_list,q_entity_id_list,question
        ) = prepare_sample(device, sample)

        top_K_triples = []

        if len(h_id_tensor) > 0:
            pred_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor,
                q_emb, entity_embs, num_non_text_entities,
                relation_embs, topic_entity_one_hot
            )
            pred_scores = torch.sigmoid(pred_logits).reshape(-1)
            top_K = torch.topk(pred_scores, min(args.max_K, len(pred_scores)))

            for j, tid in enumerate(top_K.indices.tolist()):
                top_K_triples.append((
                    entity_list[h_id_tensor[tid].item()],
                    relation_list[r_id_tensor[tid].item()],
                    entity_list[t_id_tensor[tid].item()],
                    float(top_K.values[j])
                ))

        pred_dict[i] = {
            "question": raw_sample["question"],  # 
            "q_entity": raw_sample["q_entity_id_list"],
            "a_entity": raw_sample["a_entity_id_list"],
            "scored_triples": top_K_triples,
        }

    save_path = os.path.join(os.path.dirname(args.path), "retrieval_result_tcm_50trples_withcpm.pth")
    torch.save(pred_dict, save_path)
    print("\nSaved:", save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Jan05-06:42:07/cpt.pth")
    parser.add_argument("--max_K", type=int, default=50)
    args = parser.parse_args()
    main(args)
