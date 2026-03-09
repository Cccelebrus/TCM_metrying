import argparse
import json
import re
import torch
import os
import math


def normalize(s: str) -> str:
    return s.strip()


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def get_ans_list(prediction: str):
    match_obj = re.search(r'ans:\s*\[(.*?)\]', prediction)
    if not match_obj:
        return []
    items = match_obj.group(1).split(',')
    return [item.strip() for item in items]


def eval_f1(pred_list, answer):
    if not pred_list:
        return 0, 0, 0
    matched = sum(any(match(a, p) for p in pred_list) for a in answer)
    precision = matched / len(pred_list)
    recall = matched / len(answer) if answer else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


# ---------------------------
# ⭐ 新增：@K 评估函数（方案B）
# ---------------------------
def eval_at_k(pred_list, answer, k):
    if not pred_list:
        return 0, 0, 0

    topk = pred_list[:min(k, len(pred_list))]

    matched = sum(any(match(a, p) for p in topk) for a in answer)

    precision = matched / len(topk)
    recall = matched / len(answer) if answer else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# ---------------------------
# ⭐ 新增：NDCG@K
# ---------------------------
def ndcg_at_k(pred_list, answer, k):
    if not pred_list:
        return 0.0

    topk = pred_list[:min(k, len(pred_list))]
    dcg = 0.0

    for idx, p in enumerate(topk):
        rel = 1 if any(match(a, p) for a in answer) else 0
        if rel > 0:
            dcg += 1 / math.log2(idx + 2)

    ideal_rels = [1] * min(len(answer), k)
    idcg = 0.0
    for idx, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(idx + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
#                    主评估逻辑
# ============================================================
def eval_results(predict_file, pth_file, cal_f1=True):
    f1_list, precision_list, recall_list = [], [], []

    Ks = [1, 3, 5, 8]
    precision_k = {k: [] for k in Ks}
    recall_k = {k: [] for k in Ks}
    f1_k = {k: [] for k in Ks}
    ndcg_k = {k: [] for k in Ks}  # ⭐ 新增

    print(f"加载真实答案: {pth_file}")
    samples = torch.load(pth_file, map_location='cpu')

    base, ext = os.path.splitext(predict_file)
    detailed_eval_file = base + "_detailed_eval_result.jsonl"

    with open(predict_file, 'r', encoding='utf-8') as f, \
         open(detailed_eval_file, 'w', encoding='utf-8') as f2:

        for line in f:
            try:
                data = json.loads(line)
            except:
                print("无效行:", line)
                continue

            id = data['id']
            prediction_raw = data['prediction']
            pred_list = get_ans_list(prediction_raw)

            sample_key = f'id_{id}'
            if sample_key in samples:
                answer = samples[sample_key].get('answers', samples[sample_key].get('ground_truth', []))
            else:
                answer = []

            f1_score, precision_score, recall_score = eval_f1(pred_list, answer)

            f1_list.append(f1_score)
            precision_list.append(precision_score)
            recall_list.append(recall_score)

            # ---------------------------
            # ⭐ 计算 topK + NDCG
            # ---------------------------
            k_results = {}
            for k in Ks:
                pk, rk, fk = eval_at_k(pred_list, answer, k)
                precision_k[k].append(pk)
                recall_k[k].append(rk)
                f1_k[k].append(fk)

                nk = ndcg_at_k(pred_list, answer, k)
                ndcg_k[k].append(nk)

                k_results[f"precision@{k}"] = pk
                k_results[f"recall@{k}"] = rk
                k_results[f"f1@{k}"] = fk
                k_results[f"ndcg@{k}"] = nk  # ⭐ 新增

            f2.write(json.dumps({
                'id': id,
                'prediction': pred_list,
                'ground_truth': answer,
                'f1': f1_score,
                'precision': precision_score,
                'recall': recall_score,
                **k_results
            }, ensure_ascii=False) + '\n')

    n = len(f1_list)
    avg_f1 = sum(f1_list) * 100 / n if n else 0
    avg_precision = sum(precision_list) * 100 / n if n else 0
    avg_recall = sum(recall_list) * 100 / n if n else 0

    summary_k = {}
    for k in Ks:
        summary_k[f"f1@{k}"] = sum(f1_k[k]) * 100 / n if n else 0
        summary_k[f"precision@{k}"] = sum(precision_k[k]) * 100 / n if n else 0
        summary_k[f"recall@{k}"] = sum(recall_k[k]) * 100 / n if n else 0
        summary_k[f"ndcg@{k}"] = sum(ndcg_k[k]) * 100 / n if n else 0  # ⭐新增

    print(f"F1: {avg_f1:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}")
    for k in Ks:
        print(f"F1@{k}: {summary_k[f'f1@{k}']:.2f}, "
              f"P@{k}: {summary_k[f'precision@{k}']:.2f}, "
              f"R@{k}: {summary_k[f'recall@{k}']:.2f}, "
              f"NDCG@{k}: {summary_k[f'ndcg@{k}']:.2f}")

    eval_result_path = base + "_eval_gt_result.txt"
    with open(eval_result_path, 'w', encoding='utf-8') as f:
        f.write(f"F1: {avg_f1:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}\n")
        for k in Ks:
            f.write(f"F1@{k}: {summary_k[f'f1@{k}']:.2f}, "
                    f"Precision@{k}: {summary_k[f'precision@{k}']:.2f}, "
                    f"Recall@{k}: {summary_k[f'recall@{k}']:.2f}, "
                    f"NDCG@{k}: {summary_k[f'ndcg@{k}']:.2f}\n")

    return avg_f1, avg_precision, avg_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--predict_file", type=str, default="/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_20251129_1752.jsonl")
    # parser.add_argument("--predict_file", type=str, default="/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_tcm30triples_no_score.jsonl")
    parser.add_argument("--predict_file", type=str, default="/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_20260105_tcm50triples.jsonl")
    parser.add_argument("--pth_file", type=str, default="/home/gyj/local/SubgraphRAG-main/retrieve/data_files/tcm_embedding/test_with_idsS.pth")    
    args = parser.parse_args()
    eval_results(args.predict_file, args.pth_file)
