"""
TCM知识图谱问答预测评估脚本
特点：
1. 预测统一格式: ans:[中药1,中药2,...]
2. JSONL文件仅含预测，需要从.pth文件获取真实答案
3. 计算Hit/F1/Precision/Recall
"""

import argparse
import json
import re
import torch
import os

def normalize(s: str) -> str:
    """中文直接去首尾空格，不做英文大小写/标点处理"""
    return s.strip()

def match(s1: str, s2: str) -> bool:
    """判断s2是否在s1中"""
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def get_ans_list(prediction: str):
    """从ans:[...]格式中提取中药列表"""
    match_obj = re.search(r'ans:\s*\[(.*?)\]', prediction)
    if not match_obj:
        return []
    items = match_obj.group(1).split(',')
    return [item.strip() for item in items]

def eval_acc(pred_list, answer):
    """计算命中比例"""
    if not pred_list:
        return 0
    matched = sum(any(match(a, p) for p in pred_list) for a in answer)
    return matched / len(answer) if answer else 0

def eval_hit(pred_list, answer):
    """命中任意一个即为Hit"""
    if not pred_list:
        return 0
    for a in answer:
        if any(match(a, p) for p in pred_list):
            return 1
    return 0

def eval_f1(pred_list, answer):
    """计算F1/Precision/Recall"""
    if not pred_list:
        return 0, 0, 0
    matched = sum(any(match(a, p) for p in pred_list) for a in answer)
    precision = matched / len(pred_list)
    recall = matched / len(answer) if answer else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def eval_results(predict_file, pth_file, cal_f1=True):
    """评估预测文件"""
    acc_list, hit_list, f1_list, precision_list, recall_list = [], [], [], [], []

    # 加载真实标签
    print(f"加载真实答案: {pth_file}")
    samples = torch.load(pth_file, map_location='cpu')

    base, ext = os.path.splitext(predict_file)
    detailed_eval_file = base + "_detailed_eval_result.jsonl"
    with open(predict_file, 'r', encoding='utf-8') as f, open(detailed_eval_file, 'w', encoding='utf-8') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print("无效行:", line)
                continue
            id = data['id']
            prediction = data['prediction']
            ans_list = get_ans_list(prediction)  # ans_list 就是 ['百合', '朱砂', ...]
            prediction = ans_list  # 后面 eval_f1 或 eval_acc 都可以直接用这个列表
            
            # 获取真实答案
            # sample_key = f'{id}'  # 注意加上前缀
            sample_key = f'id_{id}'  # 注意加上前缀
            if sample_key in samples:
                answer = samples[sample_key].get('answers', samples[sample_key].get('ground_truth', []))
            else:
                answer = []

            if cal_f1:
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precision_list.append(precision_score)
                recall_list.append(recall_score)
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({
                    'id': id,
                    'prediction': prediction,
                    'ground_truth': answer,
                    'acc': acc,
                    'hit': hit,
                    'f1': f1_score,
                    'precision': precision_score,
                    'recall': recall_score
                }, ensure_ascii=False) + '\n')
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({
                    'id': id,
                    'prediction': prediction,
                    'ground_truth': answer,
                    'acc': acc,
                    'hit': hit
                }, ensure_ascii=False) + '\n')

    # 汇总统计
    n = len(hit_list)
    avg_hit = sum(hit_list) * 100 / n if n else 0
    avg_f1 = sum(f1_list) * 100 / n if n else 0
    avg_precision = sum(precision_list) * 100 / n if n else 0
    avg_recall = sum(recall_list) * 100 / n if n else 0

    print(f"Hit: {avg_hit:.2f}, F1: {avg_f1:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}")

    eval_result_path = base + "_eval_gt_result.txt"
    with open(eval_result_path, 'w', encoding='utf-8') as f:
        f.write(f"Hit: {avg_hit:.2f}, F1: {avg_f1:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}\n")

    return avg_hit, avg_f1, avg_precision, avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", type=str, default="/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_20251129_1752.jsonl")
    parser.add_argument("--pth_file", type=str, default="/home/gyj/local/SubgraphRAG-main/retrieve/data_files/tcm_embedding/test_with_ids.pth")
    parser.add_argument("--cal_f1", default=True, help="是否计算F1/Precision/Recall")
    args = parser.parse_args()
    eval_results(args.predict_file, args.pth_file, cal_f1=True)
