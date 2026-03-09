import torch
import json
import numpy as np
from collections import defaultdict

# ============================
# 配置路径
# ============================
pred_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Nov29-07:57:06/retrieval_result_tcm_100triples.pth"
gold_json_path = "/home/gyj/local/SubgraphRAG-main/retrieve/data_files/tcm/test.jsonl"

# ============================
# 读取预测结果 (list 顺序)
# ============================
pred_data = list(torch.load(pred_path).values())

# ============================
# 读取 gold JSONL 文件 (list 顺序)
# ============================
gold_data = []
with open(gold_json_path, "r", encoding="utf8") as f:
    for line in f:
        if line := line.strip():
            gold_data.append(json.loads(line))

# 确保样本数一致
n = min(len(pred_data), len(gold_data))
print(f"有效评估样本数：{n}")

# ============================
# 统计指标
# ============================
hit_ratios = []
avg_ranks = []
rank_count = defaultdict(int)

for i in range(n):

    pred_item = pred_data[i]
    gold_item = gold_data[i]

    gold_answers = gold_item["answers"]
    pred_herbs = [t[2] for t in pred_item["scored_triples"]]

    found_ranks = []

    for herb in gold_answers:
        if herb in pred_herbs:
            r = pred_herbs.index(herb) + 1
            found_ranks.append(r)
            rank_count[r] += 1
        else:
            rank_count["NOT_FOUND"] += 1

    # 命中率
    hit_ratios.append(len(found_ranks) / len(gold_answers))

    if found_ranks:
        avg_ranks.append(np.mean(found_ranks))


# ============================
# 输出结果
# ============================
print("\n================= 结果统计 =================")
print(f"样本数：{n}")
print(f"平均命中率：{np.mean(hit_ratios):.4f}")
print(f"平均排名（仅统计命中部分）：{np.mean(avg_ranks):.2f}")

print("\n===== 各排名出现次数分布 =====")
sorted_ranks = sorted([r for r in rank_count if r != "NOT_FOUND"])
for r in sorted_ranks:
    print(f"排名 {r}: {rank_count[r]} 次")
print(f"未出现：{rank_count['NOT_FOUND']} 次")
