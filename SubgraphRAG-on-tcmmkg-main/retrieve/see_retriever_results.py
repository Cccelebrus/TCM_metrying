import torch
import pprint  # 更美观地打印

# pth_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Nov26-08:03:36/retrieval_result_tcm.pth"
pth_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Nov29-07:57:06/retrieval_result_tcm_100triples.pth"

# 加载
pred_dict = torch.load(pth_path)

print(f"共有 {len(pred_dict)} 个样本\n")

# pprint 打印前几个样本
for i, (sample_id, sample_dict) in enumerate(pred_dict.items()):
    print(f"样本 {i} (key={sample_id}):")
    pprint.pprint(sample_dict)
    print("\n" + "-"*50 + "\n")
    if i >= 1:  # 只看前 5 个样本
        break

# 如果想只看 scored_triples
# for sample_id, sample_dict in list(pred_dict.items())[:5]:
#     print(sample_dict['scored_triples'])
