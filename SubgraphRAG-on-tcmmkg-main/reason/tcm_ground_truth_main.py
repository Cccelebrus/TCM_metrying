import os
import json
import torch
from tqdm import tqdm
from llm_utils_tcm_qwen import llm_init

# ------------------ 读取 retriever 输出，仅获取 question ------------------
def load_questions_from_retriever(path):
    data = torch.load(path, map_location="cpu")
    q_map = {}
    for sample_id, v in data.items():
        q_map[sample_id] = v.get("question", "")
    return q_map

# ------------------ 读取 ground truth triples ------------------
def load_ground_truth(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sid = item["sample_id"]
            gt[sid] = item["ground_truth"]
    return gt

# ------------------ 格式化三元组 ------------------
def triples_to_text(triples):
    return "\n".join([f"{h}-{r}->{t}" for h, r, t in triples])

# ------------------ 构造 prompt ------------------
def build_prompt(question, triples):
    triples_text = triples_to_text(triples)
    return f"""
你是一名中医知识图谱推理助手。
下面是知识图谱中真实的 ground-truth 相关三元组（结构化知识）：

{triples_text}

请基于以上知识推理回答下面的问题：

问题：{question}

请给出你的中医推理解释，并最终给出明确答案，最终推荐的中药统一格式为"ans:[中药1,中药2,……]"。
注意：每个问题推荐的中药不超过 20 种。
"""

# ------------------ 主函数 ------------------
def main():
    gt_path = "test_gt_triples.jsonl"
    retriever_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Nov26-08:03:36/retrieval_result_tcm.pth"
    output_path = "/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_GT_predictions.jsonl"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5"
    batch_size = 1
    model_name = "/mnt/local/ccl/qwen2.5-7b-instruct/"

    print("[1] Loading retriever questions...")
    q_map = load_questions_from_retriever(retriever_path)

    print("[2] Loading ground truth triples...")
    gt_triples = load_ground_truth(gt_path)

    print(f"[INFO] 共加载 {len(gt_triples)} 条 GT 样本。")

    print("[3] Initializing LLM...")
    llm = llm_init(
        model_name=model_name,
        tensor_parallel_size=1,
        max_seq_len_to_capture=8192,
        max_tokens=1536,
        seed=0,
        temperature=0.0,
        frequency_penalty=1.0
    )

    fout = open(output_path, "w", encoding="utf-8")

    # 保证和原推理顺序一致
    keys = list(gt_triples.keys())

    for i in tqdm(range(0, len(keys), batch_size)):
        batch = keys[i:i+batch_size]
        for sid in batch:
            triples = gt_triples[sid]
            idx = int(sid.split("_")[1])
            question = q_map.get(idx, "")
            prompt = build_prompt(question, triples)

            try:
                with torch.cuda.amp.autocast():
                    messages = [
                        {"role": "system", "content": "你是一名中医知识图谱推理助手。"},
                        {"role": "user", "content": prompt}
                    ]
                    outputs = llm(messages)
                    pred = outputs[0].outputs[0].text.strip()
            except RuntimeError as e:
                print(f"[ERROR] sample {sid}: {e}")
                pred = "推理失败：显存不足或其他错误"

            fout.write(json.dumps({
                "id": sid,
                "question": question,
                "triples": triples,
                "prediction": pred
            }, ensure_ascii=False) + "\n")

    fout.close()
    print("Done. Saved to", output_path)

if __name__ == "__main__":
    main()
