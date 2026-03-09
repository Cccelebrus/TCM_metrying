import os
import json
import torch
from tqdm import tqdm
from llm_utils_tcm_qwen import llm_init, llm_inf_all   # 注意：这里用的是 TCM 下的 llm_utils

# ------------------ 读取 retriever 输出并清洗 ------------------
def load_retriever_output_no_score(path):
    """
    对比实验：只保留 h,r,t，不保留 score
    """
    data = torch.load(path)
    samples = []

    for sample_id, v in data.items():
        clean_triples = []
        for triple in v.get("scored_triples", []):
            if len(triple) >= 3:
                h, r, t = triple[:3]
                clean_triples.append((h, r, t))  # 不再包含 score

        samples.append({
            "id": sample_id,
            "triples": clean_triples,              # 四元组 -> 三元组
            "q_entity": v.get("q_entity", []),
            "a_entity": v.get("a_entity", []),
            "question": v.get("question", ""),
        })
    return samples


# ------------------ 格式化三元组为文本 ------------------
def triples_to_text_no_score(triples):
    """
    只输出 h-r-t，不显示 score
    """
    lines = []
    for h, r, t in triples:
        lines.append(f"{h}-{r}->{t}")
    return "\n".join(lines)


# ------------------ 构造 LLM prompt ------------------
def build_prompt_no_score(question, triples):
    triples_text = triples_to_text_no_score(triples)
    prompt = f"""
你是一名中医知识图谱推理助手。
下面是从知识图谱中检索到的相关信息（结构化三元组）：  

{triples_text}

请基于以上知识推理回答下面的问题：

问题：{question}

请给出你的中医推理解释，并最终给出明确答案，最终推荐的中药统一格式为"ans:[中药1,中药2,……]"。注意：每个问题推荐的中药不超过 20 种。
"""
    return prompt


# ------------------ 主函数 ------------------
def main():
    # ----------- 用户可修改 -----------
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    retriever_output_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Nov29-07:57:06/retrieval_result_tcm.pth"
    output_path = "/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_tcm30triples_no_score.jsonl"
    batch_size = 1
    # ----------------------------------

    model_name = "/mnt/local/ccl/qwen2.5-7b-instruct/"

    llm = llm_init(
        model_name=model_name,
        tensor_parallel_size=1,
        max_seq_len_to_capture=8192,
        max_tokens=1536,
        seed=0,
        temperature=0.0,
        frequency_penalty=1.0
    )

    samples = load_retriever_output_no_score(retriever_output_path)

    fout = open(output_path, "w", encoding="utf-8")

    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i:i+batch_size]

        for sample in batch:
            q = sample["question"]
            triples = sample["triples"]
            prompt = build_prompt_no_score(q, triples)

            try:
                with torch.cuda.amp.autocast():
                    messages = [
                        {"role": "system", "content": "你是一名中医知识图谱推理助手。"},
                        {"role": "user", "content": prompt}
                    ]

                    outputs = llm(messages)
                    pred = outputs[0].outputs[0].text.strip()
            except RuntimeError as e:
                print(f"[ERROR] sample {sample['id']}: {e}")
                pred = "推理失败：显存不足或其他错误"

            fout.write(json.dumps({
                "id": sample["id"],
                "question": q,
                "triples": triples,  # 现在 triples 不包含 score
                "prediction": pred
            }, ensure_ascii=False) + "\n")

    fout.close()
    print("Done. Saved to", output_path)


if __name__ == "__main__":
    main()
