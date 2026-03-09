import os
import json
import torch
from tqdm import tqdm
from llm_utils_tcm_qwen import llm_init, llm_inf_all


# ------------------ 读取 retriever 输出并清洗 ------------------
def load_retriever_output_no_score(path):
    """
    只读出 (h, r, t)，不包含 score
    """
    data = torch.load(path)
    samples = []

    for sample_id, v in data.items():
        clean_triples = []
        for triple in v.get("scored_triples", []):
            if len(triple) >= 3:
                h, r, t = triple[:3]
                clean_triples.append((h, r, t))  # 三元组格式

        samples.append({
            "id": sample_id,
            "triples": clean_triples,
            "q_entity": v.get("q_entity", []),
            "a_entity": v.get("a_entity", []),
            "question": v.get("question", ""),
        })
    return samples


# ------------------ 仅保留三元组的尾实体 t ------------------
def triples_to_text_only_t(triples):
    """
    将三元组列表转换为仅由 t 构成的文本，每行一个 t，并自动去重
    """
    uniq_t = set()
    for h, r, t in triples:
        uniq_t.add(t)

    # 排序让 prompt 输出更稳定
    return "\n".join(sorted(uniq_t))


# ------------------ 构造 LLM prompt：仅使用 t 实体 ------------------
def build_prompt_only_t(question, triples):
    triples_text = triples_to_text_only_t(triples)
    prompt = f"""
你是一名中医知识图谱推理助手。
下面是从知识图谱中检索到的相关实体（仅显示知识三元组中的尾实体 t）：  

{triples_text}

请基于以上信息推理回答下面的问题：

问题：{question}

请给出中医推理解释，并最终给出明确答案。
最终推荐的中药统一格式为 "ans:[中药1,中药2,……]"。
注意：每个问题推荐的中药不超过 20 种。
"""
    return prompt


# ------------------ 主运行逻辑 ------------------
def main():
    # ----------- 用户可修改 ----------- 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    retriever_output_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Nov29-07:57:06/retrieval_result_tcm.pth"
    output_path = "/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_tcm30triples_only_t.jsonl"
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
            prompt = build_prompt_only_t(q, triples)

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
                "triples_t_only": triples_to_text_only_t(triples),  # 输出给你看
                "prediction": pred
            }, ensure_ascii=False) + "\n")

    fout.close()
    print("Done. Saved to", output_path)


if __name__ == "__main__":
    main()
