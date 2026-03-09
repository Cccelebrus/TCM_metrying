import os
import json
import torch
from tqdm import tqdm
from llm_utils_tcm_qwen import llm_init, llm_inf_all   # 注意：这里用的是 TCM 下的 llm_utils

# ------------------ 读取 retriever 输出并清洗 ------------------
def load_retriever_output(path):
    data = torch.load(path)
    samples = []

    for sample_id, v in data.items():

        # 清洗 scored_triples：保留 score，并保留三位小数
        clean_triples = []
        for triple in v.get("scored_triples", []):
            if len(triple) >= 4:
                h, r, t, score = triple[:4]
                score = round(float(score), 3)  # → 保留三位小数
                clean_triples.append((h, r, t, score))
                
        samples.append({
            "id": sample_id,
            "triples": clean_triples,              # 用新的字段，不再叫 scored_triples
            "q_entity": v.get("q_entity", []),
            "a_entity": v.get("a_entity", []),
            "question": v.get("question", ""),
        })
    return samples


# ------------------ 格式化三元组为文本 ------------------
def triples_to_text(triples):
    lines = []
    for h, r, t, s in triples:   # 直接按四元组解包
        lines.append(f"{h}-{r}->{t} (score={s})")
    return "\n".join(lines)



# ------------------ 构造 LLM prompt ------------------
def build_prompt(question, triples):
    triples_text = triples_to_text(triples)
    prompt = f"""
你是一名中医知识图谱推理助手。
下面是从知识图谱中检索到的相关信息（结构化三元组）：

{triples_text}

请基于以上知识推理回答下面的问题：

问题：{question}

请给出你的中医推理解释，并最终给出明确答案，最终推荐的中药统一格式为"ans:[中药1,中药2,……]"。注意：每个问题推荐的中药不超过 20 种。
"""
#     prompt = f"""
#     你是一名中医知识图谱推理助手。

# 下面是从知识图谱中检索到的相关三元组（格式：症状-关系->中药 (score=数值)）：
# {triples_text}

# 请执行逐步推理（慢思考），严谨分析但不要省略推理步骤：

# 【推理步骤】
# 1. 根据三元组推断当前症状可能涉及的证候（如火热、湿毒、气滞、瘀血等）。
# 2. 对每个证候：
#    - 说明该证候为何出现（结合症状逻辑）。
#    - 从提供的三元组中挑选相关中药，并按照 score 从高到低排序。
#    - 给出“核心药”（强相关，高 score）与“辅助药”（中等 score）。
# 3. 汇总所有证候的候选中药，进行再次筛选：
#    - 保留与最多证候重复出现的药；
#    - 优先保留 score ≥ 0.1 的药；
#    - 去除重复药材；
#    - 限制最终药材数量 ≤ 20。

# 【最终输出格式要求（非常重要）】
# - 推理过程请正常输出。
# - 最终答案必须单独一行输出，格式为：

# ans:[药材1, 药材2, 药材3, ...]

# 严格保持此格式，不得添加多余说明、符号或换行。

# 现在开始推理并作答。
#     """
    return prompt


# ------------------ 主函数 ------------------
def main():
    # ----------- 用户可修改 -----------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # retriever_output_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Nov29-07:57:06/retrieval_result_tcm.pth"
    retriever_output_path = "/home/gyj/local/SubgraphRAG-main/retrieve/tcm_retriever_Jan05-06:42:07/retrieval_result_tcm_50trples_withcpm.pth"
    output_path = "/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_20260105_tcm50triples.jsonl"
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

    samples = load_retriever_output(retriever_output_path)

    fout = open(output_path, "w", encoding="utf-8")

    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i:i+batch_size]

        for sample in batch:
            q = sample["question"]
            triples = sample["triples"]
            prompt = build_prompt(q, triples)

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
                "triples": triples,  # 此时 triples 已包含 score
                "prediction": pred
            }, ensure_ascii=False) + "\n")

    fout.close()
    print("Done. Saved to", output_path)


if __name__ == "__main__":
    main()
