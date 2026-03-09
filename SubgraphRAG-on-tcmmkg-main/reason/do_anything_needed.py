# import json  ####################打印/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_20251129_1752.jsonl，预测文件中ans列表的数量统计情况
# import re

# input_file = "/home/gyj/local/SubgraphRAG-main/reason/results/llm_tcm_reason_predictions_20251129_1752.jsonl"

# # 更强的 ans 提取正则：
# # - 支持英文冒号 ":" 和中文冒号 "："
# # - 支持有方括号 ans:[...] / ans: [...] / ans：[...] 
# # - 支持没有方括号直接 ans: a, b, c
# # - 只取到行尾（避免抓到之后的解释段）
# ans_pattern = re.compile(r"ans[:：]\s*(?:\[(.*?)\]|([^\n]+))", re.IGNORECASE)

# def parse_ans_list(prediction: str):
#     """从 prediction 中尽量提取 ans 列表，返回一个字符串列表（去空项）"""
#     if not prediction:
#         return []
#     m = ans_pattern.search(prediction)
#     if not m:
#         return []

#     # group1: content inside brackets; group2: no-bracket content up to newline
#     content = m.group(1) if m.group(1) is not None else m.group(2)
#     if not content:
#         return []

#     # 有时结尾有句号或解释，去掉末尾多余符号
#     content = content.strip().rstrip("。.;，,。")

#     # 把常见分隔符统一处理：英文逗号、中文逗号、顿号、分号、斜杠
#     parts = re.split(r"[,，、；;/\n]", content)

#     # 清洗：去掉空白和左右可能残留的中英括号/引号
#     ans_list = []
#     for p in parts:
#         p = p.strip()
#         # 去掉可能的中英文引号或括号
#         p = p.strip(" \"'“”‘’()（）[]【】")
#         if p:
#             ans_list.append(p)
#     return ans_list

# # 主循环：打印长度为0/1/2的样本（完整prediction字段）
# len_count = {}
# samples_len_0 = []
# samples_len_1 = []
# samples_len_2 = []

# with open(input_file, "r", encoding="utf8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             obj = json.loads(line)
#         except Exception as e:
#             # 如果整行不是 JSON，跳过（或按需打印）
#             # print("无法解析 JSON 行：", line)
#             continue

#         pred = obj.get("prediction", "")
#         ans_list = parse_ans_list(pred)
#         L = len(ans_list)
#         len_count[L] = len_count.get(L, 0) + 1

#         if L == 0:
#             samples_len_0.append({"id": obj.get("id"), "question": obj.get("question"), "prediction": pred})
#         elif L == 1:
#             samples_len_1.append({"id": obj.get("id"), "question": obj.get("question"), "prediction": pred, "ans_list": ans_list})
#         elif L == 2:
#             samples_len_2.append({"id": obj.get("id"), "question": obj.get("question"), "prediction": pred, "ans_list": ans_list})

# # 打印统计
# print("====== 统计结果 ======")
# total = sum(len_count.values())
# print("总样本数:", total)
# for k in sorted(len_count.keys()):
#     print(f"长度 {k}: {len_count[k]} 条样本")

# # 打印长度为0/1/2的样本（简洁版，免去大体输出）
# print("\n====== 长度为 0 的样本（示例，最多打印 50 条） ======")
# for s in samples_len_0[:2]:
#     print(f"id={s['id']}\nquestion={s['question']}\nprediction={s['prediction']}\n---")

# print("\n====== 长度为 1 的样本（示例） ======")
# for s in samples_len_1[:2]:
#     print(json.dumps(s, ensure_ascii=False, indent=2))

# print("\n====== 长度为 2 的样本（示例） ======")
# for s in samples_len_2[:]:
#     print(json.dumps(s, ensure_ascii=False, indent=2))


import json  ############在打印真实test文件的答案列表数量统计情况！

input_file = "/home/gyj/local/SubgraphRAG-main/retrieve/data_files/tcm/test.jsonl"

len_count = {}
samples_len_0 = []
samples_len_1 = []
samples_len_2 = []

with open(input_file, "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            print("无法解析 JSON 行：", line)
            continue

        answers = obj.get("answers", [])
        L = len(answers)

        len_count[L] = len_count.get(L, 0) + 1

        if L == 0:
            samples_len_0.append(obj)
        elif L == 1:
            samples_len_1.append(obj)
        elif L == 2:
            samples_len_2.append(obj)

# ====== 打印统计 ======
print("====== 统计结果 ======")
total = sum(len_count.values())
print("总样本数:", total)
for k in sorted(len_count.keys()):
    print(f"长度 {k}: {len_count[k]} 条")

# # ====== 打印长度为 0/1/2 的样本 ======
# print("\n====== 长度为 0 的样本（最多打印 50 条）======")
# for s in samples_len_0[:50]:
#     print(json.dumps(s, ensure_ascii=False, indent=2))

# print("\n====== 长度为 1 的样本（最多打印 50 条）======")
# for s in samples_len_1[:50]:
#     print(json.dumps(s, ensure_ascii=False, indent=2))

# print("\n====== 长度为 2 的样本（最多打印 50 条）======")
# for s in samples_len_2[:50]:
#     print(json.dumps(s, ensure_ascii=False, indent=2))