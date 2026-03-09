import os
import time
import openai
from vllm import LLM, SamplingParams
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192,
             max_tokens=4000, seed=0, temperature=0, frequency_penalty=0):

    # 本地模型全部走 vLLM（包含 Qwen）
    if "gpt" not in model_name:
        print(f"[llm_init] Using local model via vLLM: {model_name}")

        client = LLM(
            model=model_name,
            dtype="float16",
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len_to_capture=max_seq_len_to_capture
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty
        )

        # Qwen 需要用 role-based chat 格式
        def qwen_chat(messages):
            # vLLM 的 chat API 是标准 openai 格式
            return client.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)

        llm = qwen_chat

    else:
        # 远程 GPT（你现在不用）
        client = openai.OpenAI()
        llm = partial(client.chat.completions.create,
                      model=model_name, seed=seed,
                      temperature=temperature, max_tokens=max_tokens)

    return llm


def get_outputs(outputs, model_name):
    """统一抽取输出字符串"""

    # vLLM / 本地模型
    if "gpt" not in model_name:
        # Qwen 的 vLLM 输出格式：
        # outputs[0].outputs[0].text
        return outputs[0].outputs[0].text.strip()

    # OpenAI GPT
    return outputs.choices[0].message.content


def llm_inf(llm, prompts, mode, model_name):
    res = []

    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    # ---------- First turn ----------
    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    # ---------- CoT ----------
    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "not available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if 'gpt' in model_name:
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)
