import torch
from pprint import pprint

def inspect_pth(path, max_items=3):
    print(f"\n=== Loading: {path} ===")
    data = torch.load(path, map_location="cpu")

    print(f"Total items: {len(data)}")
    print("Type of data:", type(data))

    # 列出所有 key（一般是 sample 的 id）
    keys = list(data.keys())
    print("\nFirst few keys:", keys[:max_items])

    # 检查几个样本的内容
    for k in keys[:max_items]:
        print(f"\n--- Sample ID: {k} ---")
        item = data[k]

        if not isinstance(item, dict):
            print("Unexpected entry type:", type(item))
            pprint(item)
            continue

        # 打印每个字段
        for field in ["q_emb", "entity_embs", "relation_embs"]:
            if field not in item:
                print(f"{field} missing!")
                continue

            value = item[field]

            if torch.is_tensor(value):
                print(f"{field}: Tensor shape = {tuple(value.shape)}")
            elif isinstance(value, list):
                print(f"{field}: List of {len(value)} tensors")
                if len(value) > 0 and torch.is_tensor(value[0]):
                    print(f" - First tensor shape = {tuple(value[0].shape)}")
            else:
                print(f"{field}: {type(value)}")
                pprint(value)

if __name__ == "__main__":
    base = "/home/gyj/local/SubgraphRAG-main/retrieve/data_files/tcm_embedding"

    inspect_pth(f"{base}/test.pth")
    # 你也可以检查 val/train：
    # inspect_pth(f"{base}/train.pth")
    # inspect_pth(f"{base}/val.pth")
