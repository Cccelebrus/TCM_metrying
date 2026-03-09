import torch

pth_path = "/home/gyj/local/SubgraphRAG-on-tcmmkg-main/retrieve/data_files/tcm_embedding/val_with_ids.pth"   # 你的路径

data = torch.load(pth_path, map_location="cpu")

print("=== 文件加载成功 ===")
print("数据类型:", type(data))

# 如果是 dict
if isinstance(data, dict):
    print("\n字典的键:")
    print(list(data.keys()))
    print("\n每个 key 的前几个内容：")
    for k, v in data.items():
        print(f"\n--- key: {k} ---")
        if isinstance(v, torch.Tensor):
            print("Tensor shape:", tuple(v.shape))
            print("前几个值：", v.flatten()[:10])
        elif isinstance(v, list):
            print("List length:", len(v))
            print("前几个：", v[:5])
        else:
            print("类型:", type(v))
            print("内容:", v)

# 如果是 tensor
elif isinstance(data, torch.Tensor):
    print("Tensor shape:", tuple(data.shape))
    print("前几个值:", data.flatten()[:10])

# 如果是 list
elif isinstance(data, list):
    print("List length:", len(data))
    print("前几个：", data[:5])

else:
    print("未知数据类型，内容：", data)
