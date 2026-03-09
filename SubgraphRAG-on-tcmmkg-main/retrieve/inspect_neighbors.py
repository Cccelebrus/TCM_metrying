# check_neighbors.py
# 用 dedup 后的 ID 版 triples_dedup_ids.tsv 构建图，并支持打印任意节点的一跳出边

import sys
from collections import defaultdict

TRIPLE_FILE = "./triples_dedup_ids.tsv"

def load_graph(triple_file):
    print(f"Loading triples from {triple_file} ...")

    triples_by_head = defaultdict(list)

    with open(triple_file, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            h, r, t = int(h), int(r), int(t)
            triples_by_head[h].append((r, t))

    print(f"Loaded {sum(len(v) for v in triples_by_head.values())} triples.")
    print(f"Distinct heads: {len(triples_by_head)}")
    print("=" * 60)
    return triples_by_head


def print_neighbors(triples_by_head, head_id, max_show=50):
    if head_id not in triples_by_head:
        print(f"节点 {head_id} 没有任何出边。")
        return

    edges = triples_by_head[head_id]
    print(f"节点 {head_id} 的一跳出边数量: {len(edges)}")
    print(f"前 {min(max_show, len(edges))} 条：")

    for i, (r, t) in enumerate(edges[:max_show]):
        print(f"  [{i}]  (h={head_id}) --r={r}--> (t={t})")

    if len(edges) > max_show:
        print(f"... 共 {len(edges)} 条，仅显示前 {max_show} 条")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("    python check_neighbors.py <head_id>")
        sys.exit(1)

    head_id = int(sys.argv[1])
    triples_by_head = load_graph(TRIPLE_FILE)
    print_neighbors(triples_by_head, head_id)
