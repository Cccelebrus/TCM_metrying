# 2025-11-25
# file: tcm_retriever2.py  （替换你的旧文件）
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ============================
#   Position Encoding Convolution
# ============================
class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')
    # PyG 的惯例是先传 node features x，再传 edge_index。
    def forward(self, x, edge_index):
        # x: [num_nodes, feat_dim]
        # edge_index: [2, num_edges]
        return self.propagate(edge_index, x=x)

    # message 接收的是 source 节点特征 x_j（已经对齐为边的 source）
    def message(self, x_j):
        return x_j


# ============================
#  Dual Directional Expansion (DDE)
# ============================
class DDE(nn.Module):
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()

        self.layers = nn.ModuleList([PEConv() for _ in range(num_rounds)])
        self.reverse_layers = nn.ModuleList([PEConv() for _ in range(num_reverse_rounds)])

    def forward(self, topic_one_hot, edge_index, reverse_edge_index):
        """
        topic_one_hot: 2D [num_nodes, 2]
        edge_index: [2, num_edges] (heads, tails) for forward direction
        reverse_edge_index: [2, num_edges] (tails, heads) for reverse direction
        返回：list of tensors，每个 tensor 形状 [num_nodes, 2]（或与 topic_one_hot 的 feat_dim 一致）
        """

        h_pe = topic_one_hot  # 已经是 [N_triples, 2]

        result_list = []

        # ---- 正向扩散 ----
        cur = h_pe
        # print(f"[Forward] Layer 0: cur shape={cur.shape}")
        # num_print_rows = min(100, cur.size(0))
        # num_print_cols = min(2, cur.size(1))
        # print(cur[:num_print_rows, :num_print_cols])
        for layer_idx, layer in enumerate(self.layers):
            # debug 打印：非零数量与前几个位置（便于观察传播效果）
            # # 注意参数顺序：x 在前，edge_index 在后（符合 PyG 习惯）
            cur = layer(cur, edge_index)
            # ---- 打印原始 cur 前几行几列 ----
            # print(f"[Forward] Layer {layer_idx+1}: cur shape={cur.shape}")
            # print(cur[:num_print_rows, :num_print_cols])
            result_list.append(cur)

        # ---- 反向扩散 ----
        h_pe_rev = topic_one_hot.clone()
        cur_rev = h_pe_rev
        # print(f"[Backward] Layer 0: cur_rev shape={cur_rev.shape}")
        # print(cur_rev[:num_print_rows, :num_print_cols])
        for layer_idx, layer in enumerate(self.reverse_layers):
            cur_rev = layer(cur_rev, reverse_edge_index)
            # print(f"[Backward] Layer {layer_idx+1}: cur_rev shape={cur_rev.shape}")
            # print(cur_rev[:num_print_rows, :num_print_cols])
            result_list.append(cur_rev)

        return result_list


# ============================
#      Retriever Module
# ============================
class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe=True,
        DDE_kwargs=None
    ):
        super().__init__()

        if DDE_kwargs is None:
            DDE_kwargs = dict(num_rounds=3, num_reverse_rounds=3)

        self.topic_pe = topic_pe
        self.non_text_entity_emb = nn.Embedding(1, emb_size)

        # DDE 位置编码扩散
        self.dde = DDE(**DDE_kwargs)

        # ===== 输入维度计算 =====
        pred_in_size = 4 * emb_size   
        if topic_pe:
            pred_in_size += 2 * 2  

        # DDE outputs
        num_pe = DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds']
        pred_in_size += 2 * 2 * num_pe

        # ===== 得分网络 =====
        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )


    def forward(
        self,
        h_id_tensor,               # [num_triple]
        r_id_tensor,               # [num_triple]
        t_id_tensor,               # [num_triple]
        q_emb,                     # [emb]
        entity_embs,               # [num_entity, emb]
        num_non_text_entities,
        relation_embs,             # [num_rel, emb]
        topic_one_hot       # [num_entity_total, 2]    — 保持和预处理一致
    ):

        device = entity_embs.device
        # # ========= 0.Debug: 查看 BFS 子图结构 =========
        # print("relation_embs.shape",relation_embs.shape)
        # print(f"num_triple = {len(h_id_tensor)}")
        # q_nodes = torch.where(topic_one_hot[:, 1] > 0)[0]  # 第2列=1的就是 query
        # print(f"⚠ 当前样本 query 节点数量: {len(q_nodes)}")
        # print("query 节点ID列表：", q_nodes.tolist())
        # # -------- 统计第一跳数量 --------
        # if len(q_nodes) > 0:
        #     mask1 = (h_id_tensor.unsqueeze(1) == q_nodes.unsqueeze(0)).any(dim=1)
        #     first_hop_t = t_id_tensor[mask1]

        #     print(f"第一跳数量: {first_hop_t.size(0)}")
        # ============================
        # 1. 构造最终实体 embedding 表
        # ============================
        non_text_e = self.non_text_entity_emb(
            torch.LongTensor([0]).to(device)
        ).expand(num_non_text_entities, -1)
        if entity_embs.dim() == 1:
            entity_embs = entity_embs.unsqueeze(0)  # [1, emb_size]

        h_e = torch.cat([entity_embs, non_text_e], dim=0)   # [num_text_entities + non_text_entities, emb]
        h_e_list = [h_e]
        # print("len(h_e_list) =", len(h_e_list))
        # for i, t in enumerate(h_e_list):
            # print(f"h_e_list[{i}] shape:", t.shape)
        if self.topic_pe:
            # 注意：这里期望 topic_one_hot 是 [num_entities, 2]
            # 直接加入到列表以供 later concat
            # topic_one_hot shape: [num_entities, 2]
            h_e_list.append(topic_one_hot)
            # print("len(h_e_list) =", len(h_e_list))
            # for i, t in enumerate(h_e_list):
            #     print(f"h_e_list[{i}] shape:", t.shape)
            # print("topic_one_hot.shape",topic_one_hot.shape)
        # ============================
        # 2.  构造边
        # ============================
        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)

        # 打印正反两个方向的edge_index
        # print("edge_index shape:", edge_index.shape)
        # print("reverse_edge_index shape:", reverse_edge_index.shape)

        # ============================
        # 3.  计算 DDE（邻域位置编码）
        # ============================
        # DDE 接受 (topic_one_hot, edge_index, reverse_edge_index)
        dde_list = self.dde(topic_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)
        # print("len(h_e_list) =", len(h_e_list))
        # for i, t in enumerate(h_e_list):
        #     print(f"h_e_list[{i}] shape:", t.shape)
        # =========== 拼接最终实体 embedding ===========
        # 这里 concat 维度是 dim=1（feature 维度），要求 h_e_list 中每个元素在
        # 第0维长度都一致（即节点数一致）
        h_e = torch.cat(h_e_list, dim=1)
        # print("h_e shape:", h_e.shape)
        
        # ============================
        # 4.  构造 triple 特征
        # ============================
        h_q = q_emb                          # [emb]
        h_r = relation_embs[r_id_tensor]     # [num_triple, emb]
        # print(f"h_q shape: {h_q.shape}")                    
        # print(f"h_r shape: {h_r.shape}") 
        # # ===== 拼接前检查 =====
        # print("Expect concat dim =", h_q.expand(len(h_r), -1).shape,
        #     h_e[h_id_tensor].shape,
        #     h_r.shape,
        #     h_e[t_id_tensor].shape)

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),        # q → [num_triple, emb]
            h_e[h_id_tensor],                # head emb
            h_r,                             # relation emb
            h_e[t_id_tensor]                 # tail emb
        ], dim=1)

        # ============================
        # 5.  得分
        # ============================
        return self.pred(h_triple)           # [num_triple, 1]
