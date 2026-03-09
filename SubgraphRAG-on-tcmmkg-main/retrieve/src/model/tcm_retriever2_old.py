import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ============================
#   Position Encoding Convolution
# ============================
class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

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
        print("topic_one_hot.shape",topic_one_hot.shape)
        print("topic_one_hot",topic_one_hot)
        result_list = []

        # 正向扩散
        h_pe = topic_one_hot.unsqueeze(-1) ##########加上.unsqueeze(-1)变为二维[5782, 1]
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            # 打印非零数量
            nz = h_pe.nonzero(as_tuple=False)
            print(f"h_pe non-zero count: {nz.size(0)}")
            # 打印前几个非零位置
            if nz.size(0) > 0:
                print("first few non-zero positions:", nz[:50].tolist())
            result_list.append(h_pe)

        # 反向扩散
        h_pe_rev = topic_one_hot.unsqueeze(-1)  ##########加上.unsqueeze(-1)变为二维[5782, 1]
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            # 打印非零数量
            nz = h_pe_rev.nonzero(as_tuple=False)
            print(f"h_pe_rev non-zero count: {nz.size(0)}")

            # 打印前几个非零位置
            if nz.size(0) > 0:
                print("first few non-zero positions:", nz[:50].tolist())
            result_list.append(h_pe_rev)

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
            DDE_kwargs = dict(num_rounds=2, num_reverse_rounds=2)

        self.topic_pe = topic_pe
        self.non_text_entity_emb = nn.Embedding(1, emb_size)

        # DDE 位置编码扩散
        self.dde = DDE(**DDE_kwargs)

        # ===== 输入维度计算 =====
        pred_in_size = 4 * emb_size   # q_emb, h_emb, r_emb, t_emb

        if topic_pe:
            pred_in_size += 2 * 2  # topic_one_hot for h and t

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
        topic_entity_one_hot       # [num_entity_total, 2]
    ):

        device = entity_embs.device

        # ============================
        # 1. 构造最终实体 embedding 表
        # ============================
        non_text_e = self.non_text_entity_emb(
            torch.LongTensor([0]).to(device)
        ).expand(num_non_text_entities, -1)
        if entity_embs.dim() == 1:
            entity_embs = entity_embs.unsqueeze(0)  # [1, emb_size] 
        # print("num_non_text_entities",num_non_text_entities)
        # print("entity_embs.shape:", entity_embs.shape)
        # print("non_text_e.shape:", non_text_e.shape)
        h_e = torch.cat([entity_embs, non_text_e], dim=0)
        h_e_list = [h_e]

        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)


        # ============================
        # 2.  构造边
        # ============================
        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)

        # 打印正反两个方向的edge_index
        print("edge_index shape:", edge_index.shape)
        print("reverse_edge_index shape:", reverse_edge_index.shape)
        # print("edge_index:", edge_index)
        # print("reverse_edge_index:", reverse_edge_index)
        # ============================
        # 3.  计算 DDE（邻域位置编码）
        # ============================
        # print("topic_entity_one_hot.shape",topic_entity_one_hot.shape)
        # print("topic_entity_one_hot",topic_entity_one_hot)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        # ↓打印 DDE 输出
        print(f"len(dde_list) = {len(dde_list)}")
        for i, x in enumerate(dde_list):
            print(f"dde_list[{i}] shape: {x.shape}")
        print("====== DDE列表非零元素 ======")
        for i, x in enumerate(dde_list):
            nz = x.nonzero(as_tuple=False)   # [k, 2]
            v = x[nz[:, 0], nz[:, 1]]

            print(f"\n-- dde_list[{i}] non-zero count: {nz.size(0)}")

            # if nz.size(0) > 0:
            #     limit = min(3, nz.size(0))
            #     for j in range(limit):
            #         print(f"  index={tuple(nz[j].tolist())}, value={v[j].item()}")
            # else:
            #     print("  *** all zeros ***")
        # ↑打印 DDE 输出
        h_e_list.extend(dde_list)

        # =========== 拼接最终实体 embedding ===========
        h_e = torch.cat(h_e_list, dim=1)


        # ============================
        # 4.  构造 triple 特征
        # ============================
        h_q = q_emb                          # [emb]
        h_r = relation_embs[r_id_tensor]     # [num_triple, emb]

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
