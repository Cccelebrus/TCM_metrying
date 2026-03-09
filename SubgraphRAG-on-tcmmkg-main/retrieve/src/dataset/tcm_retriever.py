import torch
from torch.utils.data import Dataset

"""
2025.11.24
基于 _with_ids.pth 的 Dataset，直接使用预先计算好的
- q_entity_id_list
- a_entity_id_list
- h_id_list, r_id_list, t_id_list
- non_text_entity_list
- topic_entity_one_hot
- target_triple_probs（来自 build_triple_ids.py）

2025.11.27 tcm_inference要求，加入q_entity_id_list
"""

class TCMRetrieverDataset(Dataset):
    def __init__(self, emb_file: str):
        data = torch.load(emb_file)
        self.data_list = list(data.values())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        h_id_tensor = torch.tensor(sample['h_id_list'], dtype=torch.long)
        r_id_tensor = torch.tensor(sample['r_id_list'], dtype=torch.long)
        t_id_tensor = torch.tensor(sample['t_id_list'], dtype=torch.long)

        q_emb = sample['q_emb']
        question = sample['question']
        entity_embs = sample['entity_embs']
        relation_embs = sample['relation_embs']

        num_non_text_entities = len(sample['non_text_entity_list'])

        # topic entity one-hot (直接从预处理文件读取)
        topic_entity_one_hot = sample['topic_entity_one_hot']

        # 直接使用 build_triple_ids.py 生成的 target_triple_probs
        target_triple_probs = sample['target_triple_probs']

        return {
            'h_id_tensor': h_id_tensor,
            'r_id_tensor': r_id_tensor,
            't_id_tensor': t_id_tensor,
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs,
            'num_non_text_entities': num_non_text_entities,
            'topic_entity_one_hot': topic_entity_one_hot,
            'target_triple_probs': target_triple_probs,
            'a_entity_id_list': sample['a_entity_id_list'],
            'q_entity_id_list':sample['q_entity_id_list'],
            'question': question
        }


def collate_fn_tcm(data):
    sample = data[0]
    return (
        sample['h_id_tensor'],
        sample['r_id_tensor'],
        sample['t_id_tensor'],
        sample['q_emb'],
        sample['entity_embs'],
        sample['num_non_text_entities'],
        sample['relation_embs'],
        sample['topic_entity_one_hot'],
        sample['target_triple_probs'],
        sample['a_entity_id_list'],
        sample['q_entity_id_list'],
        sample['question']
    )
