# file: retrieve/src/model/text_encoders/gte_large_zh.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
class GTELargeZH:
    """
    中文 text encoder wrapper.
    支持任意 HuggingFace/transformers 模型（取 last_hidden_state[:,0] 作为句向量）。
    自动读取 model.config.hidden_size 作为输出维度。
    """
    def __init__(self, device, model_path="/mnt/local2/gyj/text2vec-large-chinese", normalize=True):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 使用 trust_remote_code=True 仅在必要时；大部分中文向量模型不需要它
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.normalize = normalize
        # infer output dim
        try:
            self.dim = self.model.config.hidden_size
        except:
            # fallback
            self.dim = 1024

    @torch.no_grad()
    def embed(self, text_list, max_length=1024):
        """
        text_list: list[str]
        返回 cpu tensor shape (len(text_list), dim)
        """
        if len(text_list) == 0:
            return torch.zeros(0, self.dim)

        batch = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**batch)
        # last_hidden_state may be a tensor or a BaseModelOutput - we already have it
        last = outputs.last_hidden_state  # (B, T, D)
        emb = last[:, 0, :]  # [CLS] token or first token pooling

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu()

    def __call__(self, q_text, text_entity_list, relation_list):
        """
        保持和原 GTELargeEN 的接口一致：返回 q_emb, entity_embs, relation_embs
        """
        # q_text 可能是 str 或 list[str]
        q_emb = self.embed([q_text]) if isinstance(q_text, str) else self.embed(q_text)

        # text_entity_list 可能是 list[list[str]]，flatten
        flat_entities = []
        for x in text_entity_list:
            if isinstance(x, list):
                flat_entities.extend(x)
            else:
                flat_entities.append(x)
        entity_embs = self.embed(flat_entities)

        # relation_list 同理
        flat_relations = []
        for x in relation_list:
            if isinstance(x, list):
                flat_relations.extend(x)
            else:
                flat_relations.append(x)
        relation_embs = self.embed(flat_relations)

        return q_emb, entity_embs, relation_embs