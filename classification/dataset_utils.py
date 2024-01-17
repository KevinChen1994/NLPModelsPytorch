from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import jieba
from collections import defaultdict


class ToutiaoClassificationDataset(Dataset):
    def __init__(self, data_file, label_dict) -> None:
        self.label_dict = label_dict
        self.raw_dataset = load_dataset("text", streaming=False, data_files=data_file, split='train')

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        item_split = item['text'].replace('\n', '').split('\t')
        query = item_split[1]
        label = self.label_dict[item_split[0]]
        return query, label
    
@dataclass
class DataCollator(DataCollatorWithPadding):
    # 通过结巴分词，然后转成id，需要进行padding
    query_max_len: int = 50
    word_dict: defaultdict = None
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        query = [f[0] for f in features]
        labels = torch.tensor([f[1] for f in features], dtype=torch.int32)

        q_collated = self.my_tokenizer(
            query,
            max_length=self.query_max_len,
        )
        return {"query": q_collated, "labels":labels}
    
    # 不使用huggingface的tokenizer，使用自己定义的tokenizer
    def my_tokenizer(self, query, max_length=50):
        # query是一个List
        query_tensors = []
        for q in query:
            q_tokens = jieba.lcut(q)
            q_token = [self.word_dict[w] if w in self.word_dict else self.word_dict["[UNK]"] for w in q_tokens]
            q_token = q_token[:max_length]
            q_token = q_token + [0] * (max_length - len(q_token))
            query_tensors.append(torch.tensor(q_token, dtype=torch.int32))
        return torch.stack(query_tensors)