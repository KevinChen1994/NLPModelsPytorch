# -*- coding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2023/09/07 15:09:51
@Author  :   chenmeng
@Desc    :   None
'''

from dataclasses import dataclass
from typing import Any, Dict, List
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import defaultdict
import datasets
from transformers import DataCollatorWithPadding
import torch

label_dict = {"positive": 0, "negative": 1, 0: "positive", 1: "negative"}

    
class SIMCSEDataset(Dataset):
    def __init__(self, data_file) -> None:
        self.raw_dataset = load_dataset("text", streaming=False, data_files=data_file, split='train')

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        item_split = item['text'].replace('\n', '').split('\t')
        query = item_split[1]
        passage = item_split[2]
        label = label_dict[item_split[0]]
        return query, passage, label
    
@dataclass
class DataCollator(DataCollatorWithPadding):
    query_max_len: int = 12
    passage_max_len: int = 400
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        labels = torch.tensor([f[2] for f in features], dtype=torch.float32)

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated, "labels":labels}
    

