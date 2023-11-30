# -*- coding: utf-8 -*-
'''
@File    :   mismatch_classification.py
@Time    :   2023/09/28 15:24:28
@Author  :   chenmeng
@Desc    :   None
'''
from typing import Dict, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig


class RBT6SimCSEModel(nn.Module):
    def __init__(self, pretrained_model, sentence_pooling_method='cls', temperature=0.05) -> None:
        super(RBT6SimCSEModel, self).__init__()

        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        config = BertConfig.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(
            pretrained_model, config=config)
        
        
    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, labels=None):
        query_out = self.bert(**query)
        passage_out = self.bert(**passage)
        query_sentence_embedding = self.sentence_embedding(query_out.last_hidden_state, query['attention_mask'])
        passage_sentence_embedding = self.sentence_embedding(passage_out.last_hidden_state, passage['attention_mask'])

        if self.training:
            scores = self.compute_similarity(query_sentence_embedding, passage_sentence_embedding)
            # 得到y_pred对应的label, 也就是对角线，因为训练集中全是正例，但是在batch中，除去自己的正例，都为负例
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (query_sentence_embedding.size(0) // passage_sentence_embedding.size(0))

            loss = self.compute_loss(scores, target)
        else:
            # 在预测阶段，不需要计算batch中的两两相似度，而是只计算pair的相似度就行。
            scores = F.cosine_similarity(query_sentence_embedding, passage_sentence_embedding)
            target = labels
            loss = self.compute_loss(scores, target)

        outputs = {}
        outputs['loss'] = loss
        outputs['scores'] = scores
        return outputs

    def compute_similarity(self, query_embedding, passage_embedding):
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(query_embedding.unsqueeze(1), passage_embedding.unsqueeze(0), dim=-1)
        # 相似度矩阵除以温度系数
        sim = sim / self.temperature
        return sim
    

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)