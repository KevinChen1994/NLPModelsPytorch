# -*- coding:utf-8 -*-
# author: chenmeng
# datetime:2023/5/12 18:53
# Description:

import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import scipy.stats
import os
from tqdm import tqdm
from loss_function import simcse_loss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))
model_path = "./checkpoints/bert-base-chinese"
save_path = "./checkpoints/bert_simcse_supervised/128_pooler_supervised_best_model.pth"
tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)

output_way = 'pooler'
batch_size = 10
learning_rate = 2e-5
maxlen = 64


class TrainDataset(Dataset):
    def __init__(self, file_path, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = open(file_path, 'r', encoding='utf-8').readlines()
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def text_to_id(self, source):
        source_json = json.loads(source)
        origin = source_json['query']
        entailment = source_json['title']
        contradiction = source_json['neg_title']
        sample = self.tokenizer([origin, entailment, contradiction], max_length=self.maxlen, truncation=True,
                                padding='max_length', return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


class TestDataset:
    def __init__(self, file_path, tokenizer, maxlen):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.all_data = self.load_data()

    def load_data(self):
        all_data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:1000]:
                line_json = json.loads(line)
                all_data.append([line_json["sentence1"], line_json["sentence2"], line_json["label"]])
        return all_data

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.text_to_id(self.all_data[idx][0]), self.text_to_id(self.all_data[idx][1]), int(self.all_data[idx][2])


class NeuralNetwork(nn.Module):
    def __init__(self, model_path, output_way):
        super(NeuralNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, config=Config)
        self.output_way = output_way
        assert output_way in ['cls', 'pooler']

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output

class BertSimCSEModel(nn.Module):
    def __init__(self,
                 config,
                 pooling):
        super(BertSimCSEModel, self).__init__()

        self.bert = BertModel.from_pretrained(model_path, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        elif self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        elif self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=first.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=first.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


# model = NeuralNetwork(model_path, output_way).to(device)
model = BertSimCSEModel(model_path, output_way).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

training_data = TrainDataset('data/simclue/train_rank.json', tokenizer, maxlen)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

testing_data = TestDataset('data/simclue/test_public.json', tokenizer, maxlen)
test_dataloader = DataLoader(testing_data, batch_size)


def compute_loss(y_pred, lamda=0.05):
    row = torch.arange(0, y_pred.shape[0], 3, device=device)
    col = torch.arange(y_pred.shape[0], device=device)
    col = torch.where(col % 3 != 0)[0]
    y_true = torch.arange(0, len(col), 2, device=device)
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # torch自带的快速计算相似度矩阵的方法
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)
    # 屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    # 论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def test(dataloader, model):
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_tensor = torch.tensor([], device=device)
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            source_pred = model(source['input_ids'].view(len(source['input_ids']), -1).to(device),
                                source['attention_mask'].view(len(source['attention_mask']), -1).to(device),
                                source['token_type_ids'].view(len(source['token_type_ids']), -1).to(device))
            target_pred = model(target['input_ids'].view(len(source['input_ids']), -1).to(device),
                                target['attention_mask'].view(len(source['attention_mask']), -1).to(device),
                                target['token_type_ids'].view(len(source['token_type_ids']), -1).to(device))
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            # concat
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label = torch.tensor(label, device=device)
            label_tensor = torch.cat((label_tensor, label), dim=0)
    # 相似度大于0.9认为相似，赋值为int 1
    pred = (sim_tensor > 0.9) + 0
    correct = (pred == label_tensor).sum()
    acc = correct.item() / label_tensor.size()[0]
    return acc


def train(dataloader, test_dataloader, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    max_corrcoef = 0
    not_up_batch = 0
    for batch, data in enumerate(dataloader):
        input_ids = data['input_ids'].view(len(data['input_ids']) * 3, -1).to(device)
        attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 3, -1).to(device)
        token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 3, -1).to(device)
        pred = model(input_ids, attention_mask, token_type_ids)
        # loss = compute_loss(pred)
        loss = simcse_loss.simcse_sup_loss(pred, device)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * int(len(input_ids) / 3)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            model.eval()  # 意味着显示关闭丢弃dropout
            corrcoef = test(test_dataloader, model)
            model.train()
            print(f"corrcoef_test: {corrcoef:>4f}")
            if corrcoef > max_corrcoef:
                not_up_batch = 0
                max_corrcoef = corrcoef
                torch.save(model.state_dict(), save_path)
                print(f"Higher corrcoef: {(max_corrcoef):>4f}%, Saved PyTorch Model State to model.pth")
            else:
                not_up_batch += 1
                if not_up_batch > 10:
                    print(f"Corrcoef didn't up for %s batch, early stop!" % not_up_batch)
                    break


if __name__ == '__main__':
    epochs = 1
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, test_dataloader, model, optimizer)
    # print("Train_Done!")
    print("Deving_start!")
    model.load_state_dict(torch.load(save_path, map_location=device))
    acc = test(test_dataloader, model)
    print(acc)
