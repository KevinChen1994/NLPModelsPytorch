# -*- coding:utf-8 -*-
# author: chenmeng
# datetime:2023/5/9 18:46
# Description:
import os
import random

from loss_function import simcse_loss
from tqdm import tqdm
from collections import defaultdict
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import AdamW, get_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
parser.add_argument("--output_path", type=str, default='output')
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size_train", type=int, default=4)
parser.add_argument("--batch_size_eval", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
parser.add_argument("--max_length", type=int, default=64, help="max length of input")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--train_file", type=str, default="data/lcqmc/train.tsv")
parser.add_argument("--dev_file", type=str, default="data/lcqmc/dev.tsv")
parser.add_argument("--test_file", type=str, default="data/lcqmc/test.tsv")
parser.add_argument("--pretrain_model_path", type=str,
                    default="checkpoints/bert-base-chinese")
parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                    default='cls', help='pooler to use')
parser.add_argument("--do_train", action='store_true', default=True)
parser.add_argument("--do_predict", action='store_true', default=True)

args = parser.parse_args()


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {args.device} device')
seed_everything(args.seed)

tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)


class TrainDataSet(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        all_data = []
        negative_dict = defaultdict(list)
        positive_dict = defaultdict(list)
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.strip().split('\t')
                label = line_split[2]
                if label == '0':
                    negative_dict[line_split[0]].append(line_split[1])
                    negative_dict[line_split[1]].append(line_split[0])
                elif label == '1':
                    positive_dict[line_split[0]].append(line_split[1])
                    positive_dict[line_split[1]].append(line_split[0])

        for source, positive_list in positive_dict.items():
            if source in negative_dict:
                positive_index = random.randrange(0, len(positive_list))
                negative_index = random.randrange(0, len(negative_dict[source]))
                all_data.append([source, positive_list[positive_index], negative_dict[source][negative_index]])

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestDataSet(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        all_data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:100]:
                line_split = line.strip().split('\t')
                sentence1 = line_split[0]
                sentence2 = line_split[1]
                label = int(line_split[2])
                all_data.append([sentence1, sentence2, label])
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_collote_fn(batch_samples):
    source_batch = []
    positive_batch = []
    negative_batch = []
    for sample in batch_samples:
        # 这里将同一个句子使用两次的目的就是通过dropout拿到相同句子不同的embedding
        source_batch.append(sample[0])
        positive_batch.append(sample[1])
        negative_batch.append(sample[2])
    source_data = tokenizer(source_batch, padding='max_length', truncation=True, max_length=args.max_length,
                            return_tensors='pt')
    positive_data = tokenizer(positive_batch, padding='max_length', truncation=True, max_length=args.max_length,
                              return_tensors='pt')
    negative_data = tokenizer(negative_batch, padding='max_length', truncation=True, max_length=args.max_length,
                              return_tensors='pt')
    source_input_ids = source_data['input_ids'].view(-1, args.max_length).to(args.device)
    source_attention_mask = source_data['attention_mask'].view(-1, args.max_length).to(args.device)
    source_token_type_ids = source_data['token_type_ids'].view(-1, args.max_length).to(args.device)

    positive_input_ids = positive_data['input_ids'].view(-1, args.max_length).to(args.device)
    positive_attention_mask = positive_data['attention_mask'].view(-1, args.max_length).to(args.device)
    positive_token_type_ids = positive_data['token_type_ids'].view(-1, args.max_length).to(args.device)

    negative_input_ids = negative_data['input_ids'].view(-1, args.max_length).to(args.device)
    negative_attention_mask = negative_data['attention_mask'].view(-1, args.max_length).to(args.device)
    negative_token_type_ids = negative_data['token_type_ids'].view(-1, args.max_length).to(args.device)

    return torch.cat([source_input_ids, positive_input_ids, negative_input_ids], 0), \
           torch.cat([source_attention_mask, positive_attention_mask, negative_attention_mask], 0), \
           torch.cat([source_token_type_ids, positive_token_type_ids, negative_token_type_ids], 0)


def test_collote_fn(batch_samples):
    sentence_batch_1, sentence_batch_2, label_batch = [], [], []
    for sample in batch_samples:
        sentence_batch_1.append(sample[0])
        sentence_batch_2.append(sample[1])
        label_batch.append(sample[2])
    source_data = tokenizer(sentence_batch_1, padding='max_length', truncation=True, max_length=args.max_length,
                            return_tensors='pt')
    target_data = tokenizer(sentence_batch_2, padding='max_length', truncation=True, max_length=args.max_length,
                            return_tensors='pt')
    source_input_ids = source_data['input_ids'].view(-1, args.max_length).to(args.device)
    source_attention_mask = source_data['attention_mask'].view(-1, args.max_length).to(args.device)
    source_token_type_ids = source_data['token_type_ids'].view(-1, args.max_length).to(args.device)
    target_input_ids = target_data['input_ids'].view(-1, args.max_length).to(args.device)
    target_attention_mask = target_data['attention_mask'].view(-1, args.max_length).to(args.device)
    target_token_type_ids = target_data['token_type_ids'].view(-1, args.max_length).to(args.device)
    label_tensor = torch.tensor(label_batch, device=args.device)
    return [source_input_ids, source_attention_mask, source_token_type_ids], \
           [target_input_ids, target_attention_mask, target_token_type_ids], label_tensor


class BertSimCSEModel(nn.Module):
    def __init__(self,
                 config,
                 pooling):
        super(BertSimCSEModel, self).__init__()

        self.bert = BertModel.from_pretrained(args.pretrain_model_path, config=config)
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


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, (input_ids, attention_mask, token_type_ids) in enumerate(dataloader, start=1):
        pred = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_loss.simcse_sup_loss(pred, args.device)

        # 梯度归零
        optimizer.zero_grad()
        # 计算梯度值
        loss.backward()
        # 反向传播
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    model.eval()
    sim_tensor = torch.tensor([], device=args.device)
    label_tensor = torch.tensor([], device=args.device)
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            source_pred = model(source[0], source[1], source[2])
            target_pred = model(target[0], target[1], target[2])
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            # concat
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_tensor = torch.cat((label_tensor, label), dim=0)
    # 相似度大于0.9认为相似，赋值为int 1
    pred = (sim_tensor > 0.9) + 0
    correct = (pred == label_tensor).sum()
    acc = correct.item() / label_tensor.size()[0]
    return acc


if __name__ == '__main__':
    train_data = TrainDataSet(args.train_file)

    test_data = TestDataSet(args.dev_file)

    train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size_train, shuffle=True,
                                  collate_fn=train_collote_fn)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size_eval, shuffle=False,
                                 collate_fn=test_collote_fn)

    config = BertConfig.from_pretrained(args.pretrain_model_path)
    config.attention_probs_dropout_prob = args.dropout
    config.hidden_dropout_prob = args.dropout
    bertSimCSEModel = BertSimCSEModel(config, args.pooler).to(args.device)

    optimizer = AdamW(bertSimCSEModel.parameters(), lr=args.lr)
    # 动态调整学习率
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(train_dataloader),
    )
    total_loss = 0
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}\n-------------------------------")
        # total_loss = train_loop(train_dataloader, bertSimCSEModel, optimizer, lr_scheduler, args.epochs, total_loss)
        valid_acc = test_loop(test_dataloader, bertSimCSEModel, 'Test')
        print('valid acc' + str(valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(bertSimCSEModel.state_dict(), './checkpoints/bert_simcse_supervised/model_weights.bin')
        print("Done!")
