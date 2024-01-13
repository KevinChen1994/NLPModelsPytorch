# -*- coding:utf-8 -*-
# author: chenmeng
# datetime:2023/4/13 17:08
# Description:
import os
import random

from loss_function import simcse_loss
from tqdm import tqdm
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
parser.add_argument("--eval_step", type=int, default=10, help="every eval_step to evaluate model")
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
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.strip().split('\t')
                all_data.append(line_split[0])
                all_data.append(line_split[1])
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 这里将同一个句子使用两次的目的就是通过dropout拿到相同句子不同的embedding
        return tokenizer([self.data[idx][0], self.data[idx][1]], max_length=args.max_length, truncation=True,
                         padding='max_length', return_tensors='pt')


class TestDataSet(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        all_data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.strip().split('\t')
                sentence1 = line_split[0]
                sentence2 = line_split[1]
                label = int(line_split[2])
                all_data.append([sentence1, sentence2, label])
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = tokenizer(self.data[idx][0], max_length=args.max_length, truncation=True, padding='max_length',
                           return_tensors='pt')
        target = tokenizer(self.data[idx][1], max_length=args.max_length, truncation=True, padding='max_length',
                           return_tensors='pt')
        source_input_ids = source['input_ids'].view(args.max_length).to(args.device)
        source_attention_mask = source['attention_mask'].view(args.max_length).to(args.device)
        source_token_type_ids = source['token_type_ids'].view(args.max_length).to(args.device)
        target_input_ids = target['input_ids'].view(args.max_length).to(args.device)
        target_attention_mask = target['attention_mask'].view(args.max_length).to(args.device)
        target_token_type_ids = target['token_type_ids'].view(args.max_length).to(args.device)
        source_input = {'input_ids': source_input_ids, 'attention_mask': source_attention_mask,
                        'token_type_ids': source_token_type_ids}
        target_input = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask,
                        'token_type_ids': target_token_type_ids}
        return [source_input, target_input, torch.tensor(self.data[idx][2], device=args.device)]


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


def train_loop(train_dataloader, test_dataloader, model, optimizer, lr_scheduler, epoch, total_loss, best_acc):
    progress_bar = tqdm(range(len(train_dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(train_dataloader)

    model.train()
    for step, inputs in enumerate(train_dataloader, start=1):
        input_ids = inputs['input_ids'].view(len(inputs['input_ids']) * 2, -1).to(args.device)
        attention_mask = inputs['attention_mask'].view(len(inputs['attention_mask']) * 2, -1).to(args.device)
        token_type_ids = inputs['token_type_ids'].view(len(inputs['token_type_ids']) * 2, -1).to(args.device)
        pred = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_loss.simcse_unsup_loss(pred, args.device)

        # 梯度归零
        optimizer.zero_grad()
        # 计算梯度值
        loss.backward()
        # 反向传播
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description('loss: {:.7f}'.format(total_loss / (finish_step_num + step)))
        progress_bar.update(1)
        if step % args.eval_step == 0:
            model.eval()
            eval_acc = test_loop(test_dataloader, model)
            print("acc_test:" + "{:.4f}".format(eval_acc))
            if eval_acc > best_acc:
                best_acc = eval_acc
                print('saving new weights...\n')
                torch.save(bertSimCSEModel.state_dict(), './checkpoints/bert_simcse_unsupervised/model_weights.bin')
            model.train()
    return total_loss, best_acc


def test_loop(dataloader, model, mode='Test'):
    model.eval()
    sim_tensor = torch.tensor([], device=args.device)
    label_tensor = torch.tensor([], device=args.device)
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            source_pred = model(source['input_ids'], source['attention_mask'], source['token_type_ids'])
            target_pred = model(target['input_ids'], target['attention_mask'], target['token_type_ids'])
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

    train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size_train, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size_eval, shuffle=False)

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
        total_loss, best_acc = train_loop(train_dataloader, test_dataloader, bertSimCSEModel, optimizer, lr_scheduler, epoch,
                                total_loss, best_acc)
    print("Done!")
