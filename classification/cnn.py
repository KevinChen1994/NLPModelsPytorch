''''
max_pool和rele也可以使用nn.functional中的方法，https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextCNN.py
个人还是习惯使用nn下的方法，这样代码看起来比较统一
'''
import os
import torch
import numpy as np
import random
import logging
import torch.nn as nn
import torch.nn.functional as F
from arguments import get_args
from transformers import  EvalPrediction, BertTokenizer
from arguments import DATASETS
from model.simcse_model import RBT6SimCSEModel
from dataset_utils import ToutiaoClassificationDataset, DataCollator
import evaluate
from collections import defaultdict
import transformers
from transformers import Trainer


transformers.logging.set_verbosity_info()
logging.disable(logging.WARNING)

label_dict = {
    "故事": 0, 
    "文化": 1,
    "娱乐": 2,
    "体育": 3,
    "财经": 4,
    "新时代": 5,
    "房产": 6,
    "汽车": 7,
    "教育": 8,
    "科技": 9 ,
    "军事": 10,
    "旅游": 11,
    "国际": 12,
    "证券": 13,
    "三农": 14,
    "电竞": 15
    }


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


def compute_metrics(p: EvalPrediction):
    # preditons中有两个元组，第一个是模型预测的结果，第二个是归一化后的结果，我们选择第二个
    preds = p.predictions[1] if isinstance(
        p.predictions, tuple) else p.predictions

    preds = np.argmax(preds, axis=1)
    metric = evaluate.load('accuracy')
    result = metric.compute(predictions=preds, references=p.label_ids)
    return result

def read_word_dict(file_path):
    word_dict = defaultdict(int)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, word in enumerate(lines):
            word_dict[word.strip()] = i
    return word_dict

class CnnClassificationModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, max_len, dorpout):
        super(CnnClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_len - 3 + 1, 1)) # shape: torch.Size([1, 256, 1, 1])
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(4, embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_len - 4 + 1, 1))  # shape: torch.Size([1, 256, 1, 1])
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(5, embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_len - 5 + 1, 1))  # shape: torch.Size([1, 256, 1, 1])
        )

        self.dropout = nn.Dropout(dorpout)

        self.out = nn.Linear(256 * 3, num_classes)

    def forward(self, query, labels):
        x = self.embedding(query)
        x = x.unsqueeze(1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        out = torch.cat([conv1, conv2, conv3], dim=1)
        out = out.view(out.size(0), -1) 
        out = self.dropout(out)
        output = self.out(out)

        loss = self.compute_loss(output, labels)
        preds = F.softmax(output, dim=1)
        outputs = {}
        outputs['loss'] = loss
        outputs['output'] = output
        outputs['preds'] = preds
        return outputs
    
    def compute_loss(self, logits, labels):
        return self.cross_entropy(logits, labels)
    

    
if __name__ == '__main__':
    model_args, data_args, training_args = get_args()
    seed_everything(training_args.seed)

    word_dict = read_word_dict('../dataset/toutiao_classification/word_dict.txt')

    model = CnnClassificationModel(num_classes=len(label_dict), vocab_size=len(word_dict), embedding_dim=100, max_len=data_args.query_max_seq_length, dorpout=0.5)

    train_dataset = ToutiaoClassificationDataset(data_args.train_file, label_dict)
    test_dataset = ToutiaoClassificationDataset(data_args.test_file, label_dict)

    # 为了使用transformers的Trainer，我们需要传入一个None的Tokenizer。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        data_collator=DataCollator(
            query_max_len=data_args.query_max_seq_length,
            word_dict=word_dict,
            tokenizer=None

        ),
        compute_metrics=compute_metrics,
    )

    trainer.train()