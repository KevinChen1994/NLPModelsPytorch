# -*- coding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2023/09/07 15:22:14
@Author  :   chenmeng
@Desc    :   None
'''

import os
import torch
import numpy as np
import random
import logging
from arguments import get_args
from transformers import  EvalPrediction, BertTokenizer
from simcse.dataset import SIMCSEDataset, DataCollator
from arguments import DATASETS
from model.simcse_model import RBT6SimCSEModel
import evaluate
from collections import defaultdict
import transformers
from transformers import Trainer


transformers.logging.set_verbosity_info()
logging.disable(logging.WARNING)

label_dict = {"positive": 0, "negative": 1, 0: "positive", 1: "negative"}


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
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    # 如果大于0.9则为1，也就是相似，否则为0，不相似
    preds = (preds > 0.9) + 0
    metric = evaluate.load('accuracy')
    result = metric.compute(predictions=preds, references=p.label_ids)
    return result

# 如果使用raw_data.map()，可以调用这个方法
def prepare_dataset(examples):
    result = defaultdict(list)
    for example in examples['text']:
        example_split = example.replace('\n', '').split('\t')
        query = example_split[1]
        passage = example_split[2]
        query_feature = tokenizer(query, max_length=data_args.max_seq_length,
                                  truncation=True, padding='max_length', return_tensors='pt')
        passage_feature = tokenizer(passage, max_length=data_args.max_seq_length,
                                    truncation=True, padding='max_length', return_tensors='pt')

        result['query_feature'].append(query_feature)
        result['passage_feature'].append(passage_feature)
        result['labels'].append(label_dict[example_split[0]])
    return result


if __name__ == '__main__':
    model_args, data_args, training_args = get_args()
    seed_everything(training_args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name) if model_args.tokenizer_name else BertTokenizer.from_pretrained(model_args.model_name_or_path)

    model = RBT6SimCSEModel(model_args.model_name_or_path)

    train_dataset = SIMCSEDataset(data_args.train_file)
    eval_dataset = SIMCSEDataset(data_args.test_file)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=DataCollator(
            tokenizer=tokenizer,
            query_max_len=data_args.query_max_seq_length,
            passage_max_len=data_args.passage_max_seq_length
        ),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
