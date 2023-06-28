# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:KevinChen1994
# datetime:2023/4/4 07:16

import torch
import torch.nn.functional as F


def simcse_unsup_loss_bak(y_pred, device, temp=0.05):
    """
    用于无监督SimCSE训练的loss，在构造标签和计算余弦相似度上有不同的实现，最终结果一致。
    """
    # 构造标签 [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    batch_size = y_pred.size(0)
    y_true = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long).unsqueeze(1),
                        torch.arange(0, batch_size, step=2, dtype=torch.long).unsqueeze(1)],
                       dim=1).reshape([batch_size, ])

    # 计算score和loss
    norm_emb = F.normalize(y_pred, dim=1, p=2)
    # L2正则化后再进行内积即可得到余弦相似度，也可以直接通过api进行计算余弦相似度
    sim_score = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
    sim_score = sim_score - torch.eye(batch_size, device=device) * 1e12
    sim_score = sim_score / temp
    loss = F.cross_entropy(sim_score, y_true)
    return torch.mean(loss)


def simcse_unsup_loss(y_pred, device, temp=0.05):
    """
    无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / temp
    # 计算相似度矩阵与y_true的交叉熵损失
    # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def simcse_sup_loss_bak(y_pred, device, lamda=0.05):
    """
    有监督损失函数，获取相似度矩阵使用不同的方法，最终结果一致。
    """
    # 构造标签
    row = torch.arange(0, y_pred.shape[0], 3, device=device)
    col = torch.arange(0, y_pred.shape[0], device=device)
    col = col[col % 3 != 0]
    y_true = torch.arange(0, len(col), 2, device=device)

    # 计算相似度
    similarities = F.cosine_similarity(y_pred.unsqueeze(0), y_pred.unsqueeze(1), dim=-1)
    similarities = similarities[row, :]
    similarities = similarities[:, col]
    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def simcse_sup_loss(y_pred, device, lamda=0.05):
    """
    有监督损失函数
    """
    row = torch.arange(0, y_pred.shape[0], 3, device=device)  # [0,3]
    col = torch.arange(y_pred.shape[0], device=device)  # [0,1,2,3,4,5]
    # 这里[(0,1,2),(3,4,5)]代表二组样本，
    # 其中0,1是相似句子，0,2是不相似的句子
    # 其中3,4是相似句子，3,5是不相似的句子
    col = torch.where(col % 3 != 0)[0]  # [1,2,4,5]
    y_true = torch.arange(0, len(col), 2, device=device)  # 生成真实的label  = [0,2]
    # 计算各句子之间的相似度，形成下方similarities 矩阵，其中xij 表示第i句子和第j个句子的相似度
    # [[ x00,x01,x02,x03,x04 ,x05  ]
    # [ x10,x11,x12,x13,x14 ,x15  ]
    # [ x20,x21,x22,x23,x24 ,x25  ]
    # [ x30,x31,x32,x33,x34 ,x35  ]
    # [ x40,x41,x42,x43,x44 ,x45  ]
    # [ x50,x51,x52,x53,x54 ,x55  ]]
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # 这里将similarities 做切片处理，形成下方矩阵
    # [[ x01,x02,x04 ,x05 ]
    # [x31,x32,x34 ,x35 ]]
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)
    # 论文中除以 temperature 超参
    similarities = similarities / lamda
    # 下面这一行计算的是相似矩阵每一行和y_true = [0, 2] 的交叉熵损失
    # [[ x01,x02,x04 ,x05 ]   label = 0 含义：第0个句子应该和第1个句子的相似度最高,  即x01越接近1越好
    # [x31,x32,x34 ,x35 ]]  label = 2 含义：第3个句子应该和第4个句子的相似度最高   即x34越接近1越好
    # 这行代码就是simcse的核心部分，和正例句子向量相似度应该越大
    # 越好，和负例句子之间向量的相似度越小越好
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


if __name__ == '__main__':
    y_pred = torch.rand((6, 6), device="cuda")
    loss = simcse_unsup_loss_bak(y_pred, 'cpu', 0.05)
    print(loss)
    loss = simcse_unsup_loss(y_pred, 'cpu', 0.05)
    print(loss)
    loss = simcse_sup_loss_bak(y_pred, 'cpu', 0.05)
    print(loss)
    loss = simcse_sup_loss(y_pred, 'cpu', 0.05)
    print(loss)
