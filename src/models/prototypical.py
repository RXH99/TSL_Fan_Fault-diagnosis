"""
原型网络核心逻辑：损失函数、预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def prototypical_loss(encoder, support_x, support_y, query_x, query_y, device):
    """
    原型网络前向 + 损失计算

    步骤:
    1. Encoder 提取 support 和 query 特征
    2. 每类 support 特征求均值 → 类原型
    3. query 与各原型算欧氏距离 → Softmax → CrossEntropy

    返回: (loss, acc)
    """
    support_emb = encoder(support_x.to(device))
    query_emb = encoder(query_x.to(device))

    ways = len(torch.unique(support_y))
    shot = support_x.size(0) // ways

    # 计算类原型
    prototypes = []
    for cls in range(ways):
        cls_mask = (support_y == cls)
        proto = support_emb[cls_mask].mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes)  # (WAYS, D)

    # 欧氏距离
    dists = torch.cdist(query_emb.unsqueeze(1),
                        prototypes.unsqueeze(0)).squeeze(1)  # (Nq, WAYS)

    # 负距离做 Softmax
    loss = F.cross_entropy(-dists, query_y.to(device))

    # 准确率
    _, preds = torch.min(dists, dim=1)
    acc = (preds == query_y.to(device)).float().mean().item()

    return loss, acc
