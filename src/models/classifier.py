"""
全量分类模型：CNN1D（带分类头），用于 W2 全量训练基线
"""

import torch.nn as nn
from .encoder import CNNEncoder


class CNN1DClassifier(nn.Module):
    """
    1D-CNN + 全连接分类头
    用于全量训练基线
    """
    def __init__(self, num_classes=4, encoder_dim=64):
        super().__init__()
        self.encoder = CNNEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

    def get_encoder(self):
        """返回去掉分类头的特征提取器，用于小样本训练"""
        return self.encoder
