"""
特征提取器（Encoder）

当前版本：CNN1D（旧模型，作为基线）
后续可以在这里加 TCN、Transformer、GNN 等
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    1D-CNN 特征提取器（去掉分类头）
    输入: (B, 1, 1024) → 输出: (B, 64) 特征向量

    这是旧项目的 CNN1D.features 部分，作为基线
    """
    def __init__(self, in_channels=1, hidden_dims=[16, 32, 64],
                 kernel_sizes=[15, 7, 3]):
        super().__init__()
        layers = []

        # Conv1
        layers.extend([
            nn.Conv1d(in_channels, hidden_dims[0],
                      kernel_size=kernel_sizes[0], stride=2, padding=7),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ])

        # Conv2
        layers.extend([
            nn.Conv1d(hidden_dims[0], hidden_dims[1],
                      kernel_size=kernel_sizes[1], stride=1, padding=3),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ])

        # Conv3
        layers.extend([
            nn.Conv1d(hidden_dims[1], hidden_dims[2],
                      kernel_size=kernel_sizes[2], stride=1, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        ])

        self.features = nn.Sequential(*layers)
        self.output_dim = hidden_dims[2]

    def forward(self, x):
        # x: (B, 1, L) → (B, 64)
        return self.features(x).squeeze(-1)


def create_encoder(encoder_type="cnn", **kwargs):
    """工厂方法：方便未来扩展其他 Encoder"""
    if encoder_type == "cnn":
        return CNNEncoder(**kwargs)
    # elif encoder_type == "tcn":
    #     return TCNEncoder(**kwargs)    # 后续添加
    # elif encoder_type == "transformer":
    #     return TransformerEncoder(**kwargs)  # 后续添加
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
