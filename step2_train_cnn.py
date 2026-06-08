"""
Step 2: 全量 CNN 训练（监督基线）

在 153 类训练集上训练 CNN 分类器
从训练集内部 80/20 分割做验证（因为 test/val 是没见过的类别，CNN 无法预测）

目的：确认数据可训练，记录全量监督的上限准确率
同时训练好的 Encoder 权重可迁移到 Step 3 小样本训练

运行: python step2_train_cnn.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
from torch.utils.data import random_split

from src.config import Config
from src.data.dataset import FaultDataset
from src.models.classifier import CNN1DClassifier
from src.training.train_cnn import train_cnn


def run():
    config_path = "configs/baseline.yaml"
    cfg = Config(config_path)
    device = cfg.device

    npz_path = os.path.join(cfg.cfg["data"]["processed_dir"], "preprocessed.npz")

    if not os.path.exists(npz_path):
        print(f"❌ 找不到 {npz_path}")
        print(f"   请先运行 step1_preprocess.py")
        sys.exit(1)

    print(f"设备: {device}")
    print(f"配置文件: {config_path}")

    # 加载训练集（只有 153 类）
    train_dataset = FaultDataset(npz_path, split="train")

    # 训练标签不连续（如 3, 7, 12...），需要重映射为 0~152
    unique_labels = sorted(train_dataset.y.unique().tolist())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    # 重映射标签
    remapped_y = torch.tensor([label_map[l.item()] for l in train_dataset.y])
    train_dataset.y = remapped_y

    print(f"CNN 可预测类别数: {num_classes}")
    print(f"标签范围: {min(unique_labels)}~{max(unique_labels)} → 0~{num_classes-1}")

    # 从训练集内部 80/20 分
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    print(f"CNN 验证: 从训练类内分 {train_size}/{val_size}")

    # 构建模型
    model = CNN1DClassifier(
        num_classes=num_classes,
        encoder_dim=cfg.cfg["model"]["encoder_dim"]
    ).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    model, history = train_cnn(model, train_subset, val_subset,
                                cfg.cfg, device)

    # 最终总结
    best = max(history['val_acc'])
    last = history['val_acc'][-1]
    print(f"\n{'='*50}")
    print("📊 Step 2 结果")
    print(f"{'='*50}")
    print(f"  全量 CNN (训练类内 80/20 验证)")
    print(f"  最高准确率: {best:.1f}%")
    print(f"  最终准确率: {last:.1f}%")
    print()
    if best > 85:
        print("  ✅ 数据可训练，继续 Step 3")
    else:
        print("  ⚠️ 准确率偏低，可能需要调参或检查数据")
    print(f"\n✅ 全量 CNN 训练完成，可继续执行 step3_train_fewshot.py")


if __name__ == "__main__":
    run()
