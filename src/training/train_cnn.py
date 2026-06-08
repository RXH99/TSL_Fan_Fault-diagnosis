"""
全量 CNN 训练基线（对应旧项目的 step3）

先在风机数据上跑一遍全量 CNN，确认数据可训练。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np


def train_cnn(model, train_dataset, val_dataset, config, device):
    batch_size = config["training"]["cnn"]["batch_size"]
    epochs = config["training"]["cnn"]["epochs"]
    lr = config["training"]["cnn"]["lr"]
    output_dir = config["paths"]["output_dir"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*50}")
    print("全量 CNN 训练")
    print(f"{'='*50}")
    print(f"训练: {len(train_dataset)} 样本 | 验证: {len(val_dataset)} 样本")
    print(f"设备: {device}")

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        loss_sum = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)
        history["train_loss"].append(avg_loss)

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, preds = torch.max(model(inputs), 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / total
        history["val_acc"].append(acc)
        best_acc = max(best_acc, acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    print(f"\n✅ 全量训练完成！最高准确率: {best_acc:.2f}%")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "cnn_full.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    return model, history
