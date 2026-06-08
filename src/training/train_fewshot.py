"""
小样本原型网络训练（对应旧项目的 step4）
"""

import torch
import torch.optim as optim
import numpy as np
import os


def train_fewshot(encoder, train_dataset, val_dataset, config, device):
    episodes = config["training"]["fewshot"]["episodes"]
    val_episodes = config["training"]["fewshot"]["val_episodes"]
    ways = config["training"]["fewshot"]["ways"]
    shot = config["training"]["fewshot"]["shot"]
    query = config["training"]["fewshot"]["query"]
    lr = config["training"]["fewshot"]["lr"]

    # 从数据集推断可用类别
    from ..data.dataset import EpisodicSampler

    train_sampler = EpisodicSampler(
        train_dataset, ways=ways, shot=shot, query=query)
    val_sampler = EpisodicSampler(
        val_dataset, ways=ways, shot=shot, query=query)

    optimizer = optim.Adam(encoder.parameters(), lr=lr)

    from ..models.prototypical import prototypical_loss

    print(f"\n{'='*50}")
    print("小样本原型网络训练")
    print(f"{'='*50}")
    print(f"配置: {ways}-way {shot}-shot, {episodes} episodes")

    best_val_acc = 0.0
    log_interval = 100

    for ep in range(1, episodes + 1):
        encoder.train()
        s_x, s_y, q_x, q_y = train_sampler.sample_episode()
        loss, train_acc = prototypical_loss(encoder, s_x, s_y, q_x, q_y, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % log_interval == 0 or ep == 1:
            encoder.eval()
            val_accs = []
            with torch.no_grad():
                for _ in range(val_episodes):
                    s_x, s_y, q_x, q_y = val_sampler.sample_episode()
                    _, acc = prototypical_loss(encoder, s_x, s_y, q_x, q_y, device)
                    val_accs.append(acc)

            mean_val_acc = np.mean(val_accs) * 100
            best_val_acc = max(best_val_acc, mean_val_acc)

            print(f"Episode {ep:4d}/{episodes} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | "
                  f"Val Acc: {mean_val_acc:.1f}% (best: {best_val_acc:.1f}%)")

    print(f"\n✅ 小样本训练完成！最高验证准确率: {best_val_acc:.1f}%")

    # 保存
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "fewshot_encoder.pth")
    torch.save(encoder.state_dict(), model_path)
    print(f"编码器已保存到 {model_path}")

    return encoder, best_val_acc
