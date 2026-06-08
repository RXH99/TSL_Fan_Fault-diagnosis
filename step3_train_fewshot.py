"""
Step 3: 小样本原型网络训练

153 类训练 → 18 类验证调参 → 18 类最终测试
设定: 5-way 5-shot 标准 FSL

运行: python step3_train_fewshot.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np

from src.config import Config
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder
from src.models.prototypical import prototypical_loss
from src.training.train_fewshot import train_fewshot


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

    # 加载数据
    train_dataset = FaultDataset(npz_path, split="train")
    val_dataset = FaultDataset(npz_path, split="val")
    test_dataset = FaultDataset(npz_path, split="test")

    ways = cfg.cfg["training"]["fewshot"]["ways"]
    shot = cfg.cfg["training"]["fewshot"]["shot"]
    query = cfg.cfg["training"]["fewshot"]["query"]
    print(f"设定: {ways}-way {shot}-shot")

    # 构建编码器
    encoder = create_encoder("cnn").to(device)

    # 尝试加载 Step 2 的全量 CNN 预训练权重
    pretrained_path = os.path.join(cfg.cfg["paths"]["output_dir"], "cnn_full.pth")
    if os.path.exists(pretrained_path):
        from src.models.classifier import CNN1DClassifier
        full_model = CNN1DClassifier(
            num_classes=len(train_dataset.y.unique()),
            encoder_dim=cfg.cfg["model"]["encoder_dim"])
        full_model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        encoder.load_state_dict(full_model.encoder.state_dict(), strict=False)
        print("✅ 已加载 Step 2 预训练权重")
    else:
        print("⚠️ 无预训练权重，从头训练")
        print("   建议先运行 step2_train_cnn.py 获得预训练权重")

    # 训练（用 val 集调参）
    encoder, best_val_acc = train_fewshot(
        encoder, train_dataset, val_dataset, cfg.cfg, device)

    # 最终在 test 集上评估
    print(f"\n{'='*50}")
    print("最终测试集评估 (18 类)")
    print(f"{'='*50}")

    test_sampler = EpisodicSampler(
        test_dataset,
        ways=ways, shot=shot, query=query)

    encoder.eval()
    test_accs = []
    with torch.no_grad():
        for _ in range(200):
            s_x, s_y, q_x, q_y = test_sampler.sample_episode()
            _, acc = prototypical_loss(encoder, s_x, s_y, q_x, q_y, device)
            test_accs.append(acc)

    mean_acc = np.mean(test_accs) * 100
    std_acc = np.std(test_accs) * 100

    print(f"  Test Acc: {mean_acc:.1f}% ± {std_acc:.1f}% (200 episodes)")
    print()
    print(f"📊 Step 3 结果")
    print(f"  5-way 5-shot")
    print(f"  Val Acc (best): {best_val_acc:.1f}%")
    print(f"  Test Acc:       {mean_acc:.1f}%")
    print(f"\n✅ 小样本训练完成，可继续执行 step4_ig_analysis.py")


if __name__ == "__main__":
    run()
