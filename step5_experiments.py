"""
Step 5: 多设定小样本实验

在训练好的编码器上评估多种 FSL 设定（1-shot / 5-shot / 10-way）
每种设定跑 10 次取平均，输出稳定结果

运行: python step5_experiments.py
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


def evaluate(encoder, test_dataset, ways, shot, query, device, num_episodes=200):
    """在测试集上评估一个设定，返回平均准确率 ± 标准差"""
    sampler = EpisodicSampler(test_dataset, ways=ways, shot=shot, query=query)
    encoder.eval()

    accs = []
    with torch.no_grad():
        for _ in range(num_episodes):
            s_x, s_y, q_x, q_y = sampler.sample_episode()
            _, acc = prototypical_loss(encoder, s_x, s_y, q_x, q_y, device)
            accs.append(acc)

    return np.mean(accs) * 100, np.std(accs) * 100


def run():
    config_path = "configs/baseline.yaml"
    cfg = Config(config_path)
    device = cfg.device

    npz_path = os.path.join(cfg.cfg["data"]["processed_dir"], "preprocessed.npz")
    encoder_path = os.path.join(cfg.cfg["paths"]["output_dir"], "fewshot_encoder.pth")

    if not os.path.exists(encoder_path):
        print(f"❌ 找不到 {encoder_path}，请先运行 step3_train_fewshot.py")
        sys.exit(1)

    test_dataset = FaultDataset(npz_path, split="test")
    encoder = create_encoder("cnn").to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    print(f"✅ 加载编码器\n")

    # 实验设定
    experiments = [
        ("5-way 1-shot",  5, 1, 15),   # 1-shot: 每类15个query（20-1=19够用）
        ("5-way 5-shot",  5, 5, 5),    # 5-shot: 原有设定
        ("10-way 1-shot", 10, 1, 10),  # 10-way 1-shot
        ("10-way 5-shot", 10, 5, 5),   # 10-way 5-shot
    ]
    RUNS = 10  # 每个设定跑 10 次
    EPISODES = 200

    print(f"{'='*60}")
    print(f"{'设定':<18} {'准确率':<16} {'跑10次':<10}")
    print(f"{'='*60}")

    results = []
    for name, ways, shot, query in experiments:
        all_runs = []
        for run_idx in range(RUNS):
            mean_acc, std_acc = evaluate(encoder, test_dataset, ways, shot, query,
                                          device, EPISODES)
            all_runs.append(mean_acc)

        overall_mean = np.mean(all_runs)
        overall_std = np.std(all_runs)
        results.append((name, overall_mean, overall_std))

        bar = "█" * int(overall_mean / 2)
        print(f"{name:<18} {overall_mean:>5.1f}% ± {overall_std:.1f}%     {bar:<10}")

    print(f"{'='*60}")
    print(f"\n📊 实验总结（测试集 18 类，各跑 {RUNS} 次取平均）")
    print(f"{'='*60}")
    for name, mean, std in results:
        print(f"  {name:<18} → {mean:.1f}% ± {std:.1f}%")

    # 保存结果
    output_dir = cfg.cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "experiments_results.txt")
    with open(save_path, "w") as f:
        f.write("FSL_fan 多设定实验结果\n")
        f.write("测试集: 18 个新类别\n")
        f.write(f"每个设定跑 {RUNS} 次，每次 {EPISODES} episodes\n\n")
        for name, mean, std in results:
            f.write(f"{name:<18} {mean:.1f}% ± {std:.1f}%\n")
    print(f"\n✅ 结果已保存到 {save_path}")


if __name__ == "__main__":
    run()
