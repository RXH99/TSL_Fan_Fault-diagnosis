"""
Step 4: IG 归因可解释性分析
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.data.dataset import FaultDataset, EpisodicSampler
from src.models.encoder import create_encoder


def compute_ig(encoder, query_x, support_x, support_y, device, steps=50):
    """计算 IG 归因，用"到预测原型的负距离"作为评分函数"""
    encoder.eval()
    with torch.no_grad():
        s_emb = encoder(support_x.to(device))
        ways = len(torch.unique(support_y))
        prototypes = []
        for c in range(ways):
            mask = (support_y == c)
            proto = s_emb[mask].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)

    q = encoder(query_x.to(device))
    dists = torch.cdist(q.unsqueeze(1), prototypes.unsqueeze(0)).squeeze(1)
    pred = dists.argmin(dim=1).item()

    x = query_x.clone().detach().to(device).requires_grad_(True)
    baseline = torch.zeros_like(x)
    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline)
                     for i in range(steps + 1)]

    grad_sum = torch.zeros_like(x)
    for i in range(0, steps + 1, 10):
        batch = torch.cat(scaled_inputs[i:i+10], dim=0).to(device)
        batch.requires_grad_(True)
        emb = encoder(batch)
        d = torch.cdist(emb, prototypes[pred:pred+1])
        score = -d.squeeze()
        grads = torch.autograd.grad(torch.sum(score), batch, retain_graph=True)[0]
        grad_sum += grads.sum(dim=0, keepdim=True)

    avg_grad = grad_sum / (steps + 1)
    attributions = (x - baseline) * avg_grad
    return attributions.squeeze().detach().cpu().numpy(), pred


def analyze_ig(attr):
    abs_attr = np.abs(attr)
    pos_ratio = (attr > 0).mean() * 100
    threshold = abs_attr.mean() + 2 * abs_attr.std()
    peaks = np.where(abs_attr > threshold)[0]
    cv = abs_attr.std() / (abs_attr.mean() + 1e-8)
    cr = abs_attr[peaks].sum() / (abs_attr.sum() + 1e-8) if len(peaks) > 0 else 0
    return pos_ratio, len(peaks), cv, cr


def run():
    config_path = "configs/baseline.yaml"
    cfg = Config(config_path)
    device = cfg.device

    npz_path = os.path.join(cfg.cfg["data"]["processed_dir"], "preprocessed.npz")
    encoder_path = os.path.join(cfg.cfg["paths"]["output_dir"], "fewshot_encoder.pth")

    if not os.path.exists(npz_path) or not os.path.exists(encoder_path):
        print("❌ 预处理文件或模型不存在，请先运行前的步骤")
        sys.exit(1)

    ways = cfg.cfg["training"]["fewshot"]["ways"]
    test_dataset = FaultDataset(npz_path, split="test")

    encoder = create_encoder("cnn").to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    print(f"✅ 加载小样本编码器")

    test_sampler = EpisodicSampler(test_dataset, ways=ways, shot=5, query=5)
    s_x, s_y, q_x, q_y = test_sampler.sample_episode()
    print(f"\n采样: {len(s_x)} 支持集 | {len(q_x)} 查询集")

    # 从每个类选一个查询样本
    selected = []
    for c in range(ways):
        mask = (q_y == c)
        idx = torch.where(mask)[0][0].item()
        selected.append(q_x[idx:idx+1])

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    output_dir = os.path.join(cfg.cfg["paths"]["output_dir"], "ig_analysis")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    results = []

    for i, x in enumerate(selected[:6]):
        attr, pred = compute_ig(encoder, x, s_x, s_y, device)
        pos_ratio, peaks, cv, cr = analyze_ig(attr)
        results.append((pos_ratio, peaks, cv, cr))

        ok = "✅" if pred == i else "❌"
        ax = axes[i]
        ax.plot(attr, color='steelblue', linewidth=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_title(f'类#{i} {ok} 预测#{pred}', fontsize=11)
        ax.set_xlabel('采样点', fontsize=9)
        ax.set_ylabel('归因值', fontsize=9)
        ax.grid(True, alpha=0.3)

        abs_attr = np.abs(attr)
        th = abs_attr.mean() + 2 * abs_attr.std()
        peaks_idx = np.where(abs_attr > th)[0]
        if len(peaks_idx) > 0:
            ax.scatter(peaks_idx, attr[peaks_idx],
                      color='red', s=8, alpha=0.6, label=f'{len(peaks_idx)} 峰值')
            ax.legend(fontsize=8)

    for j in range(len(selected), 6):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ig_attributions.png"), dpi=150)
    print(f"✅ 图已保存到 {output_dir}/ig_attributions.png")

    print(f"\n{'='*50}")
    print("IG 归因统计")
    print(f"{'='*50}")
    for i, (pr, pk, cv, cr) in enumerate(results):
        print(f"  类#{i}: pos={pr:.1f}%, 峰值={pk}, CV={cv:.2f}, cr={cr:.2f}")


if __name__ == "__main__":
    run()
