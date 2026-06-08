"""
集成梯度 (Integrated Gradients) 归因分析
"""

import torch
import numpy as np


class IntegratedGradients:
    """
    IG 归因计算

    在输入空间计算每个点的归因分数，
    不受 GAP/非线性分类头的限制。
    """
    def __init__(self, model, device="cpu", steps=50):
        self.model = model
        self.device = device
        self.steps = steps

    def attribute(self, x, target_class=None):
        """
        计算 IG 归因

        参数:
            x: (1, 1, L) 输入信号
            target_class: 目标类别，默认取模型预测类别

        返回:
            attributions: (L,) 归因分数
        """
        self.model.eval()
        x = x.to(self.device).requires_grad_(True)

        if target_class is None:
            with torch.no_grad():
                target_class = self.model(x).argmax(dim=1).item()

        # 基线：全零信号
        baseline = torch.zeros_like(x)

        # Riemann 近似
        scaled_inputs = [
            baseline + (float(i) / self.steps) * (x - baseline)
            for i in range(self.steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)

        # 计算梯度
        grad_sum = torch.zeros_like(x)
        for i in range(0, self.steps + 1, 10):
            batch = scaled_inputs[i:i+10].to(self.device)
            batch.requires_grad_(True)
            outputs = self.model(batch)
            # 对于分类头模型
            if outputs.ndim == 2:
                scores = outputs[:, target_class]
            else:
                scores = outputs
            grads = torch.autograd.grad(
                torch.sum(scores), batch, retain_graph=True)[0]
            grad_sum += grads.sum(dim=0, keepdim=True)

        # 平均梯度 × (x - baseline)
        avg_grad = grad_sum / (self.steps + 1)
        attributions = (x - baseline) * avg_grad
        return attributions.squeeze().detach().cpu().numpy()


def analyze_ig_results(attributions, class_name=""):
    """
    分析 IG 归因结果，计算统计指标
    """
    attr = attributions.flatten()
    abs_attr = np.abs(attr)

    pos_ratio = (attr > 0).mean() * 100

    # 峰值检测（超过均值+2倍标准差）
    threshold = abs_attr.mean() + 2 * abs_attr.std()
    peaks = np.where(abs_attr > threshold)[0]
    peak_count = len(peaks)

    # 集中度 (CV)
    cv = abs_attr.std() / (abs_attr.mean() + 1e-8)

    # 集中比 cr
    if len(peaks) > 0:
        peak_mass = abs_attr[peaks].sum()
        total_mass = abs_attr.sum()
        cr = peak_mass / total_mass if total_mass > 0 else 0
    else:
        cr = 0

    return {
        "class": class_name,
        "pos_ratio": pos_ratio,
        "peak_count": peak_count,
        "cv": cv,
        "cr": cr,
    }
