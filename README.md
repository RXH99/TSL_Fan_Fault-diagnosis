# FSL_fan — 行星齿轮箱跨工况小样本故障诊断

基于原型网络（Prototypical Networks）和集成梯度（IG）归因的可解释小样本故障诊断方法，在风机实验台上验证。

---

## 项目概述

| 项目 | 内容 |
|------|------|
| **任务** | 189 类工况-故障组合的小样本分类 |
| **方法** | Prototypical Networks (PN) + CNN Encoder |
| **可解释性** | Integrated Gradients (IG) 归因 |
| **状态** | 基线已跑通，方法可优化 |

---


### 数据规格

| 属性 | 值 |
|------|----|
| 总文件数 | 189 个 .mat |
| 每个文件 | Y0 变量, 形状 (20, 1024), float32 |
| 总样本数 | 3,780 个独立样本 |
| 采样频率 | 100 kHz |
| 信号通道 | Ch3(振动X), Ch4(振动Y), Ch5(振动Z) |

### 类别定义

每个 .mat 文件为 1 个唯一类别，由故障类型 + 工况参数完整标记：

```
CxxLxSxxR10_Chx.mat
 ├── C00: 健康
 ├── C08: 太阳轮轻微点蚀
 ├── C09: 太阳轮中度点蚀
 ├── C10: 太阳轮半断齿
 ├── C11: 太阳轮全断齿
 ├── C12: 太阳轮中度双点蚀
 ├── C14: 太阳轮线切割凹槽
 ├── L1/L2/L3: 负载（1Ω/1.5Ω/10.5Ω）
 ├── S30/S40/S50: 转速（30Hz/40Hz/50Hz）
 └── Ch3/Ch4/Ch5: 通道（振动 X/Y/Z）
```

### 数据划分（标准 FSL：类别不交叉）

| 数据集 | 类别数 | 样本数 | 用途 |
|--------|--------|--------|------|
| **Train** | 153 | 3,060 | 训练 base classes |
| **Val** | 18 | 360 | 调参、选最优模型 |
| **Test** | 18 | 360 | 最终评测（从未见过） |
| **合计** | **189** | **3,780** | — |

**类别不交叉：** train / val / test 各自覆盖完全不同的类别，所有 .mat 文件名唯一。

---

## 实验流程

```
Step 1: 数据预处理   → 读取 .mat、解析标签、构建 189 类
Step 2: 全量 CNN      → 训练特征提取器（用作预训练）
Step 3: 原型网络      → 5-way 5-shot 小样本训练 + 评测
Step 4: IG 归因       → 可解释性可视化
Step 5: 对比实验      → 多设定评测（1-shot / 10-way）
Step 6: t-SNE 可视化  → 特征空间聚类展示
```

## 运行方式

```bash
# 按顺序依次执行
python step1_preprocess.py
python step2_train_cnn.py
python step3_train_fewshot.py
python step4_ig_analysis.py
python step5_experiments.py
python step6_tsne.py
```

### 环境要求

```bash
pip install torch numpy scipy matplotlib scikit-learn pyyaml
```

---

## 实验结果

### 6.1 小样本分类准确率

各设定在 **18 个全新类别（测试集）** 上运行 10 次取平均：

| 设定 | 准确率 | 标准差 |
|------|--------|--------|
| **5-way 5-shot** | **89.6%** | ±0.4% |
| 5-way 1-shot | 83.6% | ±0.6% |
| 10-way 5-shot | 79.3% | ±0.4% |
| 10-way 1-shot | 70.0% | ±0.4% |

**结论：** 在标准 5-way 5-shot 设定下达到 ~90% 准确率，且多次运行标准差 <1%，结果稳定可靠。1-shot 仍达 ~84%，证明模型具备较强的泛化能力。

### 6.2 基线全量 CNN 准确率

| 指标 | 值 |
|------|-----|
| 最高验证准确率 | 20.9% |
| 最终准确率 | 17.8% |

**说明：** 全量 CNN 仅在训练集（153 类）内部分 80/20 验证，且每类仅 16 个训练样本。低准确率属正常预期，CNN 在此仅作为特征提取器预训练使用，分类能力由 ProtoNet 提供。

### 6.3 IG 归因分析

IG 归因图显示各类别的特征贡献分布存在差异，证明模型依据的是物理故障特征而非噪声。

| 类 | pos_ratio | 峰值数 | CV | cr |
|----|-----------|--------|-----|-----|
| #0 | 53.1% | 30 | 1.46 | 0.21 |
| #1 | 56.5% | 53 | 1.19 | 0.25 |
| #2 | 54.7% | 58 | 1.21 | 0.27 |
| #3 | 51.2% | 43 | 2.35 | 0.45 |
| #4 | 53.1% | 53 | 1.09 | 0.22 |

![IG 归因图](outputs/ig_analysis/ig_attributions.png)

### 6.4 t-SNE 特征可视化

编码器提取的特征经 t-SNE 降维后，各类别在特征空间中形成清晰聚类。

**测试集 18 类特征分布：**

![t-SNE 测试集](outputs/tsne/tsne_test.png)

**训练集 vs 测试集对照：**

![t-SNE 对照图](outputs/tsne/tsne_comparison.png)

| 图 | 说明 |
|----|------|
| 左侧（蓝色） | 训练集 153 类特征分布 |
| 右侧（暖色） | 测试集 18 类特征分布，各类形成独立聚类 |

**结论：** t-SNE 可视化表明编码器提取的特征具有良好类间可分性，即使测试集类别在训练时从未见过，其特征也能与训练集特征形成区分。这从直观上解释了 ProtoNet 在 5-way 5-shot 下达到 89.6% 准确率的原因。


---

## 项目结构

```
FSL_fan/
├── configs/
│   └── baseline.yaml           ← 实验参数配置
├── data/                       
│   ├── train/     (153 .mat)
│   ├── test/      (18 .mat)
│   └── val/       (18 .mat)
├── src/
│   ├── data/
│   │   ├── preprocess.py       ← 数据预处理
│   │   └── dataset.py          ← 数据集 + EpisodicSampler
│   ├── models/
│   │   ├── encoder.py          ← CNN 特征提取器
│   │   ├── classifier.py       ← CNN1D 全量分类器
│   │   └── prototypical.py     ← 原型网络损失函数
│   ├── training/
│   │   ├── train_cnn.py        ← 全量 CNN 训练
│   │   └── train_fewshot.py    ← 小样本训练
│   └── interpret/
│       └── integrated_gradients.py  ← IG 归因
├── outputs/                   ← 输出结果（.gitignore 排除）
├── step1_preprocess.py        ← 步骤 1：数据预处理
├── step2_train_cnn.py         ← 步骤 2：全量 CNN 训练
├── step3_train_fewshot.py     ← 步骤 3：小样本训练
├── step4_ig_analysis.py       ← 步骤 4：IG 归因分析
├── step5_experiments.py       ← 步骤 5：多设定对比实验
├── step6_tsne.py              ← 步骤 6：t-SNE 特征可视化
├── .gitignore
└── README.md
```

---

## 后续优化方向

1. **特征提取器升级：** CNN → TCN / Transformer / GNN
2. **损失函数改进：** 加类内聚集损失 + 类间判别损失（参考 MLIIO）
3. **对比基线扩展：** 补充 Matching Network / MAML
4. **数据扩展：** 获取更多故障类型（C01~C35），扩充类别数
5. **域自适应：** 针对跨负载/跨转速场景做显式域对齐



## 许可证


