# 01 基础网络家族：感知机与 MLP（Fundamentals）

> 深度学习的"地基"：从这里建立训练全流程的心智模型——前向传播 → 损失 → 反向传播 → 优化器 → 评估。

## 学习目标

1. 理解感知机 → MLP 的演进（线性不可分问题）
2. 手推/手写反向传播（NumPy 版可选，PyTorch autograd 对照验证）
3. 掌握激活函数、归一化、优化器、正则化的作用与选型
4. 建立 PyTorch 标准训练范式（Dataset/DataLoader/训练循环/评估）

## 算法清单（学习检查表）✔ 全部完成

- [x] Perceptron（感知机、线性可分、无法解 XOR）→ `01_Perceptron`
- [x] MLP（隐藏层、万能近似定理）→ `02_MLP_MNIST` + `03` PartB2（sin 拟合 mse=0.00019）
- [x] 反向传播（链式法则、计算图）→ `03` PartA（NumPy 手推 vs autograd，误差 1e-8）
- [x] 激活函数：Sigmoid / Tanh / ReLU / GELU / SiLU（dead ReLU 问题）→ `03` PartD（ReLU 死亡比例 42.3% 实测）
- [x] 归一化：BatchNorm / LayerNorm / RMSNorm（为什么 Transformer 用 LN 而不是 BN）→ `03` PartC②+PartE（batch=2 时 BN -21.6pt / LN 无感）
- [x] 优化器：SGD / Momentum / RMSProp / Adam / AdamW → `03` PartC①（SGD lr 是命门：58%→96%）
- [x] 正则化：Dropout / Weight Decay / Label Smoothing / 早停 → `02` + `03` PartC③（wd 过强会反噬）
- [x] 损失函数：MSE / CrossEntropy（softmax 数值稳定版）→ `03` PartB（朴素 softmax nan 现场 + log-sum-exp）

## 项目规划

| 编号 | 项目 | 覆盖内容 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_Perceptron` | 感知机、线性可分 | 玩具二维数据 / Iris | 手写感知机，演示 XOR 失效 |
| 02 | `02_MLP_MNIST` | MLP + 完整训练流程 | MNIST / Fashion-MNIST | 98%+ 准确率基线 |
| 03 | `03_Training_Tricks_Ablation` | 优化器/归一化/正则化对比 | 同上（同一 MLP） | SGD vs Adam vs AdamW；有无 BN/LN/Dropout 的对比曲线 |

## 学习顺序

01 → 02 → 03

## 进度

- ✅ `01_Perceptron`（手写感知机：线性可分 2 轮收敛 100% / XOR 不收敛 ~56% / Iris 测试 100%，sklearn 对照一致）
- ✅ `02_MLP_MNIST`（784-256-128-10 + Dropout0.2，MNIST 测试 98.14%；XOR 同台对照：感知机 46.8% vs MLP 96.0%；超参小网格：无 Dropout 97.98% → 过拟合信号）
- ✅ `03_Training_Tricks_Ablation`（14 组消融：BN +1.85pt 最大单项；SGD lr 命门 58%→96%；dead ReLU 42.3% 实测；BN batch=2 暴跌 21.6pt；手推反向传播 vs autograd 误差 1e-8。**家族清单 8 项全部收口**）

## 与其他家族的关系

- **后续**：02_CNN_Family（加归纳偏置）、05_Transformer_NLP（LN/优化器的现代用法）
