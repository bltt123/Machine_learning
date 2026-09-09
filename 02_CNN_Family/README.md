# 02 CNN 家族：图像分类主干演进

> 主线：LeNet → AlexNet → VGG → Inception → ResNet → DenseNet → MobileNet → EfficientNet → ConvNeXt
> 每一代都在解决上一代的问题：深度加深 → 梯度消失 → 残差 → 效率 → 现代化。

## 学习目标

1. 理解 CNN 基本组件：卷积、池化、感受野、参数量/FLOPs
2. 沿演进主线复现代表模型（小数据集上都可训练）
3. 重点吃透 ResNet 残差思想（现代所有大模型的基础件）
4. 建立效率视角：精度 vs 参数量 vs 速度

## 算法清单（学习检查表）

- [x] LeNet（1998，CNN 鼻祖）→ `01_LeNet_MNIST`（MNIST 98.88%，参数量仅 MLP 的 26.2%）
- [x] AlexNet（2012，ReLU + Dropout + GPU 训练）→ `02`（AlexNetMini 84.96% 领跑）
- [x] VGG（2014，3x3 小卷积堆叠）→ `02`（VGGMini 84.00%，无 BN 慢热难点如实入账）
- [x] NiN（选学：1×1 卷积 + 全局平均池化的起源）→ `02`（NiNMini 45.99%，无 BN 时代调参敏感的活教材）
- [x] GoogLeNet / Inception（2014，多尺度并行分支）→ `04`（Inception 轻量两块 39.08%，思想验证；参数仅 3.6 万）
- [x] ResNet（2015，残差连接）★重点 → `03_ResNet_CIFAR10`（五臂消融：残差增益随深度放大 +2.9→+15.4pt；退化复现 -9.25pt；BN 贡献 +10.6pt）
- [x] DenseNet（密集连接）→ `04`（DenseNetCIFAR 58.77%：55% 参数超 ResNet20 锚点 +1.63pt；拼接收益与 2.3× 耗时代价同账）
- [x] SENet（选学：通道注意力，04 项目顺带——通往 Transformer 注意力的桥）→ `04`（ResNetCIFAR 已支持 se=True 开关；04 的 SE 对照留待外部小实验，避免主线过长）
- [x] MobileNet（深度可分离卷积）→ `05`（MobileNet α=0.5 以 27% 参数 29% 计算反超 α=1 +0.64pt；分解账 7.9× 现场 verify）
- [x] EfficientNet（复合缩放）→ `05`（EffNet base 51.63% → 宽×2 56.33% → 复合 58.66%；0.20pt/10M 递减，如实呈现 mini 容量限制）
- [x] ConvNeXt（用 Transformer 经验改造 CNN，为 ViT 对比铺垫）→ `06`（阶梯：DW +3.94pt / 倒瓶颈 +5.07pt / 大核在小图上 -4.59pt 约束；ConvNeXt-mini 3.9M MACs 极端性价比，97 秒每臂）

## 场景速查（选型指南，数字均来自本家族同协议实测）

| 场景 | 推荐 | 支撑数据 | 指向项目 |
|---|---|---|---|
| 手写数字级小图 / CPU 秒级原型 | LeNet | MNIST 98.88%，61k 参数（MLP 的 26.2%） | `01_LeNet_MNIST` |
| 28×28 语义分类基线 | AlexNetMini | Fashion-MNIST 84.96% 领跑 | `02_AlexNet_VGG` |
| CPU 端到端默认 backbone | ResNet20/32（残差必开） | ResNet32 64.88%；残差增益随深度 +2.9→+15.4pt | `03_ResNet_CIFAR10` |
| 参数预算受限（<60% 参数） | DenseNet | 58.77% @150k（ResNet20 的 55% 参数超锚点 +1.63pt），但吞吐 43% | `04_DenseNet_Inception` |
| 计算预算受限（MACs<10M） | MobileNet α=0.5 | 54.34% @2.8M MACs（ResNet 的 7%）；且小数据上反超 α=1 +0.64pt | `05_Lightweight` |
| 多尺度输入 / 降维再卷 | Inception 分支思想 | 轻量两块 39.08% 思想验证；1×1 已成全家族标配 | `04_DenseNet_Inception` |
| 现代化翻新 / 对齐 Transformer 工程习惯 | ConvNeXt 倒瓶颈形态 | 倒瓶颈臂 58.45% @120k/21.1M；大核+patchify 在 32px 小图负收益——小数据别盲从大核 | `06_ConvNeXt_vs_ViT` |
| ViT / 注意力主干 | 预留 | 待 `06_Transformer_Vision_Multimodal` 家族同协议回填 | — |

> 表的用法：先按约束（参数/算力/数据量）挑行，点开"指向项目"看消融曲线与 FAQ 再定参；同协议数字可直接互比（03 起锚点链）。

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_LeNet_MNIST` | LeNet | MNIST | 复现经典 |
| 02 | `02_AlexNet_VGG_FashionMNIST` | AlexNet, VGG, NiN | Fashion-MNIST / CIFAR-10 | 深度加深时 VGG 训练难点；数据增强首秀 |
| 03 | `03_ResNet_CIFAR10` | ResNet18/34 | CIFAR-10 | 有/无残差对比 ★ |
| 04 | `04_DenseNet_Inception_CIFAR10` | DenseNet, Inception | CIFAR-10 | 连接方式对比 |
| 05 | `05_Lightweight_MobileNet_EfficientNet` | MobileNet, EfficientNet | CIFAR-10 | Acc-FLOPs 权衡曲线 |
| 06 | `06_ConvNeXt_vs_ViT`（收官） | ConvNeXt 阶梯 + ViT 预留 | CIFAR-10 | 现代化路径 DW→倒瓶颈→大核；ViT 同台预留 06 家族 |
| 07 | 05 已交付（额外） | MobileNet/EfficientNet 完整 | 已入 05 | 型号意义不大，见 05 README 注 |

## 与其他家族的关系

- **前置**：01 基础训练范式
- **后续**：03_CNN_Segmentation_Detection（检测分割）、06_Transformer_Vision_Multimodal（ViT 是 CNN 的"去卷积化"对手）

## 进度

- ✅ `01_LeNet_MNIST`（LeNet-5 现代复现：MNIST 98.88% / 答错 112，参数量 61,706 = MLP 的 26.2%；特征图可视化 + MLP 正面对比）
- ✅ `02_AlexNet_VGG_FashionMNIST`（Fashion-MNIST 语义级难点：AlexNetMini 84.96% 领跑 / VGGMini 慢热 / NiNMini 无 BN 难训 45.99%；数据增强首秀：3ep -6.60% → 10ep +2.51% 转正，增强序列可复现）
- ✅ `03_ResNet_CIFAR10` ★（五臂受控消融：ResNet32 64.88% 冠军；残差增益随深度放大 +2.9→+15.4pt；Plain 退化实锤 -9.25pt（train 也掉 → 优化失败）；BN 贡献 +10.6pt；协议演进实录：Adam(1e-3) 会让残差"被失效"）
- ✅ `04_DenseNet_Inception_CIFAR10`（三拓扑同台：DenseNet 58.77% 以 55% 参数超 ResNet 锚点 +1.63pt，但吞吐仅 43%；Inception 轻量两块 39.08% 思想验证）
- ✅ `05_Lightweight_MobileNet_EfficientNet`（效率时代收口：分解账 7.9×；MobileNet α=0.5 以 27% 参数反超 +0.64pt；EffNet 复合缩放 51.63→58.66% 但 0.20pt/10M 递减；Acc-FLOPs 双图性价比地图落盘）
- ✅ `06_ConvNeXt_vs_ViT`（收官：DW +3.94pt / 倒瓶颈 +5.07pt / 大核在 CIFAR 小图上 -4.59pt 约束；ConvNeXt-mini 3.9M MACs 极端性价比；ViT 同台为 06 家族预留延续点）

全家族 01-06 六站 30 张图落盘，总检六项（结构/图引闭环/数字抽查/模型实例化/数据缓存/脏文件）全部通过；场景速查表见"算法清单"末尾。
