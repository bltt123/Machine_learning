# 02 CNN 家族：图像分类主干演进

> 主线：LeNet → AlexNet → VGG → Inception → ResNet → DenseNet → MobileNet → EfficientNet → ConvNeXt
> 每一代都在解决上一代的问题：深度加深 → 梯度消失 → 残差 → 效率 → 现代化。

## 学习目标

1. 理解 CNN 基本组件：卷积、池化、感受野、参数量/FLOPs
2. 沿演进主线复现代表模型（小数据集上都可训练）
3. 重点吃透 ResNet 残差思想（现代所有大模型的基础件）
4. 建立效率视角：精度 vs 参数量 vs 速度

## 算法清单（学习检查表）

- [ ] LeNet（1998，CNN 鼻祖）
- [ ] AlexNet（2012，ReLU + Dropout + GPU 训练）
- [ ] VGG（2014，3x3 小卷积堆叠）
- [ ] GoogLeNet / Inception（2014，多尺度并行分支）
- [ ] ResNet（2015，残差连接）★重点
- [ ] DenseNet（密集连接）
- [ ] MobileNet（深度可分离卷积）
- [ ] EfficientNet（复合缩放）
- [ ] ConvNeXt（用 Transformer 经验改造 CNN，为 ViT 对比铺垫）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_LeNet_MNIST` | LeNet | MNIST | 复现经典 |
| 02 | `02_AlexNet_VGG_FashionMNIST` | AlexNet, VGG | Fashion-MNIST / CIFAR-10 | 深度加深时 VGG 训练难点 |
| 03 | `03_ResNet_CIFAR10` | ResNet18/34 | CIFAR-10 | 有/无残差对比 ★ |
| 04 | `04_DenseNet_Inception_CIFAR10` | DenseNet, Inception | CIFAR-10 | 连接方式对比 |
| 05 | `05_Lightweight_MobileNet_EfficientNet` | MobileNet, EfficientNet | CIFAR-10 | Acc-FLOPs 权衡曲线 |
| 06 | `06_ConvNeXt_vs_ViT`（预留） | ConvNeXt | CIFAR-10 | 与 06 目录的 ViT 同台对比 |

## 与其他家族的关系

- **前置**：01 基础训练范式
- **后续**：03_CNN_Segmentation_Detection（检测分割）、06_Transformer_Vision_Multimodal（ViT 是 CNN 的"去卷积化"对手）
