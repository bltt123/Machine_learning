# 07 生成式模型家族：Autoencoder / VAE / GAN / Diffusion / Flow

> 生成式模型是一条非常清晰的演进线：重建 → 概率生成 → 对抗生成 → 扩散生成。

## 学习目标

1. 理解“编码-解码”与“生成分布”的基本思想
2. 认识 VAE 的重参数技巧、GAN 的对抗训练、Diffusion 的逐步去噪
3. 通过小数据集实现可视化生成实验

## 算法清单（学习检查表）

- [ ] Autoencoder
- [ ] VAE
- [ ] GAN（可含 DCGAN）
- [ ] Diffusion Model（DDPM）
- [ ] Normalizing Flow（可选拓展）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_AE_VAE_MNIST` | AE, VAE | MNIST | 重建 vs 采样 |
| 02 | `02_GAN_MNIST_CIFAR10` | GAN, DCGAN | MNIST / CIFAR-10 | 生成效果对比 |
| 03 | `03_Diffusion_MNIST` | Diffusion | MNIST | 逐步去噪可视化 |
| 04 | `04_Flow_Optional` | Flow-based | toy / 小图像 | 可选拓展 |

## 与其他家族的关系

- **前置**：01 的 Autoencoder 思想、02 的卷积骨干、05/06 的 Transformer 生成能力
- **后续**：Diffusion 中常见 U-Net 骨干与 03 分割家族呼应
