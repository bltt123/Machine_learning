# 07 生成式模型家族：Autoencoder / VAE / GAN / Diffusion / Flow

> **一句话定位**：前面 01~06 全是“判别”（图进→标签出：这是什么），07 反过来做 **“生成”**（潜变量进→图出：画一个出来）——演进线：**重建(AE) → 概率生成(VAE) → 对抗生成(GAN) → 扩散生成(Diffusion)**。
> 状态：✅ 01-04 全完成（07 家族收官，含可选拓展）| 数据：MNIST 子集（02 家族本地缓存，无外网）| 环境：`LLM_learning`（torch 2.5.1+cpu；4 站 20 图）

## 进度

- ✅ `01_AE_VAE_MNIST`（ConvAE 271k test MSE 80.6 vs ConvVAE 371k ELBO 118；走格渐变+随机采样全新数字；5 图，CPU 55s）
- ✅ `02_GAN_MNIST_CIFAR10`（MLP-GAN vs DCGAN 同 MNIST 6000/20ep 同台：G 终值 3.47 vs 2.76、D(real)≈0.9、多样性 std/pdist 0.24/6.7 vs 0.31/10.1——卷积保局部性红利 measured，灰雾期如实记录；5 图，CPU 230s）
- ✅ `03_Diffusion_MNIST`（DDPM UNet-lite 108k：T=100 12ep epsilon 0.55→0.13 + T=20 4ep 对照；前向雾化+去噪轨迹+末期样本+T 对比 5 图，CPU 160s）
- ✅ `04_Flow_Optional`（RealNVP 6 耦合层 2.2M：8ep NLL 439→-728，往返 8.9e-07，test NLL -710 精确似然+似然直方图；5 图，CPU 32s）

## 场景速查（生成家族选型）

| 场景 | 推荐 | 支撑数据 | 要点 |
|---|---|---|---|
| 复现/压缩/异常检测 | AE | 07-01 test MSE 80.6 | 确定性=锐 |
| 连续潜空间编辑/插值 | VAE | 07-01 走格渐变+采样新图 | KL 换连续性 |
| 高保真图像生成 | DCGAN 起步 | 07-02 G 2.76、多样性 1.6× | 对抗最难稳，看多样性别只看 loss |
| 训练稳、逐步去噪 | Diffusion | → 07-03 | 采样慢但稳 |

## 通俗理解：从“复印”到“画画”

| 阶段 | 通俗说 | 类比 | 家族内落点 |
| :--- | :--- | :--- | :--- |
| **AE** | 压成小纸条再展开，看丢了多少 | 复印机：28×28 压 32 维再还原 | `01`：ConvAE 271k，重建锐（test MSE 标准化域 80.6） |
| **VAE** | 纸条改成“骰子”（均值+方差采样），摇出新图 | 摇骰子复印机：KL 压成正态小区 | `01`：ConvVAE 371k，走格渐变+随机采样全新数字 |
| **GAN** | 造假画的 vs 鉴假的，互卷 | 印假钞 vs 验钞机 minimax | `02`：MLP-GAN vs DCGAN 同台，G 3.47 vs 2.76、多样性 1.6×，D(fake)≈0.1 灰雾期如实记录 |
| **Diffusion** | 一步步加噪再一步步擦干净 | 雾玻璃擦雾：T 步去噪链 | `03`：DDPM 108k 猜雾 epsilon 0.13，去噪轨迹+末期样本，训练稳 vs GAN 震荡 |
| **Flow** | 可逆变形记，能算概率 | 双向拉链：z↔x 一一对应 | `04`：RealNVP 2.2M 往返 8.9e-07、NLL 精确似然，似然直方图（GAN 给不出、VAE 只给下界） |

## 学习目标

1. 理解“编码-解码”与“生成分布”的基本思想
2. 认识 VAE 的重参数技巧、GAN 的对抗训练、Diffusion 的逐步去噪
3. 通过小数据集实现可视化生成实验

## 算法清单（学习检查表）

- [x] Autoencoder → `01_AE_VAE_MNIST`（ConvAE 271k，32 维瓶颈，test 重建 MSE 80.6 标准化域）
- [x] VAE → `01`（ConvVAE 371k，重参数+KL，ELBO 118，走格渐变+随机采样新图）
- [x] GAN（可含 DCGAN）→ `02_GAN_MNIST_CIFAR10`（MLP 551k/534k vs DCGAN 553k/139k，同 z 同 20ep：G 3.47 vs 2.76、多样性 1.6×，卷积红利；D(fake)≈0.1 灰雾期诚实记录）
- [x] Diffusion Model（DDPM）→ `03_Diffusion_MNIST`（UNet-lite 108k，一步加噪+猜雾+反向链：epsilon 0.55→0.13，T=100/20 对比，轨迹+样本 5 图）
- [x] Normalizing Flow（可选拓展→已做）→ `04_Flow_Optional`（RealNVP 6 耦合层 2.2M：往返 8.9e-07、NLL train -756/test -710 精确似然，似然直方图 5 图，CPU 32s）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 | 状态 |
|---|---|---|---|---|---|
| 01 | `01_AE_VAE_MNIST` | AE, VAE | MNIST | 重建 vs 采样 | ✅（AE 80.6 / VAE 走格+采样，5 图，CPU 55s） |
| 02 | `02_GAN_MNIST_CIFAR10` | GAN, DCGAN | MNIST / CIFAR-10 | 生成效果对比 | ✅（G 2.76 vs 3.47、多样性 1.6×，5 图，CPU 230s） |
| 03 | `03_Diffusion_MNIST` | Diffusion | MNIST | 逐步去噪可视化 | ✅（epsilon 0.13、轨迹+样本，5 图，CPU 160s） |
| 04 | `04_Flow_Optional` | Flow-based | toy / 小图像 | 可选拓展 | ✅（往返 8.9e-07、NLL 精确似然，5 图，CPU 32s） |

## 与其他家族的关系

- **前置**：01 的 Autoencoder 思想、02 的卷积骨干、05/06 的 Transformer 生成能力
- **后续**：Diffusion 中常见 U-Net 骨干与 03 分割家族呼应
