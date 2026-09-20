# 深度学习算法练习项目总览（Deep Learning Practice Roadmap）

> **核心原则**：按"家族"学、按"任务/数据集"练、按"代表模型"做对比。
> 每个家族一个目录，家族内按"项目（编号）"推进；每个项目最终沉淀一份 README：原理 → 代码 → 实验结果 → 误差分析 → 总结。

---

## 一、学习路线（总览）

| 顺序 | 目录 | 主线 | 覆盖算法（代表） | 练习任务与数据集 |
|---|---|---|---|---|
| 1 | `01_Fundamentals_MLP` | 打地基 | 感知机、MLP、反向传播、激活/归一化/优化器/正则化 | MNIST / Fashion-MNIST 分类 |
| 2 | `02_CNN_Family` | 视觉主干演进 | LeNet → AlexNet → VGG → Inception → ResNet → DenseNet → MobileNet → EfficientNet → ConvNeXt | MNIST / Fashion-MNIST / CIFAR-10 分类 |
| 3 | `03_CNN_Segmentation_Detection` | 视觉下游任务 | FCN、U-Net、YOLO、Faster R-CNN、RetinaNet、Mask R-CNN | 图像分割（Oxford-IIIT Pet）、目标检测 |
| 4 | `04_Sequence_Models` | 序列建模演化 | **HMM → CRF → RNN → LSTM/GRU → Seq2Seq → Attention** | 分词/NER/文本分类/小翻译 |
| 5 | `05_Transformer_NLP` | Transformer 主干 | Transformer（自实现）、BERT、GPT、T5、BART、（拓展对照：**SSM/Mamba**） | 文本分类/小语料 LM/摘要、Mamba vs Transformer 对照 |
| 6 | `06_Transformer_Vision_Multimodal` | 视觉与多模态 | ViT、CLIP、BLIP、LLaVA | CIFAR-10、图文检索 zero-shot |
| 7 | `07_Generative_Models` | 生成式演进 | AE → VAE → GAN → Diffusion（→ Flow） | MNIST/Fashion-MNIST 生成 |
| 8 | `08_Production_Optimization` | 生产级工程 | RoPE、KV Cache、MQA/GQA、FlashAttention、LoRA/QLoRA、量化/蒸馏/剪枝、Mixed Precision、Gradient Checkpointing、MoE、RAG、**推理部署（PagedAttention/vLLM、投机解码）** | 小实验 + 原理总结 |
| 9 | `09_Domain_Models` | 领域应用落地 | 时序：Informer/PatchTST；推荐：DeepFM/DIN；语音：Whisper | 时序预测（ETT/PM2.5）、MovieLens 推荐、Whisper 转写体验 |

> **推荐顺序**：1 → 2/4 可并行 → 5 → 6/7 → 8 → 9；03 可放在 2 之后任意时间；09 领域应用也可按兴趣穿插进行。

---

## 二、目录结构

```
Machine_learning/
├── AGENTS.md                          # 项目工作规则（必读）
├── README.md                          # 本文件：总路线图 + 进度追踪
├── package.md                         # 旧环境/历史记录
├── 01_Fundamentals_MLP/
│   └── README.md                      # 家族学习计划
├── 02_CNN_Family/
│   └── README.md
├── 03_CNN_Segmentation_Detection/
│   └── README.md
├── 04_Sequence_Models/
│   └── README.md
├── 05_Transformer_NLP/
│   └── README.md
├── 06_Transformer_Vision_Multimodal/
│   └── README.md
├── 07_Generative_Models/
│   └── README.md
├── 08_Production_Optimization/
│   └── README.md
└── 09_Domain_Models/
    └── README.md
```

---

## 三、进度追踪表

| # | 家族 | 状态 | 已完成项目 | 关键收获（一句话） |
|---|------|------|-----------|-------------------|
| 01 | Fundamentals_MLP | ✅ 已完成 | `01_Perceptron` ✅（手写感知机，Iris 100%/XOR 失效）；`02_MLP_MNIST` ✅（MNIST 98.14%）；`03_Training_Tricks_Ablation` ✅（14 组消融，清单 8 项收口） | 线性天花板 → 非线性突破 → 技巧各归其位；控制变量才能归因 |
| 02 | CNN_Family | ✅ 已完成 | 01-06 六站全部 ✅（LeNet 98.88% / 增强转正 / 五臂残差消融 / DenseNet 拼接 / 分解 7.9× / ConvNeXt 阶梯）；场景速查表就位，ViT 同台留 06 家族回填 | 归纳偏置省参数；残差越深越值钱；分解与倒瓶颈是效率真红利 |
| 03 | CNN_Segmentation_Detection | ✅ 已完成 | `01_FCN_UNet_Segmentation` ✅（自训 U-Net mIoU 0.5409 vs FCN 0.5273，Dice +0.0315）；`02_Detection_torchvision` ✅（COCO 预训练推理 Faster 6/Retina 5，NMS 手写一致）；`03_YOLO_Concept` ✅（Grid/NMS/解码思想实验，5 图）——全家 15 图 3×README 10 节闭环 | 涂色→画框→快检三步走；分割靠跳跃、检测靠框IoU、实时靠网格 |
| 04 | Sequence_Models | ✅ 已完成 | `01_HMM_CRF_NER` ✅（CRF 1.0 vs HMM 0.875）；`02_RNN_LSTM_GRU_TextCls` ✅（T=30 长 0.71 vs 短 0.965，GRU 76%参换 0.695）；`03_Seq2Seq_Attention_MT` ✅（复制 0.867/翻转 0.782，对角/反对角对齐） | 统计手拉手 → 记忆传话 → 开卷对齐；顺序一反意思全反 |
| 05 | Transformer_NLP | ✅ 已完成 | 01-05 全 ✅（25 图：Transformer 手写 MHA/因果mask/PE；BERT 双向；GPT 因果；T5/BART 去噪；Mamba 线性vs二次对照） | 传话改开会，全员同时互相看；编码填空、解码续写、编码器-解码器去噪 |
| 06 | Transformer_Vision_Multimodal | ✅ 已完成 | 01-03 全 ✅（15 图：ViT 小数据 0.3614 vs CNN 0.4096；CLIP zero-shot 0.092 vs 监督 0.485；LLaVA 三段式 toy 接线） | 图切块当字读；图文双塔拉近推远；投影层当视神经 |
| 07 | Generative_Models | ✅ 已完成 | 01-04 全 ✅（20 图：AE MSE 80.6 vs VAE ELBO 118；DCGAN 多样性胜 MLP-GAN；DDPM epsilon 0.55→0.13；Flow NLL 439→-728 往返 8.9e-07） | 重建 → 概率生成 → 对抗生成 → 扩散去噪 → 可逆似然 |
| 08 | Production_Optimization | ✅ 已完成 | 01-09 全 ✅（32 图：RoPE 外推/Cache/GQA/Flash/LoRA/ckpt/INT8/MoE 激活1/4/RAG 三1.0/投机5.00 tok每轮） | 不训新模型，给 Transformer 动手术：省算省搬省参省显存省体积 |
| 09 | Domain_Models | ✅ 已完成 | 01-03 全 ✅（9 图：ETT LSTM MSE 0.0936 胜 toy Transformer；MovieLens DIN 0.6603 历史增益+0.0859；Whisper tiny 英WER 0.1786/中字级 0.5862） | 领域×主干出口：看表看人听声；短窗先 LSTM，有历史用 DIN，中文上 base |

> 状态标记：⬜ 未开始 / 🔵 进行中 / ✅ 已完成

---

## 四、算法覆盖总清单（对照你的临时记逐项核对）

### 基础层 → 01_Fundamentals_MLP
Perceptron、MLP、反向传播、激活函数（ReLU/GELU/SiLU/Sigmoid/Tanh）、BatchNorm/LayerNorm/RMSNorm、SGD/Momentum/Adam/AdamW、Dropout/Weight Decay/Label Smoothing、常见损失函数

### CNN 分类主干 → 02_CNN_Family
LeNet、AlexNet、VGG、GoogLeNet/Inception、ResNet、DenseNet、MobileNet、EfficientNet、ConvNeXt

### CNN 检测/分割 → 03_CNN_Segmentation_Detection
FCN、U-Net、YOLO 系列、Faster R-CNN、RetinaNet、Mask R-CNN

### 序列建模 → 04_Sequence_Models（含传统方法）
**HMM、CRF**、RNN、LSTM、GRU、Seq2Seq、Attention

### Transformer / NLP → 05_Transformer_NLP
Transformer 自实现（MHA、Masked/Causal Attention、Positional Encoding、Pre-LN/Post-LN、残差）、BERT、GPT、T5、BART、**SSM/Mamba**（架构对照拓展，纯 PyTorch mini 实现）

### 视觉 / 多模态 → 06_Transformer_Vision_Multimodal
ViT、CLIP、BLIP、LLaVA

### 生成式 → 07_Generative_Models
Autoencoder、VAE、GAN（含 DCGAN）、Diffusion（DDPM）、Normalizing Flow（可选）

### 生产级工程 → 08_Production_Optimization
Positional Encoding 对比、**RoPE**、**KV Cache**、**MQA/GQA**、**FlashAttention**、Sparse Attention、Pre-LN/Post-LN 复盘、LoRA/QLoRA/PEFT、量化（INT8/INT4）、剪枝、知识蒸馏、Mixed Precision、Gradient Checkpointing、MoE、RAG、**推理部署**（PagedAttention/vLLM 原理、投机解码）

### 领域应用 → 09_Domain_Models
时序预测：Informer、PatchTST；推荐系统：DeepFM、DIN；语音：Whisper（tiny 版推理 / 可选微调）

---

## 五、项目 README 统一模板（每个项目最终沉淀）

1. **任务背景与目标**
2. **模型/算法原理**（含家族演化位置：解决了前代什么问题）
3. **网络结构图**（手绘/ASCII/mermaid）
4. **核心公式**（关键 3~5 个即可）
5. **数据集介绍与预处理**
6. **训练过程与超参数**
7. **实验结果**（表格对比：家族内多模型同台）
8. **误差分析与可视化**
9. **总结与改进方向**
10. **场景速查**（实践推荐：什么场景用什么，一句话定位）

### 代码文件组织规范（Notebook 为主 + 家族级公共模块）

> 原则：**notebook 做学习与实验，公共逻辑抽成模块，绝不每个项目复制一遍训练代码。**

每个家族目录内的标准结构：

```
0X_家族/
├── README.md
├── common/                  # 家族级公共模块，全家族只写一次
│   ├── data.py              # 数据集加载 / 预处理 / 增强
│   ├── models.py            # 家族内所有模型定义（体现"家族"）
│   ├── engine.py            # 通用训练 / 评估循环
│   └── utils.py             # 随机种子、曲线绘图、参数量统计等
├── 01_子项目/
│   └── main.ipynb           # 学习主载体：实验、可视化、结论
└── 02_子项目/
    └── main.ipynb
```

- `.ipynb` 是主载体：import `common` 后跑实验、画图、记录结论（环境已有 ipykernel，Cursor 直接可跑）
- `.py` 只放公共逻辑，写一次全家族复用；08 这类"机制实验集"同样一个 `common/` + 每机制一个 notebook
- 生产视角：这正是企业"notebook 探索 → src 模块化"的两段式；将来需要批量跑时，再从 notebook 抽一个十几行的 `run.py` 入口即可
- git 提交前建议清理 notebook 输出，减少 diff 噪音

### 项目 README 模板约定

> 每个子项目的 README 固定 10 节 + 三件固定插件，新项目照此写，读者知道去哪找什么：

1. **§2 原理节开头 → 「通俗理解」**：一句话版 + 生活比喻（轻量版 2~3 句，先给直觉再上术语）
2. **§8 误差分析末尾 → 「常见坑 FAQ」**：本项目实测踩过/险些踩过的坑，新手少走弯路
3. **§9 总结内 → 「实际应用场景」**：谁在用、用在哪、历史现场——建立"江湖地位"认知
4. **§10 场景速查**：实践推荐表（场景/推荐/支撑数据/要点），读者直接抄作业
5. 全部数字必须来自 notebook 实跑输出；图片存 `figs/` 并在 README 引用

---

## 六、数据集与下载建议

- **图像分类**：MNIST、Fashion-MNIST、CIFAR-10（torchvision 自动下载；如遇网络问题用 Kaggle/ModelScope 镜像后放本地目录）
- **分割**：Oxford-IIIT Pet（torchvision 内置分割掩码）
- **检测**：先用 torchvision 预训练模型推理 + 小数据微调
- **NLP**：IMDB、ChnSentiCorp（中文情感）、CoNLL-2003（NER）、Tatoeba（小翻译对）——走 HuggingFace 镜像或 ModelScope
- **生成**：MNIST / Fashion-MNIST 起步，进阶 CelebA
- **时序**：ETT-small（ETTh1，GitHub 可下）、北京 PM2.5（UCI/Kaggle）
- **推荐**：MovieLens-100K（grouplens/Kaggle，小而经典）
- **语音**：自备音频文件 / Common Voice 小样本子集
- **原则**：**先用小数据集跑通，再考虑加大**；所有下载文件放对应项目目录内，严禁写 C 盘

---

## 七、环境与规则（摘要，详见 AGENTS.md）

- **conda 环境**：`LLM_learning`（Python 3.9 + PyTorch 2.5.1+cpu + transformers 4.47 + torchvision 0.20）
- **硬件**：RTX 3050 Laptop GPU（4GB 显存）但当前环境为 CPU 版 PyTorch
- **解释器**：`D:\LeStoreDownload\Anaconda1\envs\LLM_learning\python.exe`
- **严禁向 C 盘写入任何文件**；不擅自安装库（如 08 需要 `peft`、`seqeval` 等会先征得同意）
- **未经要求不执行 git 提交/推送**
- **CPU 优先设计**：所有项目按 CPU/小模型设计；如需 GPU，需重新安装 CUDA 版 PyTorch

---

## 八、拓展选学（暂不排期，按需再开）

> 主线（01~09）已覆盖必学主干与重点选学。以下为深度学习版图中确实存在、但**当前明确不排期**的方向：

| 方向 | 代表内容 | 说明 | 建议时机 |
|---|---|---|---|
| 强化学习 / 对齐 | DQN、Policy Gradient、PPO、RLHF/GRPO、DPO | **按你的计划：深度学习主线结束后单独开项，本路线不涉及** | 深度学习结束后单独规划 |
| 自监督预训练 | SimCLR、MoCo、BYOL、MAE、DINO | ViT/CLIP 的底层思想，做完 06 后理解会很顺 | 学完 06 可选 |
| 图神经网络 GNN | GCN、GAT、GraphSAGE | 完全独立的分支（图数据），NLP/CV 用不到 | 有图数据需求时 |
| 度量学习 | Siamese、Triplet Loss、对比损失 | 人脸识别/检索类任务的基础 | 有相关任务时 |

---

## 九、术语速查表（一页通）

> 回看旧项目时"这词当时啥意思"的急救页。按概念族分组，详见各项目 README 的通俗理解与 FAQ。

### 数据与训练单元

| 术语 | 通俗解释 |
|---|---|
| epoch / batch / iteration | 全部数据过一遍 / 一小撮（如 128 张）一起喂 / 喂一个 batch 算一次梯度。60k 数据 batch=128 → 1 epoch ≈ 469 iterations |
| 标准化 (Normalization) | 把像素从 0~255 压到均值 0 附近，让优化器不用"先学会读大数字"。MNIST 用 (0.1307, 0.3081) |
| 训练/验证/测试集 | 上课用 / 平时小测（调参依据） / 高考（只考一次）。拿测试集调参 = 背考题 |
| 数据增强 | 对训练图做保标签的随机变换（翻转/平移），等效免费扩容。**收益取决于训练时长**（02 实测：3ep 有害、10ep 转正） |

### 模型与表达

| 术语 | 通俗解释 |
|---|---|
| 假设空间 | 模型"能想到的所有候选答案"的范围。感知机的假设空间只有直线——答案不在里面，再努力也没用（01 核心） |
| 归纳偏置 | 模型出厂自带的"先验直觉"。CNN 的权值共享 = "图案在哪出现都一样"——好偏置用更少参数买到更高精度（LeNet 26% 参数反超） |
| 感受野 | 一个神经元能"看见"的输入区域。层层堆叠池化，浅层看笔画、深层看结构（LeNet 特征图可视化） |
| 参数量 / FLOPs | 旋钮个数 / 需要的计算次数。前者决定内存，后者决定速度 |
| 万能近似定理 | 单隐层够宽就能逼近任意连续函数——但只保证"存在"，不保证 SGD 找得到、更不保证经济性（03 PartB2） |

### 训练动力学

| 术语 | 通俗解释 |
|---|---|
| 损失函数 | "错了多少"的打分器。分类用 CrossEntropy（内置 softmax 数值稳定版），回归才用 MSE |
| 反向传播 | 按"对错误的贡献度"层层追责到每个参数的算法 = 计算图上的链式法则（03 PartA 手推验证到 1e-8） |
| 学习率 lr | 每次拧旋钮的幅度。**太大学飞、太小爬**——SGD(1e-3) 只有 58%，换 0.1 追平 Adam（03 实测） |
| 过拟合 / 欠拟合 | 背题（train 高 val 低）/ 没学进去（两者都低）。看 train-val 差距与 val 曲线拐点 |
| 梯度消失/爆炸 | 追责传到深层时责任书缩成 0 / 膨胀成 ∞。深层网络训不动的经典病因（sigmoid 饱和、无 BN 深堆叠） |
| dead ReLU | ReLU 神经元对某些输入永久输出 0 → 梯度也是 0 → 再也学不动。03 实测 42.3% 激活为零 |
| 收敛 | 训练到"再训也差不多"的稳定状态。感知机的收敛另有严格含义：整轮零误分 |

### 优化器与正则化

| 术语 | 通俗解释 |
|---|---|
| SGD / Momentum | 只看当前坡度下山 / 加上惯性滑行——惯性帮它冲过小坑洼（03 实测 58% → 89%） |
| Adam / AdamW | 给每个参数配自适应步长的优化器 / 修正版（weight decay 与梯度解耦，大模型默认） |
| Dropout | 训练时随机让一部分神经元"下班"，逼全网络成为多面手。推理时全员上班（所以要 `model.eval()`） |
| Weight Decay | 训练时持续把参数往 0 轻推，偏爱简单模型。**过强会反噬**（03 实测 -0.36pt） |
| Label Smoothing | 把"100% 是猫"软化为"90% 是猫"，防过度自信 |
| 早停 (Early Stopping) | val 连续 N 轮无提升就刹车——省算力，本身就是正则化 |

### 归一化层（易与"标准化"混淆）

| 术语 | 通俗解释 |
|---|---|
| BatchNorm | 用**同 batch 其他样本**的统计量给每层输入"调音"——训练/推理行为不一致、小 batch 下失灵（batch=2 暴跌 21.6pt） |
| LayerNorm / RMSNorm | 用**自己这一条样本**的特征维统计量——与 batch 无关，Transformer 的选择 |
| 归一化层 vs 标准化 vs 正则化 | 归一化层稳定训练过程（网络内的零件）；标准化预处理输入数据；正则化防过拟合（Dropout/wd/早停这一族） |
