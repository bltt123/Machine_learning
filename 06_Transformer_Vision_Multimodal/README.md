# 06 Transformer 视觉与多模态：ViT / CLIP / BLIP / LLaVA

> **一句话定位**：05 让机器“读书”（字→字），06 让机器 **“看图”（像素→字）+“图文对上”（图和字互相指）**——Transformer 跨出 NLP 的第一步。
> 状态：✅ 01-03 全完成（06 家族收官）| 数据：CIFAR-10 子集（02 家族本地缓存，无外网）| 环境：`LLM_learning`（torch 2.5.1+cpu；01 5图 + 02 5图 + 03 5图 = 15 图）

## 通俗理解：从“字序列”到“图块序列”

| 阶段 | 通俗说 | 类比 | 家族内落点 |
| :--- | :--- | :--- | :--- |
| **ViT** | 把图切成 64 块拼图当“字”，`[CLS]` 当班长通读打分 | 放大镜逐格看(CNN) vs 拼图全摊开看(ViT) | `01`：809k ViT vs 374k CNN 同台，小数据 CNN 0.4096 vs ViT 0.3614 |
| **CLIP** | 图和字各训一个编码器，拉近配对推远错配 | 相亲：cos 高即看对眼 | `02`：双塔 1.24M+InfoNCE 手写，zero-shot 0.092≈随机 vs 监督 0.485（小数据喂不饱是结论不是 bug），免标注换类才是卖点 |
| **BLIP/LLaVA** | 视觉编码器+投影层+LLM，图转成 LLM 能读的“方言” | ViT 当眼睛，LLM 当嘴，投影层当视神经 | `03`：三段式 toy 全闭环（CE 0.089 背熟但类名命中仅 0.06，图像注意力 0.018<基线——模板太规律语言先验压过视觉，教学演示接线用） |

> 整条线的结论：**看图当字(ViT) → 图文对上(CLIP) → 眼睛接大脑(BLIP/LLaVA) → 08 工程优化同样适用多模态大模型**

## 学习目标

1. ViT：patch embedding，理解图像序列化
2. CLIP：对比学习图文对齐，zero-shot 分类
3. BLIP / LLaVA：理解多模态大模型的一般架构（视觉编码器 + 投影层 + LLM）

## 算法清单（学习检查表）

- [x] ViT → `01_ViT_CIFAR10`（patchify+CLS+可学习pos 手写；与 SmallCNN 同子集同台，归纳偏置实锤）
- [x] CLIP → `02_CLIP_ZeroShot`（双塔+char 文本塔+双向 InfoNCE+可学习温度手写；zero-shot 0.0916 vs 监督 0.4851 同预算，诚实归因“对比学习喂不饱”，免标注扩展是范式卖点）
- [x] BLIP（理解为主）→ `03_Multimodal_Experience` 前半：模板描述句 + 看图说话体验（冻结 ViT + 投影 + 因果 LLM，CE 0.089 背下模板）
- [x] LLaVA（理解为主）→ `03` 三段式 toy 全闭环（整句 0.0 / 类名命中 0.06 / 图像注意力 0.018——诚实短板三件套，管道同构真 LLaVA）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_ViT_CIFAR10` | ViT | CIFAR-10 | 小型 ViT 从零训练；与 CNN 对比 |
| 02 | `02_CLIP_ZeroShot` | CLIP | CIFAR-10 / 自选图片 | zero-shot 分类 |
| 03 | `03_Multimodal_Experience` | BLIP / LLaVA | 任意图片 | 推理体验（不训练） |

## 进度

- ✅ `01_ViT_CIFAR10`（ViT-Tiny 809k vs SmallCNN 374k，CIFAR-10 子集 4000/800 同 6ep：CNN test 0.4096 > ViT 0.3614，归纳偏置小数据实锤；CLS 热力+错例 5 图，CPU 285s）
- ✅ `02_CLIP_ZeroShot`（双塔 1.24M 参手写：图像塔=01 ViT 砍头 + char 文本塔 + 双向 InfoNCE + 可学习温度；zero-shot 0.0916≈随机 vs 监督同预算 0.4851——诚实结论“对比学习靠亿级图文对喂”，免标注换类是范式价值；相似度矩阵+错例 5 图，CPU 729s）
- ✅ `03_Multimodal_Experience`（LLaVA 三段式 toy：ViT 3ep 预热冻结 + 投影 4 词 + 3 层因果 LLM；10ep CE 0.089→PPL 1.09 背下模板，但自回归整句 0.0/类名命中 0.06/图像注意力 0.018——诚实短板，管道同构真 LLaVA；5 图，CPU 220s）

## 与其他家族的关系

- **前置**：05 Transformer 基础（本章 Encoder-only 双向+CLS 与 05-02 BERT 同构，token 从字换 patch）；02_CNN_Family（数据缓存 + 归纳偏置对照组）
- **后续**：08 工程优化中 GQA/RoPE 等同样适用于多模态大模型
