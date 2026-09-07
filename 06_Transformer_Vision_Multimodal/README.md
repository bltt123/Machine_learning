# 06 Transformer 视觉与多模态：ViT / CLIP / BLIP / LLaVA

> Transformer 从 NLP 走向视觉与多模态。这一族关注“图像如何 token 化、图文如何对齐、多模态如何统一”。

## 学习目标

1. ViT：patch embedding，理解图像序列化
2. CLIP：对比学习图文对齐，zero-shot 分类
3. BLIP / LLaVA：理解多模态大模型的一般架构（视觉编码器 + 投影层 + LLM）

## 算法清单（学习检查表）

- [ ] ViT
- [ ] CLIP
- [ ] BLIP（理解为主）
- [ ] LLaVA（理解为主）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_ViT_CIFAR10` | ViT | CIFAR-10 | 小型 ViT 从零训练；与 CNN 对比 |
| 02 | `02_CLIP_ZeroShot` | CLIP | CIFAR-10 / 自选图片 | zero-shot 分类 |
| 03 | `03_Multimodal_Experience` | BLIP / LLaVA | 任意图片 | 推理体验（不训练） |

## 与其他家族的关系

- **前置**：05 Transformer 基础
- **后续**：08 工程优化中 GQA/RoPE 等同样适用于多模态大模型
