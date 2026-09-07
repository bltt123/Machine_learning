# 05 Transformer NLP 家族：Transformer / BERT / GPT / T5 / BART（拓展对照：SSM/Mamba）

> Transformer 是现代大模型的主轴。这里建议按“自实现 → 编码器 → 解码器 → 编解码器”来学；主干完成后可做 SSM/Mamba 的架构对照（与 04 的 Mamba 进阶项目共用实验）。

## 学习目标

1. 自己写一个最小可运行 Transformer（理解 MHA、Mask、FFN、残差、LayerNorm）
2. 理解 BERT / GPT / T5 / BART 的结构差异与适用任务
3. 掌握预训练 / 微调 / 推理的基本范式
4. （拓展对照）通过 mini-Mamba vs mini-Transformer 理解"线性复杂度 vs 二次复杂度"的架构取舍

## 算法清单（学习检查表）

- [ ] Transformer（基础结构自实现）
- [ ] BERT（Encoder-only，理解）
- [ ] GPT（Decoder-only，生成）
- [ ] T5（Text-to-Text）
- [ ] BART（Denoising Seq2Seq）
- [ ] SSM / Mamba（拓展对照：Mamba vs Transformer 同规模实验，进阶可回补 04 的演化链视角）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_Transformer_From_Scratch` | Transformer 基础 | toy 数据 / 小语料 | 注意力权重可视化 |
| 02 | `02_BERT_TextCls_FineTune` | BERT | IMDB / THUCNews | 微调分类 |
| 03 | `03_GPT_LM_Mini` | GPT | 小语料 | 自回归生成 |
| 04 | `04_T5_BART_Mini` | T5, BART | 摘要/翻译/文本改写 | 编解码统一范式 |
| 05 | `05_Mamba_vs_Transformer_Mini`（拓展对照） | SSM, Mamba, Transformer | 小语料 LM | 同参数量级下效果/训练速度/推理吞吐对照 |

## 与其他家族的关系

- **前置**：04 序列模型中的 Attention
- **后续**：06 视觉/多模态 Transformer、08 工程优化（RoPE/KV Cache/MQA/GQA/FlashAttention）
- **拓展**：Mamba 与 04-04 的 Mini-Mamba 项目共用实现，此处聚焦"与 Transformer 的同台对照"
