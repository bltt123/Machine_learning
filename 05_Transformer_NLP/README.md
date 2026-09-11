# 05 Transformer NLP 家族：Transformer / BERT / GPT / T5 / BART（拓展对照：SSM/Mamba）

> **一句话定位**：现代大模型的“主轴”——04 用 Attention 给 RNN 加聚光灯，05 把聚光灯抽出来去掉循环，MHA/因果 mask/位置编码全部手写，靠“全员同时开会”取代“逐人传话”。
> 状态：✅ 01-05 全完成（含拓展 05 线性 vs 二次）| 数据：复制/翻转/CLS/循环/去噪 toy | 环境：`LLM_learning`（torch 2.5.1+cpu，无外网，01 4图+02 5图+03 5图+04 6图+05 5图共25图，CPU 约 4 分钟）

## 通俗理解：从“传话”到“开会”

| 阶段 | 通俗说 | 类比 | 家族内落点 |
| :--- | :--- | :--- | :--- |
| **Attention** | 边生成边回头看底稿，聚光灯指哪看哪 | 开卷翻译 | `04-03`：Bahdanau 复制对角/翻转反对角 |
| **Transformer** | 去掉循环，所有位置互相看一遍，全并行 | 全员开会：Q我要问什么 K我能答什么 V我的内容，Q·Kᵀ打分加权 V | `01`：MHA/因果mask/PE 手写，GRU+Attn 同台双 1.0，对角/反对角 |
| **BERT** | 只看编码器，双向看完整句 | 开卷考试：前后都可见，填空 MLM | `02`：Encoder-only 双向+CLS，含0任务双 1.0（MeanPool 同满分，序敏感才拉开） |
| **GPT** | 只看解码器，因果只能看过去 | 闭卷写作：下一个词只看已写过的 | `03`：Decoder-only 因果，循环计数 next 1.0 vs 基线 1/8，续写 4/4 |
| **T5/BART** | 编解码齐全，文本进文本出 | 全能翻译官 | `04`：统一 `src→tgt`，复制 1.0 vs 去噪 0.115/0.77 难但可学 |
| **Mamba** | 线性复杂度流水线 vs 二次复杂度全员会 | 流水线 O(n) vs 全员会 O(n²) | `05`：selective SSM 线性扫，同循环双 1.0，理论 S=128 16 vs 50 倍 |

> 整条线的结论：**规则手拉手(HMM/CRF) → 记忆传递(RNN) → 门控直通(LSTM/GRU) → 聚光灯(Attention) → 全注意力(Transformer) → 去噪/自回归/统一文本(BERT/GPT/T5) → 线性流水线(Mamba)**

## 学习目标

1. 自己写一个最小可运行 Transformer（理解 MHA、Mask、FFN、残差、LayerNorm）
2. 理解 BERT / GPT / T5 / BART 的结构差异与适用任务
3. 掌握预训练 / 微调 / 推理的基本范式
4. （拓展对照）通过 mini-Mamba vs mini-Transformer 理解"线性复杂度 vs 二次复杂度"的架构取舍

## 算法清单（学习检查表）

- [x] Transformer（基础结构自实现）→ `01_Transformer_From_Scratch`（MHA+因果mask+PE 手写，与 04-03 GRU+Attn 同台双 1.0）
- [x] BERT（Encoder-only，`[CLS]` 微调范式）→ `02_BERT_TextCls_FineTune`（Encoder-only 双向，含0任务双 1.0，CLS 对 0 聚焦）
- [x] GPT（Decoder-only，自回归生成）→ `03_GPT_LM_Mini`（因果 LM 循环计数 next 1.0，贪心 4/4 续写，温度 sweep）
- [x] T5（Text-to-Text 统一）→ `04_T5_BART_Mini`（复制 1.0 复现 01，去噪 0.115/0.77）
- [x] BART（Denoising 去噪）→ `04`（25% MASK 去噪，seq 0.115 难但 tok 可学）
- [x] SSM / Mamba（拓展对照）→ `05_Mamba_vs_Transformer_Mini`（同循环双 1.0，17.3k vs 11.0k，FLOPs/实测随 S 趋势）

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

## 场景速查（选型指南，人话：能干什么活 + 支撑数据）

| 场景（能干什么活） | 推荐 | 支撑数据 | 要点 |
|---|---|---|---|
| 需双向理解/填空/分类 | BERT Encoder-only | 含0 1.0 CLS 对 0 聚焦 | `mask=None` 全可见，`h[CLS]` 分类 |
| 需自回归生成/續写对话 | GPT Decoder-only | 循环 next 1.0 vs 基线 0.125，续写 4/4 | 因果下三角必加 |
| 一切任务当翻译（翻译/摘要/问答） | T5 统一 `src→tgt` | 复制 1.0 全量复现 | 换数据即换任务 |
| 无标注预训练挖洞还原 | BART 去噪 | tok 0.77 seq 0.115 p=0.25 | 难但可学 |
| 长句线性拓展 | Mamba 线性 | 理论 S=128 16 vs 50 倍，实测 GPT 1ms 线性亦 1.0 | 流水线 O(S)，显存 O(1) |

## 进度

- ✅ `01_Transformer_From_Scratch`（MHA/因果mask/sin-cos PE/残差LN 手写闭环；S=6 复制/翻转上 Transformer 42.5k参 vs GRU+Attn 15.4k参 双 seq-acc 1.000；交叉注意力对角/反对角热力与 04-03 Bahdanau 同构；4 图，CPU 78s）
- ✅ `02_BERT_TextCls_FineTune`（Encoder-only 双向+CLS：800/200 含0任务 BERT/MeanPool 双 1.0（无序足，04-02 序敏感 0.47 vs 0.71 反例），5 图，CPU 25s）
- ✅ `03_GPT_LM_Mini`（Decoder-only 因果：循环计数 800/200 next 1.0 vs 基线 1/8，续写 4/4，温度 sweep，5 图，CPU 39s）
- ✅ `04_T5_BART_Mini`（Encoder-Decoder 统一：T5 复制 600/200 1.0 复现 01，BART 25%挖洞去噪 0.115/0.77 同骨架 42.5k，6 图，CPU 60s）
- ✅ `05_Mamba_vs_Transformer_Mini`（拓展对照 selective SSM 线性：同循环 800/200 S=8 GPT 17.3k vs Mamba 11.0k 双 next 1.0，续写 3/3，理论+实测随 S 趋势，5 图，CPU 60s；修 `F.silu` 兼容 2.5）
