# 04 序列模型家族：HMM / CRF / RNN / LSTM / GRU / Seq2Seq / Attention（进阶：SSM/Mamba）

> 这条线解决的是“顺序依赖”的问题。它从传统统计模型一路演化到深度序列模型，再过渡到 Transformer，最终延伸到 SSM/Mamba（现代 RNN 复兴）。

## 学习目标

1. 理解序列建模中的状态转移、条件依赖与标注问题
2. 弄清 HMM、CRF 与 RNN 家族在建模假设上的区别
3. 理解 RNN 难训练的原因，以及 LSTM/GRU 如何缓解梯度消失
4. 理解 Seq2Seq 与 Attention 如何为 Transformer 铺路

## 算法清单（学习检查表）

- [ ] HMM（隐马尔可夫，传统序列生成/解码）
- [ ] CRF（条件随机场，序列标注经典方法）
- [ ] RNN
- [ ] LSTM
- [ ] GRU
- [ ] Seq2Seq
- [ ] Attention（Bahdanau / Luong）
- [ ] SSM / Mamba（进阶：状态空间模型，选择性扫描 S6；建议**学完 05 后回补**，与 Transformer 形成"线性 vs 二次复杂度"对照）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_HMM_CRF_NER` | HMM, CRF | 小型 NER 数据 / toy corpus | 序列标注对比 |
| 02 | `02_RNN_LSTM_GRU_TextCls` | RNN, LSTM, GRU | IMDB / 中文情感分类 / THUCNews | 长文本上 RNN vs LSTM vs GRU |
| 03 | `03_Seq2Seq_Attention_MT` | Seq2Seq, Attention | 小型翻译对 / Tatoeba | Attention 可视化 |
| 04 | `04_Mamba_Mini_SSM` | SSM, Mamba | 小语料 LM / 长序列 toy | Mini-Mamba 纯 PyTorch 实现，与 05 的 mini Transformer 对照（速度/内存/效果） |

## 学习顺序

01 → 02 → 03 →（进阶 04：学完 05 后回补）

## 与其他家族的关系

- **后续**：05 Transformer 直接继承了 Attention 机制；HMM/CRF 是传统序列标注的思想补充
- **进阶**：SSM/Mamba 是"现代版 RNN 复兴"，与 05 的 mini Transformer 做架构对照（线性 vs 二次复杂度），建议学完 05 再回补
