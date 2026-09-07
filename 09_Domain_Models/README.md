# 09 领域应用模型：时序预测 / 推荐系统 / 语音

> 这一族不是新的"基础架构"，而是**领域知识 × 主干网络**的组合落地：时序和推荐本质上仍以 CNN/Attention/MLP 为骨干，语音则是 Transformer 的音频版。适合在主干家族（01~08）学完后做综合应用，也可按兴趣穿插进行。

## 学习目标

1. 时序预测：理解长序列依赖、时间特征编码、长期预测的挑战
2. 推荐系统：理解 Embedding 化、特征交叉、用户行为序列建模（与你已有的传统 ML 背景衔接最好）
3. 语音：理解音频 → 频谱 → token 的转换链路，体验工业级 ASR 模型

## 算法清单（学习检查表）

- [ ] Informer（ProbSparse Attention，长序列高效预测）
- [ ] PatchTST（Patch 化 + Channel Independence，当前时序强基线）
- [ ] DeepFM（FM + DNN 并联，特征自动交叉）
- [ ] DIN（Attention 对用户行为序列加权，阿里出品）
- [ ] Whisper（Encoder-Decoder 语音识别，tiny 版推理为主）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_TimeSeries_ETT` | Informer, PatchTST | ETT-small（ETTh1）/ 北京 PM2.5 | MSE 对比：LSTM（04 的模型）vs Informer vs PatchTST |
| 02 | `02_Recommendation_MovieLens` | DeepFM, DIN | MovieLens-100K | AUC 对比：LR/FM 基线 vs DeepFM vs DIN |
| 03 | `03_Whisper_ASR_Experience` | Whisper | 自备音频 / Common Voice 小样本 | tiny 模型转写体验 + 中文/英文对比（推理为主，可选微调） |

## 数据集说明

- **ETT-small**：电力变压器温度数据，2 年小时级 7 个特征，GitHub 直接下载（约 7MB），时序预测的标准测试床
- **北京 PM2.5**：UCI/Kaggle，气象+污染小时数据，适合入门
- **MovieLens-100K**：10 万评分、1000 用户 × 1700 电影，推荐系统入门标准数据集
- **语音**：先用几段自己的音频体验 tiny/base 模型；Common Voice 中文子集可选

## 与其他家族的关系

- **前置**：04 的 LSTM（时序基线）、05 的 Transformer/Attention（Informer/PatchTST/DIN 的核心机制）、02 的 CNN 思想（Patch 化）
- **后置**：无——这是主干学习的综合应用出口
- **衔接传统 ML**：推荐系统部分与你之前练过的 FM/LR/树模型思路一脉相承
