# 09 领域应用模型：时序预测 / 推荐系统 / 语音

> 这一族不是新的"基础架构"，而是**领域知识 × 主干网络**的组合落地：时序和推荐本质上仍以 CNN/Attention/MLP 为骨干，语音则是 Transformer 的音频版。适合在主干家族（01~08）学完后做综合应用，也可按兴趣穿插进行。

## 学习目标

1. 时序预测：理解长序列依赖、时间特征编码、长期预测的挑战
2. 推荐系统：理解 Embedding 化、特征交叉、用户行为序列建模（与你已有的传统 ML 背景衔接最好）
3. 语音：理解音频 → 频谱 → token 的转换链路，体验工业级 ASR 模型

## 算法清单（学习检查表）

- [x] Informer（ProbSparse Attention，长序列高效预测）→ `01_TimeSeries_ETT`（ETTh1 真数据：LSTM 0.0936 胜 Informer toy 0.2899；复杂度边界诚实声明）
- [x] PatchTST（Patch 化 + Channel Independence，当前时序强基线）→ `01`（Patch=16/stride=8：MSE 0.3638；教学简化，非论文 SOTA）
- [x] DeepFM（FM + DNN 并联，特征自动交叉）→ `02_Recommendation_MovieLens`（test-AUC=0.5607）
- [x] DIN（Attention 对用户行为序列加权，阿里出品）→ `02`（test-AUC=0.6603；历史增益 +0.0859）
- [x] Whisper（Encoder-Decoder 语音识别，tiny 版推理为主）→ `03_Whisper_ASR_Experience`（离线 tiny 37.8M：EN WER 0.1786 / ZH 字级 0.5862，forced==auto；真音频）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_TimeSeries_ETT` | Informer, PatchTST | ETT-small（ETTh1）/ 北京 PM2.5 | MSE 对比：LSTM（04 的模型）vs Informer vs PatchTST | ✅（ETTh1 17,420×8；4 图；LSTM MSE 0.0936） |
| 02 | `02_Recommendation_MovieLens` | DeepFM, DIN | MovieLens-100K | AUC 对比：LR/FM 基线 vs DeepFM vs DIN | ✅（真数据 100K；DIN 0.6603 vs DeepFM 0.5607；2 图） |
| 03 | `03_Whisper_ASR_Experience` | Whisper | 自备音频 / Common Voice 小样本 | tiny 模型转写体验 + 中文/英文对比（推理为主，可选微调） | ✅（中英双 WER+频谱；3 图；离线 35s） |

## 数据集说明

> 以下 `data/` 与 `weights/` 均已被 `.gitignore` 忽略（分别约 8MB / 6MB / 1MB / 155MB），不会推送；按下表手动下载放入对应目录即可复跑。

### 01 时序：ETT-small（ETTh1）

| 项 | 内容 |
|---|---|
| 下载链接 | https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv |
| 存放位置 | `09_Domain_Models/01_TimeSeries_ETT/data/ETTh1.csv`（约 2.5MB） |
| 校验 | 17,420 行；列 `date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`；无 NaN；`2016-07-01` 至 `2018-06-26` 小时级 |
| 备注 | 北京 PM2.5 为规划备选，本轮未使用（ETTh1 已覆盖时序对照目标） |

### 02 推荐：MovieLens-100K

| 项 | 内容 |
|---|---|
| 下载链接 | https://files.grouplens.org/datasets/movielens/ml-100k.zip |
| 存放位置 | 解压到 `09_Domain_Models/02_Recommendation_MovieLens/data/ml-100k/`（约 6MB） |
| 必需文件 | `u.data` / `u.item` / `u.user` / `u1.base` / `u1.test`（本轮只用这 5 个；其余 `u2~u5/ua/ub` 为官方交叉切分，未使用） |
| 校验 | 943 用户 / 1682 电影槽位 / 100,000 评分；`u1.base` 80,000 条 / `u1.test` 20,000 条 |

### 03 语音：LibriSpeech 英文 + 自备中文 + Whisper tiny 权重

| 项 | 内容 |
|---|---|
| 英文音频 | https://www.openslr.org/resources/12/test-clean.tar.gz（LibriSpeech test-clean；本轮只用 `1089/134686/1089-134686-0000.flac` 10.44s + 同目录 `.trans.txt` 转写；需自转 `ffmpeg -i x.flac -ar 16000 -ac 1 x.wav`） |
| 中文音频 | 自备 `A7_87.wav`（16k 单声道 9.19s）+ `A7_87.txt`（参考文本 29 字）；Common Voice 备选 https://commonvoice.mozilla.org/zh-CN/datasets（需登录，本轮未使用） |
| 存放位置 | `09_Domain_Models/03_Whisper_ASR_Experience/data/` |
| 权重下载页 | https://huggingface.co/openai/whisper-tiny/tree/main（约 155MB） |
| 存放位置 | `09_Domain_Models/03_Whisper_ASR_Experience/weights/whisper-tiny/`（9 文件缺一不可） |
| 必需文件 | `config.json` / `generation_config.json` / `preprocessor_config.json` / `tokenizer.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` / `normalizer.json` / `model.safetensors`（transformers 4.47 慢速 BPE 需要 vocab/merges/normalizer；见 03 README §8） |

## 与其他家族的关系

- **前置**：04 的 LSTM（时序基线）、05 的 Transformer/Attention（Informer/PatchTST/DIN 的核心机制）、02 的 CNN 思想（Patch 化）
- **后置**：无——这是主干学习的综合应用出口
- **衔接传统 ML**：推荐系统部分与你之前练过的 FM/LR/树模型思路一脉相承
