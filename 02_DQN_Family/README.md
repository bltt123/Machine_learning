# 02 离散值函数家族：DQN / Double / Dueling

> **一句话定位**：把 01 的表格账本换成神经网络——DQN 拿 MLP 当会计，Double 换个裁判防自夸，Dueling 把“地好不好”和“招妙不妙”分开算。
> 状态：✅ 2 站全部完成 | 环境：`CartPole-v1`（gymnasium 0.29.1），torch 2.5.1+cpu | 实跑：01（200ep×3种子+消融）+ 02（三变体×3种子），7 张图

## 通俗理解：从查表到请会计

| 阶段 | 通俗说 | 类比 | 落点 |
|---|---|---|---|
| DQN | 请神经网络当会计估值 + 错题本 + frozen 参考答案 | 查表改估价：状态连续装不下表了 | `01`：eval 55.0±27.8，target 消融 55 vs 9.3 |
| Double | 在线选动作、目标打分，换个裁判 | 运动员别兼裁判 | `02`：86.3±15.1 最稳，三种子全≥66 |
| Dueling | Q=V+(A−meanA)，场地与招式分开 | 先评地再评招 | `02`：65.0±62.1，两极分化 152/33/10 |

## 学习目标

1. 写出 DQN target 与 loss，解释回放和 target 各防什么
2. 写出 Double target，解释解耦选择与评估
3. 写出 Dueling 聚合式，解释可辨识性（减 meanA）
4. 跑出多种子 mean±std + per-seed 表，理解方差报告规范

## 算法清单

- [x] DQN（含 ReplayBuffer、TargetNet、Huber、eps-greedy）→ `01`（E1–E3）
- [x] Target 消融 → `01`（E2：loss 0.057 vs 18471，30 万倍）
- [x] Double DQN → `02`（E1–E3，最稳 86.3±15.1）
- [x] Dueling DQN → `02`（E1–E3，方差反例 ±62）
- [ ] Rainbow（集大成，思想级，不实现）

## 项目规划

| 编号 | 项目 | 覆盖内容 | 环境 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_DQN_CartPole` | DQN 基线 + target 消融 + 高估探针 | CartPole-v1 | stable 55 vs no-target 9.3；meanQ 5.54 远低 500 |
| 02 | `02_Double_Dueling_LunarLander` | 三变体同预算同台 | CartPole-v1（同环境才公平，见站内 §8 说明） | Double 86 最稳；Dueling 152/33/10 分化 |

## 场景速查

| 场景 | 推荐 | 支撑数据 | 指向 |
|---|---|---|---|
| 离散默认 | Double DQN | 86.3±15.1 | 02-E1 |
| 稳定器 | frozen target sync≥200 | 55 vs 9.3 | 01-E2 |
| 简单环境 | 慎上 Dueling | ±62 反例 | 02-E3 |
| 报结果 | ≥3 种子 + per-seed | 24~152 spread | 01/02 |

## 与其他家族的关系

- **前置**：01 表格 Q-Learning（精确解，小状态）
- **后续**：03 PPO（on-policy 规模化，MountainCar 稀疏奖励）、04 SAC（连续动作，target 双 Q 继承）、05 CQL（离线，保守 Q 继承）
