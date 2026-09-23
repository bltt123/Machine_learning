# 强化学习算法练习项目总览（RL Practice Roadmap）

> **核心原则**：按“问题类型”学、按“环境”练、按“代表算法”做消融对比。
> 每个家族一个目录，家族内按“站（编号）”推进；每站沉淀 `run.py`（可复现）+ `README.md`（原理 → 代码 → 实验 → 失败模式 → 总结）+ `figs/`（实跑图）。
> 全部按 CPU / 小环境设计，单站 CPU 总时长 < 10 分钟；表格数字 ≥5 种子 mean±std。

---

## 一、学习路线（总览）

| 顺序 | 目录 | 主线 | 覆盖算法（代表） | 练习环境 |
|---|---|---|---|---|
| 1 | `01_RL_Foundations` | 打地基 | MDP、贝尔曼方程、动态规划、MC、SARSA、Q-Learning（表格） | 手写 GridWorld + CliffWalking |
| 2 | `02_DQN_Family` | 离散值函数 | DQN → Double DQN → Dueling DQN | CartPole + LunarLander |
| 3 | `03_OnPolicy_PPO` | 同策略 | REINFORCE → A2C → TRPO（理论） → PPO-Clip | CartPole + MountainCar |
| 4 | `04_Continuous_TD3_SAC` | 连续控制 | DDPG → TD3 → SAC（含自动温度） | Pendulum + MountainCarContinuous |
| 5 | `05_Offline_RL` | 离线 | BC → CQL → IQL（含 stitching） | 自采 CartPole 离线数据集 |
| 6 | `06_RLHF_GRPO_LLM` | LLM 对齐 | DPO/ORPO/SimPO/KTO → GRPO/DAPO/GSPO → VERL toy | 自造偏好对（toy logits） |
| 7 | `07_Production_Robotics` | 生产选学 | SB3 / Tianshou / RLlib / Isaac / SERL（概念+对照） | CartPole + 文档阅读 |

> **推荐顺序**：01 → 02/03 可并行 → 04 → 05/06 → 07 选学。
> 档位：A 精简（01+02+03+04+06）/ B 标准（01~06）/ C 完整（01~07）。

---

## 二、算法覆盖清单（对照用户名单逐项核对）

### 主线必做

- 表格与地基：MDP、贝尔曼期望/最优、策略迭代/值迭代、MC、SARSA、Q-Learning → `01`
- DQN 家族：DQN、Double DQN、Dueling DQN → `02`（Rainbow 只讲思想）
- 同策略：REINFORCE、A2C、TRPO、PPO；A3C 只讲异步思想，不单独实现 → `03`
- 连续控制：DDPG、TD3、SAC → `04`
- 离线：BC、CQL、IQL → `05`
- 对齐：DPO、GRPO → `06`（ORPO/SimPO/KTO 归 DPO 站内对照；DAPO/GSPO/ARPO/GDPO 归 GRPO 站内对照）

### 工具（概念 + 对照，不深挖）

- Gymnasium（全程地基）、Stable-Baselines3 / CleanRL（标准答案对照）、Tianshou（简洁复现）、Ray RLlib（分布式概念）、VERL（LLM GRPO 流程 toy 走通）→ `07`、`06-02`
- NVIDIA Isaac Gym / Isaac Sim：GPU 并行仿真，无硬件条件，只做原理阅读 → `07-02`

### 阅读级（不开项目，README 一句话定位）

- SERL（真机机械臂栈）、DSRL / SAC-Flow（SAC 换头分支）、RL-100（榜单合集叫法）→ 各站 §9 应用场景

---

## 三、进度追踪表

| # | 家族 | 状态 | 已完成 | 关键收获（一句话） |
|---|------|------|--------|-------------------|
| 01 | RL_Foundations | ✅ 已完成 | 2 站 + common | 贝尔曼是账本；DP 全知、MC 等账单、TD 每天估账；Q 贪心 13 步最优 vs SARSA 17 步安全 |
| 02 | DQN_Family | ✅ 已完成 | 2 站 + common | target 是稳定器（55 vs 9.3）；Double 最稳 86.3±15.1；Dueling 简单环境方差大 |
| 03 | OnPolicy_PPO | ✅ 已完成 | 2 站 + common | 基线 +91 eval；PPO clip0.2 以 488.2±3.9 夺冠，KL/clipfrac 双零=收敛 |
| 04 | Continuous_TD3_SAC | ✅ 已完成 | 2 站 + common | TD3 反超高估小 10 倍；Pendulum 上只有延迟更新明确有益；SAC 熵正则被奖励尺度碾压 |
| 05 | Offline_RL | ✅ 已完成 | 2 站 + common | BC 天花板=数据质量（31.8→500）；random 上离线 RL 逆袭（BC 31.8 vs IQL 211）；CQL α=0 崩、α=2 稳 |
| 06 | RLHF_GRPO_LLM | ✅ 已完成 | 2 站 + common | DPO 家族同分 0.9，IPO 最保守；β 是 accuracy-KL 旋钮；GRPO 组基线省 critic，G=2 最差 |
| 07 | Production_Robotics | ✅ 已完成 | 2 站（01 实跑 + 02 阅读级） | SB3 交叉验证手写 DQN 通过（167.8≈177.9）；PPO 落后在调参；工具链按规模选 |

> 状态标记：⬜ 未开始 / 🔵 进行中 / ✅ 已完成

---

## 四、目录结构

```text
Machine_learning/
├── AGENTS.md
├── README.md
├── package.md
├── 01_RL_Foundations/
│   ├── README.md
│   ├── common/
│   ├── 01_MDP_DP_GridWorld/
│   └── 02_MC_TD_Cliff/
├── 02_DQN_Family/
│   ├── 01_DQN_CartPole/
│   └── 02_Double_Dueling_LunarLander/
├── 03_OnPolicy_PPO/
│   ├── 01_REINFORCE_A2C/
│   └── 02_PPO_Clip_TRPO/
├── 04_Continuous_TD3_SAC/
│   ├── 01_DDPG_TD3_Pendulum/
│   └── 02_SAC_AutoTemp/
├── 05_Offline_RL/
│   ├── 01_Collect_Dataset/
│   └── 02_BC_CQL_IQL/
├── 06_RLHF_GRPO_LLM/
│   ├── 01_DPO_Family/
│   └── 02_GRPO_VERL_Toy/
└── 07_Production_Robotics/
    ├── 01_SB3_Tianshou_Compare/
    └── 02_RLlib_Isaac_SERL_Notes/
```

---

## 五、环境与规则（摘要，详见 AGENTS.md）

- conda 环境：`LLM_learning`（Python 3.9 + PyTorch 2.5.1+cpu）
- `01` 地基零依赖（纯 numpy）；`02` 起需 `gymnasium`，安装前先征得同意
- 严禁写 C 盘；不擅自装库；未经要求不 git 提交
- 每站 `run.py` 一键复现 + `figs/` 实跑图 + README 10 节；表格数字 ≥5 种子

---

## 六、术语速查表（一页通）

| 术语 | 通俗解释 |
|---|---|
| MDP 五元组 | 世界的说明书：状态 S、动作 A、转移 P、奖励 R、折扣 gamma |
| return Gt | 从现在起未来奖励的折现总和，智能体一生追求的就是它大 |
| V(s) / Q(s,a) | 这个位置值多少 / 在这位置出这招值多少，贝尔曼账本的两页 |
| 贝尔曼方程 | 账本自洽关系：现在的值 = 眼前奖励 + 折现后的未来值 |
| 探索 / 利用 | 出去乱试找新路 / 走已知最好路；epsilon-greedy 就是掷骰子决定 |
| on-policy / off-policy | 边学边用自己数据（PPO）/ 用旧账本也能学（DQN、离线） |
| baseline / advantage | 考试平均分 / 超出平均分几分，减掉它方差立刻降 |
| 熵正则 | 逼策略别太死心眼，留条后路，SAC 的看家本领 |
| OOD 外推 | 拿没见过的招去问 Q，Q 张口就编，离线 RL 第一死因 |
| KL 约束 | 新策略别跑太远，TRPO/PPO/DPO/GRPO 共用的缰绳 |
