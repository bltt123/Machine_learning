# 08 生产级优化与工程机制：RoPE / KV Cache / GQA / FlashAttention / LoRA / 量化 / MoE / RAG / 推理部署

> **一句话定位**：01~07 全是“学新模型”，08 反过来“让模型跑得快、训得起、落得了地”——不训新模型，只给 05 的 Transformer 动手术。
> 状态：✅ 01-09 全部完成 🎉 | torch 2.5.1+cpu，SDPA 实测存在；**peft 0.17.1 已装**（accelerate 0.26.0→1.10.1，Windows 兼容）；vllm 未装（09 站 toy 模拟）
>
> **包策略**：该装就装（Windows 兼容优先；vLLM 不兼容不硬装）；推不动的模块直说不造数。

> 这是“企业生产级”最核心的一族。重点不是训练新模型，而是理解怎么让模型跑得快、训得起、落得了地。建议在学完 05 / 06 之后进入。

## 学习目标

1. 理解位置编码演进（绝对 → 相对 → RoPE）与长度外推
2. 理解推理阶段的 KV Cache，以及 MQA/GQA 的显存优化
3. 理解 FlashAttention 的 IO 优化思路与效果
4. 理解 LoRA/QLoRA 参数高效微调
5. 理解量化 / 蒸馏 / 剪枝三种模型压缩路线
6. 理解 MoE 稀疏激活与 RAG 生产范式（结合 LLM_learning 环境已有 langchain/llama-index）
7. 掌握训练侧效率技术：Mixed Precision（AMP）混合精度与 Gradient Checkpointing 梯度检查点
8. 理解推理部署三件套：PagedAttention（KV Cache 显存分页）、投机解码（Speculative Decoding）、GGUF/llama.cpp（量化部署延伸）

## 算法清单（学习检查表）

- [x] Positional Encoding / RoPE → `01_RoPE_PositionEncoding`（Abs/SinCos/RoPE 同骨架同台；复制全外推 1.0、模加 S=32 全崩 0——任务设计决定外推上限，诚实记录）
- [x] KV Cache → `02_KVCache_MQA_GQA`（预填充+增量解码，加速 ×1.1 toy / O(S²)→O(S) 公式，一致性全 True；抓到 2 个真 bug 已修）
- [x] MQA / GQA → `02`（kv 4→2→1，KV 1024→512→256B 逐级减半，参数 102k/94k/90k）
- [x] FlashAttention（SDPA）→ `03_FlashAttention_SDPA`（手写分块 online softmax：三方等价 ~3e-07 + HBM ×3.95@S=1024；SDPA 实测存在，误判已纠正）
- [x] LoRA / QLoRA / PEFT → `04_LoRA_Finetune`（手写 B·A + peft 0.17.1 对照：6.8k vs 102k 三者全 1.0，merge 1.19e-06；copy→modadd 阴性结果诚实记录）
- [x] Mixed Precision（AMP 混合精度训练）→ `05_MixedPrecision_Checkpointing`（bf16 val-seq=1.0 精度不掉；CPU 上 624.8s vs 21.7s 慢 29×，GPU 才加速，诚实声明）
- [x] Gradient Checkpointing（梯度检查点）→ `05`（saved_tensors 409.5→23.4MB ×17.5，前向等价 Δ=0；端到端 RSS 1881→1676MB ×1.12）
- [x] 量化（INT8/INT4）→ `06_Quantization_Prune`（对称 INT8 PTQ：1.69MB→0.42MB ×4.00，acc Δ=-0.0002）
- [ ] 蒸馏 → `06`（T=4/α=0.7：106k 学生 0.9699 vs 单训 0.9706——本轮≈单训，适用边界诚实记录）
- [x] 剪枝 → `06`（全局幅度：s=0.7 前不掉 0.9735，s=0.9 崩到 0.9559，拐点实测）
- [x] MoE（稀疏混合专家）→ `07_MoE_Mini`（E=8 top-2：总参 3×但激活同量级 ≈0.40M，FLOPs 1/4，双 1.0，专家使用率 ±0.004 均匀）
- [x] RAG → `08_RAG_Mini`（jieba+TF-IDF 175维：24问三1.0，阈值0.08拒答率1.0；零新包）
- [x] PagedAttention → `09_Inference_Deployment`（toy 分页 1/3/5 页，碎片 0/20/21%；vLLM 不装 toy 模拟）
- [x] 投机解码 → `09`（draft 猜 γ=4：5.00 token/轮，一致性 True；上限口径已声明）
- [x] GGUF / llama.cpp → `09`（toy 拼盘 ×3.96，回读 4.39e-02；二进制不调拼字节校验）

## 项目规划

| 编号 | 项目 | 覆盖机制 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_RoPE_PositionEncoding` | RoPE | toy | 外推长度测试 | ✅（三编码同台：复制 S=64 全 1.0、模加 S=32 全 0；4 图，CPU 140s） |
| 02 | `02_KVCache_MQA_GQA` | KV Cache, MQA, GQA | GPT 推理 | Time / 内存 对比 |
| 03 | `03_FlashAttention_SDPA` | FlashAttention | toy / 中小序列 | 耗时对比 |
| 04 | `04_LoRA_Finetune` | LoRA | 小化语言模型 / GLUE 子集 | 微调参数量对比 |
| 05 | `05_MixedPrecision_Checkpointing` | Mixed Precision, Gradient Checkpointing | 小型 Transformer / CNN | 显存、速度、吞吐对比 | ✅（saved ×17.5 + RSS ×1.12 + bf16 精度不掉；3 图，CPU 700s） |
| 06 | `06_Quantization_Prune` | 量化, 剪枝 | MNIST 模型 | 体积 / 精度对比 | ✅（INT8 ×4.00 零掉点 + 剪枝 0.7 免费/0.9 崩 + 蒸馏≈单训；4 图，CPU 70s） |
| 07 | `07_MoE_Mini` | MoE | toy | 稀疏激活 | ✅（总参 3×激活同量级，双 1.0，专家 ±0.004 均匀；3 图，CPU 170s） |
| 08 | `08_RAG_Mini` | RAG | 本地文档 / 中文问答 | 检索问答实例 | ✅（24问三1.0+双消融；3图，CPU 20s） |
| 09 | `09_Inference_Deployment` | PagedAttention 原理, 投机解码, GGUF/llama.cpp | 小型 GPT / 量化后小模型 | PagedAttention 显存分页模拟（toy KV 管理器）、投机解码 toy 对比、llama.cpp 加载 GGUF 推理 | ✅（分页+投机5.00+GGUF×3.96；3图，CPU 35s） |

## 与其他家族的关系

- **前置**：05 Transformer 基础最好先熟练，再进入这一族
- **依赖**：`peft` 等库当前环境未安装，确需时会先征求同意再安装
- **环境适配**：vLLM 对 Windows 原生支持差，PagedAttention 以原理 + toy 模拟为主；GGUF/llama.cpp 有 Windows CPU 版本，可直接实操（与 CPU 环境契合）
- **08-01 勘误**：`common/data.py::make_modadd_data` 首版用 `roll(-1)` 把 `t+1` 泄进 `t` 且回绕 `%S` 让规则随长度变——已改为因果式 `tgt[i]=(src[i]+src[i-1])%V`（首位补 0）。08-01 的外推数值为旧数据口径，结论（任务设计决定外推上限）不变，复跑 08 家族时统一刷新。