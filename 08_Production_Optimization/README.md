# 08 生产级优化与工程机制：RoPE / KV Cache / GQA / FlashAttention / LoRA / 量化 / MoE / RAG / 推理部署

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

- [ ] Positional Encoding / RoPE
- [ ] KV Cache
- [ ] MQA / GQA
- [ ] FlashAttention（SDPA）
- [ ] LoRA / QLoRA / PEFT
- [ ] Mixed Precision（AMP 混合精度训练）
- [ ] Gradient Checkpointing（梯度检查点）
- [ ] 量化（INT8/INT4）
- [ ] 蒸馏
- [ ] 剪枝
- [ ] MoE（稀疏混合专家）
- [ ] RAG
- [ ] PagedAttention / vLLM（原理为主，4GB 显存以体验/读源码思路为主）
- [ ] 投机解码（Draft 模型 + 目标模型并行验证）
- [ ] GGUF / llama.cpp（量化格式与 CPU/GPU 混合部署）

## 项目规划

| 编号 | 项目 | 覆盖机制 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_RoPE_PositionEncoding` | RoPE | toy | 外推长度测试 |
| 02 | `02_KVCache_MQA_GQA` | KV Cache, MQA, GQA | GPT 推理 | Time / 内存 对比 |
| 03 | `03_FlashAttention_SDPA` | FlashAttention | toy / 中小序列 | 耗时对比 |
| 04 | `04_LoRA_Finetune` | LoRA | 小化语言模型 / GLUE 子集 | 微调参数量对比 |
| 05 | `05_MixedPrecision_Checkpointing` | Mixed Precision, Gradient Checkpointing | 小型 Transformer / CNN | 显存、速度、吞吐对比 |
| 06 | `06_Quantization_Prune` | 量化, 剪枝 | MNIST 模型 | 体积 / 精度对比 |
| 07 | `07_MoE_Mini` | MoE | toy | 稀疏激活 |
| 08 | `08_RAG_Mini` | RAG | 本地文档 / 中文问答 | 检索问答实例 |
| 09 | `09_Inference_Deployment` | PagedAttention 原理, 投机解码, GGUF/llama.cpp | 小型 GPT / 量化后小模型 | PagedAttention 显存分页模拟（toy KV 管理器）、投机解码 toy 对比、llama.cpp 加载 GGUF 推理 |

## 与其他家族的关系

- **前置**：05 Transformer 基础最好先熟练，再进入这一族
- **依赖**：`peft` 等库当前环境未安装，确需时会先征求同意再安装
- **环境适配**：vLLM 对 Windows 原生支持差，PagedAttention 以原理 + toy 模拟为主；GGUF/llama.cpp 有 Windows CPU 版本，可直接实操（与 CPU 环境契合）