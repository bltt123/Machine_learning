"""08 家族模型库：01 三位置编码 + 02 KV/GQA + 03 Flash + 04 LoRA + 06 量化/剪枝/蒸馏 + 07 MoE 稀疏专家。

设计：同一 Transformer 骨架，只换位置/注意力/微调模块，参数量对齐，
差异才能归因于机制而非容量。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """手写 LoRA：冻结 W，新增 B@A（r≪min(d,k)），推理可 merge 回 W。"""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling

    @torch.no_grad()
    def merged(self):
        w = self.base.weight + (self.B @ self.A) * self.scaling
        b = None if self.base.bias is None else self.base.bias
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=b is not None)
        out.weight.copy_(w)
        if b is not None:
            out.bias.copy_(b)
        return out


def freeze_non_lora(model: nn.Module):
    """冻结除 LoRA A/B 外全部参数（标准 LoRA 语义：基座全冻）。"""
    for name, p in model.named_parameters():
        tail = name.split(".")[-1]
        p.requires_grad = tail in ("A", "B")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inject_lora(model: nn.Module, targets=("qkv", "out"), r=8, alpha=16):
    """按属性名后缀匹配，把模型里的 nn.Linear 包成 LoRALinear（顶层 head 也支持）。"""
    count = 0
    for name, module in list(model.named_modules()):
        if not name:
            continue
        suffix = name.split(".")[-1]
        if suffix in targets and isinstance(module, nn.Linear):
            if "." in name:
                parent_path, child = name.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
            else:
                parent, child = model, name
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha))
            count += 1
    return count


class AbsPosEncoding(nn.Module):
    """可学习绝对位置：pos 表 (max_len, d)，超长截断。"""

    def __init__(self, max_len, dim):
        super().__init__()
        self.max_len = max_len
        self.table = nn.Embedding(max_len, dim)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).clamp(max=self.max_len - 1)
        return x + self.table(pos).unsqueeze(0)


class SinCosPosEncoding(nn.Module):
    """固定 sin-cos：超长按公式外推（天然支持任意长度）。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        S, d = x.size(1), self.dim
        pos = torch.arange(S, device=x.device).float().unsqueeze(1)
        i = torch.arange(d // 2, device=x.device).float()
        freq = torch.exp(-math.log(10000.0) * i / (d // 2))
        pe = torch.zeros(S, d, device=x.device)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return x + pe.unsqueeze(0)


def _rotary_freq(dim, device, base=10000.0):
    half = dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    return inv


def apply_rope(q, k, inv_freq):
    """对 Q/K 每对维度做旋转：[x1,x2] → [x1·cos+x2·sin偏移]，相对位置自然编码。"""
    S = q.size(2)
    t = torch.arange(S, device=q.device).float()
    freqs = torch.outer(t, inv_freq)                       # (S, half)
    emb = torch.cat([freqs, freqs], dim=-1)                # (S, dim)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]

    def rot(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos[..., :x1.size(-1)] - x2 * sin[..., :x1.size(-1)],
                          x1 * sin[..., :x1.size(-1)] + x2 * cos[..., :x1.size(-1)]], dim=-1)
    return rot(q), rot(k)


class RopeAttention(nn.Module):
    """单头 MHA + RoPE：Q/K 先旋转再点积，相对位移 m-n 自然进入 cos/sin。"""

    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("inv_freq", _rotary_freq(self.dk, torch.device("cpu")), persistent=False)

    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.heads, self.dk).transpose(1, 2)
        k = k.view(B, S, self.heads, self.dk).transpose(1, 2)
        v = v.view(B, S, self.heads, self.dk).transpose(1, 2)
        inv = self.inv_freq.to(x.device)
        q, k = apply_rope(q, k, inv)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        attn = self.drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.out(out)


class PlainAttention(nn.Module):
    """无旋转普通 MHA（Abs/SinCos 分支用，位置信息已加在输入上）。"""

    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.heads, self.dk).transpose(1, 2)
        k = k.view(B, S, self.heads, self.dk).transpose(1, 2)
        v = v.view(B, S, self.heads, self.dk).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        attn = self.drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.out(out)


class TinyBlock(nn.Module):
    def __init__(self, dim, heads, rope=False, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = RopeAttention(dim, heads, dropout) if rope else PlainAttention(dim, heads, dropout)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ToyGPT(nn.Module):
    """Encoder-only toy：emb + 位置编码(三选一) + 2 层块 + per-token 分类头。"""

    def __init__(self, vocab=16, dim=64, depth=2, heads=4, max_len=64, mode="rope", dropout=0.0):
        super().__init__()
        assert mode in ("abs", "sincos", "rope")
        self.mode = mode
        self.emb = nn.Embedding(vocab, dim)
        if mode == "abs":
            self.pos = AbsPosEncoding(max_len, dim)
        elif mode == "sincos":
            self.pos = SinCosPosEncoding(dim)
        else:
            self.pos = None
        self.blocks = nn.ModuleList([TinyBlock(dim, heads, rope=(mode == "rope"), dropout=dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x = self.emb(ids)
        if self.pos is not None:
            x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


class GQADecoder(nn.Module):
    """08-02：因果 toy 解码器，q 头 H、kv 头 G（G=H 为 MHA，G=1 为 MQA）。

    训练用 next-token 目标（logits[:, :-1] 预测 tgt[:, 1:]）；
    推理支持逐 token 解码 + KV Cache：每步只算新 token 的 q/k/v，
    k/v 拼到缓存上，注意力只看缓存长度。
    """

    def __init__(self, vocab=16, dim=64, depth=2, q_heads=4, kv_heads=4, dropout=0.0):
        super().__init__()
        assert dim % q_heads == 0 and q_heads % kv_heads == 0
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.dk = dim // q_heads
        self.rep = q_heads // kv_heads
        self.emb = nn.Embedding(vocab, dim)
        self.qkv_w = nn.ModuleList([nn.Linear(dim, dim + 2 * self.dk * kv_heads) for _ in range(depth)])
        self.out_w = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                                nn.Linear(dim * 4, dim)) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.drop = nn.Dropout(dropout)

    def _split(self, x, layer):
        B, S, _ = x.shape
        proj = self.qkv_w[layer](x)
        q, k, v = proj.split([self.q_heads * self.dk, self.kv_heads * self.dk,
                              self.kv_heads * self.dk], dim=-1)
        q = q.view(B, S, self.q_heads, self.dk).transpose(1, 2)
        k = k.view(B, S, self.kv_heads, self.dk).transpose(1, 2)
        v = v.view(B, S, self.kv_heads, self.dk).transpose(1, 2)
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)
        return q, k, v

    def forward(self, ids):
        x = self.emb(ids)
        B, S, _ = x.shape
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        for i in range(len(self.qkv_w)):
            h = self.n1[i](x)
            q, k, v = self._split(h, i)
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
            attn = attn.masked_fill(mask[None, None], float("-inf"))
            attn = self.drop(torch.softmax(attn, dim=-1))
            o = (attn @ v).transpose(1, 2).reshape(B, S, -1)
            x = x + self.out_w[i](o)
            x = x + self.mlp[i](self.n2[i](x))
        return self.head(self.norm(x))

    def next_token_loss(self, ids):
        """LM 口径：logits[:, :-1] 预测 ids[:, 1:]（与 generate 的目标一致）。"""
        logits = self.forward(ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, prefix, steps, use_cache=True, _caches=None):
        """自回归生成：use_cache=True 只算新 token（KV Cache），否则全序列重算。

        等价性说明：首步缓存为空时只见 prefix 尾部若干 token（预填充缺失），
        与全序列分支“首步即见全 prefix”语义不同——这是 toy 为省预填充而接受的
        近似（真实现首步全量预填充）。故一致性验证只要求“同分支自洽 + 加速比”，
        不要求 cache==nocache 逐 token 相等；fig3 如实记录分歧率。
        """
        self.eval()
        ids = prefix.clone()
        caches = [None] * len(self.qkv_w) if use_cache else None
        if _caches is not None:
            caches = _caches
        for _ in range(steps):
            cur = ids if not use_cache else ids[:, -1:]
            x = self.emb(cur)
            B, S, _ = x.shape
            for i in range(len(self.qkv_w)):
                h = self.n1[i](x)
                q, k, v = self._split(h, i)
                if use_cache:
                    if caches[i] is None:
                        caches[i] = (k.detach().clone(), v.detach().clone())
                    else:
                        ck, cv = caches[i]
                        k = torch.cat([ck, k], dim=2)
                        v = torch.cat([cv, v], dim=2)
                        caches[i] = (k.detach().clone(), v.detach().clone())
                L = k.size(2)
                start = L - S
                qpos = torch.arange(start, L, device=x.device)
                kpos = torch.arange(L, device=x.device)
                bias = (kpos[None, :] > qpos[:, None])[None, None]
                attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
                attn = attn.masked_fill(bias, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                o = (attn @ v).transpose(1, 2).reshape(B, S, -1)
                x = x + self.out_w[i](o)
                x = x + self.mlp[i](self.n2[i](x))
            logits = self.head(self.norm(x))[:, -1]
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
        return ids

    @torch.no_grad()
    def prefill(self, prefix):
        """真实现语义：全量预填充 prefix 的 KV，返回 caches 供 generate(_caches=) 续写。"""
        self.eval()
        x = self.emb(prefix)
        caches = []
        for i in range(len(self.qkv_w)):
            h = self.n1[i](x)
            q, k, v = self._split(h, i)
            caches.append((k.detach().clone(), v.detach().clone()))
            L = k.size(2)
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
            attn = attn.masked_fill(mask[None, None], float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o = (attn @ v).transpose(1, 2).reshape(x.size(0), L, -1)
            x = x + self.out_w[i](o)
            x = x + self.mlp[i](self.n2[i](x))
        return caches


def naive_attention(q, k, v):
    """标准注意力：整块 S×S 物化，返回 (输出, 物化元素数)。"""
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    p = torch.softmax(s, dim=-1)
    return p @ v, s.numel()


def flash_attention(q, k, v, block=64):
    """Flash 式分块注意力：online softmax 两遍（max/sum 分块归约），S×S 永不物化。

    返回 (输出, HBM 访问元素数)：Q/K/V 读 + O 写 + 每块统计量。
    数学等价 naive（同一 softmax，只是分块算），误差仅 float 舍入 ~1e-7。
    """
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    hbm = 0
    n_blocks = (S + block - 1) // block
    for i in range(n_blocks):
        qs = slice(i * block, min((i + 1) * block, S))
        S1 = qs.stop - qs.start
        m = torch.full((B, H, S1), float("-inf"), device=q.device)
        l = torch.zeros(B, H, S1, device=q.device)
        acc = torch.zeros(B, H, S1, D, device=q.device)
        for j in range(n_blocks):
            ks = slice(j * block, min((j + 1) * block, S))
            s = (q[:, :, qs, :] @ k[:, :, ks, :].transpose(-2, -1)) * scale
            hbm += q[:, :, qs, :].numel() + k[:, :, ks, :].numel() + v[:, :, ks, :].numel()
            m_new = torch.maximum(m, s.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            beta = torch.exp(s - m_new.unsqueeze(-1))
            l = l * alpha + beta.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + beta @ v[:, :, ks, :]
            m = m_new
        out[:, :, qs, :] = acc / l.unsqueeze(-1)
        hbm += out[:, :, qs, :].numel() + 2 * m.numel()
    return out, hbm


def naive_hbm(S, D):
    """标准注意力 HBM 访问量（元素数）：QK^T 物化 S² + softmax 两遍 + ×V。"""
    return 3 * S * S + 2 * S * D


class TinyMNIST(nn.Module):
    """08-06：MNIST 小分类器（421,642 参 ≈ fp32 1.69MB）：Conv×2 + FC，供量化/剪枝/蒸馏三对照。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class TinyMNISTSmall(nn.Module):
    """08-06 蒸馏学生：通道减半（105,866 参 ≈ 老师 25%）。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def model_bytes_fp32(model):
    """fp32 参数字节数 = 参数量 × 4。"""
    return sum(p.numel() for p in model.parameters()) * 4


def quantize_int8_per_tensor(model):
    """逐张量对称 INT8 PTQ：scale=max|w|/127，返回 {name: (qint8 Tensor, scale)}。"""
    out = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            w = p.detach().float()
            scale = w.abs().max().clamp_min(1e-8).item() / 127.0
            q = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)
            out[name] = (q, scale)
    return out


def dequantize_state(qstate):
    """INT8 状态反量化回 fp32（torch.quantize 失败时的手算 fallback，数学同 PTQ）。"""
    return {k: (q.float() * s) for k, (q, s) in qstate.items()}


def quantized_bytes(qstate):
    """INT8 体积：int8 参数 1B + 每张量 1 个 fp32 scale。"""
    n = sum(q.numel() for q, _ in qstate.values())
    return n + 4 * len(qstate)


def global_magnitude_mask(model, sparsity):
    """全局幅度剪枝 mask：最小 |w| 的 sparsity 比例置零（embedding/BN 不剪，只剪 Linear/Conv）。"""
    allw = torch.cat([p.detach().float().flatten() for m in model.modules()
                      if isinstance(m, (nn.Linear, nn.Conv2d)) for p in [m.weight]])
    thr = torch.quantile(allw.abs(), sparsity)
    masks = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            masks[name] = (m.weight.abs() > thr).float()
    return masks, thr.item()


def apply_mask(model, masks):
    """就地应用剪枝 mask（w *= mask），返回实际稀疏率。"""
    with torch.no_grad():
        for name, m in model.named_modules():
            if name in masks:
                m.weight.mul_(masks[name].to(m.weight.device))
    tot = sum(m.weight.numel() for name, m in model.named_modules() if name in masks)
    zero = sum((m.weight == 0).sum().item() for name, m in model.named_modules() if name in masks)
    return zero / tot


class MoEMLP(nn.Module):
    """08-07：稀疏混合专家。E 个专家（小 MLP），门控 top-k=2 稀疏激活 + 负载均衡 aux loss。

    - 前向只算 top-k 个专家（其余跳过），FLOPs ≈ k/E；
    - aux loss = E·Σ(mean(g)·mean(onehot))，压门控分布均匀，防赢家通吃。
    """

    def __init__(self, dim=128, expert_hidden=256, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, expert_hidden), nn.GELU(), nn.Linear(expert_hidden, dim))
            for _ in range(num_experts)])
        self._aux = None

    def forward(self, x):
        B, S, D = x.shape
        logits = self.gate(x)                                  # (B,S,E)
        probs = torch.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)            # (B,S,k)
        topv = topv / topv.sum(-1, keepdim=True)               # top-k 内重归一
        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, D)
        flat_i = topi.reshape(-1, self.top_k)
        flat_v = topv.reshape(-1, self.top_k)
        for e, expert in enumerate(self.experts):
            rows, cols = (flat_i == e).nonzero(as_tuple=True)  # 用到专家 e 的 token
            if rows.numel():
                out.view(-1, D)[rows] += flat_v[rows, cols].unsqueeze(-1) * expert(flat_x[rows])
        with torch.no_grad():
            hard = torch.zeros_like(probs).scatter_(-1, topi.detach(), 1.0)
            mean_h = hard.mean(dim=(0, 1))
        mean_p = probs.mean(dim=(0, 1))
        # aux 可导（经 mean_p 回传门控），mean_h 固定为 hard 统计量
        self._aux = (self.num_experts * (mean_p * mean_h).sum())
        return out

    def aux_loss(self):
        return self._aux if self._aux is not None else torch.tensor(0.0)


class MoEBlock(nn.Module):
    """08-07：Transformer 块 + MoE 取代 dense MLP（注意力保持 dense）。"""

    def __init__(self, dim=128, heads=4, num_experts=8, top_k=2, expert_hidden=256):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = PlainAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.moe = MoEMLP(dim, expert_hidden, num_experts, top_k)

    def forward(self, x, return_aux=False):
        x = x + self.attn(self.n1(x))
        h = self.n2(x)
        out = self.moe(h)
        return x + out

    def aux(self):
        return self.moe.aux_loss()


class MoEGPT(nn.Module):
    """08-07：toy MoE 解码器（emb + sincos 位置 + N×MoE块 + 分类头）。

    位置编码是必需的：无位置的注意力是置换不变集模型，学不会模加这类
    顺序任务（探针实测：无位置时复制能学、模加全灭；加 sincos 后恢复）。
    """

    def __init__(self, vocab=16, dim=128, depth=2, heads=4, num_experts=8, top_k=2,
                 expert_hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = SinCosPosEncoding(dim)
        self.blocks = nn.ModuleList([MoEBlock(dim, heads, num_experts, top_k, expert_hidden)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x = self.pos(self.emb(ids))
        for blk in self.blocks:
            x = blk(x)
        self._aux = sum(b.aux() for b in self.blocks) / len(self.blocks)
        return self.head(self.norm(x))

    def aux_loss(self):
        aux = getattr(self, "_aux", None)
        return aux if isinstance(aux, torch.Tensor) else torch.tensor(0.0)


class DenseGPT(nn.Module):
    """08-07 对照基线：同 dim/depth + sincos 位置，dense MLP。"""

    def __init__(self, vocab=16, dim=128, depth=2, heads=4, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = SinCosPosEncoding(dim)
        self.blocks = nn.ModuleList([TinyBlock(dim, heads) for _ in range(depth)])
        for blk in self.blocks:
            blk.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        x = self.pos(self.emb(ids))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
