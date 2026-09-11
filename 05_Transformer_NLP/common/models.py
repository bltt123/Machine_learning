"""Seq2Seq Attention（Bahdanau）+ Transformer 从零实现（MLP-style，极简可教）。

目标：与 04-03 同 toy（S=6, vocab=8）同台，证明“全注意力并行”可学到对角/反对角对齐。
实现：单层 GRU 编码器 + Bahdanau 注意力 + GRUCell 解码；Transformer 版：小 Encoder-Decoder
各 2 层，4 头，d=32，FFN=64，已够在 S=6 上拟合，CPU 1~2 分钟。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


def set_torch_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)


# ---------- GRU+Attention（与 04-03 对齐，作对照） ----------
class BahdanauAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid * 2, hid, bias=False)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_h, enc_out):
        """dec_h: (B, hid), enc_out: (B, S, hid) -> (B, S) alpha, (B, hid) ctx"""
        B, S, H = enc_out.shape
        dec = dec_h.unsqueeze(1).expand(-1, S, -1)  # (B,S,hid)
        score = self.v(torch.tanh(self.W(torch.cat([dec, enc_out], dim=-1)))).squeeze(-1)  # (B,S)
        alpha = F.softmax(score, dim=-1)
        ctx = (alpha.unsqueeze(-1) * enc_out).sum(dim=1)  # (B, hid)
        return alpha, ctx


class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab, emb_dim=16, hid=32):
        super().__init__()
        self.vocab = vocab
        self.bos = vocab
        self.emb = nn.Embedding(vocab + 1, emb_dim)
        self.enc = nn.GRU(emb_dim, hid, batch_first=True)
        self.attn = BahdanauAttention(hid)
        self.dec_cell = nn.GRUCell(emb_dim + hid, hid)
        self.out = nn.Linear(hid + hid, vocab)

    def forward(self, src, tgt_input):
        """Teacher forcing：src (B,S), tgt_input (B,S) -> logits (B,S,vocab) + alphas (B,S,S)"""
        emb_e = self.emb(src)  # (B,S,E)
        enc_out, h = self.enc(emb_e)  # enc_out (B,S,H)
        dec_h = h.squeeze(0)  # (B,H)
        logits = []
        alphas = []
        for t in range(src.size(1)):
            alpha, ctx = self.attn(dec_h, enc_out)
            alphas.append(alpha)
            inp = torch.cat([self.emb(tgt_input[:, t]), ctx], dim=-1)
            dec_h = self.dec_cell(inp, dec_h)
            logit = self.out(torch.cat([dec_h, ctx], dim=-1))
            logits.append(logit)
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)

    def greedy(self, src, max_len):
        emb_e = self.emb(src)
        enc_out, h = self.enc(emb_e)
        dec_h = h.squeeze(0)
        prev = torch.full((src.size(0),), self.bos, dtype=torch.long, device=src.device)
        preds = []
        for _ in range(max_len):
            _, ctx = self.attn(dec_h, enc_out)
            inp = torch.cat([self.emb(prev), ctx], dim=-1)
            dec_h = self.dec_cell(inp, dec_h)
            logit = self.out(torch.cat([dec_h, ctx], dim=-1))
            prev = logit.argmax(dim=-1)
            preds.append(prev)
        return torch.stack(preds, dim=1)


# ---------- 极简 Transformer Encoder-Decoder ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape
        Q = self.q(q).view(B, Lq, self.n_head, self.d_k).transpose(1, 2)  # (B,h,Lq,dk)
        K = self.k(k).view(B, Lk, self.n_head, self.d_k).transpose(1, 2)
        V = self.v(v).view(B, Lk, self.n_head, self.d_k).transpose(1, 2)
        scale = 1 / math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B,h,Lq,Lk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Lq, -1)
        out = self.o(ctx)
        return out, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerTiny(nn.Module):
    def __init__(self, vocab, d_model=32, n_head=4, n_layer=2, d_ff=64, max_len=32):
        super().__init__()
        self.vocab = vocab
        self.bos = vocab
        self.emb = nn.Embedding(vocab + 1, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers_enc = nn.ModuleList()
        self.layers_dec = nn.ModuleList()
        for _ in range(n_layer):
            self.layers_enc.append(nn.ModuleDict({
                "self_attn": MultiHeadAttention(d_model, n_head),
                "ln1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)),
                "ln2": nn.LayerNorm(d_model),
            }))
            self.layers_dec.append(nn.ModuleDict({
                "self_attn": MultiHeadAttention(d_model, n_head),
                "cross_attn": MultiHeadAttention(d_model, n_head),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
                "ln3": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)),
            }))
        self.out = nn.Linear(d_model, vocab)
        self.n_layer = n_layer

    def encode(self, src):
        x = self.pos(self.emb(src))
        for lyr in self.layers_enc:
            a, _ = lyr["self_attn"](x, x, x, mask=None)
            x = lyr["ln1"](x + a)
            x = lyr["ln2"](x + lyr["ffn"](x))
        return x

    def decode_layerwise(self, tgt_input, enc_out, causal_mask):
        x = self.pos(self.emb(tgt_input))
        for lyr in self.layers_dec:
            a, _ = lyr["self_attn"](x, x, x, mask=causal_mask)
            x = lyr["ln1"](x + a)
            b, _ = lyr["cross_attn"](x, enc_out, enc_out, mask=None)
            x = lyr["ln2"](x + b)
            x = lyr["ln3"](x + lyr["ffn"](x))
        return x

    def forward(self, src, tgt_input):
        enc_out = self.encode(src)
        B, S = tgt_input.shape
        causal = torch.tril(torch.ones(S, S, device=src.device)).view(1, 1, S, S)
        dec = self.decode_layerwise(tgt_input, enc_out, causal)
        logits = self.out(dec)
        return logits

    def greedy(self, src, max_len):
        B = src.size(0)
        device = src.device
        enc_out = self.encode(src)
        preds = torch.full((B, 0), 0, dtype=torch.long, device=device)
        # incremental: feed [BOS, pred...] each step (simple, not cached KV)
        seq = torch.full((B, 1), self.bos, dtype=torch.long, device=device)
        out_list = []
        for _ in range(max_len):
            S = seq.size(1)
            causal = torch.tril(torch.ones(S, S, device=device)).view(1, 1, S, S)
            dec = self.decode_layerwise(seq, enc_out, causal)
            logit = self.out(dec[:, -1:])
            nxt = logit.argmax(dim=-1).squeeze(1)
            out_list.append(nxt)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        return torch.stack(out_list, dim=1)

    def greedy_with_attn(self, src, max_len):
        """Return preds and cross-attn (last layer, mean heads)."""
        B = src.size(0)
        device = src.device
        enc_out = self.encode(src)
        seq = torch.full((B, 1), self.bos, dtype=torch.long, device=device)
        preds = []
        attns = []
        for _ in range(max_len):
            S = seq.size(1)
            x = self.pos(self.emb(seq))
            causal = torch.tril(torch.ones(S, S, device=device)).view(1, 1, S, S)
            # run decoder layers, capture last cross attn
            last_cross = None
            for idx, lyr in enumerate(self.layers_dec):
                a, _ = lyr["self_attn"](x, x, x, mask=causal)
                x = lyr["ln1"](x + a)
                b, attn = lyr["cross_attn"](x, enc_out, enc_out, mask=None)
                if idx == self.n_layer - 1:
                    last_cross = attn  # (B,h,S,Ssrc)
                x = lyr["ln2"](x + b)
                x = lyr["ln3"](x + lyr["ffn"](x))
            dec = x
            logit = self.out(dec[:, -1:])
            nxt = logit.argmax(dim=-1).squeeze(1)
            # last token's cross attn (S-1)
            # avg heads
            mean_attn = last_cross.mean(dim=1)[:, -1, :]  # (B, Ssrc)
            attns.append(mean_attn.detach().cpu().float().numpy() if len(mean_attn.shape)==2 else mean_attn)
            preds.append(nxt)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1), torch.stack([torch.tensor(a) if isinstance(a, list) else a for a in attns], dim=1)


# ---------- GPT Decoder-only（自回归 LM：下一个词预测） ----------
class GPTForLM(nn.Module):
    """Decoder-only：因果自注意力（下三角 mask），无 Encoder/cross，LM 头 tie 可选。"""

    def __init__(self, vocab, d_model=32, n_head=4, n_layer=2, d_ff=64, max_len=64, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleDict({
                "self_attn": MultiHeadAttention(d_model, n_head),
                "ln1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)),
                "ln2": nn.LayerNorm(d_model),
            }))
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.n_layer = n_layer

    def forward(self, x, return_attn=False):
        """x: (B, S) → logits (B, S, vocab)，因果 mask 下三角。"""
        h = self.pos(self.emb(x))
        B, S = x.shape
        causal = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
        attns = []
        for lyr in self.layers:
            a, attn = lyr["self_attn"](h, h, h, mask=causal)
            if return_attn:
                attns.append(attn)
            h = lyr["ln1"](h + a)
            h = lyr["ln2"](h + lyr["ffn"](h))
        h = self.dropout(h)
        logits = self.lm_head(h)
        if return_attn:
            return logits, attns
        return logits

    def generate(self, prefix, max_new=8, temperature=1.0, top_k=None):
        """自回归生成：prefix (B, S0) → 续写 max_new 个 token（贪心/采样）。"""
        self.eval()
        seq = prefix.clone()
        with torch.no_grad():
            for _ in range(max_new):
                logits = self.forward(seq)  # (B, S, vocab)
                nxt_logits = logits[:, -1, :] / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(nxt_logits, min(top_k, nxt_logits.size(-1)))
                    nxt_logits[nxt_logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(nxt_logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1) if temperature != 0 else probs.argmax(dim=-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
        return seq


# ---------- Mamba/Mini-SSM（线性复杂度 selective scan 极简 toy 版） ----------
class MambaBlock(nn.Module):
    """极简 selective SSM：h_t = exp(A·dt)·h_{t-1} + B·x_t；y = C·h + D·x。"""

    def __init__(self, d_model, d_state=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.2 - 1.0)
        self.D = nn.Parameter(torch.ones(d_model) * 0.5)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B,S,D) → (B,S,D)  线性扫，一次前向 O(S·D·N)。"""
        B, S, D = x.shape
        xz = self.in_proj(x)  # (B,S,2D)
        x1, z = xz.chunk(2, dim=-1)
        x1 = F.silu(x1)
        # depthwise conv 沿序列
        x1 = self.conv(x1.transpose(1, 2)).transpose(1, 2)
        x1 = F.silu(x1)
        dt = F.softplus(self.dt_proj(x1))  # (B,S,D)
        Bc = self.B_proj(x1)  # (B,S,N)
        Cc = self.C_proj(x1)  # (B,S,N)
        A = -torch.exp(self.A_log)  # (D,N) 负
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(S):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B,D,1)
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)  # (B,D,N)
            B_bar = Bc[:, t, :].unsqueeze(1) * dt_t  # (B,D,N)
            h = h * A_bar + B_bar * x1[:, t, :].unsqueeze(-1)
            y_t = (h * Cc[:, t, :].unsqueeze(1)).sum(dim=-1) + self.D * x1[:, t, :]
            y_t = y_t * F.silu(z[:, t, :])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = self.out_proj(y)
        return self.norm(y + x)


class MambaForLM(nn.Module):
    """Mamba LM：Embedding + PE + MambaBlock×L + LM head，无注意力，O(n)。"""

    def __init__(self, vocab, d_model=32, n_layer=2, d_state=8, max_len=64):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        h = self.pos(self.emb(x))
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.lm_head(h)

    def generate(self, prefix, max_new=8, temperature=0):
        self.eval()
        seq = prefix.clone()
        with torch.no_grad():
            for _ in range(max_new):
                logits = self.forward(seq)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True) if temperature == 0 else torch.multinomial(F.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1), 1)
                seq = torch.cat([seq, nxt], dim=1)
        return seq


# ---------- BERT Encoder-only (微调范式：[CLS] 分类) ----------
class BERTForCls(nn.Module):
    """Encoder-only：前置 [CLS]，双向自注意力无因果 mask，取 CLS 做分类。"""

    def __init__(self, vocab, d_model=32, n_head=4, n_layer=2, d_ff=64, num_labels=2, max_len=64, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.cls_id = vocab  # [CLS] 放在词表末尾
        self.emb = nn.Embedding(vocab + 1, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleDict({
                "self_attn": MultiHeadAttention(d_model, n_head),
                "ln1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)),
                "ln2": nn.LayerNorm(d_model),
            }))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)
        self.n_layer = n_layer

    def forward(self, x, return_attn=False):
        """x: (B, S) 整数 id，自动前置 CLS。return logits (B, num_labels)."""
        B = x.size(0)
        cls = torch.full((B, 1), self.cls_id, dtype=torch.long, device=x.device)
        x = torch.cat([cls, x], dim=1)  # (B, S+1)
        h = self.pos(self.emb(x))
        attns = []
        for lyr in self.layers:
            a, attn = lyr["self_attn"](h, h, h, mask=None)
            if return_attn:
                attns.append(attn)
            h = lyr["ln1"](h + a)
            h = lyr["ln2"](h + lyr["ffn"](h))
        h = self.dropout(h)
        logits = self.classifier(h[:, 0])  # CLS
        if return_attn:
            return logits, attns
        return logits
