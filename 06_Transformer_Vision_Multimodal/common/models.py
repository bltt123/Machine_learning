"""ViT-Tiny 与 SmallCNN 对照模型（06 复用 05 的 MHA 骨架思想，图像版）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Conv2d(patch, stride=patch) 一步切块+投影：32×32 & patch4 → 64 token。"""

    def __init__(self, img_size=32, patch=4, in_ch=3, dim=128):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)              # (B, dim, 8, 8)
        return x.flatten(2).transpose(1, 2)  # (B, 64, dim)


class TransformerBlock(nn.Module):
    """Pre-LN Encoder 块：x = x + MHA(LN(x)); x = x + MLP(LN(x))。"""

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        h = self.norm1(x)
        a, attn = self.attn(h, h, h, need_weights=return_attn, average_attn_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, attn
        return x


class ViTTiny(nn.Module):
    """小 ViT：patch embed + 可学习 pos + CLS + Encoder×L + CLS 分类头。"""

    def __init__(self, img_size=32, patch=4, in_ch=3, num_classes=10,
                 dim=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, in_ch, dim)
        n = self.patch.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def encode_cls(self, x):
        """返回归一化前的 CLS 句向量，供 CLIP 图像塔复用。"""
        B = x.size(0)
        tokens = torch.cat([self.cls.expand(B, -1, -1), self.patch(x)], dim=1) + self.pos
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)[:, 0]

    def forward(self, x, return_attn=False):
        B = x.size(0)
        tokens = self.drop(torch.cat([self.cls.expand(B, -1, -1), self.patch(x)], dim=1) + self.pos)
        attns = []
        for blk in self.blocks:
            if return_attn:
                tokens, attn = blk(tokens, return_attn=True)
                attns.append(attn)
            else:
                tokens = blk(tokens)
        h = self.norm(tokens)
        logits = self.head(h[:, 0])
        if return_attn:
            return logits, attns
        return logits


class MiniTextEncoder(nn.Module):
    """字符级文本塔：char emb + 可学习位置 + Encoder 层，取 EOS 位表示。

    学龄前设定：词汇包含 ``a-z0-9 `` 与 4 个特殊符，序列固定 ``[:max_len]``
    并以 ``<EOS>`` 截断；输出首/末位置的 EOS 向量作为整句表示。
    """

    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self, max_len=64, dim=128, depth=2, heads=4, dropout=0.1):
        super().__init__()
        self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .:")
        self.vocab = {"<pad>": self.PAD, "<s>": self.START, "</s>": self.END, "<unk>": self.UNK}
        for index, char in enumerate(self.chars, start=4):
            self.vocab[char] = index
        self.max_len = max_len
        self.emb = nn.Embedding(len(self.vocab), dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def tokenize(self, texts, max_len=None):
        """把英文短句转为 ``(B, max_len)`` ids，含 ``<s>`` 开头和 ``</s>`` 结尾。"""
        max_len = max_len or self.max_len
        out = torch.full((len(texts), max_len), self.PAD, dtype=torch.long)
        for row, text in enumerate(texts):
            ids = [self.START]
            for char in str(text).lower()[: max_len - 2]:
                ids.append(self.vocab.get(char, self.UNK))
            ids.append(self.END)
            out[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def forward(self, ids):
        tokens = self.emb(ids) + self.pos[:, : ids.size(1)]
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        mask = (ids == self.END).float()
        positions = mask.argmax(dim=1).clamp(min=1)
        rows = torch.arange(ids.size(0), device=ids.device)
        return tokens[rows, positions]


def tokenize_prompts(prompts, max_len=64):
    """便捷函数：把 10 条类名模板转为同一 tokenizer 的 ids。"""
    helper = MiniTextEncoder(max_len=max_len)
    return helper.tokenize(prompts, max_len=max_len).numpy()


def tokenize_texts(texts, max_len=64):
    """便捷函数：把任意短句列表转为同一字符词表的 (N,max_len) ids（06-03 描述句用）。"""
    helper = MiniTextEncoder(max_len=max_len)
    return helper.tokenize(texts, max_len=max_len)


class CLIPMini(nn.Module):
    """极简 CLIP：图像 ViT 塔 + 字符文本塔，双向 InfoNCE 对比训练。"""

    def __init__(self, img_dim=128, img_depth=4, heads=4, txt_dim=128,
                 txt_depth=2, proj_dim=64, max_text_len=64):
        super().__init__()
        self.image_encoder = ViTTiny(dim=img_dim, depth=img_depth, heads=heads)
        self.text_encoder = MiniTextEncoder(max_len=max_text_len, dim=txt_dim, depth=txt_depth, heads=heads)
        self.image_proj = nn.Linear(img_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(txt_dim, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)

    def encode_image(self, images):
        return self.image_encoder.encode_cls(images)

    def encode_text(self, token_ids):
        return self.text_encoder(token_ids.to(next(self.text_encoder.parameters()).device))

    def forward(self, images, token_ids):
        device = next(self.parameters()).device
        images = images.to(device)
        token_ids = token_ids.to(device)
        image_feats = F.normalize(self.image_proj(self.encode_image(images)), dim=-1)
        text_feats = F.normalize(self.text_proj(self.encode_text(token_ids)), dim=-1)
        logits = torch.exp(self.logit_scale.clamp(max=4.6052)) * image_feats @ text_feats.t()
        labels = torch.arange(images.size(0), device=device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class SmallCNN(nn.Module):
    """3 层 3×3 CNN + BN + GAP：约 0.4M 参，作归纳偏置对照。"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                       # 16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),                       # 8
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


class CausalBlock(nn.Module):
    """Pre-LN 因果自注意力块（MiniLLM 用，is_causal=True 下三角遮未来）。"""

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        h = self.norm1(x)
        a, attn = self.attn(h, h, h, attn_mask=torch.triu(
            torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device), 1),
            need_weights=return_attn, average_attn_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return (x, attn) if return_attn else x


class MiniLLM(nn.Module):
    """LLaVA 三段式 toy：冻结视觉塔 + 投影层 + 因果字符 LLM，看图生成模板描述句。

    与真 LLaVA 同构：vision_encoder(frozen) -> projector -> LLM(自回归)。
    本实现只训 projector + LLM，视觉塔用已监督训练好的 ViTTiny 编码 CLS 向量。
    """

    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self, vision: ViTTiny, llm_dim=128, llm_depth=3, heads=4,
                 img_dim=128, n_img_tokens=1, max_len=80, dropout=0.1):
        super().__init__()
        self.vision = vision          # 冻结，CLS 提供图像语义
        for p in self.vision.parameters():
            p.requires_grad = False
        self.tokenizer = MiniTextEncoder()          # 复用字符词表（仅取 vocab/tokenize）
        self.vocab = self.tokenizer.vocab
        self.inv = {v: k for k, v in self.vocab.items()}
        self.n_img = n_img_tokens
        self.projector = nn.Sequential(
            nn.Linear(img_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim * n_img_tokens))
        self.emb = nn.Embedding(len(self.vocab), llm_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, llm_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([CausalBlock(llm_dim, heads, dropout=dropout) for _ in range(llm_depth)])
        self.norm = nn.LayerNorm(llm_dim)
        self.lm_head = nn.Linear(llm_dim, len(self.vocab), bias=False)
        self.llm_dim = llm_dim
        self.max_len = max_len

    def _img_tokens(self, images):
        with torch.no_grad():
            v = self.vision.encode_cls(images)              # (B, img_dim)
        proj = self.projector(v).view(-1, self.n_img, self.llm_dim)  # (B, n_img, D)
        return proj

    def forward(self, images, cap_ids, return_attn=False):
        """cap_ids: (B, Lc) 含 START..END。teacher forcing 预测下一字符（next-token CE，忽略 PAD）。"""
        img_tk = self._img_tokens(images)                   # (B, n_img, D)
        txt_tk = self.emb(cap_ids)                          # (B, Lc, D)
        x = torch.cat([img_tk, txt_tk], dim=1)              # 图像前缀 + 文本
        S = x.size(1)
        x = x + self.pos[:, :S]
        attns = []
        for blk in self.blocks:
            if return_attn:
                x, attn = blk(x, return_attn=True); attns.append(attn)
            else:
                x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)                            # (B, S, V)
        if return_attn:
            return logits, attns
        return logits

    def caption_loss(self, images, cap_ids):
        """把 [IMG]+cap 的 [:-1] 作输入，预测 cap[1:]；忽略 PAD(-100)。

        输入 (B, n_img+Lc-1) → logits (B, n_img+Lc-1, V)；
        第 n_img-1 位（最后图像词）预测 cap[0]=START，对齐 cap[1:]。
        """
        B, Lc = cap_ids.shape
        logits = self.forward(images, cap_ids)               # (B, n_img+Lc, V)
        text_logits = logits[:, self.n_img - 1:-1]           # (B, Lc, V) 预测下一字，最后一位丢弃
        targets = cap_ids.clone()
        targets[targets == self.PAD] = -100
        return F.cross_entropy(text_logits.transpose(1, 2), targets, ignore_index=-100)

    @torch.no_grad()
    def generate(self, images, temperature=0.0):
        """自回归续写描述：从 [IMG, <s>] 起，逐字生成到 </s> 或到 max_len。"""
        self.eval()
        B = images.size(0)
        ids = torch.full((B, 1), self.START, dtype=torch.long, device=images.device)
        img_tk = self._img_tokens(images)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)
        outs = ids.clone()
        for _ in range(self.max_len - self.n_img - 2):
            S = ids.size(1)
            txt_tk = self.emb(ids)
            x = torch.cat([img_tk, txt_tk], dim=1)
            x = x + self.pos[:, :x.size(1)]
            for blk in self.blocks:
                x = blk(x)
            logits = self.lm_head(self.norm(x))[:, -1]      # (B, V)
            if temperature == 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(1)
            outs = torch.cat([outs, nxt.unsqueeze(1)], dim=1)
            ids = outs[:, 1:]
            finished |= nxt == self.END
            if bool(finished.all()):
                break
        return outs

    def decode(self, ids):
        """把一行 char id 还原成字符串（去特殊符）。"""
        chars = []
        for i in ids.tolist():
            if i == self.END:
                break
            if i in (self.START, self.PAD):
                continue
            chars.append(self.inv.get(i, "?"))
        return "".join(chars)
