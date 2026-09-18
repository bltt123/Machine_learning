"""08-05：AMP(bf16/fp32) 与梯度检查点的内存/精度/耗时口径工具（CPU 上以内存+重算为主）。

- measure_peak：子进程跑 fn，psutil 读进程 Peak RSS（Windows 上 WorkingSet 最稳）；
  Windows spawn 要求 fn 可 pickle → 约定传模块级函数（common.engine 注册的 task_*）。
- GradCkptBlock / CkptMLP：普通 vs 检查点两种前向，saved_tensors_hooks 精确量激活字节。
- 08-06（同文件追加）：MNIST 分类器训练/评估 + 蒸馏 loss，供量化/剪枝/蒸馏三对照。
"""
import multiprocessing as mp

import torch
import torch.nn as nn


def _worker(queue, target, args, kwargs):
    import os, psutil
    proc = psutil.Process(os.getpid())
    result = target(*args, **kwargs)
    peak = proc.memory_info().peak_wset
    queue.put((result, peak))


def measure_peak(target, *args, **kwargs):
    """子进程跑 target，返回 (返回值, 峰值内存 bytes)。

    Windows spawn 只能 pickle 模块级函数：target 必须是 import 可得的全局函数
    （如 engine.task_ckpt_full），不能是 notebook 内联闭包。
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(queue, target, args, kwargs))
    p.start()
    result, peak = queue.get()
    p.join()
    return result, peak


def task_ckpt_train(use_ckpt, epochs=2):
    """模块级可 pickle 任务：短训 CkptMLP，返回末步 loss。"""
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn
    torch.manual_seed(0)
    m = CkptMLP(dim=512, depth=24, vocab=16)
    loader = DataLoader(TensorDataset(torch.randn(64, 16, 512),
                                      torch.randint(0, 16, (64, 16))), batch_size=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = None
    for _ in range(epochs):
        for src, tgt in loader:
            logits = m(src, use_ckpt=use_ckpt)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 16), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


class GradCkptBlock(nn.Module):
    """带可选梯度检查点的 Transformer 块：checkpointing 时前向不存激活，反向重算。"""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def _forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))

    def forward(self, x, use_ckpt=False):
        from torch.utils.checkpoint import checkpoint
        if use_ckpt and self.training and x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class CkptMLP(nn.Module):
    """K 层块堆叠，测激活内存口径（dim=512，K=24）。"""

    def __init__(self, dim=512, depth=24, vocab=16):
        super().__init__()
        self.layers = nn.ModuleList([GradCkptBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, x, use_ckpt=False):
        for layer in self.layers:
            x = layer(x, use_ckpt=use_ckpt)
        return self.head(x)


def activation_memory(model, src, use_ckpt=False):
    """用 saved_tensors_hooks 精确量前向保存的激活字节数，返回 (bytes, loss张量)。"""
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        logits = model(src, use_ckpt=use_ckpt)
        loss = logits.sum()
    return sum(saved), loss


@torch.no_grad()
def mnist_accuracy(model, loader, device="cpu"):
    """MNIST test 口径：整集准确率。"""
    model.eval()
    ok, tot = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        ok += (model(xb).argmax(-1) == yb).sum().item(); tot += len(xb)
    return ok / tot


def fit_mnist(model, train_loader, epochs=5, lr=1e-3, device="cpu"):
    """MNIST 短训：Adam + CE，打印每轮 loss（teacher/student 共用）。"""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        print(f"epoch {ep:02d} | loss {tot / n:.4f}", flush=True)
    return model


def fit_distill(student, teacher, train_loader, epochs=5, lr=1e-3, T=4.0, alpha=0.7, device="cpu"):
    """蒸馏：α·T²·KL(soft) + (1-α)·CE(hard)，T=4 软化，α=0.7 偏向老师。"""
    teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        student.train(); tot, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                t_logits = teacher(xb)
            s_logits = student(xb)
            soft = nn.functional.kl_div(
                nn.functional.log_softmax(s_logits / T, dim=-1),
                nn.functional.softmax(t_logits / T, dim=-1),
                reduction="batchmean") * (T * T)
            hard = nn.functional.cross_entropy(s_logits, yb)
            loss = alpha * soft + (1 - alpha) * hard
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        print(f"distill ep {ep:02d} | loss {tot / n:.4f}", flush=True)
    return student
