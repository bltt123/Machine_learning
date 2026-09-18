"""08 家族数据：toy 复制/模加任务 + 复用 02 家族 MNIST 缓存（08-06 用）。

- toy 任务长度可变，供外推测试（S_train → S_test）；
- MNIST 走 02_CNN_Family 本地缓存，子集切分固定可复现。
"""
import numpy as np
import torch


def make_copy_data(n, seq_len, vocab=16, seed=0):
    """复制任务：src==tgt，测试模型“记住顺序”的能力。"""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, vocab, size=(n, seq_len))
    return torch.tensor(src, dtype=torch.long), torch.tensor(src, dtype=torch.long)


def make_modadd_data(n, seq_len, vocab=16, seed=0):
    """因果模加任务：tgt[0]=0，tgt[i]=(src[i]+src[i-1])%vocab（i>=1），需相对位置推理。

    只用左侧信息：旧版 roll(-1) 把 t+1 泄进 t，首 token 又需右边界，
    teacher forcing 会学出“不可测首位”，外推/迁移全部失真（08-04 探针实测）。
    """
    rng = np.random.default_rng(seed)
    src = rng.integers(0, vocab, size=(n, seq_len))
    prev = np.concatenate([np.zeros((n, 1), dtype=np.int64), src[:, :-1]], axis=1)
    tgt = (src + prev) % vocab
    tgt[:, 0] = 0
    return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def load_mnist_local(n_train=8000, n_val=1000, seed=0, root=None):
    """从 02 家族缓存加载 MNIST 并切 train/val/test（08-06 用）。

    返回 (X_train, y_train, X_val, y_val, X_test, y_test)：
    图像 (N,1,28,28) float32 已标准化（0.1307/0.3081），标签 int64。
    """
    from pathlib import Path
    from common import _compat
    cnn_data = _compat.cnn_data
    if root is None:
        root = Path(__file__).resolve().parents[2] / "02_CNN_Family" / "data"
    Xtr, ytr, Xte, yte = cnn_data.load_mnist_torch(str(root))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train + n_val]
    return (
        Xtr[tr_idx], ytr[tr_idx],
        Xtr[va_idx], ytr[va_idx],
        Xte, yte,
    )
