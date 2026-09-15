"""07 家族数据：复用 02_CNN_Family 的 MNIST 本地缓存，子集切分固定可复现。"""
from pathlib import Path

import numpy as np

from common import _compat

cnn_data = _compat.cnn_data


def load_mnist_local(n_train=6000, n_val=1000, seed=0, root=None):
    """从 02 家族缓存加载 MNIST 并切 train/val/test。

    返回 (X_train, y_train, X_val, y_val, X_test, y_test)：
    图像 (N,1,28,28) float32 已标准化（0.1307/0.3081），标签 int64。
    """
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
