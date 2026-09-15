"""06 家族数据：复用 02_CNN_Family 的 CIFAR-10 本地缓存，子集切分固定可复现。"""
from pathlib import Path

import numpy as np
import torch

from common import _compat

cnn_data = _compat.cnn_data

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_cifar10_local(n_train=4000, n_val=800, seed=0, root=None):
    """从 02 家族缓存加载 CIFAR-10 并切 train/val/test。

    返回 (X_train, y_train, X_val, y_val, X_test, y_test)：
    图像 (N,3,32,32) float32 已标准化，标签 int64。
    """
    if root is None:
        root = Path(__file__).resolve().parents[2] / "02_CNN_Family" / "data"
    Xtr, ytr, Xte, yte = cnn_data.load_cifar10_torch(str(root))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train + n_val]
    return (
        Xtr[tr_idx], ytr[tr_idx],
        Xtr[va_idx], ytr[va_idx],
        Xte, yte,
    )


def class_prompts(classes=None):
    """返回与 CIFAR 类别一一对应的零样本文本提示。"""
    return [f"a photo of a {name}" for name in (classes or CIFAR10_CLASSES)]


CAPTION_TMPL = {
    "airplane": "a photo of a airplane flying in the sky.",
    "automobile": "a photo of a automobile driving on the road.",
    "bird": "a photo of a bird perching on a branch.",
    "cat": "a photo of a cat sleeping on a soft rug.",
    "deer": "a photo of a deer standing in a green forest.",
    "dog": "a photo of a dog running on the grass.",
    "frog": "a photo of a frog sitting by a quiet pond.",
    "horse": "a photo of a horse galloping across a field.",
    "ship": "a photo of a ship sailing on the sea.",
    "truck": "a photo of a truck parked near a warehouse.",
}

INSTRUCTION = "Tell me about the image:"


def captions_from_labels(y, classes=None):
    """标签 → 模板描述句（toy 版“标注数据”，供 captioner 训练）。"""
    classes = classes or CIFAR10_CLASSES
    return [CAPTION_TMPL[classes[int(i)]] for i in y]
