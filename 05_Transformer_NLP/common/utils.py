"""05 家族通用工具：随机种子、中文字体、注意力热力图。"""
import random

import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_attention(ax, attn, src_tokens, tgt_tokens, title="", cmap="viridis"):
    """单条注意力矩阵热力图：行=tgt 步，列=src 位置。"""
    im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels([str(t) for t in src_tokens], fontsize=8)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels([str(t) for t in tgt_tokens], fontsize=8)
    ax.set_xlabel("src 位置")
    ax.set_ylabel("解码步")
    if title:
        ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return im


def seq_accuracy(pred, gold):
    """整句全对率。"""
    total = len(gold)
    correct = sum(p == g for p, g in zip(pred, gold))
    return correct / total if total else 0.0


def token_accuracy(pred, gold):
    """token 级准确率（忽略长度差异按总数）。"""
    correct = sum(
        1
        for p, g in zip(pred, gold)
        for pi, gi in zip(p, g)
        if pi == gi
    )
    total = sum(len(g) for g in gold)
    return correct / total if total else 0.0
