"""04 序列家族通用工具：随机种子、中文字体、NER 可视化配色。"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


# BIO 标签配色：B 深、I 浅，O 灰
TAG_COLORS = {
    "O": "#D5D8DC",
    "B-PER": "#E74C3C", "I-PER": "#F5B7B1",
    "B-LOC": "#2E86C1", "I-LOC": "#AED6F1",
    "B-ORG": "#1E8449", "I-ORG": "#A9DFBF",
}


def draw_tokens(ax, tokens, tags, title="", fontsize=11, tag_label=True):
    """把一句 (tokens, tags) 画成彩色 token 方块，标签写在方块下方。"""
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1.6 if tag_label else 1.0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        color = TAG_COLORS.get(tag, "#FFFFFF")
        ax.add_patch(plt.Rectangle((i + 0.04, 0.55), 0.92, 0.62,
                                   facecolor=color, edgecolor="#555555", linewidth=0.8))
        ax.text(i + 0.5, 0.86, tok, ha="center", va="center", fontsize=fontsize)
        if tag_label:
            ax.text(i + 0.5, 0.28, tag, ha="center", va="center",
                    fontsize=fontsize - 3, color="#333333")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
