"""09 家族工具：种子、参数量、中文字体。"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def setup_chinese_font():
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        try:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
