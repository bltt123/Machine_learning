"""08 家族通用工具：种子、中文字体、参数统计。"""
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


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
