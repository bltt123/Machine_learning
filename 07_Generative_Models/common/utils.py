"""07 家族通用工具：种子、中文字体、参数统计、MNIST 显示。"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_mnist(x: torch.Tensor):
    """单张 (1,28,28) 标准化图 → [0,1] numpy，供 imshow。"""
    arr = x.detach().cpu().numpy()[0] * MNIST_STD + MNIST_MEAN
    return np.clip(arr, 0, 1)
