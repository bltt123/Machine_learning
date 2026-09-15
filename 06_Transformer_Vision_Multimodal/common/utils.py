"""06 家族通用工具：种子、中文字体、参数统计、反归一化。"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unnormalize(x: torch.Tensor):
    """CIFAR-10 反归一化到 [0,1] HWC numpy，供 imshow。"""
    arr = (x.detach().cpu().numpy() * CIFAR_STD + CIFAR_MEAN).transpose(1, 2, 0)
    return np.clip(arr, 0, 1)
