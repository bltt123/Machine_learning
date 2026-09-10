"""03 分割家族通用工具：随机种子、中文字体、参数量统计。"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size=(1, 3, 128, 128)) -> int:
    """统计单张图前向 MACs（Conv2d 含 groups + Linear）。"""
    flops = [0]

    def conv_hook(m, inp, out):
        kh, kw = m.kernel_size
        flops[0] += int(out.numel() * (m.in_channels / m.groups) * kh * kw)

    def linear_hook(m, inp, out):
        flops[0] += int(m.in_features * m.out_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_size))
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return flops[0]
