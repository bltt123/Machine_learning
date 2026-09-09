"""02 CNN 家族通用工具：随机种子、中文字体、参数统计。"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_chinese_font():
    """Windows 下 matplotlib 中文字体，避免标题/图例显示为方框。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size=(1, 3, 32, 32)) -> int:
    """统计单张图前向 MACs（乘加次数）。

    Conv2d: Cout·Hout·Wout·(Cin/groups)·k·k；Linear: Cin·Cout；BN/激活/池化按惯例忽略。
    口径说明：MACs × 2 ≈ FLOPs，README/图表统一用 MACs（模型间相对比较不受口径影响）。
    """
    handles, total = [], [0]

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += out.numel() * (m.in_channels // m.groups) * k

    def linear_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_size))
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return total[0]


def count_flops(model, input_size=(1, 3, 32, 32)) -> int:
    """统计一次前向的乘加次数（MACs，≈2×FLOPs 的习惯口径）。

    覆盖 Conv2d（含 depthwise/groups：per_out = k²·in/groups·h·w）与 Linear。
    用法：count_flops(model)，input_size 需与模型实际输入一致。
    """
    import torch.nn as nn

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
        model(torch.zeros(input_size))
    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return flops[0]
