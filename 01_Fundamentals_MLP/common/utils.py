"""01 家族通用工具：随机种子、决策边界绘图、训练曲线。"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_chinese_font():
    """Windows 下 matplotlib 中文字体，避免标题/图例显示为方框。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_data(X: np.ndarray, y: np.ndarray, title: str = "", ax=None):
    """二维散点图。y 约定取值 {-1,+1} 或 {0,1} 均可。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    y01 = (np.asarray(y) > 0).astype(int)
    ax.scatter(X[y01 == 0, 0], X[y01 == 0, 1], c="#4C72B0", s=18, edgecolors="k", linewidths=0.3, label="class -1")
    ax.scatter(X[y01 == 1, 0], X[y01 == 1, 1], c="#DD8452", s=18, edgecolors="k", linewidths=0.3, label="class +1")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "",
    ax=None,
    h: float = 0.02,
    to_pm1=None,
):
    """二维决策边界。y 约定取值 {-1,+1} 或 {0,1} 均可。

    to_pm1: 可选回调，把模型输出映射为 ±1（如 predict 返回 {0,1} 时）。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if to_pm1 is not None:
        zz = to_pm1(model, grid)
    else:
        zz = model.predict(grid)
    zz = np.asarray(zz).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, levels=[-2, 0, 2], colors=["#4C72B0", "#DD8452"])
    ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1)
    y01 = (np.asarray(y) > 0).astype(int)
    ax.scatter(X[y01 == 0, 0], X[y01 == 0, 1], c="#4C72B0", s=18, edgecolors="k", linewidths=0.3, label="class -1")
    ax.scatter(X[y01 == 1, 0], X[y01 == 1, 1], c="#DD8452", s=18, edgecolors="k", linewidths=0.3, label="class +1")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_history(hist: dict, keys=("train_loss", "val_loss"), ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    for k in keys:
        if k in hist:
            ax.plot(hist[k], label=k)
    ax.set_xlabel("epoch")
    ax.legend()
    return ax


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def activation_zero_fractions(model, xb: torch.Tensor) -> list:
    """逐激活层统计输出为 0 的比例（dead ReLU 诊断用），按 net 顺序返回。"""
    fractions, hooks = [], []

    def hook(_module, _inp, out):
        fractions.append(float((out == 0).float().mean()))

    for m in model.net:
        if isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU)):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(xb)
    for h in hooks:
        h.remove()
    return fractions
