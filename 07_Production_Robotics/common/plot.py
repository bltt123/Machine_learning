"""Plotting utils for 07. Agg backend, English labels."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def curve_with_band(ax, xs, mean, std, label, color=None):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    (line,) = ax.plot(xs, mean, label=label, color=color)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=line.get_color())


def savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
