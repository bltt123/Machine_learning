"""Plotting + misc utils. Agg backend, English labels, mean±std bands."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(x, w=21):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def curve_with_band(ax, xs, mean, std, label, color=None):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    (line,) = ax.plot(xs, mean, label=label, color=color)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=line.get_color())


def savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def grid_values_4x4(V, fmt="{:6.1f}"):
    lines = []
    for r in range(4):
        lines.append(" ".join(fmt.format(V[r * 4 + c]) for c in range(4)))
    return "\n".join(lines)


def cliff_policy_arrows(pi):
    sym = {0: "^", 1: ">", 2: "v", 3: "<"}
    rows = []
    for r in range(4):
        row = []
        for c in range(12):
            if (r, c) == (3, 0):
                row.append("S")
            elif (r, c) == (3, 11):
                row.append("G")
            elif r == 3 and 1 <= c <= 10:
                row.append("x")
            else:
                row.append(sym[int(pi[r * 12 + c])])
        rows.append(" ".join(row))
    return "\n".join(rows)
