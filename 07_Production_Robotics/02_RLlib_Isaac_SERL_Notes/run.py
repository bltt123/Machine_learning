"""02_RLlib_Isaac_SERL_Notes run.py — production toolchain landscape (reading-level).

No training: this station is a concept map. It renders two figures that summarize
the RL toolchain ecosystem and the compute-scale ladder, matching the README notes.
Experiments:
  E1: toolchain feature heatmap (maturity/docs/distributed/multi-agent/GPU-sim/LLM/robot)
  E2: compute-scale ladder (env-steps/sec, log scale)
Figures: figs/fig1_toolchain_heatmap.png, figs/fig2_scale_ladder.png
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.plot import savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

# 0 = weak / not supported, 1 = partial, 2 = strong / native
TOOLS = ["SB3", "Tianshou", "CleanRL", "RLlib", "VERL", "Isaac Gym", "SERL"]
FEATURES = ["maturity", "docs", "distributed", "multi-agent", "GPU-sim", "LLM-RL", "real-robot"]
SCORES = np.array([
    #  mat doc dis mab gpu llm rob
    [2, 2, 0, 0, 0, 0, 0],   # SB3
    [1, 1, 0, 1, 0, 0, 0],   # Tianshou
    [1, 1, 0, 0, 0, 0, 0],   # CleanRL (single-file, educational)
    [2, 1, 2, 2, 1, 1, 0],   # RLlib
    [1, 1, 2, 0, 2, 2, 0],   # VERL
    [1, 1, 2, 0, 2, 0, 1],   # Isaac Gym
    [0, 0, 0, 0, 0, 0, 2],   # SERL
], dtype=float)

SCALE_TOOLS = ["SB3\n(1 CPU)", "RLlib\n(multi-CPU)", "Isaac Gym\n(1 GPU sim)", "Isaac+RLlib\n(GPU farm)"]
SCALE_EPS = [1e4, 1e5, 1e7, 1e9]  # env steps/sec, illustrative orders of magnitude


def main():
    print("[config] concept-map station (no training); rendering toolchain landscape")
    print(f"[E1] tools={TOOLS}")
    print(f"[E1] features={FEATURES}")

    # Fig1: heatmap
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(SCORES, cmap="YlGnBu", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(TOOLS)))
    ax.set_yticklabels(TOOLS)
    for i in range(len(TOOLS)):
        for j in range(len(FEATURES)):
            ax.text(j, i, int(SCORES[i, j]), ha="center", va="center", color="black")
    ax.set_title("RL toolchain landscape (0 weak / 1 partial / 2 native)")
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    savefig(os.path.join(FIGS, "fig1_toolchain_heatmap.png"))

    # Fig2: scale ladder
    plt.figure(figsize=(9, 4))
    x = np.arange(len(SCALE_TOOLS))
    plt.bar(x, SCALE_EPS, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    plt.yscale("log")
    plt.xticks(x, SCALE_TOOLS)
    plt.ylabel("env steps / sec (log, illustrative)")
    plt.title("Compute-scale ladder: why GPU-parallel sim matters")
    for i, v in enumerate(SCALE_EPS):
        plt.text(i, v * 1.2, f"{v:.0e}", ha="center")
    plt.ylim(1e3, 1e10)
    savefig(os.path.join(FIGS, "fig2_scale_ladder.png"))

    print("[E2] scale ladder (env steps/sec):", dict(zip(SCALE_TOOLS, SCALE_EPS)))
    print("[note] Isaac Gym needs NVIDIA GPU + hardware; no local GPU -> reading only.")
    print(f"[done] figs -> {FIGS} (2 png)")


if __name__ == "__main__":
    main()
