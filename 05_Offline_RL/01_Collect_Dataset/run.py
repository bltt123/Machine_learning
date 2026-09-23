"""01_Collect_Dataset run.py — collect CartPole offline datasets (random/hard/mixed/expert).

Saves .npz + stats.json to ./data. No learning here, just data + statistics.
Experiments:
  E1: dataset quality table (mean return, ep len, action balance)
  E2: episode-return histograms per dataset (quality separation)
  E3: action distribution + episode length per dataset
Figures: figs/fig1_return_hist.png, figs/fig2_action_len.png, figs/fig3_quality_bar.png
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
from common.dataset import collect_dataset, save_dataset
from common.plot import savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
DATA = os.path.join(HERE, "data")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

ENV = "CartPole-v1"
N_TRANSITIONS = 20000
POLICIES = ["random", "mixed", "hard", "expert"]


def main():
    print(f"[config] {ENV}, n_transitions={N_TRANSITIONS} per dataset, policies={POLICIES}")
    all_stats, all_returns, all_lens, all_actions = {}, {}, {}, {}
    for p in POLICIES:
        data, stats, rets, lens = collect_dataset(ENV, p, N_TRANSITIONS, seed=0)
        save_dataset(DATA, p, data, stats)
        all_stats[p] = stats
        all_returns[p] = rets
        all_lens[p] = lens
        all_actions[p] = data["act"]
        print(f"[{p:6s}] transitions={stats['n_transitions']} eps={stats['n_episodes']} "
              f"mean_return={stats['mean_return']:.1f}±{stats['std_return']:.1f} "
              f"mean_len={stats['mean_ep_len']:.1f} action_frac_1={stats['action_frac_1']:.3f}")

    # Fig1: return histograms
    plt.figure(figsize=(9, 4))
    for p in POLICIES:
        plt.hist(all_returns[p], bins=30, alpha=0.5, label=f"{p} (mean {all_stats[p]['mean_return']:.0f})")
    plt.xlabel("episode return")
    plt.ylabel("count")
    plt.title("Offline dataset quality: episode-return distribution")
    plt.legend()
    savefig(os.path.join(FIGS, "fig1_return_hist.png"))

    # Fig2: action balance + ep length
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(POLICIES))
    fr1 = [all_stats[p]["action_frac_1"] for p in POLICIES]
    axes[0].bar(x, fr1, label="frac action=1")
    axes[0].axhline(0.5, linestyle="--", color="gray", label="balanced (0.5)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(POLICIES)
    axes[0].set_ylabel("fraction")
    axes[0].set_title("Action balance per dataset")
    axes[0].legend()
    axes[1].boxplot([all_lens[p] for p in POLICIES], labels=POLICIES)
    axes[1].set_ylabel("episode length")
    axes[1].set_title("Episode length per dataset")
    savefig(os.path.join(FIGS, "fig2_action_len.png"))

    # Fig3: quality bar (mean return)
    plt.figure(figsize=(7, 4))
    means = [all_stats[p]["mean_return"] for p in POLICIES]
    stds = [all_stats[p]["std_return"] for p in POLICIES]
    plt.bar(POLICIES, means, yerr=stds, capsize=5)
    plt.ylabel("mean episode return")
    plt.title("Dataset quality (CartPole ceiling 500)")
    savefig(os.path.join(FIGS, "fig3_quality_bar.png"))

    print(f"[done] data -> {DATA} (4 npz), figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()