"""01_DQN_CartPole run.py — vanilla DQN on CartPole-v1 + ablations.

CPU budget: 200 episodes x 3 seeds (~8 min). Single train.py shared by all stations.
Experiments:
  E1: vanilla DQN learning curve (train + greedy eval)
  E2: target-network ablation: sync=500 (stable) vs no_target (every step)
  E3: overestimation probe: batch mean Q(s,a) over training vs env ceiling 500
Figures: figs/fig1_dqn_curve.png, figs/fig2_target_ablation.png,
         figs/fig3_q_overestimation.png, figs/fig4_eval_hist.png
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.train import train_dqn
from common.plot import curve_with_band, savefig, smooth

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [0, 1, 2]
EPISODES = 200


def main():
    print(f"[config] CartPole-v1, vanilla DQN, episodes={EPISODES}, seeds={SEEDS}")
    print("[config] hidden=128, lr=1e-3, gamma=0.99, buffer=20000, batch=64, "
          "target_sync=500, eps 1.0->0.05/160eps")

    # E1: vanilla DQN
    hists = [train_dqn(env_id="CartPole-v1", variant="dqn", seed=s,
                       total_episodes=EPISODES, eps_decay_episodes=160) for s in SEEDS]
    ep_ret = np.array([h["ep_returns"] for h in hists])
    ev_ep = hists[0]["eval_episodes"]
    ev_ret = np.array([h["eval_returns"] for h in hists])
    print(f"[E1] train last-30: {ep_ret[:, -30:].mean():.1f}±{ep_ret[:, -30:].mean(axis=1).std():.1f}")
    print(f"[E1] greedy eval final: {ev_ret[:, -1].mean():.1f}±{ev_ret[:, -1].std():.1f}")
    for k, e in enumerate(ev_ep):
        print(f"[E1] eval@{e}: {ev_ret[:, k].mean():.1f}±{ev_ret[:, k].std():.1f}")

    # E2: no-target ablation (same 3 seeds, same budget)
    h_notarget = [train_dqn(env_id="CartPole-v1", variant="dqn", seed=s,
                            total_episodes=EPISODES, eps_decay_episodes=160,
                            no_target=True) for s in SEEDS]
    nt_ev = np.array([h["eval_returns"] for h in h_notarget])
    st_loss = np.array([h["losses"] for h in hists])
    nt_loss = np.array([h["losses"] for h in h_notarget])
    print(f"[E2] stable final eval: {ev_ret[:, -1].mean():.1f}±{ev_ret[:, -1].std():.1f} | "
          f"no-target: {nt_ev[:, -1].mean():.1f}±{nt_ev[:, -1].std():.1f}")
    print(f"[E2] mean loss last-30: stable={st_loss[:, -30:].mean():.4f} "
          f"no-target={nt_loss[:, -30:].mean():.4f}")

    # E3: overestimation probe
    qm = np.array([h["q_means"] for h in hists])
    n = EPISODES
    print(f"[E3] meanQ early={qm[:, :n // 20].mean():.2f} mid={qm[:, n // 2 - 10:n // 2 + 10].mean():.2f} "
          f"late={qm[:, -20:].mean():.2f} (ceiling 500)")

    # Fig1: train + eval curves
    xs = np.arange(1, EPISODES + 1)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    curve_with_band(ax, xs, smooth(ep_ret.mean(0)), smooth(ep_ret.std(0)), "train return (eps-greedy)")
    curve_with_band(ax, ev_ep, ev_ret.mean(0), ev_ret.std(0), "greedy eval return")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title("CartPole DQN: train vs greedy eval (3 seeds, mean±std)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_dqn_curve.png"))

    # Fig2: target ablation eval
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    curve_with_band(ax, ev_ep, ev_ret.mean(0), ev_ret.std(0), "target sync=500")
    curve_with_band(ax, ev_ep, nt_ev.mean(0), nt_ev.std(0), "no target (sync every step)")
    ax.set_xlabel("episode")
    ax.set_ylabel("greedy eval return")
    ax.set_title("Ablation: frozen target vs no target (3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig2_target_ablation.png"))

    # Fig3: mean Q over training
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    curve_with_band(ax, xs, smooth(qm.mean(0)), smooth(qm.std(0)), "mean batch Q(s,a)")
    ax.axhline(500.0, linestyle="--", label="env ceiling (500)")
    ax.set_xlabel("episode")
    ax.set_ylabel("mean Q")
    ax.set_title("Overestimation probe: batch mean Q vs env ceiling (3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig3_q_overestimation.png"))

    # Fig4: final eval per seed
    plt.figure(figsize=(7, 4))
    plt.bar([f"seed {s}" for s in SEEDS], ev_ret[:, -1])
    plt.ylabel("final greedy eval return")
    plt.title(f"Seed robustness: final eval per seed (mean={ev_ret[:, -1].mean():.1f})")
    savefig(os.path.join(FIGS, "fig4_eval_hist.png"))

    print(f"[done] figs -> {FIGS} (4 png)")
    print(f"[torch] {torch.__version__}")


if __name__ == "__main__":
    main()
