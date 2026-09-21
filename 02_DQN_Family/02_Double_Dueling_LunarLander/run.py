"""02_Double_Dueling run.py — DQN vs Double vs Dueling on CartPole-v1.

Same budget for all: 200 eps x 3 seeds, identical hypers, only variant differs.
Experiments:
  E1: three-way learning curve (greedy eval mean±std)
  E2: final-eval table per seed
  E3: overestimation: batch mean-Q trajectories per variant
Figures: figs/fig1_threeway_eval.png, figs/fig2_threeway_train.png, figs/fig3_meanq_compare.png
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
from common.train import train_dqn
from common.plot import curve_with_band, savefig, smooth

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [0, 1, 2]
EPISODES = 200
VARIANTS = ["dqn", "double", "dueling"]


def main():
    print(f"[config] CartPole-v1, variants={VARIANTS}, episodes={EPISODES}, seeds={SEEDS}")
    all_h = {}
    for v in VARIANTS:
        all_h[v] = [train_dqn(env_id="CartPole-v1", variant=v, seed=s,
                              total_episodes=EPISODES, eps_decay_episodes=160) for s in SEEDS]
        ev = np.array([h["eval_returns"] for h in all_h[v]])
        tr = np.array([h["ep_returns"] for h in all_h[v]])
        print(f"[E1-{v}] train last-30: {tr[:, -30:].mean():.1f}±{tr[:, -30:].mean(axis=1).std():.1f} | "
              f"eval final: {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")

    ev_ep = all_h["dqn"][0]["eval_episodes"]
    xs = np.arange(1, EPISODES + 1)

    # E2: final table per seed
    print("[E2] final greedy eval per seed:")
    for s_i, s in enumerate(SEEDS):
        row = " | ".join(f"{v}={all_h[v][s_i]['eval_returns'][-1]:.0f}" for v in VARIANTS)
        print(f"  seed {s}: {row}")

    # E3: mean-Q
    for v in VARIANTS:
        qm = np.array([h["q_means"] for h in all_h[v]])
        print(f"[E3-{v}] meanQ early={qm[:, :10].mean():.2f} mid={qm[:, 90:110].mean():.2f} "
              f"late={qm[:, -20:].mean():.2f}")

    # Fig1: eval three-way
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for v in VARIANTS:
        ev = np.array([h["eval_returns"] for h in all_h[v]])
        curve_with_band(ax, ev_ep, ev.mean(0), ev.std(0), v)
    ax.set_xlabel("episode")
    ax.set_ylabel("greedy eval return")
    ax.set_title("CartPole: DQN vs Double vs Dueling (3 seeds, same budget)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_threeway_eval.png"))

    # Fig2: train three-way
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for v in VARIANTS:
        tr = np.array([h["ep_returns"] for h in all_h[v]])
        curve_with_band(ax, xs, smooth(tr.mean(0)), smooth(tr.std(0)), v + " train")
    ax.set_xlabel("episode")
    ax.set_ylabel("train return (smoothed)")
    ax.set_title("Train curves three-way (eps-greedy, 3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig2_threeway_train.png"))

    # Fig3: mean-Q compare
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for v in VARIANTS:
        qm = np.array([h["q_means"] for h in all_h[v]])
        curve_with_band(ax, xs, smooth(qm.mean(0)), smooth(qm.std(0)), v + " meanQ")
    ax.axhline(500.0, linestyle="--", label="env ceiling (500)")
    ax.set_xlabel("episode")
    ax.set_ylabel("mean batch Q")
    ax.set_title("Overestimation compare: batch mean Q per variant")
    ax.legend()
    savefig(os.path.join(FIGS, "fig3_meanq_compare.png"))

    print(f"[done] figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()
