"""01_DDPG_TD3_Pendulum run.py — DDPG vs TD3 + three-move ablation on Pendulum.

Env: Pendulum-v1 (dense reward, target ~ near 0). CPU budget: 5 configs x 2 seeds x 20000 steps.
Eval by gradient step every eval_every=2000 (10 points/seed), n_eval=5.
Experiments:
  E1: DDPG vs TD3 greedy eval (TD3 overtakes; DDPG overestimates more)
  E2: TD3 three-move ablation: full vs no_twin / no_delay / no_smooth
  E3: overestimation probe: mean batch Q (DDPG higher, TD3 twin more conservative)
Figures: figs/fig1_ddpg_vs_td3.png, figs/fig2_td3_ablation.png,
         figs/fig3_q_probe.png, figs/fig4_return_curves.png
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
from common.train import train_offpolicy
from common.plot import curve_with_band, savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

ENV = "Pendulum-v1"
SEEDS = [0, 1]
STEPS = 20000
MAX_STEP = 200
HIDDEN = 128
BATCH = 128
EVAL_EVERY = 2000
N_EVAL = 5

CONFIGS = {
    "ddpg": dict(algo="ddpg"),
    "td3": dict(algo="td3"),
    "td3_no_twin": dict(algo="td3", td3_twin=False),
    "td3_no_delay": dict(algo="td3", td3_delay=False),
    "td3_no_smooth": dict(algo="td3", td3_smooth=False),
}


def run(cfg, seed):
    return train_offpolicy(env_id=ENV, seed=seed, steps=STEPS, max_steps=MAX_STEP,
                           hidden=HIDDEN, batch_size=BATCH, eval_every=EVAL_EVERY,
                           n_eval=N_EVAL, **cfg)


def main():
    print(f"[config] {ENV}, steps={STEPS}, seeds={SEEDS}, hidden={HIDDEN}, batch={BATCH}, "
          f"eval_every={EVAL_EVERY}, n_eval={N_EVAL}")
    print(f"[config] variants = {list(CONFIGS)}")

    all_h = {}
    for name, cfg in CONFIGS.items():
        all_h[name] = [run(cfg, s) for s in SEEDS]
        ev = np.array([h["eval_returns"] for h in all_h[name]])
        qm = np.array([h["q_means"] for h in all_h[name]])
        print(f"[{name:14s}] eval last: {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f} | "
              f"meanQ last: {qm[:, -10:].mean():.1f}")

    ev_x = all_h["ddpg"][0]["eval_steps"]
    qx = np.arange(1, len(all_h["ddpg"][0]["q_means"]) + 1)

    # Fig1: DDPG vs TD3
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name in ["ddpg", "td3"]:
        ev = np.array([h["eval_returns"] for h in all_h[name]])
        curve_with_band(ax, ev_x, ev.mean(0), ev.std(0), name)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy eval return")
    ax.set_title("Pendulum: DDPG vs TD3 (2 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_ddpg_vs_td3.png"))

    # Fig2: TD3 ablation
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name in ["td3", "td3_no_twin", "td3_no_delay", "td3_no_smooth"]:
        ev = np.array([h["eval_returns"] for h in all_h[name]])
        curve_with_band(ax, ev_x, ev.mean(0), ev.std(0), name)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy eval return")
    ax.set_title("TD3 three-move ablation (2 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig2_td3_ablation.png"))

    # Fig3: Q probe (ddpg vs td3)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name in ["ddpg", "td3"]:
        qm = np.array([h["q_means"] for h in all_h[name]])
        curve_with_band(ax, qx, qm.mean(0), qm.std(0), name + " meanQ")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("mean batch Q")
    ax.set_title("Overestimation probe: DDPG vs TD3 twin Q")
    ax.legend()
    savefig(os.path.join(FIGS, "fig3_q_probe.png"))

    # Fig4: all return curves (train ep returns already in eval_returns; plot all 5)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name in CONFIGS:
        ev = np.array([h["eval_returns"] for h in all_h[name]])
        curve_with_band(ax, ev_x, ev.mean(0), ev.std(0), name)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy eval return")
    ax.set_title("All variants: eval return")
    ax.legend()
    savefig(os.path.join(FIGS, "fig4_return_curves.png"))

    print(f"[done] figs -> {FIGS} (4 png)")


if __name__ == "__main__":
    main()