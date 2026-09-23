"""02_SAC_AutoTemp run.py — SAC auto vs fixed temp + TD3 reference (Pendulum, light).

Env: Pendulum-v1 (dense reward). CPU budget: sac(auto/fixed) + td3 x 2 seeds x 20000 steps.
Experiments:
  E1: SAC auto-temp vs fixed alpha=0.2 (eval return)
  E2: entropy + learned alpha trajectory under auto-temp
  E3: SAC (auto) vs TD3 same budget
Figures: figs/fig1_sac_temps.png, figs/fig2_entropy_alpha.png, figs/fig3_sac_vs_td3.png
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
from common.plot import curve_with_band, savefig, smooth

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


def run_sac(seed, auto, alpha=0.2):
    return train_offpolicy(env_id=ENV, algo="sac", seed=seed, steps=STEPS,
                           max_steps=MAX_STEP, hidden=HIDDEN, batch_size=BATCH,
                           eval_every=EVAL_EVERY, n_eval=N_EVAL,
                           sac_auto_temp=auto, sac_alpha=alpha)


def main():
    print(f"[config] {ENV} SAC, steps={STEPS}, seeds={SEEDS}, eval_every={EVAL_EVERY}")
    h_auto, h_fixed, h_td3 = [], [], []
    for s in SEEDS:
        h_auto.append(run_sac(s, True))
        h_fixed.append(run_sac(s, False))
        h_td3.append(train_offpolicy(env_id=ENV, algo="td3", seed=s, steps=STEPS,
                                     max_steps=MAX_STEP, hidden=HIDDEN, batch_size=BATCH,
                                     eval_every=EVAL_EVERY, n_eval=N_EVAL))
    ev_a = np.array([h["eval_returns"] for h in h_auto])
    ev_f = np.array([h["eval_returns"] for h in h_fixed])
    ev_t = np.array([h["eval_returns"] for h in h_td3])
    print(f"[E1-auto ] eval last: {ev_a[:, -1].mean():.1f}±{ev_a[:, -1].std():.1f}")
    print(f"[E1-fixed] eval last: {ev_f[:, -1].mean():.1f}±{ev_f[:, -1].std():.1f}")
    print(f"[E3-sac /td3] {ev_a[:, -1].mean():.1f}±{ev_a[:, -1].std():.1f} / "
          f"{ev_t[:, -1].mean():.1f}±{ev_t[:, -1].std():.1f}")

    ent = np.array([h["entropies"] for h in h_auto])
    alp = np.array([h["alphas"] for h in h_auto])
    ent_f = np.array([h["entropies"] for h in h_fixed])
    print(f"[E2-auto] entropy first={ent[:, :10].mean():.3f} last={ent[:, -10:].mean():.3f} | "
          f"alpha first={alp[:, :10].mean():.3f} last={alp[:, -10:].mean():.3f}")
    print(f"[E2-fix ] entropy first={ent_f[:, :10].mean():.3f} last={ent_f[:, -10:].mean():.3f}")

    ev_x = h_auto[0]["eval_steps"]
    qx = np.arange(1, len(ent[0]) + 1)

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    curve_with_band(ax, ev_x, ev_a.mean(0), ev_a.std(0), "SAC auto-temp")
    curve_with_band(ax, ev_x, ev_f.mean(0), ev_f.std(0), "SAC fixed alpha=0.2")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy eval return")
    ax.set_title("SAC auto vs fixed temperature (2 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_sac_temps.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    curve_with_band(axes[0], qx, smooth(ent.mean(0)), smooth(ent.std(0)), "auto entropy")
    curve_with_band(axes[0], qx, smooth(ent_f.mean(0)), smooth(ent_f.std(0)), "fixed entropy")
    axes[0].set_xlabel("gradient step")
    axes[0].set_ylabel("mean entropy")
    axes[0].set_title("Entropy over training")
    axes[0].legend()
    curve_with_band(axes[1], qx, smooth(alp.mean(0)), smooth(alp.std(0)), "alpha")
    axes[1].set_xlabel("gradient step")
    axes[1].set_ylabel("alpha")
    axes[1].set_title("Learned temperature (auto)")
    axes[1].legend()
    savefig(os.path.join(FIGS, "fig2_entropy_alpha.png"))

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    curve_with_band(ax, ev_x, ev_a.mean(0), ev_a.std(0), "SAC auto")
    curve_with_band(ax, ev_x, ev_t.mean(0), ev_t.std(0), "TD3")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy eval return")
    ax.set_title("Same budget: SAC vs TD3")
    ax.legend()
    savefig(os.path.join(FIGS, "fig3_sac_vs_td3.png"))

    print(f"[done] figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()