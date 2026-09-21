"""02_PPO_Clip_TRPO run.py — PPO-Clip ablations (clip eps + KL diagnostics) on CartPole.

CPU budget: 3 clips x 3 seeds x 60 updates (rollout 1024, epochs 4).
Experiments:
  E1: clip 0.1 / 0.2 / 0.3 three-way (greedy eval)
  E2: KL + clip-fraction + entropy diagnostics per clip
  E3: TRPO note: trust-region theory (KL-constrained) vs PPO clip (first-order approx)
Figures: figs/fig1_clip_threeway.png, figs/fig2_kl_clipfrac.png, figs/fig3_losses.png
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
from common.train import train_ppo
from common.plot import curve_with_band, savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [0, 1, 2]
UPDATES = 60
CLIPS = [0.1, 0.2, 0.3]


def main():
    print(f"[config] CartPole-v1 PPO, updates={UPDATES}, clips={CLIPS}, seeds={SEEDS}")
    print("[config] rollout=1024, epochs=4, batch=256, lr=3e-4, gamma=0.99, lam=0.95, "
          "vf=0.5, ent=0.01, eval every 10 updates n_eval=10")

    all_h = {}
    for c in CLIPS:
        hs = [train_ppo(env_id="CartPole-v1", seed=s, total_updates=UPDATES,
                        rollout_steps=1024, clip_eps=c, eval_interval=10) for s in SEEDS]
        all_h[c] = hs
        ev = np.array([h["eval_returns"] for h in hs])
        kl = np.array([h["kls"] for h in hs])
        cf = np.array([h["clip_fracs"] for h in hs])
        en = np.array([h["ents"] for h in hs])
        print(f"[E1-clip={c}] eval final: {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")
        print(f"[E2-clip={c}] kl mean={kl.mean():.4f} final={kl[:, -1].mean():.4f} | "
              f"clipfrac mean={cf.mean():.3f} final={cf[:, -1].mean():.3f} | "
              f"entropy mean={en.mean():.4f} final={en[:, -1].mean():.4f}")
    print("[E3] TRPO note: TRPO solves max E[ratio*A] s.t. KL<=delta via CG + line search; "
          "PPO-Clip replaces it with first-order clamp, no second-order ops. "
          "Diagnostics above (kl/clipfrac) are the observable footprint of the trust region.")

    ev_ep = all_h[CLIPS[0]][0]["eval_episodes"]
    upd = np.arange(1, UPDATES + 1)

    # Fig1: three-way eval
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for c in CLIPS:
        ev = np.array([h["eval_returns"] for h in all_h[c]])
        curve_with_band(ax, ev_ep, ev.mean(0), ev.std(0), f"clip={c}")
    ax.set_xlabel("ppo update")
    ax.set_ylabel("greedy eval return")
    ax.set_title("PPO clip ablation: 0.1 vs 0.2 vs 0.3 (3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_clip_threeway.png"))

    # Fig2: KL + clipfrac
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for c in CLIPS:
        kl = np.array([h["kls"] for h in all_h[c]])
        curve_with_band(axes[0], upd, kl.mean(0), kl.std(0), f"clip={c} KL")
    axes[0].set_xlabel("ppo update")
    axes[0].set_ylabel("approx KL(old||new)")
    axes[0].set_title("Trust-region footprint: KL")
    axes[0].legend()
    for c in CLIPS:
        cf = np.array([h["clip_fracs"] for h in all_h[c]])
        curve_with_band(axes[1], upd, cf.mean(0), cf.std(0), f"clip={c} clipfrac")
    axes[1].set_xlabel("ppo update")
    axes[1].set_ylabel("fraction clipped")
    axes[1].set_title("Clip activity")
    axes[1].legend()
    savefig(os.path.join(FIGS, "fig2_kl_clipfrac.png"))

    # Fig3: policy/value loss + entropy
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for c in CLIPS:
        pl = np.array([h["pol_losses"] for h in all_h[c]])
        curve_with_band(axes[0], upd, pl.mean(0), pl.std(0), f"clip={c}")
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("policy loss")
    axes[0].set_title("Policy loss")
    axes[0].legend()
    for c in CLIPS:
        vl = np.array([h["val_losses"] for h in all_h[c]])
        curve_with_band(axes[1], upd, vl.mean(0), vl.std(0), f"clip={c}")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("value loss")
    axes[1].set_title("Value loss")
    axes[1].legend()
    for c in CLIPS:
        en = np.array([h["ents"] for h in all_h[c]])
        curve_with_band(axes[2], upd, en.mean(0), en.std(0), f"clip={c}")
    axes[2].set_xlabel("update")
    axes[2].set_ylabel("entropy")
    axes[2].set_title("Entropy (exploration left)")
    axes[2].legend()
    savefig(os.path.join(FIGS, "fig3_losses.png"))

    print(f"[done] figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()
