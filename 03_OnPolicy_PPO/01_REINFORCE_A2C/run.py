"""01_REINFORCE_A2C run.py — REINFORCE +/- baseline + A2C-flavoured PPO(1-step) on CartPole.

CPU budget: REINFORCE 300eps x 3seeds x2 (baseline on/off) + PPO small x3seeds.
Experiments:
  E1: REINFORCE with vs without learned V baseline (train + greedy eval)
  E2: gradient variance probe: grad_norm mean over training
  E3: A2C-flavoured reference: PPO rollout with clip off vs on (kl/clip diagnostics)
Figures: figs/fig1_reinforce_baseline.png, figs/fig2_grad_variance.png,
         figs/fig3_a2c_reference.png, figs/fig4_eval_hist.png
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
from common.train import train_ppo, train_reinforce
from common.plot import curve_with_band, savefig, smooth

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [0, 1, 2]
EPISODES = 300


def main():
    print(f"[config] CartPole-v1, REINFORCE episodes={EPISODES}, seeds={SEEDS}")
    print("[config] hidden=128 tanh, lr=3e-3, gamma=0.99, eval every 25ep n_eval=10")

    h_base, h_plain = [], []
    for s in SEEDS:
        h_base.append(train_reinforce(env_id="CartPole-v1", seed=s,
                                      total_episodes=EPISODES, use_baseline=True))
        h_plain.append(train_reinforce(env_id="CartPole-v1", seed=s,
                                       total_episodes=EPISODES, use_baseline=False))
    for name, hs in [("baseline", h_base), ("plain", h_plain)]:
        tr = np.array([h["ep_returns"] for h in hs])
        ev = np.array([h["eval_returns"] for h in hs])
        gn = np.array([h["grad_norms"] for h in hs])
        print(f"[E1-{name}] train last-30: {tr[:, -30:].mean():.1f}±{tr[:, -30:].mean(axis=1).std():.1f} | "
              f"eval final: {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")
        print(f"[E2-{name}] grad_norm mean: {gn.mean():.4f} (first50={gn[:, :50].mean():.4f} "
              f"last50={gn[:, -50:].mean():.4f})")

    ev_ep = h_base[0]["eval_episodes"]
    xs = np.arange(1, EPISODES + 1)

    # E3: A2C-flavoured reference via PPO with clip 0.2 vs 1.0 (no clip)
    ppo_clip, ppo_noclip = [], []
    for s in SEEDS:
        ppo_clip.append(train_ppo(env_id="CartPole-v1", seed=s, total_updates=30,
                                  rollout_steps=1024, clip_eps=0.2, eval_interval=10))
        ppo_noclip.append(train_ppo(env_id="CartPole-v1", seed=s, total_updates=30,
                                    rollout_steps=1024, clip_eps=1.0, eval_interval=10))
    for name, hs in [("clip0.2", ppo_clip), ("noclip", ppo_noclip)]:
        ev = np.array([h["eval_returns"] for h in hs])
        kl = np.array([h["kls"] for h in hs])
        cf = np.array([h["clip_fracs"] for h in hs])
        print(f"[E3-{name}] eval final: {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f} | "
              f"kl final={kl[:, -1].mean():.4f} clipfrac final={cf[:, -1].mean():.3f}")

    # Fig1: REINFORCE baseline vs plain (eval)
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name, hs in [("REINFORCE+baseline", h_base), ("REINFORCE plain", h_plain)]:
        ev = np.array([h["eval_returns"] for h in hs])
        curve_with_band(ax, ev_ep, ev.mean(0), ev.std(0), name)
    ax.set_xlabel("episode")
    ax.set_ylabel("greedy eval return")
    ax.set_title("REINFORCE: learned baseline vs none (3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_reinforce_baseline.png"))

    # Fig2: train curves + grad variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, hs in [("baseline", h_base), ("plain", h_plain)]:
        tr = np.array([h["ep_returns"] for h in hs])
        curve_with_band(axes[0], xs, smooth(tr.mean(0)), smooth(tr.std(0)), name)
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("train return (smoothed)")
    axes[0].set_title("Train curves")
    axes[0].legend()
    for name, hs in [("baseline", h_base), ("plain", h_plain)]:
        gn = np.array([h["grad_norms"] for h in hs])
        curve_with_band(axes[1], xs, smooth(gn.mean(0)), smooth(gn.std(0)), name + " grad_norm")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("grad norm (smoothed)")
    axes[1].set_title("Gradient variance probe")
    axes[1].legend()
    savefig(os.path.join(FIGS, "fig2_grad_variance.png"))

    # Fig3: PPO clip reference
    ppo_ep = ppo_clip[0]["eval_episodes"]
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name, hs in [("ppo clip=0.2", ppo_clip), ("ppo clip=1.0(noclip)", ppo_noclip)]:
        ev = np.array([h["eval_returns"] for h in hs])
        curve_with_band(ax, ppo_ep, ev.mean(0), ev.std(0), name)
    ax.set_xlabel("ppo update")
    ax.set_ylabel("greedy eval return")
    ax.set_title("A2C-flavoured reference: PPO clip on/off (3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig3_a2c_reference.png"))

    # Fig4: final eval per seed (baseline)
    ev_b = np.array([h["eval_returns"][-1] for h in h_base])
    ev_p = np.array([h["eval_returns"][-1] for h in h_plain])
    x = np.arange(len(SEEDS))
    w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, ev_b, w, label="baseline")
    plt.bar(x + w / 2, ev_p, w, label="plain")
    plt.xticks(x, [f"seed {s}" for s in SEEDS])
    plt.ylabel("final greedy eval return")
    plt.title("Seed robustness: REINFORCE baseline vs plain")
    plt.legend()
    savefig(os.path.join(FIGS, "fig4_eval_hist.png"))

    print(f"[done] figs -> {FIGS} (4 png)")


if __name__ == "__main__":
    main()
