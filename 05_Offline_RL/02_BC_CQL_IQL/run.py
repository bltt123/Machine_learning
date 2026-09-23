"""02_BC_CQL_IQL run.py — offline algorithms on collected CartPole datasets.

Data from ../01_Collect_Dataset/data (random / hard / mixed / expert, 20k each).
CPU budget: BC on 4 datasets + CQL/IQL on random+hard + CQL alpha ablation, 2 seeds.
Experiments:
  E1: BC on random vs hard vs mixed vs expert (BC ceiling = data quality)
  E2: BC vs CQL vs IQL on random (BC fails, offline RL wins) and on hard (BC strong)
  E3: CQL conservatism alpha ablation (0.0 / 0.5 / 1.0 / 2.0) on hard
  E4: diagnostics: CQL Q(data) vs Q(all) OOD probe; IQL advantage mean
Figures: figs/fig1_bc_quality.png, figs/fig2_offline_algo.png,
         figs/fig3_cql_alpha.png, figs/fig4_diagnostics.png
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
from common.dataset import load_dataset
from common.train import train_bc, train_cql, train_iql
from common.plot import curve_with_band, savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
DATA = os.path.join(HERE, "..", "01_Collect_Dataset", "data")
os.makedirs(FIGS, exist_ok=True)

ENV = "CartPole-v1"
SEEDS = [0, 1]
BC_STEPS = 4000
OFFLINE_STEPS = 8000
ALPHAS = [0.0, 0.5, 1.0, 2.0]


def load(name):
    return load_dataset(DATA, name)


def agg(histories, key="eval_returns"):
    arr = np.array([h[key] for h in histories])
    return arr.mean(0), arr.std(0), histories[0]["eval_steps"]


def main():
    print(f"[config] {ENV} offline, seeds={SEEDS}, bc_steps={BC_STEPS}, "
          f"offline_steps={OFFLINE_STEPS}, cql_alphas={ALPHAS}")
    d_random, d_mixed, d_hard, d_expert = load("random"), load("mixed"), load("hard"), load("expert")

    # E1: BC on datasets of increasing quality
    print("[E1] BC by dataset quality:")
    bc_by_ds = {}
    for name, data in [("random", d_random), ("hard", d_hard), ("mixed", d_mixed), ("expert", d_expert)]:
        hs = [train_bc(data, env_id=ENV, seed=s, steps=BC_STEPS) for s in SEEDS]
        bc_by_ds[name] = hs
        ev = np.array([h["eval_returns"] for h in hs])
        print(f"    BC on {name:6s}: eval final {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")

    # E2: BC vs CQL vs IQL on random and hard
    print("[E2] BC vs CQL vs IQL:")
    cql_random = [train_cql(d_random, env_id=ENV, seed=s, steps=OFFLINE_STEPS, cql_alpha=1.0) for s in SEEDS]
    iql_random = [train_iql(d_random, env_id=ENV, seed=s, steps=OFFLINE_STEPS) for s in SEEDS]
    cql_hard = [train_cql(d_hard, env_id=ENV, seed=s, steps=OFFLINE_STEPS, cql_alpha=1.0) for s in SEEDS]
    iql_hard = [train_iql(d_hard, env_id=ENV, seed=s, steps=OFFLINE_STEPS) for s in SEEDS]
    for ds_name, cql_hs, iql_hs in [("random", cql_random, iql_random), ("hard", cql_hard, iql_hard)]:
        print(f"    --- on {ds_name} ---")
        for tag, hs in [("BC", bc_by_ds[ds_name]), ("CQL(a=1.0)", cql_hs), ("IQL", iql_hs)]:
            ev = np.array([h["eval_returns"] for h in hs])
            print(f"    {tag:11s}: eval final {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")

    # E3: CQL alpha ablation on hard
    print("[E3] CQL conservatism alpha ablation (hard data):")
    cql_by_alpha = {}
    for a in ALPHAS:
        if a == 1.0:
            hs = cql_hard
        else:
            hs = [train_cql(d_hard, env_id=ENV, seed=s, steps=OFFLINE_STEPS, cql_alpha=a) for s in SEEDS]
        cql_by_alpha[a] = hs
        ev = np.array([h["eval_returns"] for h in hs])
        print(f"    CQL alpha={a}: eval final {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")

    # E4 diagnostics (hard)
    qd = np.array([h["q_data_means"] for h in cql_hard]).mean(0)
    qo = np.array([h["q_ood_means"] for h in cql_hard]).mean(0)
    adv = np.array([h["adv_means"] for h in iql_hard]).mean(0)
    print(f"[E4] CQL(hard) Q(data) last={qd[-1]:.2f} Q(all) last={qo[-1]:.2f} gap={qo[-1]-qd[-1]:.2f}")
    print(f"[E4] IQL(hard) advantage mean first={adv[:50].mean():.3f} last={adv[-50:].mean():.3f}")

    # Fig1: BC quality
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name in ["random", "hard", "mixed", "expert"]:
        m, s, xs = agg(bc_by_ds[name])
        curve_with_band(ax, xs, m, s, f"BC on {name}")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy eval return")
    ax.set_title("BC ceiling depends on data quality (2 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_bc_quality.png"))

    # Fig2: BC vs CQL vs IQL on random and hard (2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, ds_name, cql_hs, iql_hs in [
        (axes[0], "random", cql_random, iql_random),
        (axes[1], "hard", cql_hard, iql_hard),
    ]:
        for tag, hs in [("BC", bc_by_ds[ds_name]), ("CQL(a=1.0)", cql_hs), ("IQL", iql_hs)]:
            m, s, xs = agg(hs)
            curve_with_band(ax, xs, m, s, tag)
        ax.set_xlabel("gradient step")
        ax.set_ylabel("greedy eval return")
        ax.set_title(f"On {ds_name} data")
        ax.legend()
    fig.suptitle("BC vs CQL vs IQL: random (BC fails) vs hard (BC strong)")
    savefig(os.path.join(FIGS, "fig2_offline_algo.png"))

    # Fig3: CQL alpha ablation
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for a in ALPHAS:
        m, s, xs = agg(cql_by_alpha[a])
        curve_with_band(ax, xs, m, s, f"alpha={a}")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("greedy eval return")
    ax.set_title("CQL conservatism alpha ablation (hard data, 2 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig3_cql_alpha.png"))

    # Fig4: diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    qx = np.arange(1, len(qd) + 1) * 200
    axes[0].plot(qx, qd, label="Q(data action)")
    axes[0].plot(qx, qo, label="Q(all actions, OOD)")
    axes[0].fill_between(qx, qd, qo, alpha=0.15, color="red")
    axes[0].set_xlabel("gradient step")
    axes[0].set_ylabel("mean Q")
    axes[0].set_title("CQL OOD probe: data Q vs all-action Q (hard)")
    axes[0].legend()
    axes[1].plot(np.arange(1, len(adv) + 1), adv, label="IQL advantage mean")
    axes[1].axhline(0.0, linestyle="--", color="gray")
    axes[1].set_xlabel("gradient step")
    axes[1].set_ylabel("mean advantage")
    axes[1].set_title("IQL advantage over training (hard)")
    axes[1].legend()
    savefig(os.path.join(FIGS, "fig4_diagnostics.png"))

    print(f"[done] figs -> {FIGS} (4 png)")


if __name__ == "__main__":
    main()