"""01_DPO_Family run.py — SFT vs DPO / IPO / cDPO / ORPO on toy alignment task.

Task: toy policy maps prompt -> token; correct token = argmax(prompt @ W).
Data: preference pairs (chosen=target, rejected=random other).
CPU budget: 5 variants x 2 seeds x 600 steps + DPO beta ablation. Seconds.
Experiments:
  E1: accuracy curves: SFT (supervised upper bound) vs DPO family
  E2: KL-to-ref (DPO's implicit constraint)
  E3: implicit reward margin + beta ablation for DPO
Figures: figs/fig1_acc.png, figs/fig2_kl.png, figs/fig3_beta.png
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
from common.toy import build_task
from common.algos import train_dpo, train_sft
from common.plot import curve_with_band, savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [0, 1]
STEPS = 600
PROMPT_DIM, VOCAB = 8, 8
N_TRAIN, N_EVAL = 256, 32


def main():
    print(f"[config] toy task prompt_dim={PROMPT_DIM} vocab={VOCAB}, "
          f"n_train={N_TRAIN} n_eval={N_EVAL}, steps={STEPS}, seeds={SEEDS}")
    prompts, targets, W = build_task(seed=0, n_prompts=N_TRAIN, prompt_dim=PROMPT_DIM, vocab=VOCAB)
    ev_prompts, ev_targets, _ = build_task(seed=1, n_prompts=N_EVAL, prompt_dim=PROMPT_DIM, vocab=VOCAB, W=W)

    import torch

    p_t = torch.as_tensor(prompts)
    t_t = torch.as_tensor(targets)
    ep_t = torch.as_tensor(ev_prompts)
    et_t = torch.as_tensor(ev_targets)

    # E1: SFT + DPO family
    print("[E1] SFT + DPO family:")
    results = {}
    sft_h = [train_sft(prompts, targets, seed=s, steps=STEPS, hidden=128,
                       eval_prompts=ep_t, eval_targets=et_t) for s in SEEDS]
    results["sft"] = sft_h
    acc = np.array([h["eval_acc"] for h in sft_h])
    kl = np.array([h["eval_kl"] for h in sft_h])
    print(f"    sft  : eval acc final {acc[:, -1].mean():.3f}±{acc[:, -1].std():.3f} | "
          f"KL final {kl[:, -1].mean():.3f}")
    for variant in ["dpo", "cdpo", "ipo", "orpo"]:
        hs = [train_dpo(prompts, targets, VOCAB, seed=s, variant=variant, beta=0.1, hidden=128,
                        steps=STEPS, eval_prompts=ep_t, eval_targets=et_t) for s in SEEDS]
        results[variant] = hs
        acc = np.array([h["eval_acc"] for h in hs])
        kl = np.array([h["eval_kl"] for h in hs])
        print(f"    {variant:5s}: eval acc final {acc[:, -1].mean():.3f}±{acc[:, -1].std():.3f} | "
              f"KL final {kl[:, -1].mean():.3f}")

    # E3: DPO beta ablation under noisy preferences (shows the tradeoff)
    print("[E3] DPO beta ablation (noisy preferences noise=0.2):")
    betas = [0.01, 0.1, 0.5]
    beta_h = {}
    for b in betas:
        hs = [train_dpo(prompts, targets, VOCAB, seed=s, variant="dpo", beta=b, hidden=128,
                        steps=STEPS, noise=0.2, eval_prompts=ep_t, eval_targets=et_t) for s in SEEDS]
        beta_h[b] = hs
        acc = np.array([h["eval_acc"] for h in hs])
        kl = np.array([h["eval_kl"] for h in hs])
        print(f"    beta={b}: acc final {acc[:, -1].mean():.3f} | KL final {kl[:, -1].mean():.3f}")

    # Fig1: accuracy
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name in ["sft", "dpo", "cdpo", "ipo", "orpo"]:
        arr = np.array([h["eval_acc"] for h in results[name]])
        xs = results[name][0]["ev_steps"]
        curve_with_band(ax, xs, arr.mean(0), arr.std(0), name)
    ax.set_xlabel("step")
    ax.set_ylabel("eval accuracy (greedy == target)")
    ax.set_title("Toy alignment: SFT vs DPO family (2 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_acc.png"))

    # Fig2: KL to ref
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for name in ["sft", "dpo", "cdpo", "ipo", "orpo"]:
        arr = np.array([h["eval_kl"] for h in results[name]])
        xs = results[name][0]["ev_steps"]
        curve_with_band(ax, xs, arr.mean(0), arr.std(0), name)
    ax.set_xlabel("step")
    ax.set_ylabel("KL(policy || ref)")
    ax.set_title("KL to reference (all methods drift on the deterministic task)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig2_kl.png"))

    # Fig3: beta ablation (acc + KL)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for b in betas:
        arr = np.array([h["eval_acc"] for h in beta_h[b]])
        xs = beta_h[b][0]["ev_steps"]
        curve_with_band(axes[0], xs, arr.mean(0), arr.std(0), f"beta={b}")
        kl = np.array([h["eval_kl"] for h in beta_h[b]])
        curve_with_band(axes[1], xs, kl.mean(0), kl.std(0), f"beta={b}")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("eval accuracy")
    axes[0].set_title("DPO beta ablation: accuracy")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("KL to ref")
    axes[1].set_title("DPO beta ablation: KL")
    axes[1].legend()
    savefig(os.path.join(FIGS, "fig3_beta.png"))

    print(f"[done] figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()