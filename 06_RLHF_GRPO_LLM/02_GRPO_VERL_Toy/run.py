"""02_GRPO_VERL_Toy run.py — GRPO group-baseline vs REINFORCE + KL/group ablations.

Same toy task as 01. GRPO samples a group of responses per prompt and uses the
group mean/std as baseline (no value net). VERL-style pipeline is described in E4.
CPU budget: GRPO/REINFORCE x 2 seeds + KL + group ablations. Seconds.
Experiments:
  E1: GRPO vs REINFORCE(baseline) vs REINFORCE(no baseline): accuracy + reward
  E2: GRPO KL penalty beta ablation
  E3: group size ablation (2 / 4 / 8 / 16)
  E4: VERL pipeline note + entropy/KL diagnostics
Figures: figs/fig1_grpo_vs_reinforce.png, figs/fig2_kl_entropy.png, figs/fig3_group_size.png
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.toy import build_task
from common.algos import train_grpo
from common.plot import curve_with_band, savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [0, 1]
STEPS = 600
PROMPT_DIM, VOCAB = 8, 8
N_TRAIN, N_EVAL = 256, 32


def main():
    import torch

    print(f"[config] toy task prompt_dim={PROMPT_DIM} vocab={VOCAB}, n_train={N_TRAIN} "
          f"n_eval={N_EVAL}, steps={STEPS}, seeds={SEEDS}")
    prompts, targets, W = build_task(seed=0, n_prompts=N_TRAIN, prompt_dim=PROMPT_DIM, vocab=VOCAB)
    ev_prompts, ev_targets, _ = build_task(seed=1, n_prompts=N_EVAL, prompt_dim=PROMPT_DIM, vocab=VOCAB, W=W)
    ep_t = torch.as_tensor(ev_prompts)
    et_t = torch.as_tensor(ev_targets)

    def run(algo, seed, group=8, beta=0.04, use_baseline=True):
        return train_grpo(prompts, targets, VOCAB, seed=seed, algo=algo, group=group, beta=beta,
                          steps=STEPS, batch=32, hidden=128, use_baseline=use_baseline,
                          eval_prompts=ep_t, eval_targets=et_t)

    # E1: GRPO vs REINFORCE
    print("[E1] GRPO vs REINFORCE:")
    grpo_h = [run("grpo", s) for s in SEEDS]
    rf_base_h = [run("reinforce", s, use_baseline=True) for s in SEEDS]
    rf_plain_h = [run("reinforce", s, use_baseline=False) for s in SEEDS]
    for tag, hs in [("GRPO", grpo_h), ("REINFORCE+base", rf_base_h), ("REINFORCE", rf_plain_h)]:
        acc = np.array([h["eval_acc"] for h in hs])
        rew = np.array([h["eval_rew"] for h in hs])
        print(f"    {tag:14s}: acc final {acc[:, -1].mean():.3f}±{acc[:, -1].std():.3f} | "
              f"reward final {rew[:, -1].mean():.3f}")

    # E2: KL penalty beta ablation
    print("[E2] GRPO KL beta ablation:")
    betas = [0.0, 0.04, 0.2]
    beta_h = {}
    for b in betas:
        hs = grpo_h if b == 0.04 else [run("grpo", s, beta=b) for s in SEEDS]
        beta_h[b] = hs
        acc = np.array([h["eval_acc"] for h in hs])
        kl = np.array([h["eval_kl"] for h in hs])
        print(f"    beta={b}: acc final {acc[:, -1].mean():.3f} | KL final {kl[:, -1].mean():.3f}")

    # E3: group size ablation
    print("[E3] GRPO group size ablation:")
    groups = [2, 4, 8, 16]
    group_h = {}
    for g in groups:
        hs = grpo_h if g == 8 else [run("grpo", s, group=g) for s in SEEDS]
        group_h[g] = hs
        acc = np.array([h["eval_acc"] for h in hs])
        print(f"    group={g:2d}: acc final {acc[:, -1].mean():.3f}±{acc[:, -1].std():.3f}")

    # E4: VERL pipeline note
    print("[E4] VERL pipeline (toy):")
    print("    actor=policy; rollout=sample G responses/prompt; reward_fn=task reward;")
    print("    adv=(r-mean_group)/std_group; loss=-(adv*logp)+beta*KL(ref); no critic net.")

    # Fig1: GRPO vs REINFORCE (acc + reward)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for tag, hs in [("GRPO", grpo_h), ("REINFORCE+base", rf_base_h), ("REINFORCE", rf_plain_h)]:
        acc = np.array([h["eval_acc"] for h in hs])
        xs = hs[0]["ev_steps"]
        curve_with_band(axes[0], xs, acc.mean(0), acc.std(0), tag)
        rew = np.array([h["eval_rew"] for h in hs])
        curve_with_band(axes[1], xs, rew.mean(0), rew.std(0), tag)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("eval accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("train reward")
    axes[1].set_title("Reward")
    axes[1].legend()
    fig.suptitle("GRPO group baseline vs REINFORCE (2 seeds)")
    savefig(os.path.join(FIGS, "fig1_grpo_vs_reinforce.png"))

    # Fig2: KL beta ablation + entropy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for b in betas:
        acc = np.array([h["eval_acc"] for h in beta_h[b]])
        xs = beta_h[b][0]["ev_steps"]
        curve_with_band(axes[0], xs, acc.mean(0), acc.std(0), f"beta={b}")
        kl = np.array([h["eval_kl"] for h in beta_h[b]])
        curve_with_band(axes[1], xs, kl.mean(0), kl.std(0), f"beta={b}")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("eval accuracy")
    axes[0].set_title("KL beta ablation: accuracy")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("KL to ref")
    axes[1].set_title("KL beta ablation: KL")
    axes[1].legend()
    savefig(os.path.join(FIGS, "fig2_kl_entropy.png"))

    # Fig3: group size
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for g in groups:
        acc = np.array([h["eval_acc"] for h in group_h[g]])
        xs = group_h[g][0]["ev_steps"]
        curve_with_band(ax, xs, acc.mean(0), acc.std(0), f"group={g}")
    ax.set_xlabel("step")
    ax.set_ylabel("eval accuracy")
    ax.set_title("GRPO group size ablation (2 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig3_group_size.png"))

    print(f"[done] figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()
