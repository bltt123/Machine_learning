"""01_MDP_DP_GridWorld run.py — DP on 4x4 GridWorld, numpy only.

Experiments:
  E1: Policy Iteration vs Value Iteration (optimal V, iters/sweeps, max diff)
  E2: Convergence speed (delta per sweep)
  E3: Uniform random policy V vs optimal V (cost of dithering)
Figures: figs/fig1_optimal_values.png, figs/fig2_delta_curve.png, figs/fig3_uniform_vs_optimal.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.mdp import GridWorld4x4
from common.solvers import policy_evaluation, policy_iteration, value_iteration

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


def vi_with_history(env, gamma=1.0, theta=1e-6):
    V = np.zeros(env.nS)
    deltas = []
    while True:
        delta = 0.0
        for s in range(env.nS):
            q = np.zeros(env.nA)
            for a in range(env.nA):
                for p, ns, r, done in env.transitions(s, a):
                    q[a] += p * (r + gamma * (0.0 if done else V[ns]))
            best = float(np.max(q))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        deltas.append(delta)
        if delta < theta:
            break
    return V, deltas


def pe_deltas_uniform(env, gamma=1.0, theta=1e-6):
    policy = np.ones((env.nS, env.nA)) / env.nA
    V = np.zeros(env.nS)
    deltas = []
    while True:
        delta = 0.0
        for s in range(env.nS):
            v = 0.0
            for a in range(env.nA):
                for p, ns, r, done in env.transitions(s, a):
                    v += policy[s, a] * p * (r + gamma * (0.0 if done else V[ns]))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        deltas.append(delta)
        if delta < theta:
            break
    return V, deltas


def main():
    rng_note = "gamma=1.0, theta=1e-6, rewards -1/step"
    print(f"[config] {rng_note}")
    env = GridWorld4x4()

    # E1: PI vs VI
    pi_pi, V_pi, pi_iters = policy_iteration(env)
    pi_vi, V_vi, vi_sweeps = value_iteration(env)
    max_diff = float(np.max(np.abs(V_pi - V_vi)))
    pol_same = bool(np.array_equal(pi_pi, pi_vi))
    print(f"[E1] policy_iteration: outer_iters={pi_iters}")
    print(f"[E1] value_iteration: sweeps={vi_sweeps}")
    print(f"[E1] max|V_pi - V_vi| = {max_diff:.2e}, policies identical = {pol_same}")
    print(f"[E1] V_opt[5] = {V_vi[5]:.4f} (state 5=(1,1), 2 steps to terminal 0)")
    print(f"[E1] V_opt grid:\n{np.round(V_vi.reshape(4, 4), 1)}")

    # E2: delta curves
    _, vi_deltas = vi_with_history(env)
    _, pe_deltas = pe_deltas_uniform(env)
    print(f"[E2] VI converged in {len(vi_deltas)} sweeps, final delta={vi_deltas[-1]:.2e}")
    print(f"[E2] Uniform-PE converged in {len(pe_deltas)} sweeps, final delta={pe_deltas[-1]:.2e}")

    # E3: uniform vs optimal
    uniform = np.ones((env.nS, env.nA)) / env.nA
    V_unif = policy_evaluation(env, uniform)
    print(f"[E3] V_uniform[5] = {V_unif[5]:.4f} vs V_opt[5] = {V_vi[5]:.4f}")
    print(f"[E3] mean V_uniform(non-terminal) = {V_unif[1:15].mean():.4f}")
    print(f"[E3] mean V_opt(non-terminal) = {V_vi[1:15].mean():.4f}")

    # Fig1: optimal values heatmap + policy arrows
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(V_vi.reshape(4, 4), cmap="viridis")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("Optimal V* (4x4 GridWorld)")
    for r in range(4):
        for c in range(4):
            axes[0].text(c, r, f"{V_vi[r * 4 + c]:.0f}", ha="center", va="center", color="w", fontsize=9)
    sym = {0: "^", 1: ">", 2: "v", 3: "<"}
    axes[1].set_title("Optimal greedy policy")
    axes[1].set_xlim(-0.5, 3.5)
    axes[1].set_ylim(3.5, -0.5)
    axes[1].set_aspect("equal")
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    for s in range(16):
        r, c = divmod(s, 4)
        if s in (0, 15):
            axes[1].text(c, r, "T", ha="center", va="center", fontsize=12)
        else:
            axes[1].text(c, r, sym[int(np.argmax(pi_vi[s]))], ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig1_optimal_values.png"), dpi=150)
    plt.close()

    # Fig2: delta curves (log scale)
    plt.figure(figsize=(7, 4))
    plt.semilogy(vi_deltas, label=f"value iteration ({len(vi_deltas)} sweeps)")
    plt.semilogy(pe_deltas, label=f"uniform policy eval ({len(pe_deltas)} sweeps)")
    plt.xlabel("sweep")
    plt.ylabel("max delta (log)")
    plt.title("DP convergence: VI vs uniform policy evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig2_delta_curve.png"), dpi=150)
    plt.close()

    # Fig3: uniform vs optimal bar
    x = np.arange(env.nS)
    w = 0.35
    plt.figure(figsize=(9, 4))
    plt.bar(x - w / 2, V_unif, w, label="uniform random")
    plt.bar(x + w / 2, V_vi, w, label="optimal")
    plt.xlabel("state")
    plt.ylabel("V(s)")
    plt.title("Uniform policy vs optimal values (terminals 0/15 = 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "fig3_uniform_vs_optimal.png"), dpi=150)
    plt.close()

    print(f"[done] figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()
