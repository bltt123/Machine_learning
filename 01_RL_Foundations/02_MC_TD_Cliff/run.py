"""02_MC_TD_Cliff run.py — MC vs TD prediction + SARSA vs Q-Learning control.

Env: CliffWalking 4x12, gamma=0.9 (discounted so uniform-policy values stay bounded).
Prediction uses uniform random policy; control uses eps=0.1, alpha=0.5.
Experiments:
  E1: MC vs TD(0) prediction (MSE to discounted DP truth, 5 seeds, budgets 30/100/300)
  E2: SARSA vs Q-Learning control, 300 episodes x 5 seeds (return mean±std)
  E3: Cliff falls per episode (exploration cost)
  E4: Greedy rollout demo from learned Q (seed=0): steps, return, falls
Figures: figs/fig1_mc_td_mse.png, figs/fig2_sarsa_qlearning_return.png,
         figs/fig3_cliff_falls.png, figs/fig4_greedy_path.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.mdp import CliffWalking, greedy_policy_from_Q
from common.solvers import (
    mc_prediction,
    policy_evaluation,
    q_learning,
    sarsa,
    sarsa_falls,
    td_zero_prediction,
)
from common.utils import cliff_policy_arrows, curve_with_band, savefig, smooth

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_EP_CTRL = 300
ALPHA, GAMMA, EPS = 0.5, 0.9, 0.1
MAX_STEPS = 200


def uniform_policy_fn(s, rng):
    return int(rng.integers(4))


def main():
    print(f"[config] CliffWalking 4x12, alpha={ALPHA}, gamma={GAMMA}, eps={EPS}, "
          f"seeds={SEEDS}, max_steps={MAX_STEPS}")
    env = CliffWalking()

    # DP truth for uniform policy, discounted -> bounded, converges fast
    uniform = np.ones((env.nS, env.nA)) / env.nA
    V_star = policy_evaluation(env, uniform, gamma=GAMMA, theta=1e-6)
    print(f"[truth] V_uniform(start, gamma=0.9) = {V_star[36]:.2f}")

    # E1: MC vs TD sample efficiency
    budgets = [30, 100, 300]
    mc_mses = np.zeros((len(SEEDS), len(budgets)))
    td_mses = np.zeros((len(SEEDS), len(budgets)))
    for i, sd in enumerate(SEEDS):
        for j, b in enumerate(budgets):
            V_mc, _ = mc_prediction(env, uniform_policy_fn, n_episodes=b,
                                    gamma=GAMMA, seed=sd, max_steps=MAX_STEPS)
            V_td = td_zero_prediction(env, uniform_policy_fn, n_episodes=b,
                                      alpha=0.1, gamma=GAMMA, seed=sd, max_steps=MAX_STEPS)
            mc_mses[i, j] = float(np.mean((V_mc - V_star) ** 2))
            td_mses[i, j] = float(np.mean((V_td - V_star) ** 2))
    for j, b in enumerate(budgets):
        print(f"[E1] budget={b:5d} MC_MSE={mc_mses[:, j].mean():.2f}±{mc_mses[:, j].std():.2f} "
              f"TD_MSE={td_mses[:, j].mean():.2f}±{td_mses[:, j].std():.2f}")

    # E2/E3: control, 5 seeds
    sarsa_rets, ql_rets, ql_falls = [], [], []
    for sd in SEEDS:
        _, r_s, _ = sarsa(CliffWalking(), N_EP_CTRL, ALPHA, GAMMA, EPS,
                          seed=sd, max_steps=MAX_STEPS)
        _, r_q, _, f_q = q_learning(CliffWalking(), N_EP_CTRL, ALPHA, GAMMA, EPS,
                                    seed=sd, max_steps=MAX_STEPS)
        sarsa_rets.append(r_s)
        ql_rets.append(r_q)
        ql_falls.append(f_q)
    sarsa_rets = np.array(sarsa_rets)
    ql_rets = np.array(ql_rets)
    ql_falls = np.array(ql_falls)
    print(f"[E2] last-50 return: SARSA={sarsa_rets[:, -50:].mean():.1f}±"
          f"{sarsa_rets[:, -50:].mean(axis=1).std():.1f} "
          f"Q-Learning={ql_rets[:, -50:].mean():.1f}±{ql_rets[:, -50:].mean(axis=1).std():.1f}")
    print(f"[E3] total cliff falls over {N_EP_CTRL}eps x 5seeds: Q-Learning={int(ql_falls.sum())}")

    # E4: greedy demo from seed-0 Q
    Q_s0, _, _, _ = q_learning(CliffWalking(), N_EP_CTRL, ALPHA, GAMMA, EPS,
                               seed=0, max_steps=MAX_STEPS)
    Q_sarsa0, _, _ = sarsa(CliffWalking(), N_EP_CTRL, ALPHA, GAMMA, EPS,
                           seed=0, max_steps=MAX_STEPS)
    pi_q = greedy_policy_from_Q(Q_s0)
    pi_s = greedy_policy_from_Q(Q_sarsa0)
    print("[E4] Q-Learning greedy policy:\n" + cliff_policy_arrows(pi_q))
    print("[E4] SARSA greedy policy:\n" + cliff_policy_arrows(pi_s))

    def rollout(Q):
        e = CliffWalking()
        s = e.reset()
        steps, G, falls, path = 0, 0.0, 0, [s]
        done = False
        while not done and steps < MAX_STEPS:
            a = int(np.argmax(Q[s]))
            s, r, done = e.step(a)
            G += r
            steps += 1
            path.append(s)
            if r == -100.0:
                falls += 1
        return steps, G, falls, path

    st, Gt, ft, path_q = rollout(Q_s0)
    print(f"[E4] Q-Learning greedy rollout: steps={st}, return={Gt:.0f}, falls={ft}")
    st2, Gt2, ft2, _ = rollout(Q_sarsa0)
    print(f"[E4] SARSA greedy rollout: steps={st2}, return={Gt2:.0f}, falls={ft2}")
    fq_eval = sarsa_falls(CliffWalking(), Q_s0, 20)
    fs_eval = sarsa_falls(CliffWalking(), Q_sarsa0, 20)
    print(f"[E4] 20 greedy eval episodes falls: Q-Learning={fq_eval}, SARSA={fs_eval}")

    # Fig1: MC vs TD MSE
    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    curve_with_band(ax, budgets, mc_mses.mean(0), mc_mses.std(0), "MC first-visit")
    curve_with_band(ax, budgets, td_mses.mean(0), td_mses.std(0), "TD(0)")
    ax.set_xscale("log")
    ax.set_xlabel("episodes (log)")
    ax.set_ylabel("MSE to DP truth")
    ax.set_title("MC vs TD(0) prediction (uniform policy, gamma=0.9, 5 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_mc_td_mse.png"))

    # Fig2: control return curves (per-episode mean±std across seeds)
    xs = np.arange(1, N_EP_CTRL + 1)
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    curve_with_band(ax, xs, smooth(sarsa_rets.mean(0)), smooth(sarsa_rets.std(0)), "SARSA")
    curve_with_band(ax, xs, smooth(ql_rets.mean(0)), smooth(ql_rets.std(0)), "Q-Learning")
    ax.set_xlabel("episode")
    ax.set_ylabel("return (smoothed)")
    ax.set_title("CliffWalking control: SARSA vs Q-Learning (eps=0.1, 5 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig2_sarsa_qlearning_return.png"))

    # Fig3: falls per episode (Q-Learning, mean over seeds)
    plt.figure(figsize=(8, 3.5))
    plt.plot(xs, smooth(ql_falls.mean(0)), label="Q-Learning falls/ep (5-seed mean, smoothed)")
    plt.xlabel("episode")
    plt.ylabel("cliff falls")
    plt.title("Exploration cost: Q-Learning keeps falling while chasing optimal edge")
    plt.legend()
    savefig(os.path.join(FIGS, "fig3_cliff_falls.png"))

    # Fig4: greedy path sketch (Q-Learning seed 0)
    grid = np.zeros((4, 12))
    for s in path_q:
        grid[s // 12, s % 12] += 1
    plt.figure(figsize=(10, 3.5))
    plt.imshow(grid, cmap="hot")
    plt.colorbar(label="visits")
    plt.title(f"Q-Learning greedy path visits (steps={st}, return={Gt:.0f})")
    plt.xlabel("col")
    plt.ylabel("row")
    savefig(os.path.join(FIGS, "fig4_greedy_path.png"))

    print(f"[done] figs -> {FIGS} (4 png)")


if __name__ == "__main__":
    main()
