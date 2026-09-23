"""01_SB3_Tianshou_Compare run.py — SB3 (standard answer) vs self-impl DQN/PPO on CartPole.

Goal: cross-validate our hand-written 02/03 algorithms against Stable-Baselines3.
Same env, same budget, same seeds; compare eval curves + final scores.
Experiments:
  E1: PPO — SB3 vs mine (eval return curves)
  E2: DQN — SB3 vs mine
  E3: summary bar (final eval) + sample efficiency
Figures: figs/fig1_ppo_sb3_vs_mine.png, figs/fig2_dqn_sb3_vs_mine.png, figs/fig3_summary.png
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
from common.algos import train_dqn, train_ppo
from common.plot import curve_with_band, savefig

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

ENV = "CartPole-v1"
SEEDS = [0, 1, 2]
TOTAL = 40000
EVAL_EVERY = 5000
N_EVAL = 10


def run_sb3_ppo(seed):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    ev_env = gym.make(ENV)

    class CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.steps, self.rets = [], []

        def _on_step(self):
            if self.n_calls % EVAL_EVERY == 0:
                r = _eval_sb3(ev_env, self.model)
                self.steps.append(self.n_calls)
                self.rets.append(r)
            return True

    model = PPO("MlpPolicy", ENV, seed=seed, n_steps=1024, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4, verbose=0)
    cb = CB()
    model.learn(total_timesteps=TOTAL, callback=cb)
    ev_env.close()
    return {"algo": "ppo(sb3)", "seed": seed,
            "ev_steps": np.array(cb.steps), "eval_returns": np.array(cb.rets)}


def run_sb3_dqn(seed):
    import gymnasium as gym
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback

    ev_env = gym.make(ENV)

    class CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.steps, self.rets = [], []

        def _on_step(self):
            if self.n_calls % EVAL_EVERY == 0:
                r = _eval_sb3(ev_env, self.model)
                self.steps.append(self.n_calls)
                self.rets.append(r)
            return True

    model = DQN("MlpPolicy", ENV, seed=seed, learning_rate=1e-3, buffer_size=50000,
                learning_starts=1000, batch_size=64, gamma=0.99, target_update_interval=500,
                train_freq=1, gradient_steps=1, verbose=0)
    cb = CB()
    model.learn(total_timesteps=TOTAL, callback=cb)
    ev_env.close()
    return {"algo": "dqn(sb3)", "seed": seed,
            "ev_steps": np.array(cb.steps), "eval_returns": np.array(cb.rets)}


def _eval_sb3(env, model, n_eval=N_EVAL):
    rets = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=12345 + i)
        G, done, t = 0.0, False, 0
        while not done and t < 500:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(a))
            done = bool(term or trunc)
            G += r
            t += 1
        rets.append(G)
    return float(np.mean(rets))


def main():
    print(f"[config] {ENV}, budget={TOTAL} steps, seeds={SEEDS}, eval_every={EVAL_EVERY}")
    print("[E1] PPO: SB3 vs mine")
    ppo_sb3 = [run_sb3_ppo(s) for s in SEEDS]
    ppo_mine = [train_ppo(env_id=ENV, seed=s, total_updates=TOTAL // 1024,
                          rollout_steps=1024, epochs=10, batch=64,
                          eval_interval=max(1, EVAL_EVERY // 1024),
                          n_eval=N_EVAL) for s in SEEDS]
    for tag, hs in [("ppo(sb3)", ppo_sb3), ("ppo(mine)", ppo_mine)]:
        ev = np.array([h["eval_returns"] for h in hs])
        print(f"    {tag}: eval final {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")

    print("[E2] DQN: SB3 vs mine")
    dqn_sb3 = [run_sb3_dqn(s) for s in SEEDS]
    dqn_mine = [train_dqn(env_id=ENV, seed=s, steps=TOTAL, eval_every=EVAL_EVERY,
                          n_eval=N_EVAL) for s in SEEDS]
    for tag, hs in [("dqn(sb3)", dqn_sb3), ("dqn(mine)", dqn_mine)]:
        ev = np.array([h["eval_returns"] for h in hs])
        print(f"    {tag}: eval final {ev[:, -1].mean():.1f}±{ev[:, -1].std():.1f}")

    # Fig1: PPO
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for tag, hs in [("SB3 PPO", ppo_sb3), ("my PPO", ppo_mine)]:
        arr = np.array([h["eval_returns"] for h in hs])
        curve_with_band(ax, hs[0]["ev_steps"], arr.mean(0), arr.std(0), tag)
    ax.set_xlabel("env steps")
    ax.set_ylabel("greedy eval return")
    ax.set_title("PPO: SB3 vs self-implemented (3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig1_ppo_sb3_vs_mine.png"))

    # Fig2: DQN
    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    for tag, hs in [("SB3 DQN", dqn_sb3), ("my DQN", dqn_mine)]:
        arr = np.array([h["eval_returns"] for h in hs])
        curve_with_band(ax, hs[0]["ev_steps"], arr.mean(0), arr.std(0), tag)
    ax.set_xlabel("env steps")
    ax.set_ylabel("greedy eval return")
    ax.set_title("DQN: SB3 vs self-implemented (3 seeds)")
    ax.legend()
    savefig(os.path.join(FIGS, "fig2_dqn_sb3_vs_mine.png"))

    # Fig3: summary bar
    labels = ["PPO sb3", "PPO mine", "DQN sb3", "DQN mine"]
    finals = [np.array([h["eval_returns"][-1] for h in hs]).mean()
              for hs in [ppo_sb3, ppo_mine, dqn_sb3, dqn_mine]]
    errs = [np.array([h["eval_returns"][-1] for h in hs]).std()
            for hs in [ppo_sb3, ppo_mine, dqn_sb3, dqn_mine]]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, finals, yerr=errs, capsize=5)
    plt.ylabel("final eval return")
    plt.title("Final score: SB3 vs self-impl (CartPole ceiling 500)")
    plt.ylim(0, 520)
    savefig(os.path.join(FIGS, "fig3_summary.png"))

    print(f"[done] figs -> {FIGS} (3 png)")


if __name__ == "__main__":
    main()