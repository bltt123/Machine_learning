"""Offline dataset collection (CartPole) + save/load + stats. numpy only.

Policies:
  random   : uniform random actions (worst quality, low return)
  mixed    : eps-greedy PD heuristic (suboptimal, has good segments)
  expert   : PD heuristic on (theta + 0.5*theta_dot), near-optimal

Stored as .npz with keys: obs, act, rew, nobs, done (+ meta in stats.json).
"""

import json
import os

import numpy as np


def heuristic_action(obs):
    """Simple PD controller: push in the direction the pole is falling."""
    x, x_dot, theta, theta_dot = obs
    return 1 if (theta + 0.5 * theta_dot) > 0 else 0


def make_policy(name, rng):
    if name == "random":
        return lambda obs: int(rng.integers(2))
    if name == "expert":
        return lambda obs: heuristic_action(obs)
    if name == "mixed":
        def mixed(obs):
            if rng.random() < 0.4:
                return int(rng.integers(2))
            return heuristic_action(obs)
        return mixed
    if name == "hard":
        def hard(obs):
            if rng.random() < 0.75:
                return int(rng.integers(2))
            return heuristic_action(obs)
        return hard
    raise ValueError(f"unknown policy {name}")


def collect_dataset(env_id, policy_name, n_transitions, seed, max_steps=500):
    """Collect transitions + per-episode returns/lengths. Returns (data, stats, ep_returns, ep_lens)."""
    import gymnasium as gym

    rng = np.random.default_rng(seed)
    policy = make_policy(policy_name, rng)
    env = gym.make(env_id)
    obs_buf, act_buf, rew_buf, nobs_buf, done_buf = [], [], [], [], []
    ep_returns, ep_lens = [], []
    total = 0
    ep = 0
    while total < n_transitions:
        obs, _ = env.reset(seed=seed * 100000 + ep)
        done = False
        G, t = 0.0, 0
        while not done and t < max_steps:
            a = policy(obs)
            nobs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            obs_buf.append(obs)
            act_buf.append(a)
            rew_buf.append(r)
            nobs_buf.append(nobs)
            done_buf.append(float(done))
            obs = nobs
            G += r
            t += 1
            total += 1
        ep_returns.append(G)
        ep_lens.append(t)
        ep += 1
    env.close()
    data = {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "act": np.asarray(act_buf, dtype=np.int64),
        "rew": np.asarray(rew_buf, dtype=np.float32),
        "nobs": np.asarray(nobs_buf, dtype=np.float32),
        "done": np.asarray(done_buf, dtype=np.float32),
    }
    stats = {
        "policy": policy_name,
        "env_id": env_id,
        "n_transitions": int(total),
        "n_episodes": int(ep),
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "mean_ep_len": float(total / max(1, ep)),
        "action_frac_1": float(np.mean(data["act"])),
    }
    return data, stats, np.asarray(ep_returns), np.asarray(ep_lens)


def save_dataset(dir_path, name, data, stats):
    os.makedirs(dir_path, exist_ok=True)
    np.savez_compressed(os.path.join(dir_path, f"{name}.npz"), **data)
    with open(os.path.join(dir_path, f"{name}_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_dataset(dir_path, name):
    z = np.load(os.path.join(dir_path, f"{name}.npz"))
    return {k: z[k] for k in z.files}
