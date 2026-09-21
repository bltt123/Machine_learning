"""Rollout storage + discounted returns + GAE. Numpy buffers, torch batch output."""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, capacity, obs_dim, seed=0):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity,), dtype=np.int64)
        self.rew = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.logp = np.zeros((self.capacity,), dtype=np.float32)
        self.val = np.zeros((self.capacity,), dtype=np.float32)
        self.n = 0

    def __len__(self):
        return self.n

    def reset(self):
        self.n = 0

    def add(self, o, a, r, d, logp, v):
        i = self.n
        self.obs[i] = np.asarray(o, dtype=np.float32)
        self.act[i] = int(a)
        self.rew[i] = float(r)
        self.done[i] = float(d)
        self.logp[i] = float(logp)
        self.val[i] = float(v)
        self.n += 1

    def compute_returns_advantages(self, last_val, gamma, lam, normalize_adv=True):
        n = self.n
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                nv = float(last_val)
            else:
                nv = self.val[t + 1]
            nd = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * nv * nd - self.val[t]
            last_gae = delta + gamma * lam * nd * last_gae
            adv[t] = last_gae
        ret = adv + self.val[:n]
        if normalize_adv and n > 1:
            m, s = float(adv.mean()), float(adv.std() + 1e-8)
            adv = (adv - m) / s
        return ret, adv

    def batches(self, returns, advs, batch_size, device="cpu"):
        idx = self.rng.permutation(self.n)
        to = lambda x: torch.as_tensor(x).to(device)
        obs = to(self.obs[: self.n])
        act = to(self.act[: self.n])
        logp = to(self.logp[: self.n])
        ret = to(returns)
        adv = to(advs)
        bs = int(batch_size)
        for s in range(0, self.n, bs):
            j = idx[s : s + bs]
            yield obs[j], act[j], logp[j], ret[j], adv[j]
