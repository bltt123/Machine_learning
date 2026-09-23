"""ReplayBuffer: uniform sampling, numpy storage, torch batch output (same as 02)."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, seed=0):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.nobs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, o, a, r, no, d):
        i = self.idx % self.capacity
        self.obs[i] = np.asarray(o, dtype=np.float32)
        self.act[i] = np.asarray(a, dtype=np.float32)
        self.rew[i] = float(r)
        self.nobs[i] = np.asarray(no, dtype=np.float32)
        self.done[i] = float(d)
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        j = self.rng.integers(0, self.size, size=int(batch_size))
        to = lambda x: torch.as_tensor(x).to(device)
        return (
            to(self.obs[j]),
            to(self.act[j]),
            to(self.rew[j]),
            to(self.nobs[j]),
            to(self.done[j]),
        )