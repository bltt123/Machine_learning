"""OfflineDataset: wraps collected transitions, torch batch sampling. numpy + torch."""

import numpy as np
import torch


class OfflineDataset:
    def __init__(self, data, seed=0):
        self.obs = np.asarray(data["obs"], dtype=np.float32)
        self.act = np.asarray(data["act"], dtype=np.int64)
        self.rew = np.asarray(data["rew"], dtype=np.float32)
        self.nobs = np.asarray(data["nobs"], dtype=np.float32)
        self.done = np.asarray(data["done"], dtype=np.float32)
        self.n = len(self.rew)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def sample(self, batch_size, device="cpu"):
        j = self.rng.integers(0, self.n, size=int(batch_size))
        to = lambda x: torch.as_tensor(x).to(device)
        return (
            to(self.obs[j]),
            to(self.act[j]),
            to(self.rew[j]),
            to(self.nobs[j]),
            to(self.done[j]),
        )
