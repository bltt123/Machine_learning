"""Nets for offline RL (discrete actions): QNet + ValueNet + PolicyNet. torch, CPU-first."""

import torch
import torch.nn as nn


class QNet(nn.Module):
    """Q(s, .): MLP -> n_actions."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    """V(s): MLP -> scalar (IQL)."""

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PolicyNet(nn.Module):
    """pi(.|s): MLP -> logits (BC / IQL policy extraction)."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
