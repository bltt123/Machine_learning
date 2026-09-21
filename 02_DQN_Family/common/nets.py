"""Q-networks: MLP QNet + DuelingQNet. Init, shapes, and greedy action."""

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    """Two-hidden-layer MLP Q(s,a). in: obs_dim, out: n_actions."""

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
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling head: Q = V + (A - mean A). Same io as QNet."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.v = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, act_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.feat(x)
        v = self.v(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=-1, keepdim=True))


def greedy_action(qnet, obs, device="cpu"):
    qnet.eval()
    with torch.no_grad():
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
        return int(torch.argmax(qnet(o), dim=1).item())


def count_params(net):
    return sum(p.numel() for p in net.parameters())
