"""Policy + value nets for on-policy methods. Discrete actions (CartPole/MountainCar)."""

import numpy as np
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """MLP logits pi(a|s)."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
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
    """MLP V(s)."""

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
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


@torch.no_grad()
def sample_action(policy, obs, device="cpu"):
    policy.eval()
    o = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
    logits = policy(o)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    a = int(np.random.choice(len(probs), p=probs))
    logp = float(np.log(probs[a] + 1e-8))
    return a, logp


@torch.no_grad()
def greedy_action(policy, obs, device="cpu"):
    policy.eval()
    o = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
    return int(torch.argmax(policy(o), dim=1).item())
