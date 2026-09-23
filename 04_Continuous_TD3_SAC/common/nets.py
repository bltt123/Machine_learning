"""Actor (deterministic + stochastic) and Critic (twin Q) for continuous control.

Encoder: tanh MLP with ReLU hidden layers.
DeterministicActor: mu(s) -> tanh(a) in [-1, 1].
SquashedGaussianActor: mu + sigma*eps -> tanh -> log_prob (SAC reparameterized).
TwinQCritic: two Q heads; q_min for TD3/min over two for SAC.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeterministicActor(nn.Module):
    """mu(s): obs -> action in [-1, 1]. tanh squashed."""

    def __init__(self, obs_dim, act_dim, hidden=256, act_limit=1.0):
        super().__init__()
        self.act_limit = act_limit
        self.mlp = MLP(obs_dim, act_dim, hidden)

    def forward(self, x):
        return self.act_limit * torch.tanh(self.mlp(x))


class SquashedGaussianActor(nn.Module):
    """mu(s), log_std(s); pi(a|s) = tanh(N(mu, sigma)); pi.log_prob supported."""

    def __init__(self, obs_dim, act_dim, hidden=256, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.act_dim = act_dim
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.mlp = MLP(obs_dim, act_dim * 2, hidden)

    def forward(self, x):
        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample_action(self, x):
        """Reparameterized sample a ~ pi(.|s). Returns (a in [-1,1], log_prob(a))."""
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        noise = torch.randn_like(mu)
        u = mu + std * noise
        a = torch.tanh(u)
        normal = torch.distributions.Normal(mu, std)
        logp = normal.log_prob(u) - (1 - a.pow(2) + 1e-6).log()
        return a, logp.sum(-1, keepdim=True)

    def greedy(self, x):
        """Deterministic eval: tanh(mu), no noise."""
        mu, _ = self.forward(x)
        return torch.tanh(mu)


class TwinQCritic(nn.Module):
    """Two Q heads: q1(s,a), q2(s,a). q_min = min of both (TD3/SAC target)."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, s, a, both=False):
        x = torch.cat([s, a], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        if not both:
            return q1
        q2 = self.q2(x).squeeze(-1)
        return q1, q2

    def q_min(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return torch.min(self.q1(x).squeeze(-1), self.q2(x).squeeze(-1))