"""Toy "LLM" alignment task. torch, CPU, no real language model.

A toy policy maps a prompt vector to a distribution over `vocab` tokens
(single-token "responses"). The correct token is a deterministic function of
the prompt (argmax of a fixed linear map), so the task is learnable and has a
clean optimum. This lets us exercise DPO / GRPO machinery end-to-end cheaply.

Pieces:
  ToyPolicy      : MLP(prompt_dim) -> logits(vocab); log_probs(); sample()
  build_task     : prompts + targets (+ hidden reward map W)
  reward         : 1.0 if token == target else 0.0
  make_pairs     : (prompt, chosen=target, rejected=random other token)
"""

import numpy as np
import torch
import torch.nn as nn


class ToyPolicy(nn.Module):
    def __init__(self, prompt_dim, vocab, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prompt_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, vocab),
        )

    def forward(self, x):
        return self.net(x)

    def log_probs(self, x):
        return torch.log_softmax(self.forward(x), dim=-1)

    @torch.no_grad()
    def sample_tokens(self, x, group=1):
        """Sample `group` tokens per prompt (no grad). Returns tokens (N, G)."""
        logits = self.forward(x)  # (N, V)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample((group,)).transpose(0, 1)  # (N, G)

    def log_probs_of(self, x, tokens):
        """log pi(tokens | x) with grad. tokens: (N, G) -> (N, G)."""
        logp = self.log_probs(x)  # (N, V)
        g = tokens.shape[1]
        return logp.unsqueeze(1).expand(-1, g, -1).gather(2, tokens.unsqueeze(-1)).squeeze(-1)


def build_task(seed=0, n_prompts=128, prompt_dim=8, vocab=8, W=None):
    """Generate prompts + targets. Pass the same W to share the reward map across splits."""
    rng = np.random.default_rng(seed)
    if W is None:
        W = rng.normal(0, 1, size=(prompt_dim, vocab)).astype(np.float32)
    prompts = rng.normal(0, 1, size=(n_prompts, prompt_dim)).astype(np.float32)
    targets = (prompts @ W).argmax(axis=1).astype(np.int64)
    return prompts, targets, W


def reward_tokens(tokens, targets):
    """tokens, targets: torch tensors same shape -> float reward tensor."""
    return (tokens == targets).float()


def make_pairs(prompts, targets, vocab, n_pairs, seed=0, noise=0.0):
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(prompts), size=n_pairs)
    chosen = targets[idx].copy()
    rejected = rng.integers(0, vocab, size=n_pairs)
    bad = rejected == chosen
    rejected[bad] = (rejected[bad] + 1) % vocab
    if noise > 0.0:
        swap = rng.random(n_pairs) < noise
        chosen, rejected = np.where(swap, rejected, chosen), np.where(swap, chosen, rejected)
    return prompts[idx], chosen, rejected


@torch.no_grad()
def accuracy(policy, prompts, targets):
    pred = policy.forward(prompts).argmax(dim=1)
    return float((pred == targets).float().mean().item())


@torch.no_grad()
def kl_to_ref(policy, ref, prompts):
    """Mean forward KL approx: E_pi[log pi - log ref]."""
    lp = policy.log_probs(prompts)
    lr = ref.log_probs(prompts)
    pi = lp.exp()
    return float((pi * (lp - lr)).sum(dim=1).mean().item())
