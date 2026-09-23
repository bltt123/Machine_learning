"""Alignment losses + trainers: SFT, DPO/IPO/cDPO/ORPO, GRPO/REINFORCE. CPU toy.

All trainers return eval accuracy curve, KL-to-ref, entropy, and loss curves.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .toy import ToyPolicy, accuracy, kl_to_ref, make_pairs, reward_tokens


def _entropy(policy, prompts):
    lp = policy.log_probs(prompts)
    return float(-(lp.exp() * lp).sum(dim=1).mean().item())


def train_sft(prompts, targets, seed=0, steps=600, lr=1e-2, batch=64,
              eval_prompts=None, eval_targets=None, eval_every=100,
              hidden=64):
    torch.manual_seed(seed)
    policy = ToyPolicy(prompts.shape[1], int(targets.max()) + 1, hidden)
    ref = ToyPolicy(prompts.shape[1], int(targets.max()) + 1, hidden)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    p_t = torch.as_tensor(prompts)
    t_t = torch.as_tensor(targets)
    n = len(prompts)
    rng = np.random.default_rng(seed)
    losses, ev_steps, ev_acc, ev_kl, ev_ent = [], [], [], [], []
    for s in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        logits = policy(p_t[idx])
        loss = F.cross_entropy(logits, t_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if s % eval_every == 0 or s == steps:
            ev_steps.append(s)
            ev_acc.append(accuracy(policy, eval_prompts, eval_targets))
            ev_kl.append(kl_to_ref(policy, ref, eval_prompts))
            ev_ent.append(_entropy(policy, eval_prompts))
    return {"algo": "sft", "losses": np.array(losses), "ev_steps": np.array(ev_steps),
            "eval_acc": np.array(ev_acc), "eval_kl": np.array(ev_kl),
            "eval_ent": np.array(ev_ent), "model": policy}


def dpo_loss(policy, ref, prompts, chosen, rejected, beta=0.1, variant="dpo", label_smooth=0.0):
    lp_c = policy.log_probs(prompts).gather(1, chosen.unsqueeze(1)).squeeze(1)
    lp_r = policy.log_probs(prompts).gather(1, rejected.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        lr_c = ref.log_probs(prompts).gather(1, chosen.unsqueeze(1)).squeeze(1)
        lr_r = ref.log_probs(prompts).gather(1, rejected.unsqueeze(1)).squeeze(1)
    logit = beta * ((lp_c - lr_c) - (lp_r - lr_r))
    if variant == "dpo":
        loss = -F.logsigmoid(logit).mean()
    elif variant == "cdpo":
        loss = -((1 - label_smooth) * F.logsigmoid(logit) + label_smooth * F.logsigmoid(-logit)).mean()
    elif variant == "ipo":
        h = (lp_c - lp_r) - (lr_c - lr_r)
        loss = ((h - 1.0 / (2.0 * beta)) ** 2).mean()
    else:
        raise ValueError(variant)
    margin = (logit > 0).float().mean()
    return loss, logit, margin


def orpo_loss(policy, prompts, chosen, rejected, lam=1.0):
    lp_c = policy.log_probs(prompts).gather(1, chosen.unsqueeze(1)).squeeze(1)
    lp_r = policy.log_probs(prompts).gather(1, rejected.unsqueeze(1)).squeeze(1)
    sft = -lp_c.mean()
    p_c = lp_c.exp().clamp(max=1 - 1e-6)
    p_r = lp_r.exp().clamp(max=1 - 1e-6)
    log_odds_c = lp_c - torch.log1p(-p_c)
    log_odds_r = lp_r - torch.log1p(-p_r)
    or_loss = -F.logsigmoid(log_odds_c - log_odds_r).mean()
    return sft + lam * or_loss, sft, or_loss


def train_dpo(prompts, targets, vocab, seed=0, variant="dpo", beta=0.1, lr=1e-2,
              steps=600, batch=64, n_pairs=4096, label_smooth=0.0, lam=1.0,
              eval_prompts=None, eval_targets=None, eval_every=100, hidden=64, noise=0.0):
    torch.manual_seed(seed)
    policy = ToyPolicy(prompts.shape[1], vocab, hidden)
    ref = ToyPolicy(prompts.shape[1], vocab, hidden)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    pp, cc, rr = make_pairs(prompts, targets, vocab, n_pairs, seed, noise)
    pp_t = torch.as_tensor(pp)
    cc_t = torch.as_tensor(cc)
    rr_t = torch.as_tensor(rr)
    n = len(pp)
    rng = np.random.default_rng(seed)
    losses, ev_steps, ev_acc, ev_kl, ev_ent, ev_margin = [], [], [], [], [], []
    for s in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        if variant == "orpo":
            loss, _, _ = orpo_loss(policy, pp_t[idx], cc_t[idx], rr_t[idx], lam)
            margin = torch.tensor(0.0)
        else:
            loss, _, margin = dpo_loss(policy, ref, pp_t[idx], cc_t[idx], rr_t[idx],
                                       beta, variant, label_smooth)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if s % eval_every == 0 or s == steps:
            ev_steps.append(s)
            ev_acc.append(accuracy(policy, eval_prompts, eval_targets))
            ev_kl.append(kl_to_ref(policy, ref, eval_prompts))
            ev_ent.append(_entropy(policy, eval_prompts))
            ev_margin.append(float(margin.item()))
    return {"algo": variant, "losses": np.array(losses), "ev_steps": np.array(ev_steps),
            "eval_acc": np.array(ev_acc), "eval_kl": np.array(ev_kl),
            "eval_ent": np.array(ev_ent), "eval_margin": np.array(ev_margin), "model": policy}


def train_grpo(prompts, targets, vocab, seed=0, algo="grpo", group=8, beta=0.04,
               lr=1e-2, steps=600, batch=32, eval_prompts=None, eval_targets=None,
               eval_every=100, hidden=64, use_baseline=True):
    torch.manual_seed(seed)
    policy = ToyPolicy(prompts.shape[1], vocab, hidden)
    ref = ToyPolicy(prompts.shape[1], vocab, hidden)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    p_t = torch.as_tensor(prompts)
    t_t = torch.as_tensor(targets)
    n = len(prompts)
    rng = np.random.default_rng(seed)
    losses, ev_steps, ev_acc, ev_kl, ev_ent, ev_rew = [], [], [], [], [], []
    for s in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        b_p = p_t[idx]
        b_t = t_t[idx]
        tokens = policy.sample_tokens(b_p, group=group)  # (B, G) no grad
        lp = policy.log_probs_of(b_p, tokens)  # (B, G) with grad
        rew = reward_tokens(tokens, b_t.unsqueeze(1).expand_as(tokens))  # (B, G)
        if algo == "grpo":
            adv = (rew - rew.mean(dim=1, keepdim=True)) / (rew.std(dim=1, keepdim=True) + 1e-6)
        else:  # reinforce
            base = rew.mean(dim=1, keepdim=True) if use_baseline else 0.0
            adv = rew - base
        # KL penalty to ref (approx, per sampled token)
        with torch.no_grad():
            lr = ref.log_probs_of(b_p, tokens)
        kl = (lp - lr)
        loss = -(adv * lp).mean() + beta * kl.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if s % eval_every == 0 or s == steps:
            ev_steps.append(s)
            ev_acc.append(accuracy(policy, eval_prompts, eval_targets))
            ev_kl.append(kl_to_ref(policy, ref, eval_prompts))
            ev_ent.append(_entropy(policy, eval_prompts))
            ev_rew.append(float(rew.mean().item()))
    return {"algo": algo, "losses": np.array(losses), "ev_steps": np.array(ev_steps),
            "eval_acc": np.array(ev_acc), "eval_kl": np.array(ev_kl),
            "eval_ent": np.array(ev_ent), "eval_rew": np.array(ev_rew), "model": policy}
