"""Offline trainers: BC, CQL, IQL (discrete CartPole). CPU-first.

All train only on the fixed offline dataset; eval is an online greedy rollout.
- train_bc : supervised cross-entropy pi(a|s).
- train_cql: conservative Q-learning, alpha * (logsumexp_a Q - Q_data) penalty.
- train_iql: expectile V + no-OOD Q (uses V(s') target) + AWR policy extraction.

Each returns eval_returns (online greedy) + diagnostics + training curves.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import OfflineDataset
from .nets import PolicyNet, QNet, ValueNet


def _eval(env_id, act_fn, n_eval=20, max_steps=500, base_seed=999):
    import gymnasium as gym

    env = gym.make(env_id)
    rets = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=base_seed + i)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            a = act_fn(obs)
            obs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            G += r
            t += 1
        rets.append(G)
    env.close()
    return float(np.mean(rets))


def train_bc(data, env_id="CartPole-v1", seed=0, hidden=128, lr=1e-3,
             steps=4000, batch_size=256, eval_interval=1000, n_eval=20,
             device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = OfflineDataset(data, seed=seed)
    obs_dim = data["obs"].shape[1]
    act_dim = int(data["act"].max()) + 1
    policy = PolicyNet(obs_dim, act_dim, hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    losses, eval_steps, eval_returns = [], [], []
    for t in range(1, steps + 1):
        b_o, b_a, _, _, _ = ds.sample(batch_size, device)
        logits = policy(b_o)
        loss = F.cross_entropy(logits, b_a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if t % eval_interval == 0 or t == steps:
            ev = _eval(env_id, lambda o: int(torch.argmax(policy(
                torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)), dim=1).item()),
                n_eval=n_eval)
            eval_steps.append(t)
            eval_returns.append(ev)
    return {
        "algo": "bc",
        "seed": seed,
        "losses": np.array(losses),
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "model": policy,
    }


def train_cql(data, env_id="CartPole-v1", seed=0, hidden=128, lr=1e-3, gamma=0.99,
              steps=8000, batch_size=256, target_sync=500, cql_alpha=1.0,
              eval_interval=2000, n_eval=20, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = OfflineDataset(data, seed=seed)
    obs_dim = data["obs"].shape[1]
    act_dim = int(data["act"].max()) + 1
    q = QNet(obs_dim, act_dim, hidden).to(device)
    q_target = QNet(obs_dim, act_dim, hidden).to(device)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    td_losses, cql_losses, q_data_means, q_ood_means = [], [], [], []
    eval_steps, eval_returns = [], []
    for t in range(1, steps + 1):
        b_o, b_a, b_r, b_no, b_d = ds.sample(batch_size, device)
        qsa = q(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qn = q_target(b_no).max(dim=1).values
            y = b_r + gamma * (1.0 - b_d) * qn
        td = F.mse_loss(qsa, y)
        # discrete CQL: push down all actions, push up data action
        q_all = q(b_o)
        cql = (torch.logsumexp(q_all, dim=1) - qsa).mean()
        loss = td + cql_alpha * cql
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % target_sync == 0:
            q_target.load_state_dict(q.state_dict())
        td_losses.append(float(td.item()))
        cql_losses.append(float(cql.item()))
        if t % 200 == 0:
            with torch.no_grad():
                q_data_means.append(float(qsa.mean().item()))
                q_ood_means.append(float(q_all.mean().item()))
        if t % eval_interval == 0 or t == steps:
            ev = _eval(env_id, lambda o: int(torch.argmax(q(
                torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)), dim=1).item()),
                n_eval=n_eval)
            eval_steps.append(t)
            eval_returns.append(ev)
    return {
        "algo": "cql",
        "seed": seed,
        "td_losses": np.array(td_losses),
        "cql_losses": np.array(cql_losses),
        "q_data_means": np.array(q_data_means),
        "q_ood_means": np.array(q_ood_means),
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "cql_alpha": cql_alpha,
        "model": q,
    }


def train_iql(data, env_id="CartPole-v1", seed=0, hidden=128, lr=1e-3, gamma=0.99,
              steps=8000, batch_size=256, expectile=0.7, beta=3.0, tau=0.005,
              eval_interval=2000, n_eval=20, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = OfflineDataset(data, seed=seed)
    obs_dim = data["obs"].shape[1]
    act_dim = int(data["act"].max()) + 1
    q = QNet(obs_dim, act_dim, hidden).to(device)
    v = ValueNet(obs_dim, hidden).to(device)
    policy = PolicyNet(obs_dim, act_dim, hidden).to(device)
    opt_q = torch.optim.Adam(q.parameters(), lr=lr)
    opt_v = torch.optim.Adam(v.parameters(), lr=lr)
    opt_p = torch.optim.Adam(policy.parameters(), lr=lr)

    v_losses, q_losses, p_losses, adv_means = [], [], [], []
    eval_steps, eval_returns = [], []
    for t in range(1, steps + 1):
        b_o, b_a, b_r, b_no, b_d = ds.sample(batch_size, device)
        with torch.no_grad():
            qsa_t = q(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
        v_s = v(b_o)
        # expectile regression: weight by asymmetric tau
        diff = qsa_t - v_s
        weight = torch.where(diff > 0, torch.full_like(diff, 1.0 - expectile),
                             torch.full_like(diff, expectile))
        loss_v = (weight * diff.pow(2)).mean()
        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()

        with torch.no_grad():
            v_next = v(b_no)
            y = b_r + gamma * (1.0 - b_d) * v_next
        qsa = q(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
        loss_q = F.mse_loss(qsa, y)
        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()

        with torch.no_grad():
            adv = (qsa - v_s).clamp(max=100.0)
            w = torch.exp(beta * adv).clamp(max=100.0)
        logits = policy(b_o)
        logp = F.log_softmax(logits, dim=-1).gather(1, b_a.unsqueeze(1)).squeeze(1)
        loss_p = -(w * logp).mean()
        opt_p.zero_grad()
        loss_p.backward()
        opt_p.step()

        v_losses.append(float(loss_v.item()))
        q_losses.append(float(loss_q.item()))
        p_losses.append(float(loss_p.item()))
        adv_means.append(float(adv.mean().item()))
        if t % eval_interval == 0 or t == steps:
            ev = _eval(env_id, lambda o: int(torch.argmax(policy(
                torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)), dim=1).item()),
                n_eval=n_eval)
            eval_steps.append(t)
            eval_returns.append(ev)
    return {
        "algo": "iql",
        "seed": seed,
        "v_losses": np.array(v_losses),
        "q_losses": np.array(q_losses),
        "p_losses": np.array(p_losses),
        "adv_means": np.array(adv_means),
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "expectile": expectile,
        "model": policy,
    }
