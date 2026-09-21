"""On-policy trainers: REINFORCE (+/- baseline) and PPO-Clip. CPU-first, discrete actions.

train_reinforce: Monte-Carlo returns, optional learned V baseline.
train_ppo: rollout + GAE + clipped surrogate, value + entropy terms, KL monitoring.
Both return dicts with ep_returns, losses, eval_returns (+ kl for PPO).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import RolloutBuffer
from .nets import PolicyNet, ValueNet, greedy_action


def _eval_policy(policy, env_id, n_eval=10, max_steps=500, device="cpu", base_seed=999):
    import gymnasium as gym

    env = gym.make(env_id)
    rets = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=base_seed + i)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            a = greedy_action(policy, obs, device)
            obs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            G += r
            t += 1
        rets.append(G)
    env.close()
    return float(np.mean(rets))


def train_reinforce(
    env_id="CartPole-v1",
    seed=0,
    total_episodes=300,
    hidden=128,
    lr=3e-3,
    gamma=0.99,
    use_baseline=True,
    max_steps=500,
    n_eval=10,
    eval_interval=25,
    device="cpu",
):
    import gymnasium as gym

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    value = ValueNet(obs_dim, hidden).to(device) if use_baseline else None
    opt_v = torch.optim.Adam(value.parameters(), lr=lr) if use_baseline else None

    ep_returns, ep_losses, grad_norms = [], [], []
    eval_episodes, eval_returns = [], []
    for ep in range(total_episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        traj = []
        done, G, t = False, 0.0, 0
        while not done and t < max_steps:
            policy.eval()
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                logits = policy(o)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            a = int(rng.choice(act_dim, p=probs))
            logp = float(np.log(probs[a] + 1e-8))
            nobs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            traj.append((obs, a, r, logp))
            obs = nobs
            G += r
            t += 1
        ep_returns.append(G)

        # discounted returns
        rets = np.zeros(len(traj), dtype=np.float32)
        g = 0.0
        for i in reversed(range(len(traj))):
            g = traj[i][2] + gamma * g
            rets[i] = g
        obs_b = torch.as_tensor(np.array([t[0] for t in traj], dtype=np.float32)).to(device)
        act_b = torch.as_tensor(np.array([t[1] for t in traj]), dtype=torch.int64).to(device)
        logp_b = torch.as_tensor(np.array([t[3] for t in traj], dtype=np.float32)).to(device)
        ret_b = torch.as_tensor(rets).to(device)

        if use_baseline:
            value.train()
            v = value(obs_b)
            adv = ret_b - v.detach()
            loss_v = F.mse_loss(v, ret_b)
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()
        else:
            adv = ret_b - ret_b.mean()

        policy.train()
        logits = policy(obs_b)
        log_probs = torch.log_softmax(logits, dim=-1)
        lp = log_probs.gather(1, act_b.unsqueeze(1)).squeeze(1)
        loss = -(lp * adv.detach()).mean()
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5))
        opt.step()
        ep_losses.append(float(loss.item()))
        grad_norms.append(gn)

        if (ep + 1) % eval_interval == 0 or ep == total_episodes - 1:
            ev = _eval_policy(policy, env_id, n_eval, max_steps, device)
            eval_episodes.append(ep + 1)
            eval_returns.append(ev)
    env.close()
    return {
        "ep_returns": np.array(ep_returns),
        "losses": np.array(ep_losses),
        "grad_norms": np.array(grad_norms),
        "eval_episodes": np.array(eval_episodes),
        "eval_returns": np.array(eval_returns),
        "use_baseline": use_baseline,
        "seed": seed,
    }


def train_ppo(
    env_id="CartPole-v1",
    seed=0,
    total_updates=60,
    rollout_steps=1024,
    hidden=128,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    epochs=4,
    batch_size=256,
    vf_coef=0.5,
    ent_coef=0.01,
    max_steps=500,
    n_eval=10,
    eval_interval=10,
    device="cpu",
):
    import gymnasium as gym

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, hidden).to(device)
    value = ValueNet(obs_dim, hidden).to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=lr)
    buf = RolloutBuffer(rollout_steps + 8, obs_dim, seed=seed)

    ep_returns = []
    eval_episodes, eval_returns = [], []
    pol_losses, val_losses, ents, kls, clip_fracs = [], [], [], [], []
    updates = 0
    obs, _ = env.reset(seed=seed * 100000)

    while updates < total_updates:
        buf.reset()
        for _ in range(rollout_steps):
            policy.eval()
            value.eval()
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                logits = policy(o)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                v = float(value(o).item())
            a = int(np.random.choice(act_dim, p=probs))
            logp = float(np.log(probs[a] + 1e-8))
            nobs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            buf.add(obs, a, r, float(done), logp, v)
            obs = nobs
            if done:
                obs, _ = env.reset()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            last_v = float(value(o).item())
        rets, advs = buf.compute_returns_advantages(last_v, gamma, lam, normalize_adv=True)

        # epochs over rollout
        e_pl, e_vl, e_en, e_kl, e_cf = 0.0, 0.0, 0.0, 0.0, 0.0
        n_b = 0
        for _ in range(epochs):
            for b_o, b_a, b_old_logp, b_ret, b_adv in buf.batches(rets, advs, batch_size, device):
                policy.train()
                value.train()
                logits = policy(b_o)
                log_probs = torch.log_softmax(logits, dim=-1)
                lp = log_probs.gather(1, b_a.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(lp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                loss_p = -torch.min(surr1, surr2).mean()
                v = value(b_o)
                loss_v = F.mse_loss(v, b_ret)
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * log_probs).sum(-1).mean()
                loss = loss_p + vf_coef * loss_v - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value.parameters()), 0.5
                )
                opt.step()
                with torch.no_grad():
                    kl = (b_old_logp - lp).mean()
                    cf = ((ratio - 1.0).abs() > clip_eps).float().mean()
                e_pl += float(loss_p.item())
                e_vl += float(loss_v.item())
                e_en += float(ent.item())
                e_kl += float(kl.item())
                e_cf += float(cf.item())
                n_b += 1
        pol_losses.append(e_pl / max(1, n_b))
        val_losses.append(e_vl / max(1, n_b))
        ents.append(e_en / max(1, n_b))
        kls.append(e_kl / max(1, n_b))
        clip_fracs.append(e_cf / max(1, n_b))
        updates += 1
        if updates % eval_interval == 0 or updates == total_updates:
            ev = _eval_policy(policy, env_id, n_eval, max_steps, device)
            eval_episodes.append(updates)
            eval_returns.append(ev)

    env.close()
    return {
        "eval_episodes": np.array(eval_episodes),
        "eval_returns": np.array(eval_returns),
        "pol_losses": np.array(pol_losses),
        "val_losses": np.array(val_losses),
        "ents": np.array(ents),
        "kls": np.array(kls),
        "clip_fracs": np.array(clip_fracs),
        "clip_eps": clip_eps,
        "seed": seed,
    }
