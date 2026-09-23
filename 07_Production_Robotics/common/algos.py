"""Compact self-implemented DQN + PPO for CartPole, used to compare against SB3.

These are deliberately small "standard-answer" reimplementations: the point of 07
is toolchain comparison, not novel algorithms. CPU-first.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def _eval(env_id, act_fn, n_eval=10, base_seed=12345):
    import gymnasium as gym

    env = gym.make(env_id)
    rets = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=base_seed + i)
        G, done, t = 0.0, False, 0
        while not done and t < 500:
            obs, r, term, trunc, _ = env.step(act_fn(obs))
            done = bool(term or trunc)
            G += r
            t += 1
        rets.append(G)
    env.close()
    return float(np.mean(rets))


def train_dqn(env_id="CartPole-v1", seed=0, steps=40000, hidden=128, lr=1e-3,
              gamma=0.99, buffer=50000, batch=64, learning_starts=1000,
              target_update=500, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.3,
              eval_every=5000, n_eval=10, device="cpu"):
    import gymnasium as gym

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    q = _mlp(obs_dim, act_dim, hidden).to(device)
    qt = _mlp(obs_dim, act_dim, hidden).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf_o = np.zeros((buffer, obs_dim), np.float32)
    buf_a = np.zeros((buffer,), np.int64)
    buf_r = np.zeros((buffer,), np.float32)
    buf_no = np.zeros((buffer, obs_dim), np.float32)
    buf_d = np.zeros((buffer,), np.float32)
    ptr, size = 0, 0
    obs, _ = env.reset(seed=seed)
    eps_decay = max(1, int(steps * eps_decay_frac))
    losses, ev_steps, ev_ret = [], [], []
    for t in range(1, steps + 1):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - t / eps_decay)
        if t < learning_starts or rng.random() < eps:
            a = int(rng.integers(act_dim))
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(obs, dtype=torch.float32).to(device)).argmax().item())
        nobs, r, term, trunc, _ = env.step(a)
        done = bool(term or trunc)
        buf_o[ptr], buf_a[ptr], buf_r[ptr], buf_no[ptr], buf_d[ptr] = obs, a, r, nobs, float(done)
        ptr = (ptr + 1) % buffer
        size = min(size + 1, buffer)
        obs = nobs
        if done:
            obs, _ = env.reset()
        if t >= learning_starts:
            j = rng.integers(0, size, size=batch)
            bo = torch.as_tensor(buf_o[j]).to(device)
            ba = torch.as_tensor(buf_a[j]).to(device)
            br = torch.as_tensor(buf_r[j]).to(device)
            bno = torch.as_tensor(buf_no[j]).to(device)
            bd = torch.as_tensor(buf_d[j]).to(device)
            qsa = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                y = br + gamma * (1 - bd) * qt(bno).max(1).values
            loss = F.mse_loss(qsa, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if t % target_update == 0:
                qt.load_state_dict(q.state_dict())
        if t % eval_every == 0 or t == steps:
            ev_steps.append(t)
            ev_ret.append(_eval(env_id, lambda o: int(q(
                torch.as_tensor(o, dtype=torch.float32).to(device)).argmax().item()), n_eval=n_eval))
    env.close()
    return {"algo": "dqn(mine)", "seed": seed, "losses": np.array(losses),
            "ev_steps": np.array(ev_steps), "eval_returns": np.array(ev_ret)}


def train_ppo(env_id="CartPole-v1", seed=0, total_updates=60, rollout_steps=1024,
              hidden=128, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4,
              batch=256, vf_coef=0.5, ent_coef=0.0, eval_interval=10, n_eval=10,
              device="cpu"):
    import gymnasium as gym

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = _mlp(obs_dim, hidden, hidden)[:-1]
            self.pi = nn.Linear(hidden, act_dim)
            self.v = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.body(x)
            return self.pi(h), self.v(h).squeeze(-1)

    net = AC().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs, _ = env.reset(seed=seed)
    ev_steps, ev_ret = [], []
    for upd in range(1, total_updates + 1):
        so, sa, sr, sd, slp, sv = [], [], [], [], [], []
        for _ in range(rollout_steps):
            o = torch.as_tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits, v = net(o.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            nobs, r, term, trunc, _ = env.step(int(a.item()))
            done = bool(term or trunc)
            so.append(obs); sa.append(int(a.item())); sr.append(r); sd.append(float(done))
            slp.append(float(lp.item())); sv.append(float(v.item()))
            obs = nobs
            if done:
                obs, _ = env.reset()
        so = torch.as_tensor(np.array(so), dtype=torch.float32).to(device)
        sa = torch.as_tensor(np.array(sa), dtype=torch.int64).to(device)
        sr = torch.as_tensor(np.array(sr), dtype=torch.float32).to(device)
        sd = torch.as_tensor(np.array(sd), dtype=torch.float32).to(device)
        slp = torch.as_tensor(np.array(slp), dtype=torch.float32).to(device)
        sv = torch.as_tensor(np.array(sv), dtype=torch.float32).to(device)
        with torch.no_grad():
            _, v_last = net(torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0))
        adv = torch.zeros_like(sr)
        last = 0.0
        for i in reversed(range(rollout_steps)):
            nonterm = 1.0 - sd[i]
            nextv = v_last.item() if i == rollout_steps - 1 else sv[i + 1]
            delta = sr[i] + gamma * nextv * nonterm - sv[i]
            last = delta + gamma * lam * nonterm * last
            adv[i] = last
        ret = adv + sv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(rollout_steps)
        for _ in range(epochs):
            rng.shuffle(idx)
            for s in range(0, rollout_steps, batch):
                b = torch.as_tensor(idx[s:s + batch], dtype=torch.int64).to(device)
                logits, v = net(so[b])
                dist = torch.distributions.Categorical(logits=logits)
                newlp = dist.log_prob(sa[b])
                ratio = (newlp - slp[b]).exp()
                unclipped = ratio * adv[b]
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
                pi_loss = -torch.min(unclipped, clipped).mean()
                v_loss = F.mse_loss(v, ret[b])
                ent = dist.entropy().mean()
                loss = pi_loss + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                opt.step()
        if upd % eval_interval == 0 or upd == total_updates:
            ev_steps.append(upd)
            ev_ret.append(_eval(env_id, lambda o: int(net(
                torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0))[0].argmax().item()),
                n_eval=n_eval))
    env.close()
    return {"algo": "ppo(mine)", "seed": seed, "ev_steps": np.array(ev_steps),
            "eval_returns": np.array(ev_ret)}
