"""DQN trainer shared by 02 stations. CPU-first, Huber loss, hard target sync.

Variants via flags:
  double=True  -> Double DQN target (online net selects, target net evaluates)
  dueling=True -> DuelingQNet head instead of plain MLP
  no_target=True -> ablation: update target every gradient step (unstable demo)

Returns dict with: ep_returns, ep_lengths, losses, q_means, eval_returns.
eval: greedy rollouts every eval_interval episodes, n_eval each, own env instance.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import ReplayBuffer
from .nets import DuelingQNet, QNet


def train_dqn(
    env_id="CartPole-v1",
    variant="dqn",
    seed=0,
    total_episodes=300,
    hidden=128,
    lr=1e-3,
    gamma=0.99,
    buffer_size=20000,
    batch_size=64,
    learning_starts=1000,
    target_sync=500,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=250,
    max_steps=500,
    n_eval=10,
    eval_interval=25,
    no_target=False,
    log_q=True,
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

    Net = DuelingQNet if variant == "dueling" else QNet
    online = Net(obs_dim, act_dim, hidden).to(device)
    target = Net(obs_dim, act_dim, hidden).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, obs_dim, seed=seed)

    def epsilon(ep):
        t = min(1.0, ep / max(1, eps_decay_episodes))
        return eps_start + t * (eps_end - eps_start)

    ep_returns, ep_lengths, losses, q_means = [], [], [], []
    eval_episodes, eval_returns = [], []
    steps = 0

    for ep in range(total_episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        done = False
        G, L = 0.0, 0
        ep_loss, ep_q, n_upd = 0.0, 0.0, 0
        while not done and L < max_steps:
            eps = epsilon(ep)
            if rng.random() < eps:
                a = int(env.action_space.sample())
            else:
                online.eval()
                with torch.no_grad():
                    o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    a = int(torch.argmax(online(o), dim=1).item())
            nobs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            buf.add(obs, a, r, nobs, float(done))
            obs = nobs
            G += r
            L += 1
            steps += 1

            if len(buf) >= max(batch_size, learning_starts):
                online.train()
                b_o, b_a, b_r, b_no, b_d = buf.sample(batch_size, device)
                q = online(b_o).gather(1, b_a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if variant == "double":
                        na = online(b_no).argmax(dim=1, keepdim=True)
                        qn = target(b_no).gather(1, na).squeeze(1)
                    else:
                        qn = target(b_no).max(dim=1).values
                    y = b_r + gamma * (1.0 - b_d) * qn
                loss = F.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                if no_target:
                    target.load_state_dict(online.state_dict())
                elif steps % target_sync == 0:
                    target.load_state_dict(online.state_dict())
                ep_loss += float(loss.item())
                if log_q:
                    ep_q += float(q.detach().mean().item())
                n_upd += 1

        ep_returns.append(G)
        ep_lengths.append(L)
        losses.append(ep_loss / max(1, n_upd))
        q_means.append(ep_q / max(1, n_upd))

        if (ep + 1) % eval_interval == 0 or ep == total_episodes - 1:
            ev = evaluate_greedy(online, env_id, n_eval=n_eval, max_steps=max_steps, device=device)
            eval_episodes.append(ep + 1)
            eval_returns.append(ev)

    env.close()
    return {
        "ep_returns": np.array(ep_returns),
        "ep_lengths": np.array(ep_lengths),
        "losses": np.array(losses),
        "q_means": np.array(q_means),
        "eval_episodes": np.array(eval_episodes),
        "eval_returns": np.array(eval_returns),
        "variant": variant,
        "seed": seed,
    }


@torch.no_grad()
def evaluate_greedy(qnet, env_id, n_eval=10, max_steps=500, device="cpu", base_seed=999):
    import gymnasium as gym

    qnet.eval()
    env = gym.make(env_id)
    rets = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=base_seed + i)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            a = int(torch.argmax(qnet(o), dim=1).item())
            obs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            G += r
            t += 1
        rets.append(G)
    env.close()
    return float(np.mean(rets))
