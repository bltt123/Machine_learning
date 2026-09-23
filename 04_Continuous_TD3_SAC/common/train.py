"""Continuous-control trainers: DDPG, TD3, SAC. CPU-first, torch.

All three share: ReplayBuffer, twin critics (DDPG uses q1 only), soft target update (tau),
greedy eval every eval_interval episodes. Actions env-normalized in [-1,1] via tanh nets.

Ablation flags for TD3:
  twin=True, delay=True, target_noise=True  -> full TD3
  turn off one -> ablation (DDPG-like / no delayed-update / no smoothing)
SAC: auto temperature (target entropy) on by default; fixed_alpha to disable.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import ReplayBuffer
from .nets import DeterministicActor, SquashedGaussianActor, TwinQCritic


def _scale_action(act_space):
    lo, hi = act_space.low, act_space.high
    return lambda a: lo + (a + 1.0) * 0.5 * (hi - lo)


def _copy(source, target):
    target.load_state_dict(source.state_dict())


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


@torch.no_grad()
def _eval(actor, env_id, n_eval=10, max_steps=1000, device="cpu", base_seed=999, to_env=None):
    import gymnasium as gym

    actor.eval()
    env = gym.make(env_id)
    rets = []
    for i in range(n_eval):
        obs, _ = env.reset(seed=base_seed + i)
        G, done, t = 0.0, False, 0
        while not done and t < max_steps:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            a_raw = actor.greedy(o) if hasattr(actor, "greedy") else actor(o)
            a = to_env(a_raw.cpu().numpy()[0]) if to_env else a_raw.cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            G += r
            t += 1
        rets.append(G)
    env.close()
    return float(np.mean(rets))


def train_offpolicy(
    env_id="Pendulum-v1",
    algo="td3",
    seed=0,
    steps=20000,
    hidden=256,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    buffer_size=100000,
    warmup=500,
    exp_noise=0.15,
    pol_lr=1e-3,
    td3_twin=True,
    td3_delay=True,
    td3_smooth=True,
    target_noise=0.2,
    target_noise_clip=0.5,
    sac_auto_temp=True,
    sac_alpha=0.2,
    sac_target_entropy=None,
    n_eval=10,
    eval_every=2000,
    max_steps=500,
    device="cpu",
):
    import gymnasium as gym

    if algo == "ddpg":
        td3_twin = False
        td3_delay = False
        td3_smooth = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    to_env = _scale_action(env.action_space)

    if algo == "sac":
        actor = SquashedGaussianActor(obs_dim, act_dim, hidden)
        target_actor = None
    else:  # ddpg / td3
        actor = DeterministicActor(obs_dim, act_dim, hidden)
        target_actor = DeterministicActor(obs_dim, act_dim, hidden)
        _copy(actor, target_actor)
    critic = TwinQCritic(obs_dim, act_dim, hidden)
    target_critic = TwinQCritic(obs_dim, act_dim, hidden)
    _copy(critic, target_critic)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=pol_lr if algo != "sac" else lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, obs_dim, act_dim, seed=seed)

    log_alpha = torch.tensor(np.log(sac_alpha), requires_grad=True, dtype=torch.float32)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)
    if sac_target_entropy is None:
        sac_target_entropy = -act_dim

    ep_returns, ep_lengths, critic_loss, q_means, entropies, alphas = [], [], [], [], [], []
    eval_steps, eval_returns = [], []
    obs, _ = env.reset(seed=seed * 100000)
    G, L, done = 0.0, 0, False
    next_eval = eval_every

    for t in range(steps):
        if algo == "sac":
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                a_raw, _ = actor.sample_action(o)
            a = a_raw.cpu().numpy()[0]
        else:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            actor.eval()
            with torch.no_grad():
                mu = actor(o).cpu().numpy()[0]
            if len(buf) < warmup:
                a = rng.uniform(-1, 1, size=act_dim)
            else:
                a = mu + exp_noise * rng.standard_normal(size=act_dim)
        env_a = to_env(a)
        nobs, r, term, trunc, _ = env.step(env_a)
        d = float(term or trunc)
        buf.add(obs, a, r, nobs, d)
        obs = nobs
        G += r
        L += 1
        if term or trunc:
            ep_returns.append(G)
            ep_lengths.append(L)
            obs, _ = env.reset(seed=seed * 100000 + len(ep_returns))
            G, L = 0.0, 0

        if len(buf) >= max(batch_size, warmup):
            b_o, b_a, b_r, b_no, b_d = buf.sample(batch_size, device)
            with torch.no_grad():
                if algo == "sac":
                    na, _ = actor.sample_action(b_no)
                    qn = target_critic.q_min(b_no, na)
                    target_q = b_r + gamma * (1.0 - b_d) * qn
                else:
                    if td3_smooth:
                        noise = torch.randn_like(b_a).clamp(-target_noise_clip, target_noise_clip) * target_noise
                        na = (target_actor(b_no) + noise).clamp(-1.0, 1.0)
                    else:
                        na = target_actor(b_no)
                    q1n, q2n = target_critic(b_no, na, both=True)
                    qn = torch.min(q1n, q2n) if td3_twin else q1n
                    target_q = b_r + gamma * (1.0 - b_d) * qn

            q1, q2 = critic(b_o, b_a, both=True) if td3_twin else (critic(b_o, b_a), None)
            critic_loss_val = F.mse_loss(q1, target_q)
            if q2 is not None:
                critic_loss_val = 0.5 * (critic_loss_val + F.mse_loss(q2, target_q))
            opt_critic.zero_grad()
            critic_loss_val.backward()
            opt_critic.step()

            # TD3 delays the actor update; SAC updates every step (no delay concept).
            delay_ok = (algo == "sac") or (not td3_delay) or (t % 2 == 0)
            if delay_ok:
                if algo == "sac":
                    na_new, nlp_new = actor.sample_action(b_o)
                    q = critic(b_o, na_new)
                    alpha = torch.exp(log_alpha.detach())
                    pi_loss = (alpha * nlp_new.squeeze(-1) - q).mean()
                    opt_actor.zero_grad()
                    pi_loss.backward()
                    opt_actor.step()
                    if sac_auto_temp:
                        loss_alpha = -(log_alpha * (nlp_new.detach().squeeze(-1) + sac_target_entropy)).mean()
                        opt_alpha.zero_grad()
                        loss_alpha.backward()
                        opt_alpha.step()
                else:
                    mu = actor(b_o)
                    q_actor = critic(b_o, mu)
                    pi_loss = -q_actor.mean()
                    opt_actor.zero_grad()
                    pi_loss.backward()
                    opt_actor.step()
                if algo != "sac":
                    _soft_update(target_actor, actor, tau)
            _soft_update(target_critic, critic, tau)

            qm = float(q1.detach().mean().item())
            q_means.append(qm)
            critic_loss.append(float(critic_loss_val.item()))
            if algo == "sac":
                entropies.append(float(-nlp_new.float().mean().item()))
                alphas.append(float(torch.exp(log_alpha.detach()).item()))

        if t >= next_eval or t == steps - 1:
            ev = _eval(actor, env_id, n_eval, max_steps, device, to_env=to_env)
            eval_steps.append(t)
            eval_returns.append(ev)
            next_eval += eval_every
            print(f"[{algo}/seed{seed}] t={t} evals={len(eval_returns)} "
                  f"last_evals={[round(float(x), 1) for x in eval_returns[-5:]]}", flush=True)

    env.close()
    return {
        "ep_returns": np.array(ep_returns),
        "ep_lengths": np.array(ep_lengths),
        "critic_loss": np.array(critic_loss),
        "q_means": np.array(q_means),
        "entropies": np.array(entropies),
        "alphas": np.array(alphas),
        "eval_steps": np.array(eval_steps),
        "eval_returns": np.array(eval_returns),
        "algo": algo,
        "seed": seed,
    }