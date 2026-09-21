"""Tabular DP / MC / TD solvers. Vectorized over states, numpy only."""

import numpy as np


def policy_evaluation(env, policy, gamma=1.0, theta=1e-6, max_iter=10000):
    V = np.zeros(env.nS)
    for _ in range(max_iter):
        delta = 0.0
        for s in range(env.nS):
            v = 0.0
            for a in range(env.nA):
                for p, ns, r, done in env.transitions(s, a):
                    v += policy[s, a] * p * (r + gamma * (0.0 if done else V[ns]))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_improvement(env, V, gamma=1.0):
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for p, ns, r, done in env.transitions(s, a):
                q[a] += p * (r + gamma * (0.0 if done else V[ns]))
        policy[s, np.argmax(q)] = 1.0
    return policy


def policy_iteration(env, gamma=1.0, theta=1e-6):
    policy = np.ones((env.nS, env.nA)) / env.nA
    iters = 0
    while True:
        iters += 1
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, V, iters


def value_iteration(env, gamma=1.0, theta=1e-6):
    V = np.zeros(env.nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for s in range(env.nS):
            q = np.zeros(env.nA)
            for a in range(env.nA):
                for p, ns, r, done in env.transitions(s, a):
                    q[a] += p * (r + gamma * (0.0 if done else V[ns]))
            best = np.max(q)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    return policy_improvement(env, V, gamma), V, sweeps


def mc_prediction(env, policy_fn, n_episodes, gamma=1.0, seed=0, first_visit=True, max_steps=500):
    rng = np.random.default_rng(seed)
    V = np.zeros(env.nS)
    returns_sum = np.zeros(env.nS)
    counts = np.zeros(env.nS)
    for _ in range(n_episodes):
        s = env.reset()
        traj, done = [], False
        while not done and len(traj) < max_steps:
            a = policy_fn(s, rng)
            ns, r, done = env.step(a)
            traj.append((s, r))
            s = ns
        G, seen = 0.0, set()
        for s_t, r_t in reversed(traj):
            G = r_t + gamma * G
            if first_visit and s_t in seen:
                continue
            seen.add(s_t)
            returns_sum[s_t] += G
            counts[s_t] += 1
    nz = counts > 0
    V[nz] = returns_sum[nz] / counts[nz]
    return V, counts


def td_zero_prediction(env, policy_fn, n_episodes, alpha=0.1, gamma=1.0, seed=0, max_steps=500):
    rng = np.random.default_rng(seed)
    V = np.zeros(env.nS)
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            a = policy_fn(s, rng)
            ns, r, done = env.step(a)
            V[s] += alpha * (r + gamma * (0.0 if done else V[ns]) - V[s])
            s = ns
    return V


def sarsa(env, n_episodes, alpha=0.5, gamma=1.0, eps=0.1, seed=0, max_steps=500):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    ep_returns, ep_steps = [], []
    from .mdp import epsilon_greedy
    for _ in range(n_episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, eps, rng)
        G, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            ns, r, done = env.step(a)
            na = epsilon_greedy(Q, ns, eps, rng)
            Q[s, a] += alpha * (r + gamma * (0.0 if done else Q[ns, na]) - Q[s, a])
            G += r
            s, a = ns, na
            steps += 1
        ep_returns.append(G)
        ep_steps.append(steps)
    return Q, np.array(ep_returns), np.array(ep_steps)


def q_learning(env, n_episodes, alpha=0.5, gamma=1.0, eps=0.1, seed=0, max_steps=500):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    ep_returns, ep_steps, falls = [], [], []
    from .mdp import epsilon_greedy
    for _ in range(n_episodes):
        s = env.reset()
        G, steps, fell, done = 0.0, 0, 0, False
        while not done and steps < max_steps:
            a = epsilon_greedy(Q, s, eps, rng)
            ns, r, done = env.step(a)
            if r == -100.0:
                fell += 1
            Q[s, a] += alpha * (r + gamma * (0.0 if done else np.max(Q[ns])) - Q[s, a])
            G += r
            s = ns
            steps += 1
        ep_returns.append(G)
        ep_steps.append(steps)
        falls.append(fell)
    return Q, np.array(ep_returns), np.array(ep_steps), np.array(falls)


def sarsa_falls(env, Q, n_eval, eps=0.0, seed=0):
    rng = np.random.default_rng(seed)
    from .mdp import epsilon_greedy
    total = 0
    for _ in range(n_eval):
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, s, eps, rng)
            ns, r, done = env.step(a)
            if r == -100.0:
                total += 1
            s = ns
    return total
