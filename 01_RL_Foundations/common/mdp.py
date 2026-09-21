"""Tabular GridWorld / CliffWalking environments, numpy only, no gymnasium.

GridWorld 4x4 (Sutton & Barto Example 4.1):
  states 0..15, row-major. Terminal: s=0 and s=15.
  actions: 0=up, 1=right, 2=down, 3=left. Off-grid moves stay.
  reward -1 per step (including into terminal).

CliffWalking 4x12 (Sutton & Barto Example 6.6):
  start=(3,0), goal=(3,11). Cliff cells (3,1..10): reward -100, back to start.
  normal step reward -1, goal reward -1 and episode ends.
  actions same 4-dir layout, bump into wall stays.
"""

import numpy as np


ACTIONS = ["up", "right", "down", "left"]
DR = np.array([-1, 0, 1, 0], dtype=int)
DC = np.array([0, 1, 0, -1], dtype=int)


class GridWorld4x4:
    nS, nA = 16, 4
    terminals = (0, 15)

    def reset(self, s=None):
        self.s = 5 if s is None else int(s)
        return self.s

    def step(self, a):
        assert 0 <= a < 4
        if self.s in self.terminals:
            return self.s, 0.0, True
        r, c = divmod(self.s, 4)
        nr = min(3, max(0, r + DR[a]))
        nc = min(3, max(0, c + DC[a]))
        self.s = nr * 4 + nc
        done = self.s in self.terminals
        return self.s, -1.0, done

    def transitions(self, s, a):
        if s in self.terminals:
            return [(1.0, s, 0.0, True)]
        r, c = divmod(s, 4)
        nr = min(3, max(0, r + DR[a]))
        nc = min(3, max(0, c + DC[a]))
        ns = nr * 4 + nc
        return [(1.0, ns, -1.0, ns in self.terminals)]


class CliffWalking:
    nrow, ncol = 4, 12
    nS, nA = 48, 4
    start = (3, 0)
    goal = (3, 11)

    def reset(self):
        self.pos = self.start
        return self._idx(self.pos)

    @staticmethod
    def _idx(pos):
        return pos[0] * 12 + pos[1]

    def step(self, a):
        assert 0 <= a < 4
        r, c = self.pos
        if (r, c) == self.goal:
            return self._idx(self.pos), 0.0, True
        nr = min(3, max(0, r + DR[a]))
        nc = min(11, max(0, c + DC[a]))
        if nr == 3 and 1 <= nc <= 10:
            self.pos = self.start
            return self._idx(self.pos), -100.0, False
        self.pos = (nr, nc)
        done = self.pos == self.goal
        return self._idx(self.pos), -1.0, done

    def transitions(self, s, a):
        r, c = divmod(s, 12)
        if (r, c) == self.goal:
            return [(1.0, s, 0.0, True)]
        nr = min(3, max(0, r + DR[a]))
        nc = min(11, max(0, c + DC[a]))
        if nr == 3 and 1 <= nc <= 10:
            ns = self._idx(self.start)
            return [(1.0, ns, -100.0, False)]
        ns = nr * 12 + nc
        done = (nr, nc) == self.goal
        return [(1.0, ns, -1.0, done)]


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[s]))


def greedy_policy_from_Q(Q):
    return np.argmax(Q, axis=1)


def run_episode(env, policy_fn, max_steps=1000):
    s = env.reset()
    traj, done, steps = [], False, 0
    while not done and steps < max_steps:
        a = policy_fn(s)
        ns, r, done = env.step(a)
        traj.append((s, a, r, ns, done))
        s = ns
        steps += 1
    return traj
