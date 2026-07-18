"""Tabular reinforcement learning on a pharmacokinetic dosing MDP.

A small, offline, deterministic control problem for Week-8 teaching: dose a drug
to hold its blood concentration inside a therapeutic window under
patient-to-patient clearance variability. The environment is a discretized
one-compartment PK model with *stochastic* elimination, so the optimal policy
must hedge against the risk of overshooting into toxicity.

Two solvers sit on top of it, and their contrast is the lesson:

* :func:`value_iteration` solves the MDP exactly from its transition kernel --
  the *model-based* ground-truth optimal policy.
* :func:`q_learning` recovers that same policy from simulated dosing episodes
  alone, never touching the kernel -- the *model-free* learner rediscovering the
  model-based optimum.

Everything is pure numpy: no gym, no deep networks, no network access. All
randomness flows through an explicit ``numpy.random.Generator`` so every run is
reproducible from its seed.
"""

from __future__ import annotations

import numpy as np


class PKDosingEnv:
    """Discretized one-compartment PK dosing MDP with stochastic clearance.

    The state is the drug concentration on a uniform grid ``[0, c_max]`` with
    step ``dz``. An action is a discrete dose level ``d``; taking it adds
    ``d * dose_unit`` to the concentration. Between doses the drug is eliminated
    by a first-order decay factor drawn each step from ``decays`` with
    probabilities ``decay_probs`` -- this is the patient-to-patient (and
    occasion-to-occasion) clearance variability that makes the problem
    stochastic:

    ``c_next = clip(decay * c + d * dose_unit, 0, c_max)``   (then snapped to grid)

    The reward keeps the concentration in the therapeutic window
    ``[c_low, c_high]``: ``+1`` inside it, a mild penalty below it
    (subtherapeutic), and a heavier penalty above it (toxic), minus a small
    per-unit dose (exposure) cost.

    Parameters
    ----------
    c_max : float, default 2.0
        Maximum representable concentration (top of the state grid).
    dz : float, default 0.1
        Concentration grid spacing.
    doses : sequence of int, default (0, 1, 2, 3)
        Available dose levels; dose ``d`` adds ``d * dose_unit``.
    dose_unit : float, default 0.3
        Concentration added per unit dose.
    decays : sequence of float, default (0.55, 0.70, 0.85)
        Candidate per-step elimination factors (inter-patient variability).
    decay_probs : sequence of float, default (0.25, 0.50, 0.25)
        Probabilities of each decay factor; must sum to 1.
    c_low, c_high : float, default 0.9, 1.1
        Therapeutic window bounds.
    toxic_penalty : float, default 3.0
        Penalty slope above the window (toxicity, punished hard).
    sub_penalty : float, default 1.0
        Penalty slope below the window (subtherapeutic).
    dose_cost : float, default 0.05
        Penalty per unit dose (drug exposure cost).
    gamma : float, default 0.95
        Discount factor for the sequential-decision objective.

    Attributes
    ----------
    states : np.ndarray, shape (n_states,)
        The concentration grid.
    doses : np.ndarray, shape (n_actions,)
        The dose levels.
    transition_matrix : np.ndarray, shape (n_states, n_actions, n_states)
        ``P[s, a, s']`` = probability of landing in state ``s'`` after taking
        action ``a`` in state ``s``. Rows ``P[s, a]`` sum to 1.
    expected_reward : np.ndarray, shape (n_states, n_actions)
        ``R[s, a]`` = expected immediate reward for action ``a`` in state ``s``.
    """

    def __init__(
        self,
        c_max: float = 2.0,
        dz: float = 0.1,
        doses: object = (0, 1, 2, 3),
        dose_unit: float = 0.3,
        decays: object = (0.55, 0.70, 0.85),
        decay_probs: object = (0.25, 0.50, 0.25),
        c_low: float = 0.9,
        c_high: float = 1.1,
        toxic_penalty: float = 3.0,
        sub_penalty: float = 1.0,
        dose_cost: float = 0.05,
        gamma: float = 0.95,
    ) -> None:
        self.states = np.round(np.arange(0.0, c_max + dz / 2, dz), 6)
        self.n_states = int(self.states.size)
        self.doses = np.asarray(doses, dtype=int)
        self.n_actions = int(self.doses.size)
        self.c_max = float(c_max)
        self.dose_unit = float(dose_unit)
        self.decays = np.asarray(decays, dtype=float)
        self.decay_probs = np.asarray(decay_probs, dtype=float)
        if not np.isclose(self.decay_probs.sum(), 1.0):
            raise ValueError("decay_probs must sum to 1.")
        if self.decays.shape != self.decay_probs.shape:
            raise ValueError("decays and decay_probs must have the same length.")
        self.c_low = float(c_low)
        self.c_high = float(c_high)
        self.toxic_penalty = float(toxic_penalty)
        self.sub_penalty = float(sub_penalty)
        self.dose_cost = float(dose_cost)
        self.gamma = float(gamma)

        self.transition_matrix, self.expected_reward = self._build_kernel()

    def _reward(self, concentration: float, dose_level: int) -> float:
        """Immediate reward for landing at ``concentration`` after ``dose_level``."""
        if self.c_low <= concentration <= self.c_high:
            base = 1.0
        elif concentration < self.c_low:
            base = -self.sub_penalty * (self.c_low - concentration) / self.c_low
        else:
            base = -self.toxic_penalty * (concentration - self.c_high) / self.c_high
        return base - self.dose_cost * float(dose_level)

    def _next_index(self, state: int, action: int, decay: float) -> int:
        """Deterministic next-state index given a realized ``decay`` draw."""
        c_next = np.clip(
            decay * self.states[state] + self.doses[action] * self.dose_unit,
            0.0,
            self.c_max,
        )
        return int(np.argmin(np.abs(self.states - c_next)))

    def _build_kernel(self) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the (S, A, S) transition kernel and (S, A) expected reward."""
        n_s, n_a = self.n_states, self.n_actions
        transition = np.zeros((n_s, n_a, n_s), dtype=float)
        reward = np.zeros((n_s, n_a), dtype=float)
        for s in range(n_s):
            for a in range(n_a):
                for decay, prob in zip(self.decays, self.decay_probs):
                    s_next = self._next_index(s, a, decay)
                    transition[s, a, s_next] += prob
                    reward[s, a] += prob * self._reward(self.states[s_next], int(self.doses[a]))
        return transition, reward

    def step(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, float]:
        """Sample one transition (model-free view used by :func:`q_learning`).

        Parameters
        ----------
        state : int
            Current state index.
        action : int
            Chosen action index.
        rng : np.random.Generator
            Source of the stochastic clearance draw.

        Returns
        -------
        tuple[int, float]
            The next state index and the immediate reward.
        """
        decay = float(self.decays[rng.choice(self.decays.size, p=self.decay_probs)])
        s_next = self._next_index(state, action, decay)
        return s_next, self._reward(self.states[s_next], int(self.doses[action]))

    def rollout(
        self,
        policy: np.ndarray,
        start_state: int,
        rng: np.random.Generator,
        horizon: int,
    ) -> np.ndarray:
        """Simulate a concentration trajectory under a fixed ``policy``.

        Parameters
        ----------
        policy : np.ndarray, shape (n_states,)
            Action index to take in each state.
        start_state : int
            Initial state index.
        rng : np.random.Generator
            Source of the stochastic clearance draws.
        horizon : int
            Number of dosing steps to simulate.

        Returns
        -------
        np.ndarray, shape (horizon + 1,)
            The concentration at each step, including the start.
        """
        s = int(start_state)
        path = [float(self.states[s])]
        for _ in range(int(horizon)):
            s, _ = self.step(s, int(policy[s]), rng)
            path.append(float(self.states[s]))
        return np.asarray(path, dtype=float)


def value_iteration(
    transition: np.ndarray,
    reward: np.ndarray,
    gamma: float,
    tol: float = 1e-10,
    max_iter: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact optimal values and greedy policy for a finite MDP.

    Iterates the Bellman optimality update ``V(s) = max_a [ R(s,a) +
    gamma * sum_s' P(s,a,s') V(s') ]`` to convergence, then reads off the greedy
    policy. This is the *model-based* ground truth the learner is checked against.

    Parameters
    ----------
    transition : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities ``P[s, a, s']``.
    reward : np.ndarray, shape (n_states, n_actions)
        Expected immediate reward ``R[s, a]``.
    gamma : float
        Discount factor in ``[0, 1)``.
    tol : float, default 1e-10
        Convergence tolerance on the max value change per sweep.
    max_iter : int, default 20000
        Maximum number of sweeps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(V, policy)`` -- the optimal value ``V`` of shape ``(n_states,)`` and
        the greedy action index ``policy`` of shape ``(n_states,)``.
    """
    n_states = reward.shape[0]
    values = np.zeros(n_states, dtype=float)
    for _ in range(max_iter):
        q = reward + gamma * transition.dot(values)
        updated = q.max(axis=1)
        if np.max(np.abs(updated - values)) < tol:
            values = updated
            break
        values = updated
    policy = (reward + gamma * transition.dot(values)).argmax(axis=1)
    return values, policy


def policy_value(
    policy: np.ndarray,
    transition: np.ndarray,
    reward: np.ndarray,
    gamma: float,
    tol: float = 1e-10,
    max_iter: int = 20000,
) -> np.ndarray:
    """Value of a fixed ``policy`` under a finite MDP (Bellman expectation).

    Solves ``V(s) = R(s, pi(s)) + gamma * sum_s' P(s, pi(s), s') V(s')`` by
    iteration. Used to score a learned policy against the optimal value.

    Parameters
    ----------
    policy : np.ndarray, shape (n_states,)
        Action index taken in each state.
    transition : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities ``P[s, a, s']``.
    reward : np.ndarray, shape (n_states, n_actions)
        Expected immediate reward ``R[s, a]``.
    gamma : float
        Discount factor in ``[0, 1)``.
    tol : float, default 1e-10
        Convergence tolerance on the max value change per sweep.
    max_iter : int, default 20000
        Maximum number of sweeps.

    Returns
    -------
    np.ndarray, shape (n_states,)
        The value of following ``policy`` from each state.
    """
    idx = np.arange(reward.shape[0])
    trans_pi = transition[idx, policy]  # (n_states, n_states)
    reward_pi = reward[idx, policy]  # (n_states,)
    values = np.zeros(reward.shape[0], dtype=float)
    for _ in range(max_iter):
        updated = reward_pi + gamma * trans_pi.dot(values)
        if np.max(np.abs(updated - values)) < tol:
            return updated
        values = updated
    return values


def q_learning(
    env: PKDosingEnv,
    episodes: int = 3000,
    horizon: int = 25,
    gamma: float | None = None,
    alpha0: float = 0.5,
    alpha_decay: float = 0.02,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    explore_frac: float = 0.7,
    seed: int | None = 0,
) -> np.ndarray:
    """Tabular Q-learning from simulated episodes (model-free).

    Learns an action-value table by interacting with ``env`` through its
    :meth:`~PKDosingEnv.step` sampler only -- it never sees the transition
    kernel. Exploration anneals linearly from ``eps_start`` to ``eps_end`` over
    the first ``explore_frac`` of training, and the per-state-action step size
    decays as ``alpha0 / (1 + alpha_decay * visits)`` so the estimates settle.

    Parameters
    ----------
    env : PKDosingEnv
        The environment to learn in.
    episodes : int, default 3000
        Number of training episodes.
    horizon : int, default 25
        Steps per episode.
    gamma : float, optional
        Discount factor; defaults to ``env.gamma`` when None.
    alpha0 : float, default 0.5
        Initial learning rate.
    alpha_decay : float, default 0.02
        Learning-rate decay per visit to a state-action pair.
    eps_start, eps_end : float, default 1.0, 0.05
        Start and floor of the epsilon-greedy exploration rate.
    explore_frac : float, default 0.7
        Fraction of episodes over which epsilon anneals to ``eps_end``.
    seed : int, optional
        Seed for the episode/exploration/transition RNG.

    Returns
    -------
    np.ndarray, shape (n_states, n_actions)
        The learned Q-table; ``Q.argmax(axis=1)`` is the greedy policy.
    """
    discount = env.gamma if gamma is None else float(gamma)
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.n_states, env.n_actions
    q_table = np.zeros((n_states, n_actions), dtype=float)
    visits = np.zeros((n_states, n_actions), dtype=float)
    anneal = max(1.0, explore_frac * episodes)

    for episode in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (episode / anneal))
        state = int(rng.integers(n_states))
        for _ in range(horizon):
            if rng.random() < eps:
                action = int(rng.integers(n_actions))
            else:
                action = int(q_table[state].argmax())
            visits[state, action] += 1.0
            alpha = alpha0 / (1.0 + alpha_decay * visits[state, action])
            next_state, reward = env.step(state, action, rng)
            target = reward + discount * q_table[next_state].max()
            q_table[state, action] += alpha * (target - q_table[state, action])
            state = next_state
    return q_table
