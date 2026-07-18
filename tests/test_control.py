"""Ground-truth tests for the tabular RL dosing MDP.

Value iteration must be a Bellman fixed point and its greedy policy must be a
sensible (dose-tapering) rule; model-free Q-learning must recover that policy's
*value* on every state and be reproducible under its seed. All tests are
deterministic and offline (pure numpy).
"""

from __future__ import annotations

import numpy as np

from ddm4bio.methods.control import (
    PKDosingEnv,
    policy_value,
    q_learning,
    value_iteration,
)


def _env() -> PKDosingEnv:
    return PKDosingEnv()


def test_transition_kernel_is_a_valid_distribution():
    env = _env()
    p = env.transition_matrix
    assert p.shape == (env.n_states, env.n_actions, env.n_states)
    # Every (state, action) row is a probability distribution.
    assert np.all(p >= 0.0)
    assert np.allclose(p.sum(axis=2), 1.0)


def test_value_iteration_is_a_bellman_fixed_point():
    env = _env()
    v, policy = value_iteration(env.transition_matrix, env.expected_reward, env.gamma)
    # Re-applying the Bellman optimality operator must not move V.
    q = env.expected_reward + env.gamma * env.transition_matrix.dot(v)
    assert np.allclose(q.max(axis=1), v, atol=1e-8)
    # The greedy policy is the argmax of that same backup.
    assert np.array_equal(policy, q.argmax(axis=1))


def test_optimal_policy_tapers_with_concentration():
    # The optimal dose should be non-increasing in concentration: dose hard when
    # the compartment is empty, taper toward the window, hold when it is high.
    env = _env()
    _, policy = value_iteration(env.transition_matrix, env.expected_reward, env.gamma)
    optimal_doses = env.doses[policy]
    assert np.all(np.diff(optimal_doses) <= 0)
    assert optimal_doses[0] == env.doses.max()  # loading dose when empty
    assert optimal_doses[-1] == 0  # hold when already toxic


def test_policy_value_matches_value_iteration_for_the_optimal_policy():
    env = _env()
    v_star, policy = value_iteration(env.transition_matrix, env.expected_reward, env.gamma)
    v_pi = policy_value(policy, env.transition_matrix, env.expected_reward, env.gamma)
    assert np.allclose(v_pi, v_star, atol=1e-8)


def test_q_learning_recovers_the_optimal_value():
    env = _env()
    v_star, _ = value_iteration(env.transition_matrix, env.expected_reward, env.gamma)

    q = q_learning(env, seed=0)
    learned_policy = q.argmax(axis=1)
    v_learned = policy_value(learned_policy, env.transition_matrix, env.expected_reward, env.gamma)
    # The model-free learner reaches the model-based optimum on every state.
    assert np.all(v_learned >= v_star - 1e-6)


def test_q_learning_is_deterministic_under_seed():
    env = _env()
    a = q_learning(env, episodes=500, seed=0)
    b = q_learning(env, episodes=500, seed=0)
    assert np.array_equal(a, b)


def test_env_step_is_reproducible_and_in_bounds():
    env = _env()
    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)
    for _ in range(50):
        s = int(rng_a.integers(env.n_states))  # advance A's stream identically...
        s_b = int(rng_b.integers(env.n_states))
        assert s == s_b
        a = 2
        na, r = env.step(s, a, rng_a)
        nb, rb = env.step(s_b, a, rng_b)
        assert na == nb and r == rb
        assert 0 <= na < env.n_states
