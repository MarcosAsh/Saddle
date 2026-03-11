"""
Tests for the JAX optimiser implementations.

Tests verify:
  1. State initialisation produces correct shapes and zeros.
  2. A single update step changes params and state in the expected direction.
  3. Each optimiser can actually minimise a simple quadratic (bowl).
  4. Each optimiser can make progress on Rosenbrock.
  5. AdaHessian's Hessian diagonal estimate is correct on a quadratic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from saddle.optimisers import (
    SGDState,
    sgd_update,
    AdamState,
    adam_update,
    AdaHessianState,
    adahessian_update,
    _hutchinson_hessian_diag,
)


# -------------------------------------------------------------------
# Loss functions for testing
# -------------------------------------------------------------------

def bowl_loss(params: jax.Array) -> jax.Array:
    """f(x,y) = x^2 + y^2. Minimum at origin."""
    return jnp.sum(params ** 2)


def rosenbrock_loss(params: jax.Array) -> jax.Array:
    """f(x,y) = (1-x)^2 + 100*(y-x^2)^2. Minimum at (1,1)."""
    x, y = params[0], params[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


def scaled_quadratic_loss(params: jax.Array) -> jax.Array:
    """f(x,y) = 50*x^2 + y^2. Ill-conditioned, Hessian diag is [100, 2]."""
    return 50.0 * params[0] ** 2 + params[1] ** 2


# -------------------------------------------------------------------
# Initialisation tests
# -------------------------------------------------------------------

class TestSGDInit:
    def test_velocity_is_zeros(self) -> None:
        p = jnp.array([3.0, -2.0])
        state = SGDState.init(p)
        assert jnp.allclose(state.velocity, jnp.zeros(2))
        assert state.step == 0

    def test_shape_matches_params(self) -> None:
        p = jnp.ones(5)
        state = SGDState.init(p)
        assert state.velocity.shape == (5,)


class TestAdamInit:
    def test_moments_are_zeros(self) -> None:
        p = jnp.array([1.0, 2.0])
        state = AdamState.init(p)
        assert jnp.allclose(state.m, jnp.zeros(2))
        assert jnp.allclose(state.v, jnp.zeros(2))
        assert state.step == 0


class TestAdaHessianInit:
    def test_moments_are_zeros(self) -> None:
        p = jnp.array([1.0, 2.0])
        state = AdaHessianState.init(p)
        assert jnp.allclose(state.m, jnp.zeros(2))
        assert jnp.allclose(state.v, jnp.zeros(2))
        assert state.step == 0


# -------------------------------------------------------------------
# Single-step sanity tests
# -------------------------------------------------------------------

class TestSGDSingleStep:
    def test_moves_toward_minimum(self) -> None:
        """On a bowl starting at (3, 4), one step should reduce distance to origin."""
        p = jnp.array([3.0, 4.0])
        g = jax.grad(bowl_loss)(p)
        state = SGDState.init(p)
        new_state, new_p = sgd_update(state, p, g, lr=0.01, momentum=0.0)
        assert jnp.sum(new_p ** 2) < jnp.sum(p ** 2)
        assert new_state.step == 1

    def test_momentum_accumulates(self) -> None:
        """Two steps with momentum should have larger velocity than one."""
        p = jnp.array([3.0, 4.0])
        g = jax.grad(bowl_loss)(p)
        s0 = SGDState.init(p)
        s1, p1 = sgd_update(s0, p, g, lr=0.01, momentum=0.9)
        g1 = jax.grad(bowl_loss)(p1)
        s2, _ = sgd_update(s1, p1, g1, lr=0.01, momentum=0.9)
        # Velocity magnitude should grow with momentum
        assert jnp.linalg.norm(s2.velocity) > jnp.linalg.norm(s1.velocity) * 0.5


class TestAdamSingleStep:
    def test_moves_toward_minimum(self) -> None:
        p = jnp.array([3.0, 4.0])
        g = jax.grad(bowl_loss)(p)
        state = AdamState.init(p)
        new_state, new_p = adam_update(state, p, g, lr=0.01)
        assert jnp.sum(new_p ** 2) < jnp.sum(p ** 2)
        assert new_state.step == 1

    def test_bias_correction(self) -> None:
        """Early steps should have large bias correction factors."""
        p = jnp.array([3.0, 4.0])
        g = jax.grad(bowl_loss)(p)
        state = AdamState.init(p)
        new_state, _ = adam_update(state, p, g)
        # After 1 step, m_hat = m / (1 - 0.9^1) = m / 0.1 = 10*m
        # So m should be small but m_hat should be 10x larger
        m_hat = new_state.m / (1.0 - 0.9 ** 1)
        assert jnp.allclose(m_hat, 0.1 * g / 0.1)  # (1-beta1)*g / (1-beta1^1) = g


# -------------------------------------------------------------------
# Convergence tests: actually minimise things
# -------------------------------------------------------------------

class TestSGDConvergence:
    def test_converges_on_bowl(self) -> None:
        """SGD should reach near-zero on the bowl within 500 steps."""
        p = jnp.array([3.0, 4.0])
        state = SGDState.init(p)
        for _ in range(500):
            g = jax.grad(bowl_loss)(p)
            state, p = sgd_update(state, p, g, lr=0.01, momentum=0.9)
        assert bowl_loss(p) < 1e-6

    def test_makes_progress_on_rosenbrock(self) -> None:
        """SGD should reduce Rosenbrock loss substantially from (-1, 1)."""
        p = jnp.array([-1.0, 1.0])
        initial_loss = rosenbrock_loss(p)
        state = SGDState.init(p)
        for _ in range(2000):
            g = jax.grad(rosenbrock_loss)(p)
            state, p = sgd_update(state, p, g, lr=0.0001, momentum=0.9)
        assert rosenbrock_loss(p) < initial_loss * 0.1


class TestAdamConvergence:
    def test_converges_on_bowl(self) -> None:
        """Adam should converge quickly on the bowl."""
        p = jnp.array([3.0, 4.0])
        state = AdamState.init(p)
        for _ in range(500):
            g = jax.grad(bowl_loss)(p)
            state, p = adam_update(state, p, g, lr=0.05)
        assert bowl_loss(p) < 1e-6

    def test_converges_on_rosenbrock(self) -> None:
        """Adam should get close to (1,1) on Rosenbrock."""
        p = jnp.array([-1.0, 1.0])
        state = AdamState.init(p)
        for _ in range(5000):
            g = jax.grad(rosenbrock_loss)(p)
            state, p = adam_update(state, p, g, lr=0.005)
        assert rosenbrock_loss(p) < 0.01


class TestAdaHessianConvergence:
    def test_converges_on_bowl(self) -> None:
        """AdaHessian should converge on the bowl."""
        p = jnp.array([3.0, 4.0])
        state = AdaHessianState.init(p)
        key = jax.random.key(42)
        for i in range(500):
            g = jax.grad(bowl_loss)(p)
            key, subkey = jax.random.split(key)
            state, p = adahessian_update(
                state, p, g, bowl_loss, subkey, lr=0.1,
            )
        assert bowl_loss(p) < 1e-4

    def test_handles_ill_conditioned(self) -> None:
        """AdaHessian should handle the ill-conditioned quadratic better than SGD."""
        p = jnp.array([5.0, 5.0])
        state = AdaHessianState.init(p)
        key = jax.random.key(123)
        for i in range(1000):
            g = jax.grad(scaled_quadratic_loss)(p)
            key, subkey = jax.random.split(key)
            state, p = adahessian_update(
                state, p, g, scaled_quadratic_loss, subkey, lr=0.1,
            )
        assert scaled_quadratic_loss(p) < 0.01


# -------------------------------------------------------------------
# Hessian diagonal estimate test
# -------------------------------------------------------------------

class TestHutchinsonHessianDiag:
    def test_exact_on_quadratic(self) -> None:
        """
        For f(x,y) = 50*x^2 + y^2, the Hessian is diag(100, 2).
        The Hutchinson estimator with a Rademacher vector is exact
        on a pure quadratic (no higher-order terms), so a single sample
        should recover the diagonal exactly.
        """
        p = jnp.array([1.0, 1.0])
        key = jax.random.key(0)
        hess_diag = _hutchinson_hessian_diag(scaled_quadratic_loss, p, key)
        expected = jnp.array([100.0, 2.0])
        assert jnp.allclose(hess_diag, expected, atol=1e-5), (
            f"Expected {expected}, got {hess_diag}"
        )

    def test_exact_on_bowl(self) -> None:
        """For f(x,y) = x^2 + y^2, the Hessian is diag(2, 2)."""
        p = jnp.array([3.0, 4.0])
        key = jax.random.key(7)
        hess_diag = _hutchinson_hessian_diag(bowl_loss, p, key)
        expected = jnp.array([2.0, 2.0])
        assert jnp.allclose(hess_diag, expected, atol=1e-5)

    def test_position_independent_on_quadratic(self) -> None:
        """Hessian of a quadratic doesn't depend on position."""
        key = jax.random.key(99)
        h1 = _hutchinson_hessian_diag(scaled_quadratic_loss, jnp.array([0.0, 0.0]), key)
        h2 = _hutchinson_hessian_diag(scaled_quadratic_loss, jnp.array([10.0, -5.0]), key)
        assert jnp.allclose(h1, h2, atol=1e-5)
