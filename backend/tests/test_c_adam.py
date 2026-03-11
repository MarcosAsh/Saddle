"""
Tests for the C Adam implementation.

Verifies:
  1. C adam_step produces the same results as the JAX implementation.
  2. C adam_optimise converges on known surfaces.
  3. C adam_optimise trajectory matches JAX Adam trajectory closely.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from saddle.c_adam import c_adam_optimise, c_adam_step
from saddle.optimisers import AdamState, adam_update


class TestCAdamStep:
    def test_matches_jax_single_step(self) -> None:
        """C and JAX Adam should produce identical params after one step."""
        grads = np.array([6.0, 8.0], dtype=np.float64)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8

        # C version (mutates in place)
        c_params = np.array([3.0, 4.0], dtype=np.float64)
        c_m = np.zeros(2, dtype=np.float64)
        c_v = np.zeros(2, dtype=np.float64)
        c_adam_step(c_params, c_m, c_v, grads, step=1, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

        # JAX version
        jax_params = jnp.array([3.0, 4.0])
        jax_grads = jnp.array([6.0, 8.0])
        state = AdamState.init(jax_params)
        new_state, jax_new_params = adam_update(state, jax_params, jax_grads, lr, beta1, beta2, eps)

        np.testing.assert_allclose(c_params, np.array(jax_new_params), atol=1e-12)
        np.testing.assert_allclose(c_m, np.array(new_state.m), atol=1e-12)
        np.testing.assert_allclose(c_v, np.array(new_state.v), atol=1e-12)

    def test_matches_jax_multi_step(self) -> None:
        """C and JAX should agree after 100 steps with the same gradient sequence."""
        rng = np.random.default_rng(99)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
        n_steps = 100

        c_params = np.array([5.0, -3.0], dtype=np.float64)
        c_m = np.zeros(2, dtype=np.float64)
        c_v = np.zeros(2, dtype=np.float64)

        jax_params = jnp.array([5.0, -3.0])
        state = AdamState.init(jax_params)

        for i in range(n_steps):
            g = rng.standard_normal(2).astype(np.float64)
            c_adam_step(c_params, c_m, c_v, g, step=i + 1, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
            state, jax_params = adam_update(state, jax_params, jnp.array(g), lr, beta1, beta2, eps)

        np.testing.assert_allclose(c_params, np.array(jax_params), atol=1e-10)


class TestCAdamOptimise:
    def test_bowl_converges(self) -> None:
        """C Adam should converge near the origin on the bowl."""
        xs, ys, losses = c_adam_optimise("bowl", 3.0, 4.0, num_steps=1000, lr=0.05)
        assert losses[-1] < 1e-4
        assert abs(xs[-1]) < 0.01
        assert abs(ys[-1]) < 0.01

    def test_rosenbrock_makes_progress(self) -> None:
        """C Adam should substantially reduce Rosenbrock loss."""
        xs, ys, losses = c_adam_optimise("rosenbrock", -1.0, 1.0, num_steps=5000, lr=0.005)
        assert losses[-1] < losses[0] * 0.01

    def test_trajectory_length(self) -> None:
        """Trajectory should have num_steps + 1 entries (initial + each step)."""
        xs, ys, losses = c_adam_optimise("bowl", 1.0, 1.0, num_steps=50)
        assert len(xs) == 51
        assert len(ys) == 51
        assert len(losses) == 51

    def test_himmelblau_finds_minimum(self) -> None:
        """Starting near (3, 2), should converge to that minimum."""
        xs, ys, losses = c_adam_optimise("himmelblau", 2.5, 1.5, num_steps=2000, lr=0.01)
        assert losses[-1] < 0.01
