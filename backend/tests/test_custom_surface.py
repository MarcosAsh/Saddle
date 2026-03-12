"""Tests for the custom surface expression evaluator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from saddle.custom_surface import eval_custom_grid_np, make_custom_jax_fn


class TestValidation:
    def test_rejects_import(self) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            eval_custom_grid_np("__import__('os')", (-1, 1, -1, 1), 5)

    def test_rejects_attribute_access(self) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            eval_custom_grid_np("x.__class__", (-1, 1, -1, 1), 5)

    def test_rejects_unknown_function(self) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            eval_custom_grid_np("eval(x)", (-1, 1, -1, 1), 5)

    def test_rejects_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            eval_custom_grid_np("z + x", (-1, 1, -1, 1), 5)


class TestEvalGrid:
    def test_simple_bowl(self) -> None:
        grid = eval_custom_grid_np("x**2 + y**2", (-1, 1, -1, 1), 10)
        assert grid.shape == (10, 10)
        # Center should be near zero
        assert grid[5, 5] < 0.1

    def test_trig(self) -> None:
        grid = eval_custom_grid_np("sin(x) * cos(y)", (-1, 1, -1, 1), 10)
        assert grid.shape == (10, 10)
        assert np.all(np.isfinite(grid))

    def test_pi_and_e(self) -> None:
        grid = eval_custom_grid_np("pi * x + e * y", (0, 1, 0, 1), 5)
        assert grid.shape == (5, 5)
        # At (0, 0) should be 0
        assert grid[0, 0] == pytest.approx(0.0, abs=0.01)


class TestJaxFn:
    def test_bowl_gradient(self) -> None:
        import jax
        import jax.numpy as jnp

        fn = make_custom_jax_fn("x**2 + y**2")
        params = jnp.array([3.0, 4.0])
        assert float(fn(params)) == pytest.approx(25.0)

        grad = jax.grad(fn)(params)
        assert float(grad[0]) == pytest.approx(6.0)
        assert float(grad[1]) == pytest.approx(8.0)

    def test_trig_is_differentiable(self) -> None:
        import jax
        import jax.numpy as jnp

        fn = make_custom_jax_fn("sin(x) * cos(y)")
        params = jnp.array([0.0, 0.0])
        grad = jax.grad(fn)(params)
        # d/dx sin(x)*cos(y) at (0,0) = cos(0)*cos(0) = 1
        assert float(grad[0]) == pytest.approx(1.0)
        # d/dy sin(x)*cos(y) at (0,0) = sin(0)*(-sin(0)) = 0
        assert float(grad[1]) == pytest.approx(0.0)
