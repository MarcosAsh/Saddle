"""
Benchmark: C Adam update step vs JAX JIT-compiled Adam update step.

Measures wall-clock time for repeated update steps, excluding gradient
computation to isolate the optimiser arithmetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from saddle.c_adam import c_adam_step
from saddle.optimisers import AdamState, adam_update


@dataclass(frozen=True)
class BenchmarkResult:
    c_total_ms: float
    c_per_step_us: float
    jax_total_ms: float
    jax_per_step_us: float
    speedup: float
    num_steps: int
    param_dim: int


def _jit_adam_step(
    m: Array, v: Array, params: Array, grads: Array, step: Array,
    lr: float, beta1: float, beta2: float, eps: float,
) -> tuple[Array, Array, Array]:
    """Wraps adam_update for JIT, returning just the arrays we need.

    step is passed as a scalar JAX array (not a Python int) so that
    it's traced rather than treated as a compile-time constant. Otherwise
    JAX recompiles for every unique step value, which is catastrophic
    for benchmarking.
    """
    state = AdamState(m=m, v=v, step=step)
    new_state, new_params = adam_update(state, params, grads, lr, beta1, beta2, eps)
    return new_state.m, new_state.v, new_params


def benchmark_adam(
    num_steps: int = 10000,
    param_dim: int = 2,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    warmup_steps: int = 100,
) -> BenchmarkResult:
    """
    Benchmark C vs JAX JIT Adam update step.

    Both implementations receive the same pre-computed random gradients
    so we're measuring pure update arithmetic, not gradient computation.
    """
    rng = np.random.default_rng(42)
    all_grads = rng.standard_normal((num_steps, param_dim))

    # ----- C benchmark -----
    c_params = np.zeros(param_dim, dtype=np.float64)
    c_m = np.zeros(param_dim, dtype=np.float64)
    c_v = np.zeros(param_dim, dtype=np.float64)

    # Warmup
    for i in range(warmup_steps):
        c_adam_step(c_params, c_m, c_v, all_grads[i % num_steps], i + 1, lr, beta1, beta2, eps)
    c_params[:] = 0.0
    c_m[:] = 0.0
    c_v[:] = 0.0

    t0 = time.perf_counter()
    for i in range(num_steps):
        c_adam_step(c_params, c_m, c_v, all_grads[i], i + 1, lr, beta1, beta2, eps)
    c_elapsed = time.perf_counter() - t0

    # ----- JAX benchmark -----
    jit_step = jax.jit(_jit_adam_step)

    jax_grads_all = jnp.array(all_grads)
    jax_params = jnp.zeros(param_dim)
    jax_m = jnp.zeros(param_dim)
    jax_v = jnp.zeros(param_dim)

    # Warmup: trigger compilation and fill caches
    for i in range(warmup_steps):
        g = jax_grads_all[i % num_steps]
        jax_m, jax_v, jax_params = jit_step(
            jax_m, jax_v, jax_params, g, jnp.array(i + 1), lr, beta1, beta2, eps,
        )
    jax_m.block_until_ready()

    jax_params = jnp.zeros(param_dim)
    jax_m = jnp.zeros(param_dim)
    jax_v = jnp.zeros(param_dim)

    t0 = time.perf_counter()
    for i in range(num_steps):
        g = jax_grads_all[i]
        jax_m, jax_v, jax_params = jit_step(
            jax_m, jax_v, jax_params, g, jnp.array(i + 1), lr, beta1, beta2, eps,
        )
    jax_params.block_until_ready()
    jax_elapsed = time.perf_counter() - t0

    c_total_ms = c_elapsed * 1000
    jax_total_ms = jax_elapsed * 1000
    c_per_step = c_total_ms * 1000 / num_steps    # microseconds
    jax_per_step = jax_total_ms * 1000 / num_steps

    return BenchmarkResult(
        c_total_ms=round(c_total_ms, 3),
        c_per_step_us=round(c_per_step, 3),
        jax_total_ms=round(jax_total_ms, 3),
        jax_per_step_us=round(jax_per_step, 3),
        speedup=round(jax_total_ms / c_total_ms, 2) if c_total_ms > 0 else float("inf"),
        num_steps=num_steps,
        param_dim=param_dim,
    )
