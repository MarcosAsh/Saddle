"""
Neural network loss landscape visualization (Li et al. 2018).

Trains a small MLP on a spiral dataset, then projects the loss landscape
onto two filter-normalized random directions around the optimum.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import flatten_util


# ---------------------------------------------------------------------------
# Spiral dataset
# ---------------------------------------------------------------------------

def _make_spirals(n: int = 200, seed: int = 42) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a 2-class spiral dataset with n points per class."""
    key = jax.random.key(seed)
    theta = jnp.linspace(0, 4 * jnp.pi, n)
    r = jnp.linspace(0.2, 1.0, n)
    k1, k2 = jax.random.split(key)
    noise1 = 0.05 * jax.random.normal(k1, (n,))
    noise2 = 0.05 * jax.random.normal(k2, (n,))

    x1 = jnp.stack([r * jnp.cos(theta) + noise1, r * jnp.sin(theta) + noise2], axis=1)
    x2 = jnp.stack([-r * jnp.cos(theta) + noise1, -r * jnp.sin(theta) + noise2], axis=1)

    X = jnp.concatenate([x1, x2], axis=0)
    y = jnp.concatenate([jnp.zeros(n), jnp.ones(n)])
    return X, y


# ---------------------------------------------------------------------------
# MLP: 2 -> 16 -> 16 -> 1
# ---------------------------------------------------------------------------

def _init_mlp(key: jax.Array) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Initialize MLP parameters with Xavier init."""
    layers = [(2, 16), (16, 16), (16, 1)]
    params = []
    for fan_in, fan_out in layers:
        key, k = jax.random.split(key)
        scale = jnp.sqrt(2.0 / fan_in)
        W = scale * jax.random.normal(k, (fan_in, fan_out))
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def _mlp_forward(params: list, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass. tanh activations, sigmoid output."""
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return jax.nn.sigmoid(x @ W + b).squeeze(-1)


def _loss_fn(params: list, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy loss."""
    pred = _mlp_forward(params, X)
    pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
    return -jnp.mean(y * jnp.log(pred) + (1 - y) * jnp.log(1 - pred))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_mlp(
    X: jnp.ndarray,
    y: jnp.ndarray,
    seed: int = 42,
    steps: int = 2000,
    lr: float = 0.003,
) -> tuple[list, list[jnp.ndarray]]:
    """Train MLP with Adam, return final params and flat param history."""
    key = jax.random.key(seed)
    params = _init_mlp(key)

    flat, unravel = flatten_util.ravel_pytree(params)
    history = [flat.copy()]

    # Simple Adam
    m = jnp.zeros_like(flat)
    v = jnp.zeros_like(flat)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    grad_fn = jax.grad(lambda f: _loss_fn(unravel(f), X, y))

    for t in range(1, steps + 1):
        g = grad_fn(flat)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        flat = flat - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        # Record every 10th step for trajectory
        if t % 10 == 0 or t == steps:
            history.append(flat.copy())

    return unravel(flat), history


# ---------------------------------------------------------------------------
# Filter-normalized random directions (Li et al. 2018)
# ---------------------------------------------------------------------------

def _filter_normalize(direction: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Normalize direction to have the same norm as reference per filter/layer."""
    # For a flat vector, just normalize to same norm
    ref_norm = jnp.linalg.norm(reference)
    dir_norm = jnp.linalg.norm(direction)
    return direction * (ref_norm / (dir_norm + 1e-10))


def _compute_directions(flat_params: jnp.ndarray, seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate two filter-normalized random directions."""
    key = jax.random.key(seed + 1000)
    k1, k2 = jax.random.split(key)
    d1 = jax.random.normal(k1, flat_params.shape)
    d2 = jax.random.normal(k2, flat_params.shape)
    d1 = _filter_normalize(d1, flat_params)
    d2 = _filter_normalize(d2, flat_params)
    return d1, d2


# ---------------------------------------------------------------------------
# Public API (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _cached_train(seed: int) -> tuple:
    """Train and cache the result."""
    X, y = _make_spirals(seed=seed)
    params, history = _train_mlp(X, y, seed=seed)
    flat_star, unravel = flatten_util.ravel_pytree(params)
    d1, d2 = _compute_directions(flat_star, seed)
    return X, y, flat_star, unravel, d1, d2, history


def compute_nn_landscape(resolution: int = 50, seed: int = 42) -> dict:
    """
    Compute the loss landscape grid projected onto two random directions.

    Returns a dict with x_min, x_max, y_min, y_max, values.
    """
    X, y, flat_star, unravel, d1, d2, _ = _cached_train(seed)

    alpha_range = 1.0
    alphas = jnp.linspace(-alpha_range, alpha_range, resolution)
    betas = jnp.linspace(-alpha_range, alpha_range, resolution)

    values = []
    for beta in betas:
        row = []
        for alpha in alphas:
            theta = flat_star + float(alpha) * d1 + float(beta) * d2
            loss = float(_loss_fn(unravel(theta), X, y))
            row.append(loss)
        values.append(row)

    return {
        "x_min": -alpha_range,
        "x_max": alpha_range,
        "y_min": -alpha_range,
        "y_max": alpha_range,
        "values": values,
    }


def compute_nn_trajectory(seed: int = 42) -> list[tuple[float, float, float]]:
    """
    Project the training trajectory onto the (d1, d2) basis.

    Returns list of (alpha, beta, loss) tuples.
    """
    X, y, flat_star, unravel, d1, d2, history = _cached_train(seed)

    trajectory = []
    for flat in history:
        delta = flat - flat_star
        alpha = float(jnp.dot(delta, d1) / (jnp.dot(d1, d1) + 1e-10))
        beta = float(jnp.dot(delta, d2) / (jnp.dot(d2, d2) + 1e-10))
        loss = float(_loss_fn(unravel(flat), X, y))
        trajectory.append((alpha, beta, loss))

    return trajectory
