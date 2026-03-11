"""
From-scratch JAX optimiser implementations for Saddle.

Each optimiser is implemented in functional style:
  - State is an immutable dataclass (frozen=True).
  - The update function takes (state, params, grads) and returns (new_state, new_params).
  - No mutation, no side effects.
  - No optax.

Optimisers implemented:
  - SGD with momentum (Polyak's heavy ball)
  - Adam (Kingma & Ba, 2014)
  - AdaHessian (Yao et al., 2021) -- uses Hessian diagonal via Hutchinson estimator
  - RMSprop (Hinton, 2012)
  - L-BFGS (Nocedal, 1980) -- quasi-Newton with backtracking Armijo line search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# SGD with momentum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SGDState:
    """Carries the velocity term for momentum."""
    velocity: Array
    step: int

    @staticmethod
    def init(params: Array) -> SGDState:
        return SGDState(
            velocity=jnp.zeros_like(params),
            step=0,
        )


def sgd_update(
    state: SGDState,
    params: Array,
    grads: Array,
    lr: float = 0.01,
    momentum: float = 0.9,
) -> tuple[SGDState, Array]:
    """
    SGD with Polyak's heavy ball momentum.

    v_t = mu * v_{t-1} + grad
    theta_t = theta_{t-1} - lr * v_t

    This is the "momentum first" convention used by PyTorch's SGD.
    """
    new_velocity = momentum * state.velocity + grads
    new_params = params - lr * new_velocity
    new_state = SGDState(velocity=new_velocity, step=state.step + 1)
    return new_state, new_params


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdamState:
    """First moment (m), second moment (v), and step counter for bias correction."""
    m: Array
    v: Array
    step: int

    @staticmethod
    def init(params: Array) -> AdamState:
        return AdamState(
            m=jnp.zeros_like(params),
            v=jnp.zeros_like(params),
            step=0,
        )


def adam_update(
    state: AdamState,
    params: Array,
    grads: Array,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[AdamState, Array]:
    """
    Adam (Kingma & Ba, 2014).

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    Step counter is incremented before bias correction (1-indexed),
    matching the original paper's convention.
    """
    new_step = state.step + 1

    # Update biased first and second moment estimates
    new_m = beta1 * state.m + (1.0 - beta1) * grads
    new_v = beta2 * state.v + (1.0 - beta2) * grads * grads

    # Bias-corrected estimates
    m_hat = new_m / (1.0 - beta1 ** new_step)
    v_hat = new_v / (1.0 - beta2 ** new_step)

    # Parameter update
    new_params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    new_state = AdamState(m=new_m, v=new_v, step=new_step)
    return new_state, new_params


# ---------------------------------------------------------------------------
# AdaHessian
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaHessianState:
    """
    Like Adam, but the second moment tracks the Hessian diagonal
    instead of squared gradients.
    """
    m: Array
    v: Array
    step: int

    @staticmethod
    def init(params: Array) -> AdaHessianState:
        return AdaHessianState(
            m=jnp.zeros_like(params),
            v=jnp.zeros_like(params),
            step=0,
        )


def _hutchinson_hessian_diag(
    loss_fn: Callable[[Array], Array],
    params: Array,
    key: Array,
) -> Array:
    """
    Estimate the diagonal of the Hessian using a Hutchinson trace estimator.

    The idea: for a random vector z drawn from Rademacher (uniform {-1, +1}),
    the Hessian-vector product H @ z can be computed efficiently with a single
    forward-over-reverse pass (jax.jvp of jax.grad). Then:

        E[z * (H @ z)] = diag(H)

    This gives us the diagonal without ever forming the full Hessian matrix.
    We use a single sample here since we are calling this at every step
    and the noise averages out over the trajectory.
    """
    # Rademacher random vector: each element is +1 or -1 with equal probability
    z = 2.0 * jax.random.bernoulli(key, shape=params.shape).astype(params.dtype) - 1.0

    # grad_fn: params -> gradient vector
    grad_fn = jax.grad(loss_fn)

    # Hessian-vector product via forward-over-reverse:
    # jvp of grad_fn gives (grad, H @ z) when the tangent is z
    _, hvp = jax.jvp(grad_fn, (params,), (z,))

    # Element-wise: z_i * (H @ z)_i is an unbiased estimate of H_ii
    hessian_diag = z * hvp

    return hessian_diag


def adahessian_update(
    state: AdaHessianState,
    params: Array,
    grads: Array,
    loss_fn: Callable[[Array], Array],
    key: Array,
    lr: float = 0.15,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-4,
    hessian_power: float = 1.0,
) -> tuple[AdaHessianState, Array]:
    """
    AdaHessian (Yao et al., 2021).

    Like Adam, but replaces g_t^2 in the second moment with an estimate
    of the squared Hessian diagonal. This gives the optimiser curvature
    information: in directions where the loss curves sharply (large H_ii),
    it takes smaller steps, and in flat directions it takes larger steps.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * D_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * m_hat / (v_hat^(p/2) + eps)

    where D_t = diag(H_t) estimated via Hutchinson's method and
    p is the hessian_power (1.0 in the paper, controls how aggressively
    curvature influences step size).

    The key parameter is a JAX PRNG key for the Rademacher random vector
    used in the Hutchinson estimator.
    """
    new_step = state.step + 1

    # Estimate the Hessian diagonal at current params
    hess_diag = _hutchinson_hessian_diag(loss_fn, params, key)

    # First moment: same as Adam, tracks gradient direction
    new_m = beta1 * state.m + (1.0 - beta1) * grads

    # Second moment: tracks squared Hessian diagonal instead of squared gradient.
    # The absolute value ensures we have a positive definite preconditioner
    # even when Hessian entries are negative (saddle points, maxima).
    new_v = beta2 * state.v + (1.0 - beta2) * hess_diag * hess_diag

    # Bias correction
    m_hat = new_m / (1.0 - beta1 ** new_step)
    v_hat = new_v / (1.0 - beta2 ** new_step)

    # Parameter update with Hessian-based preconditioning
    new_params = params - lr * m_hat / (jnp.power(v_hat, hessian_power / 2.0) + eps)

    new_state = AdaHessianState(m=new_m, v=new_v, step=new_step)
    return new_state, new_params


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RMSpropState:
    """Running average of squared gradients."""
    v: Array
    step: int

    @staticmethod
    def init(params: Array) -> RMSpropState:
        return RMSpropState(
            v=jnp.zeros_like(params),
            step=0,
        )


def rmsprop_update(
    state: RMSpropState,
    params: Array,
    grads: Array,
    lr: float = 0.01,
    alpha: float = 0.99,
    eps: float = 1e-8,
) -> tuple[RMSpropState, Array]:
    """
    RMSprop (Hinton, 2012).

    v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
    theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + eps)

    No bias correction -- unlike Adam, the original RMSprop has none.
    """
    new_v = alpha * state.v + (1.0 - alpha) * grads * grads
    new_params = params - lr * grads / (jnp.sqrt(new_v) + eps)
    new_state = RMSpropState(v=new_v, step=state.step + 1)
    return new_state, new_params


# ---------------------------------------------------------------------------
# L-BFGS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LBFGSState:
    """
    State for L-BFGS with a fixed-size circular buffer of (s, y) pairs.
    """
    prev_params: Array
    prev_grads: Array
    s_history: Array   # (m, d) -- position differences
    y_history: Array   # (m, d) -- gradient differences
    rho_history: Array # (m,) -- 1 / (y_k^T s_k)
    history_len: int
    step: int

    @staticmethod
    def init(params: Array, m: int = 5) -> LBFGSState:
        d = params.shape[0]
        return LBFGSState(
            prev_params=params,
            prev_grads=jnp.zeros_like(params),
            s_history=jnp.zeros((m, d)),
            y_history=jnp.zeros((m, d)),
            rho_history=jnp.zeros(m),
            history_len=0,
            step=0,
        )


def _lbfgs_two_loop(
    grads: Array,
    s_history: Array,
    y_history: Array,
    rho_history: Array,
    history_len: int,
) -> Array:
    """
    L-BFGS two-loop recursion to compute the search direction.

    Returns -H_k * grads, where H_k is the approximate inverse Hessian
    built from the stored (s, y) pairs.
    """
    m = s_history.shape[0]
    k = min(history_len, m)
    q = grads.copy()
    alphas = jnp.zeros(m)

    # Backward pass
    for i in range(k - 1, -1, -1):
        a_i = rho_history[i] * jnp.dot(s_history[i], q)
        alphas = alphas.at[i].set(a_i)
        q = q - a_i * y_history[i]

    # Initial Hessian approximation: H_0 = gamma * I
    if k > 0:
        latest = k - 1
        gamma = jnp.dot(s_history[latest], y_history[latest]) / (
            jnp.dot(y_history[latest], y_history[latest]) + 1e-10
        )
        gamma = jnp.clip(gamma, 1e-6, 1e6)
    else:
        gamma = 1.0

    r = gamma * q

    # Forward pass
    for i in range(k):
        beta = rho_history[i] * jnp.dot(y_history[i], r)
        r = r + (alphas[i] - beta) * s_history[i]

    return -r


def _backtracking_line_search(
    loss_fn: Callable[[Array], Array],
    params: Array,
    grads: Array,
    direction: Array,
    lr: float,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 20,
) -> float:
    """Backtracking Armijo line search. Returns the step size."""
    f0 = float(loss_fn(params))
    slope = float(jnp.dot(grads, direction))

    # If direction is not a descent direction, return a small fixed step
    if slope >= 0:
        return lr * 0.01

    step_size = lr
    for _ in range(max_iter):
        candidate = params + step_size * direction
        f_new = float(loss_fn(candidate))
        if f_new <= f0 + c1 * step_size * slope:
            return step_size
        step_size *= rho

    return step_size


def lbfgs_update(
    state: LBFGSState,
    params: Array,
    grads: Array,
    loss_fn: Callable[[Array], Array],
    lr: float = 1.0,
    m: int = 5,
) -> tuple[LBFGSState, Array]:
    """
    L-BFGS (Nocedal, 1980).

    A quasi-Newton method that approximates the inverse Hessian using
    a limited history of position and gradient differences. Uses a
    backtracking Armijo line search for step size selection.

    On the first step, falls back to a gradient descent step since
    there is no history to build the approximation from.
    """
    if state.step == 0:
        # First step: plain gradient descent, no history yet
        direction = -grads
        step_size = _backtracking_line_search(
            loss_fn, params, grads, direction, lr,
        )
        new_params = params + step_size * direction
        new_state = LBFGSState(
            prev_params=params,
            prev_grads=grads,
            s_history=state.s_history,
            y_history=state.y_history,
            rho_history=state.rho_history,
            history_len=0,
            step=1,
        )
        return new_state, new_params

    # Compute position and gradient differences
    s_k = params - state.prev_params
    y_k = grads - state.prev_grads
    sy = jnp.dot(s_k, y_k)

    # Update circular buffer (only if curvature condition holds)
    buf_size = state.s_history.shape[0]
    if float(sy) > 1e-10:
        idx = state.history_len % buf_size
        new_s = state.s_history.at[idx].set(s_k)
        new_y = state.y_history.at[idx].set(y_k)
        new_rho = state.rho_history.at[idx].set(1.0 / sy)
        new_len = min(state.history_len + 1, buf_size)
    else:
        new_s = state.s_history
        new_y = state.y_history
        new_rho = state.rho_history
        new_len = state.history_len

    # Compute search direction via two-loop recursion
    direction = _lbfgs_two_loop(grads, new_s, new_y, new_rho, new_len)

    # Line search
    step_size = _backtracking_line_search(
        loss_fn, params, grads, direction, lr,
    )
    new_params = params + step_size * direction

    new_state = LBFGSState(
        prev_params=params,
        prev_grads=grads,
        s_history=new_s,
        y_history=new_y,
        rho_history=new_rho,
        history_len=new_len,
        step=state.step + 1,
    )
    return new_state, new_params
