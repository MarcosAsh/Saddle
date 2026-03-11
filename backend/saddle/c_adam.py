"""
Ctypes wrapper around the C Adam implementation.

Exposes two things:
  - c_adam_step: a single Adam update on numpy arrays (for benchmarking)
  - c_adam_optimise: full optimisation loop in C, returns trajectory
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from saddle.surfaces import SURFACE_IDS, SurfaceName

_LIB_PATH = Path(__file__).resolve().parent.parent / "csrc" / "libsaddle.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

# void adam_step(double *params, double *m, double *v, const double *grads,
#                int n, int step, double lr, double beta1, double beta2, double eps)
_lib.adam_step.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.POINTER(ctypes.c_double),  # m
    ctypes.POINTER(ctypes.c_double),  # v
    ctypes.POINTER(ctypes.c_double),  # grads
    ctypes.c_int,                      # n
    ctypes.c_int,                      # step
    ctypes.c_double,                   # lr
    ctypes.c_double,                   # beta1
    ctypes.c_double,                   # beta2
    ctypes.c_double,                   # eps
]
_lib.adam_step.restype = None

# void adam_optimise(double x0, double y0, int num_steps,
#                    double lr, double beta1, double beta2, double eps,
#                    int surface_id,
#                    double *trajectory_x, double *trajectory_y, double *trajectory_loss)
_lib.adam_optimise.argtypes = [
    ctypes.c_double,                   # x0
    ctypes.c_double,                   # y0
    ctypes.c_int,                      # num_steps
    ctypes.c_double,                   # lr
    ctypes.c_double,                   # beta1
    ctypes.c_double,                   # beta2
    ctypes.c_double,                   # eps
    ctypes.c_int,                      # surface_id
    ctypes.POINTER(ctypes.c_double),   # trajectory_x
    ctypes.POINTER(ctypes.c_double),   # trajectory_y
    ctypes.POINTER(ctypes.c_double),   # trajectory_loss
]
_lib.adam_optimise.restype = None


def _ptr(arr: NDArray) -> ctypes.POINTER(ctypes.c_double):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def c_adam_step(
    params: NDArray[np.float64],
    m: NDArray[np.float64],
    v: NDArray[np.float64],
    grads: NDArray[np.float64],
    step: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """
    Single Adam update step via C. Modifies params, m, v in place.

    This is the function used for benchmarking against JAX jit.
    """
    n = len(params)
    _lib.adam_step(
        _ptr(params), _ptr(m), _ptr(v), _ptr(grads),
        ctypes.c_int(n), ctypes.c_int(step),
        ctypes.c_double(lr), ctypes.c_double(beta1),
        ctypes.c_double(beta2), ctypes.c_double(eps),
    )


def c_adam_optimise(
    surface: SurfaceName,
    x0: float,
    y0: float,
    num_steps: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Run full Adam optimisation in C, returning the trajectory.

    Returns (xs, ys, losses), each of shape (num_steps + 1,).
    """
    size = num_steps + 1
    traj_x = np.empty(size, dtype=np.float64)
    traj_y = np.empty(size, dtype=np.float64)
    traj_loss = np.empty(size, dtype=np.float64)

    _lib.adam_optimise(
        ctypes.c_double(x0), ctypes.c_double(y0),
        ctypes.c_int(num_steps),
        ctypes.c_double(lr), ctypes.c_double(beta1),
        ctypes.c_double(beta2), ctypes.c_double(eps),
        ctypes.c_int(SURFACE_IDS[surface]),
        _ptr(traj_x), _ptr(traj_y), _ptr(traj_loss),
    )

    return traj_x, traj_y, traj_loss
