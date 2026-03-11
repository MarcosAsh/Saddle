"""
Ctypes wrappers around the C loss surface implementations,
plus pure-numpy reference implementations for verification.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------

_LIB_PATH = Path(__file__).resolve().parent.parent / "csrc" / "libsaddle.so"

_lib = ctypes.CDLL(str(_LIB_PATH))

# Scalar evaluators: double f(double x, double y)
for _name in ("rosenbrock", "beale", "himmelblau", "bowl", "monkey_saddle"):
    _fn = getattr(_lib, _name)
    _fn.argtypes = [ctypes.c_double, ctypes.c_double]
    _fn.restype = ctypes.c_double

# Grid evaluator
_lib.eval_grid.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # out
    ctypes.c_double,                   # x_min
    ctypes.c_double,                   # x_max
    ctypes.c_double,                   # y_min
    ctypes.c_double,                   # y_max
    ctypes.c_int,                      # rows
    ctypes.c_int,                      # cols
    ctypes.c_int,                      # surface_id
]
_lib.eval_grid.restype = None

# ---------------------------------------------------------------------------
# Surface name -> id mapping
# ---------------------------------------------------------------------------

SurfaceName = Literal["rosenbrock", "beale", "himmelblau", "bowl", "monkey_saddle"]

SURFACE_IDS: dict[SurfaceName, int] = {
    "rosenbrock": 0,
    "beale": 1,
    "himmelblau": 2,
    "bowl": 3,
    "monkey_saddle": 4,
}

# ---------------------------------------------------------------------------
# C-backed Python API
# ---------------------------------------------------------------------------


def c_eval(name: SurfaceName, x: float, y: float) -> float:
    """Evaluate a loss surface at a single point using the C implementation."""
    fn = getattr(_lib, name)
    return fn(ctypes.c_double(x), ctypes.c_double(y))


def c_eval_grid(
    name: SurfaceName,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    rows: int,
    cols: int,
) -> NDArray[np.float64]:
    """
    Evaluate a loss surface over a uniform grid using C.

    Returns a (rows, cols) array where entry [i, j] is the surface value
    at the grid point (x_j, y_i).
    """
    buf = np.empty((rows, cols), dtype=np.float64)
    _lib.eval_grid(
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(x_min),
        ctypes.c_double(x_max),
        ctypes.c_double(y_min),
        ctypes.c_double(y_max),
        ctypes.c_int(rows),
        ctypes.c_int(cols),
        ctypes.c_int(SURFACE_IDS[name]),
    )
    return buf


# ---------------------------------------------------------------------------
# Numpy reference implementations
# ---------------------------------------------------------------------------


def np_rosenbrock(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    a = 1.0 - x
    b = y - x * x
    return a * a + 100.0 * b * b


def np_beale(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    y2 = y * y
    y3 = y2 * y
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * y2
    t3 = 2.625 - x + x * y3
    return t1 * t1 + t2 * t2 + t3 * t3


def np_himmelblau(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    t1 = x * x + y - 11.0
    t2 = x + y * y - 7.0
    return t1 * t1 + t2 * t2


def np_bowl(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    return x * x + y * y


def np_monkey_saddle(x: NDArray | float, y: NDArray | float) -> NDArray | float:
    return x * x * x - 3.0 * x * y * y + 0.5 * (x * x + y * y)


NP_SURFACES = {
    "rosenbrock": np_rosenbrock,
    "beale": np_beale,
    "himmelblau": np_himmelblau,
    "bowl": np_bowl,
    "monkey_saddle": np_monkey_saddle,
}


def np_eval(name: SurfaceName, x: float, y: float) -> float:
    """Evaluate a loss surface at a single point using numpy."""
    return float(NP_SURFACES[name](x, y))


def np_eval_grid(
    name: SurfaceName,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    rows: int,
    cols: int,
) -> NDArray[np.float64]:
    """
    Evaluate a loss surface over a uniform grid using numpy.

    Returns a (rows, cols) array matching the layout of c_eval_grid.
    """
    xs = np.linspace(x_min, x_max, cols)
    ys = np.linspace(y_min, y_max, rows)
    X, Y = np.meshgrid(xs, ys)
    return NP_SURFACES[name](X, Y)
