"""
Verify that the C surface implementations match the numpy references.

Tests both scalar evaluation at specific points (including known minima)
and full grid evaluation.
"""

import numpy as np
import pytest

from saddle.surfaces import (
    SurfaceName,
    c_eval,
    c_eval_grid,
    np_eval,
    np_eval_grid,
)

# Tolerance for floating-point comparison.
# Both implementations use double precision, so differences are
# only from operation ordering -- should be extremely small.
ATOL = 1e-12


# -------------------------------------------------------------------
# Scalar tests: known minima and a handful of arbitrary points
# -------------------------------------------------------------------

KNOWN_MINIMA: list[tuple[SurfaceName, float, float, float]] = [
    ("rosenbrock", 1.0, 1.0, 0.0),
    ("beale", 3.0, 0.5, 0.0),
    ("himmelblau", 3.0, 2.0, 0.0),
    ("himmelblau", -2.805118, 3.131312, 0.0),
    ("himmelblau", -3.779310, -3.283186, 0.0),
    ("himmelblau", 3.584428, -1.848126, 0.0),
    ("bowl", 0.0, 0.0, 0.0),
]


@pytest.mark.parametrize("name,x,y,expected", KNOWN_MINIMA)
def test_known_minima(name: SurfaceName, x: float, y: float, expected: float) -> None:
    """Both implementations should return ~0 at known global minima."""
    c_val = c_eval(name, x, y)
    np_val = np_eval(name, x, y)
    assert abs(c_val - expected) < 1e-4, f"C {name}({x},{y}) = {c_val}, expected {expected}"
    assert abs(np_val - expected) < 1e-4, f"np {name}({x},{y}) = {np_val}, expected {expected}"


ARBITRARY_POINTS: list[tuple[SurfaceName, float, float]] = [
    ("rosenbrock", 0.0, 0.0),
    ("rosenbrock", -1.5, 2.3),
    ("rosenbrock", 5.0, -3.0),
    ("beale", 0.0, 0.0),
    ("beale", 1.0, 1.0),
    ("beale", -2.0, 1.5),
    ("himmelblau", 0.0, 0.0),
    ("himmelblau", 1.0, 1.0),
    ("himmelblau", -4.0, 4.0),
    ("bowl", 3.0, 4.0),
    ("bowl", -1.0, -1.0),
]


@pytest.mark.parametrize("name,x,y", ARBITRARY_POINTS)
def test_c_matches_numpy_scalar(name: SurfaceName, x: float, y: float) -> None:
    """C and numpy should agree on scalar evaluation at arbitrary points."""
    c_val = c_eval(name, x, y)
    np_val = np_eval(name, x, y)
    assert abs(c_val - np_val) < ATOL, (
        f"{name}({x},{y}): C={c_val}, numpy={np_val}, diff={abs(c_val - np_val)}"
    )


# -------------------------------------------------------------------
# Grid tests
# -------------------------------------------------------------------

GRID_CASES: list[tuple[SurfaceName, float, float, float, float]] = [
    ("rosenbrock", -2.0, 2.0, -1.0, 3.0),
    ("beale", -4.5, 4.5, -4.5, 4.5),
    ("himmelblau", -5.0, 5.0, -5.0, 5.0),
    ("bowl", -3.0, 3.0, -3.0, 3.0),
]


@pytest.mark.parametrize("name,xmin,xmax,ymin,ymax", GRID_CASES)
def test_c_matches_numpy_grid(
    name: SurfaceName,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    """C and numpy grid evaluations should produce identical arrays."""
    rows, cols = 64, 64
    c_grid = c_eval_grid(name, xmin, xmax, ymin, ymax, rows, cols)
    np_grid = np_eval_grid(name, xmin, xmax, ymin, ymax, rows, cols)

    assert c_grid.shape == (rows, cols)
    assert np_grid.shape == (rows, cols)
    np.testing.assert_allclose(c_grid, np_grid, atol=ATOL, rtol=0)


# -------------------------------------------------------------------
# Sanity: known exact values
# -------------------------------------------------------------------

def test_rosenbrock_at_origin() -> None:
    # f(0,0) = (1-0)^2 + 100*(0-0)^2 = 1
    assert abs(c_eval("rosenbrock", 0.0, 0.0) - 1.0) < ATOL

def test_bowl_at_3_4() -> None:
    # f(3,4) = 9 + 16 = 25
    assert abs(c_eval("bowl", 3.0, 4.0) - 25.0) < ATOL

def test_himmelblau_at_origin() -> None:
    # f(0,0) = (0+0-11)^2 + (0+0-7)^2 = 121 + 49 = 170
    assert abs(c_eval("himmelblau", 0.0, 0.0) - 170.0) < ATOL

def test_beale_at_origin() -> None:
    # f(0,0) = 1.5^2 + 2.25^2 + 2.625^2 = 2.25 + 5.0625 + 6.890625 = 14.203125
    assert abs(c_eval("beale", 0.0, 0.0) - 14.203125) < ATOL
