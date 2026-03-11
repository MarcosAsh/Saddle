"""
FastAPI application for Saddle.

Endpoints:
  POST /optimise    -- run an optimiser on a loss surface, return trajectory
  GET  /surface     -- evaluate a loss surface on a grid for contour rendering
  GET  /surfaces    -- list available surfaces with metadata
  GET  /gradient    -- compute gradient field over a grid
  GET  /benchmark   -- run the C vs JAX Adam benchmark
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from saddle.benchmark import benchmark_adam
from saddle.c_adam import c_adam_optimise
from saddle.optimisers import (
    AdamState,
    AdaHessianState,
    SGDState,
    adam_update,
    adahessian_update,
    sgd_update,
)
from saddle.surfaces import SURFACE_IDS, SurfaceName, c_eval_grid

app = FastAPI(title="Saddle", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# JAX loss functions (for autodiff in the optimisation loop)
# ---------------------------------------------------------------------------

def _jax_rosenbrock(params: jax.Array) -> jax.Array:
    x, y = params[0], params[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


def _jax_beale(params: jax.Array) -> jax.Array:
    x, y = params[0], params[1]
    y2 = y * y
    y3 = y2 * y
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * y2
    t3 = 2.625 - x + x * y3
    return t1 * t1 + t2 * t2 + t3 * t3


def _jax_himmelblau(params: jax.Array) -> jax.Array:
    x, y = params[0], params[1]
    t1 = x * x + y - 11.0
    t2 = x + y * y - 7.0
    return t1 * t1 + t2 * t2


def _jax_bowl(params: jax.Array) -> jax.Array:
    return jnp.sum(params ** 2)


def _jax_monkey_saddle(params: jax.Array) -> jax.Array:
    x, y = params[0], params[1]
    return x ** 3 - 3.0 * x * y ** 2 + 0.5 * (x ** 2 + y ** 2)


JAX_SURFACES = {
    "rosenbrock": _jax_rosenbrock,
    "beale": _jax_beale,
    "himmelblau": _jax_himmelblau,
    "bowl": _jax_bowl,
    "monkey_saddle": _jax_monkey_saddle,
}


# ---------------------------------------------------------------------------
# Surface metadata
# ---------------------------------------------------------------------------

SURFACE_INFO: dict[str, dict] = {
    "rosenbrock": {
        "name": "Rosenbrock",
        "formula": "f(x,y) = (1-x)^2 + 100(y-x^2)^2",
        "description": "Banana-shaped valley. The minimum sits inside a long, narrow, "
                       "parabolic valley that most methods find quickly but crawl along.",
        "minima": "(1, 1)",
    },
    "beale": {
        "name": "Beale",
        "formula": "f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2",
        "description": "Sharp ridges and flat regions. Fixed step sizes overshoot the ridges, "
                       "adaptive methods navigate them.",
        "minima": "(3, 0.5)",
    },
    "himmelblau": {
        "name": "Himmelblau",
        "formula": "f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2",
        "description": "Four identical minima. Different starting points converge to different "
                       "minima, making it useful for testing sensitivity to initial conditions.",
        "minima": "(3,2), (-2.81,3.13), (-3.78,-3.28), (3.58,-1.85)",
    },
    "bowl": {
        "name": "Bowl",
        "formula": "f(x,y) = x^2 + y^2",
        "description": "Perfectly conditioned quadratic. Every optimiser converges. "
                       "Useful as a sanity check and baseline.",
        "minima": "(0, 0)",
    },
    "monkey_saddle": {
        "name": "Monkey Saddle",
        "formula": "f(x,y) = x^3 - 3xy^2 + 0.5(x^2+y^2)",
        "description": "Three valleys radiating from the origin at 120 degrees. The gradient "
                       "vanishes at the origin but it is not a minimum. First-order methods "
                       "stall, second-order methods detect the nearby negative curvature and escape.",
        "minima": "None (saddle point at origin)",
    },
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

OptimiserName = Literal["sgd", "adam", "adahessian", "c_adam"]


class OptimiseRequest(BaseModel):
    surface: SurfaceName
    optimiser: OptimiserName
    x0: float = Field(default=-1.0, description="Starting x coordinate")
    y0: float = Field(default=1.0, description="Starting y coordinate")
    num_steps: int = Field(default=500, ge=1, le=50000)
    lr: float = Field(default=0.01, gt=0)
    momentum: float = Field(default=0.9, ge=0, le=1, description="SGD momentum")
    beta1: float = Field(default=0.9, ge=0, le=1)
    beta2: float = Field(default=0.999, ge=0, le=1)
    eps: float = Field(default=1e-8, gt=0)
    hessian_power: float = Field(default=1.0, ge=0, le=2, description="AdaHessian hessian_power")


class TrajectoryPoint(BaseModel):
    x: float
    y: float
    loss: float


class OptimiseResponse(BaseModel):
    trajectory: list[TrajectoryPoint]
    optimiser: str
    surface: str


class SurfaceResponse(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    rows: int
    cols: int
    values: list[list[float]]


class SurfaceInfoResponse(BaseModel):
    name: str
    key: str
    formula: str
    description: str
    minima: str


class GradientFieldResponse(BaseModel):
    x: list[float]
    y: list[float]
    gx: list[list[float]]
    gy: list[list[float]]


class BenchmarkResponse(BaseModel):
    c_total_ms: float
    c_per_step_us: float
    jax_total_ms: float
    jax_per_step_us: float
    speedup: float
    num_steps: int
    param_dim: int


# ---------------------------------------------------------------------------
# Default surface bounds for nice visualisation
# ---------------------------------------------------------------------------

SURFACE_BOUNDS: dict[SurfaceName, tuple[float, float, float, float]] = {
    "rosenbrock": (-2.0, 2.0, -1.0, 3.0),
    "beale": (-4.5, 4.5, -4.5, 4.5),
    "himmelblau": (-5.0, 5.0, -5.0, 5.0),
    "bowl": (-5.0, 5.0, -5.0, 5.0),
    "monkey_saddle": (-2.0, 2.0, -2.0, 2.0),
}

# Default starting points that make sense per surface
SURFACE_DEFAULTS: dict[SurfaceName, tuple[float, float]] = {
    "rosenbrock": (-1.0, 1.0),
    "beale": (-1.0, -1.0),
    "himmelblau": (-4.0, 4.0),
    "bowl": (3.0, 4.0),
    "monkey_saddle": (0.1, 0.1),
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/surfaces", response_model=list[SurfaceInfoResponse])
def list_surfaces() -> list[SurfaceInfoResponse]:
    """List all available surfaces with metadata."""
    return [
        SurfaceInfoResponse(
            key=key,
            name=info["name"],
            formula=info["formula"],
            description=info["description"],
            minima=info["minima"],
        )
        for key, info in SURFACE_INFO.items()
    ]


@app.post("/optimise", response_model=OptimiseResponse)
def optimise(req: OptimiseRequest) -> OptimiseResponse:
    """Run an optimiser on a loss surface and return the full trajectory."""

    # C Adam path: entirely in C, no JAX
    if req.optimiser == "c_adam":
        xs, ys, losses = c_adam_optimise(
            surface=req.surface,
            x0=req.x0, y0=req.y0,
            num_steps=req.num_steps,
            lr=req.lr, beta1=req.beta1, beta2=req.beta2, eps=req.eps,
        )
        trajectory = [
            TrajectoryPoint(x=float(xs[i]), y=float(ys[i]), loss=float(losses[i]))
            for i in range(len(xs))
        ]
        return OptimiseResponse(
            trajectory=trajectory,
            optimiser="c_adam",
            surface=req.surface,
        )

    # JAX path
    loss_fn = JAX_SURFACES.get(req.surface)
    if loss_fn is None:
        raise HTTPException(400, f"Unknown surface: {req.surface}")

    params = jnp.array([req.x0, req.y0])
    grad_fn = jax.grad(loss_fn)

    trajectory: list[TrajectoryPoint] = [
        TrajectoryPoint(x=req.x0, y=req.y0, loss=float(loss_fn(params)))
    ]

    if req.optimiser == "sgd":
        state = SGDState.init(params)
        for _ in range(req.num_steps):
            grads = grad_fn(params)
            state, params = sgd_update(state, params, grads, lr=req.lr, momentum=req.momentum)
            trajectory.append(TrajectoryPoint(
                x=float(params[0]), y=float(params[1]), loss=float(loss_fn(params)),
            ))

    elif req.optimiser == "adam":
        state = AdamState.init(params)
        for _ in range(req.num_steps):
            grads = grad_fn(params)
            state, params = adam_update(
                state, params, grads, lr=req.lr,
                beta1=req.beta1, beta2=req.beta2, eps=req.eps,
            )
            trajectory.append(TrajectoryPoint(
                x=float(params[0]), y=float(params[1]), loss=float(loss_fn(params)),
            ))

    elif req.optimiser == "adahessian":
        state = AdaHessianState.init(params)
        key = jax.random.key(0)
        adahessian_eps = max(req.eps, 1e-4)
        for _ in range(req.num_steps):
            grads = grad_fn(params)
            key, subkey = jax.random.split(key)
            state, params = adahessian_update(
                state, params, grads, loss_fn, subkey,
                lr=req.lr, beta1=req.beta1, beta2=req.beta2,
                eps=adahessian_eps, hessian_power=req.hessian_power,
            )
            trajectory.append(TrajectoryPoint(
                x=float(params[0]), y=float(params[1]), loss=float(loss_fn(params)),
            ))

    else:
        raise HTTPException(400, f"Unknown optimiser: {req.optimiser}")

    return OptimiseResponse(
        trajectory=trajectory,
        optimiser=req.optimiser,
        surface=req.surface,
    )


@app.get("/surface", response_model=SurfaceResponse)
def surface(
    name: SurfaceName = "rosenbrock",
    resolution: int = 100,
) -> SurfaceResponse:
    """Evaluate a loss surface on a grid for contour plot rendering."""
    if resolution < 2 or resolution > 1000:
        raise HTTPException(400, "Resolution must be between 2 and 1000")

    bounds = SURFACE_BOUNDS[name]
    x_min, x_max, y_min, y_max = bounds

    grid = c_eval_grid(name, x_min, x_max, y_min, y_max, resolution, resolution)

    return SurfaceResponse(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        rows=resolution, cols=resolution,
        values=grid.tolist(),
    )


@app.get("/gradient", response_model=GradientFieldResponse)
def gradient_field(
    name: SurfaceName = "rosenbrock",
    resolution: int = 20,
) -> GradientFieldResponse:
    """
    Compute the gradient vector field over a grid.

    Returns gx[i][j] and gy[i][j], the partial derivatives at each grid point.
    Used for rendering gradient arrows on the surface plot.
    """
    if resolution < 2 or resolution > 100:
        raise HTTPException(400, "Resolution must be between 2 and 100")

    loss_fn = JAX_SURFACES.get(name)
    if loss_fn is None:
        raise HTTPException(400, f"Unknown surface: {name}")

    bounds = SURFACE_BOUNDS[name]
    x_min, x_max, y_min, y_max = bounds

    xs = np.linspace(x_min, x_max, resolution).tolist()
    ys = np.linspace(y_min, y_max, resolution).tolist()

    grad_fn = jax.grad(loss_fn)

    gx_grid: list[list[float]] = []
    gy_grid: list[list[float]] = []

    for yi in ys:
        gx_row: list[float] = []
        gy_row: list[float] = []
        for xi in xs:
            g = grad_fn(jnp.array([xi, yi]))
            gx_row.append(float(g[0]))
            gy_row.append(float(g[1]))
        gx_grid.append(gx_row)
        gy_grid.append(gy_row)

    return GradientFieldResponse(x=xs, y=ys, gx=gx_grid, gy=gy_grid)


@app.get("/benchmark", response_model=BenchmarkResponse)
def benchmark(
    num_steps: int = 10000,
    param_dim: int = 2,
) -> BenchmarkResponse:
    """Run the C vs JAX JIT Adam benchmark."""
    if num_steps < 100 or num_steps > 1000000:
        raise HTTPException(400, "num_steps must be between 100 and 1,000,000")
    if param_dim < 2 or param_dim > 10000:
        raise HTTPException(400, "param_dim must be between 2 and 10,000")

    result = benchmark_adam(num_steps=num_steps, param_dim=param_dim)

    return BenchmarkResponse(
        c_total_ms=result.c_total_ms,
        c_per_step_us=result.c_per_step_us,
        jax_total_ms=result.jax_total_ms,
        jax_per_step_us=result.jax_per_step_us,
        speedup=result.speedup,
        num_steps=result.num_steps,
        param_dim=result.param_dim,
    )
