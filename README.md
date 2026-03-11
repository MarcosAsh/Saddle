# Saddle

Interactive optimiser visualisation. Compare how SGD, Adam, L-BFGS, and more
navigate loss surfaces in real time -- with a C backend for fast numerical
baselines, JAX for autodiff, and WASM for in-browser benchmarks.

![screenshot](https://github.com/user-attachments/assets/placeholder)

## What it does

Pick a loss surface, drop two optimisers on it, and watch them race.
Toggle between a 3D surface plot and a 2D contour view. A convergence chart
tracks loss over steps, and a benchmark panel times C vs JAX (server-side)
and WASM vs JS (client-side) Adam head-to-head.

### Surfaces

| Surface | Formula | Character |
|---------|---------|-----------|
| Rosenbrock | `(1-x)^2 + 100(y-x^2)^2` | Narrow banana valley |
| Beale | `(1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2` | Sharp ridges, flat regions |
| Himmelblau | `(x^2+y-11)^2 + (x+y^2-7)^2` | Four identical minima |
| Bowl | `x^2 + y^2` | Perfect quadratic baseline |
| Monkey Saddle | `x^3 - 3xy^2 + 0.5(x^2+y^2)` | Degenerate saddle point |

### Optimisers

- **SGD + Momentum** -- Polyak heavy ball
- **Adam** -- adaptive learning rates with bias correction (Kingma & Ba, 2014)
- **AdaHessian** -- replaces squared gradients with Hessian diagonal via Hutchinson's estimator (Yao et al., 2021)
- **RMSprop** -- running average of squared gradients (Hinton, 2012)
- **L-BFGS** -- quasi-Newton with limited-memory BFGS and backtracking Armijo line search (Nocedal, 1980)
- **Adam (C)** -- same update equations in C with numerical gradients, for benchmarking

## Project structure

```
backend/
  csrc/           C loss surfaces + Adam (compiled to libsaddle.so and saddle.wasm)
  saddle/
    optimisers.py JAX implementations of SGD, Adam, AdaHessian, RMSprop, L-BFGS
    api.py        FastAPI server (5 endpoints)
    surfaces.py   ctypes bindings to C library
    c_adam.py     ctypes wrapper for C Adam
    benchmark.py  C vs JAX timing harness

frontend/         Next.js + Plotly.js + Tailwind (Catppuccin Mocha)
  lib/wasm.ts     WASM loader for client-side benchmarks
  public/wasm/    Compiled WASM binary (16KB)
```

## Setup

### Backend

```bash
cd backend/csrc
make                          # builds libsaddle.so

cd ..
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uvicorn saddle.api:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                   # http://localhost:3000
```

The frontend reads `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`).

### WASM build (optional)

Requires Emscripten. The WASM binary is already checked into `frontend/public/wasm/`,
so this is only needed if you modify the C code.

```bash
cd backend/csrc
make install-wasm             # builds and copies to frontend/public/wasm/
```

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/optimise` | Run an optimiser, return trajectory |
| GET | `/surface` | Loss values on a grid |
| GET | `/surfaces` | List surfaces with metadata |
| GET | `/gradient` | Gradient vector field |
| GET | `/benchmark` | C vs JAX Adam timing |

## Tech

- **C** -- loss surfaces and Adam inner loop (`-O2`, no allocations in hot path)
- **WebAssembly** -- same C code compiled to WASM for client-side benchmarks (16KB)
- **JAX** -- autodiff optimisers, Hessian-vector products for AdaHessian, L-BFGS
- **FastAPI** -- serves everything over REST
- **Next.js / React** -- app shell
- **Plotly.js** -- 3D surface and 2D contour rendering with trajectory overlay
- **Tailwind v4** -- Catppuccin Mocha dark theme
