const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type SurfaceName = "rosenbrock" | "beale" | "himmelblau" | "bowl" | "monkey_saddle" | "custom" | "nn_landscape";
export type OptimiserName = "sgd" | "adam" | "adahessian" | "c_adam" | "rmsprop" | "lbfgs";
export type ScheduleName = "constant" | "cosine" | "warmup_cosine" | "step_decay";

export interface TrajectoryPoint {
  x: number;
  y: number;
  loss: number;
  lr?: number;
}

export interface OptimiseRequest {
  surface: SurfaceName;
  optimiser: OptimiserName;
  x0: number;
  y0: number;
  num_steps: number;
  lr: number;
  momentum?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  hessian_power?: number;
  alpha?: number;
  lbfgs_m?: number;
  schedule?: ScheduleName;
  warmup_steps?: number;
  batch_size?: number;
  custom_expr?: string;
}

export interface OptimiseResponse {
  trajectory: TrajectoryPoint[];
  optimiser: string;
  surface: string;
}

export interface SurfaceResponse {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  rows: number;
  cols: number;
  values: number[][];
}

export interface SurfaceInfo {
  key: string;
  name: string;
  formula: string;
  description: string;
  minima: string;
}

export interface GradientFieldResponse {
  x: number[];
  y: number[];
  gx: number[][];
  gy: number[][];
}

export interface BenchmarkResponse {
  c_total_ms: number;
  c_per_step_us: number;
  jax_total_ms: number;
  jax_per_step_us: number;
  speedup: number;
  num_steps: number;
  param_dim: number;
}

export async function fetchSurface(
  name: SurfaceName,
  resolution: number = 150
): Promise<SurfaceResponse> {
  const res = await fetch(
    `${API_BASE}/surface?name=${name}&resolution=${resolution}`
  );
  if (!res.ok) throw new Error(`Surface fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchOptimise(
  req: OptimiseRequest
): Promise<OptimiseResponse> {
  const res = await fetch(`${API_BASE}/optimise`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Optimise failed: ${res.status}`);
  return res.json();
}

export async function fetchSurfaces(): Promise<SurfaceInfo[]> {
  const res = await fetch(`${API_BASE}/surfaces`);
  if (!res.ok) throw new Error(`Surfaces list failed: ${res.status}`);
  return res.json();
}

export async function fetchGradientField(
  name: SurfaceName,
  resolution: number = 20
): Promise<GradientFieldResponse> {
  const res = await fetch(
    `${API_BASE}/gradient?name=${name}&resolution=${resolution}`
  );
  if (!res.ok) throw new Error(`Gradient field failed: ${res.status}`);
  return res.json();
}

export async function fetchBenchmark(
  numSteps: number = 10000,
  paramDim: number = 2
): Promise<BenchmarkResponse> {
  const res = await fetch(
    `${API_BASE}/benchmark?num_steps=${numSteps}&param_dim=${paramDim}`
  );
  if (!res.ok) throw new Error(`Benchmark failed: ${res.status}`);
  return res.json();
}

export async function fetchCustomSurface(
  expr: string,
  resolution: number = 150,
  bounds?: { x_min: number; x_max: number; y_min: number; y_max: number }
): Promise<SurfaceResponse> {
  const params = new URLSearchParams({
    expr,
    resolution: String(resolution),
  });
  if (bounds) {
    params.set("x_min", String(bounds.x_min));
    params.set("x_max", String(bounds.x_max));
    params.set("y_min", String(bounds.y_min));
    params.set("y_max", String(bounds.y_max));
  }
  const res = await fetch(`${API_BASE}/custom-surface?${params}`);
  if (!res.ok) throw new Error(`Custom surface failed: ${res.status}`);
  return res.json();
}

export async function fetchNNLandscape(
  resolution: number = 30,
  seed: number = 42
): Promise<SurfaceResponse> {
  const res = await fetch(
    `${API_BASE}/nn-landscape?resolution=${resolution}&seed=${seed}`
  );
  if (!res.ok) throw new Error(`NN landscape failed: ${res.status}`);
  return res.json();
}

export async function fetchNNTrajectory(
  seed: number = 42
): Promise<OptimiseResponse> {
  const res = await fetch(`${API_BASE}/nn-trajectory?seed=${seed}`);
  if (!res.ok) throw new Error(`NN trajectory failed: ${res.status}`);
  return res.json();
}
