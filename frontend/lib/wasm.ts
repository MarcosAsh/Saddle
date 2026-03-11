/**
 * WASM loader for Saddle's C backend.
 *
 * Loads the Emscripten-compiled module and wraps the C functions
 * in a typed TypeScript API. Memory management (malloc/free) is
 * handled internally so callers never touch the WASM heap.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

interface EmscriptenModule {
  ccall: (name: string, ret: string, argTypes: string[], args: any[]) => any;
  cwrap: (name: string, ret: string, argTypes: string[]) => (...args: any[]) => any;
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
  HEAPF64: Float64Array;
}

export interface WasmTrajectory {
  x: number[];
  y: number[];
  loss: number[];
}

export interface WasmSaddle {
  evalGrid(
    surfaceId: number,
    xMin: number, xMax: number,
    yMin: number, yMax: number,
    rows: number, cols: number,
  ): number[][];

  adamOptimise(
    surfaceId: number,
    x0: number, y0: number,
    numSteps: number,
    lr: number, beta1: number, beta2: number, eps: number,
  ): WasmTrajectory;

  evalSurface(surfaceId: number, x: number, y: number): number;
}

export const SURFACE_IDS: Record<string, number> = {
  rosenbrock: 0,
  beale: 1,
  himmelblau: 2,
  bowl: 3,
  monkey_saddle: 4,
};

let cachedModule: EmscriptenModule | null = null;

async function loadModule(): Promise<EmscriptenModule> {
  if (cachedModule) return cachedModule;

  // Dynamically load the Emscripten glue JS from public/
  // @ts-expect-error -- loaded as a static asset, no module declaration
  const factory = (await import(/* webpackIgnore: true */ "/wasm/saddle.js")).default;
  const mod: EmscriptenModule = await factory();
  cachedModule = mod;
  return mod;
}

export async function initWasm(): Promise<WasmSaddle> {
  const mod = await loadModule();

  const _evalGrid = mod.cwrap("wasm_eval_grid", "void", [
    "number", "number", "number", "number", "number", "number", "number", "number",
  ]);
  const _adamOptimise = mod.cwrap("wasm_adam_optimise", "void", [
    "number", "number", "number", "number", "number", "number", "number", "number",
    "number", "number", "number",
  ]);

  const surfaceFns = [
    mod.cwrap("wasm_rosenbrock", "number", ["number", "number"]),
    mod.cwrap("wasm_beale", "number", ["number", "number"]),
    mod.cwrap("wasm_himmelblau", "number", ["number", "number"]),
    mod.cwrap("wasm_bowl", "number", ["number", "number"]),
    mod.cwrap("wasm_monkey_saddle", "number", ["number", "number"]),
  ];

  return {
    evalGrid(surfaceId, xMin, xMax, yMin, yMax, rows, cols) {
      const size = rows * cols;
      const ptr = mod._malloc(size * 8); // float64 = 8 bytes
      try {
        _evalGrid(ptr, xMin, xMax, yMin, yMax, rows, cols, surfaceId);
        const offset = ptr / 8;
        const result: number[][] = [];
        for (let i = 0; i < rows; i++) {
          const row: number[] = [];
          for (let j = 0; j < cols; j++) {
            row.push(mod.HEAPF64[offset + i * cols + j]);
          }
          result.push(row);
        }
        return result;
      } finally {
        mod._free(ptr);
      }
    },

    adamOptimise(surfaceId, x0, y0, numSteps, lr, beta1, beta2, eps) {
      const n = numSteps + 1;
      const ptrX = mod._malloc(n * 8);
      const ptrY = mod._malloc(n * 8);
      const ptrL = mod._malloc(n * 8);
      try {
        _adamOptimise(x0, y0, numSteps, lr, beta1, beta2, eps, surfaceId,
                      ptrX, ptrY, ptrL);
        const offX = ptrX / 8;
        const offY = ptrY / 8;
        const offL = ptrL / 8;
        const x: number[] = [];
        const y: number[] = [];
        const loss: number[] = [];
        for (let i = 0; i < n; i++) {
          x.push(mod.HEAPF64[offX + i]);
          y.push(mod.HEAPF64[offY + i]);
          loss.push(mod.HEAPF64[offL + i]);
        }
        return { x, y, loss };
      } finally {
        mod._free(ptrX);
        mod._free(ptrY);
        mod._free(ptrL);
      }
    },

    evalSurface(surfaceId, x, y) {
      if (surfaceId >= 0 && surfaceId < surfaceFns.length) {
        return surfaceFns[surfaceId](x, y);
      }
      return 0;
    },
  };
}
