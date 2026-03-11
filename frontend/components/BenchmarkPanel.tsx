"use client";

import { useState, useRef } from "react";
import { fetchBenchmark, type BenchmarkResponse } from "@/lib/api";
import { initWasm, SURFACE_IDS, type WasmSaddle } from "@/lib/wasm";

interface WasmBenchResult {
  wasmMs: number;
  wasmPerStepUs: number;
  jsMs: number;
  jsPerStepUs: number;
  speedup: number;
  numSteps: number;
}

function jsAdamBenchmark(numSteps: number): number {
  // Minimal JS Adam on Rosenbrock for timing comparison
  let x = -1.0, y = 1.0;
  let mx = 0, my = 0, vx = 0, vy = 0;
  const lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const h = 1e-7;

  const f = (a: number, b: number) => (1 - a) ** 2 + 100 * (b - a * a) ** 2;

  const t0 = performance.now();
  for (let s = 1; s <= numSteps; s++) {
    const gx = (f(x + h, y) - f(x - h, y)) / (2 * h);
    const gy = (f(x, y + h) - f(x, y - h)) / (2 * h);

    mx = b1 * mx + (1 - b1) * gx;
    my = b1 * my + (1 - b1) * gy;
    vx = b2 * vx + (1 - b2) * gx * gx;
    vy = b2 * vy + (1 - b2) * gy * gy;

    const mxh = mx / (1 - b1 ** s);
    const myh = my / (1 - b1 ** s);
    const vxh = vx / (1 - b2 ** s);
    const vyh = vy / (1 - b2 ** s);

    x -= lr * mxh / (Math.sqrt(vxh) + eps);
    y -= lr * myh / (Math.sqrt(vyh) + eps);
  }
  return performance.now() - t0;
}

export default function BenchmarkPanel() {
  const [serverResult, setServerResult] = useState<BenchmarkResponse | null>(null);
  const [wasmResult, setWasmResult] = useState<WasmBenchResult | null>(null);
  const [serverLoading, setServerLoading] = useState(false);
  const [wasmLoading, setWasmLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wasmRef = useRef<WasmSaddle | null>(null);

  async function runServerBenchmark() {
    setServerLoading(true);
    setError(null);
    try {
      const r = await fetchBenchmark(10000, 2);
      setServerResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Benchmark failed");
    } finally {
      setServerLoading(false);
    }
  }

  async function runWasmBenchmark() {
    setWasmLoading(true);
    setError(null);
    const numSteps = 10000;
    try {
      if (!wasmRef.current) {
        wasmRef.current = await initWasm();
      }
      const wasm = wasmRef.current;

      // Warm up
      wasm.adamOptimise(SURFACE_IDS.rosenbrock, -1, 1, 100, 0.01, 0.9, 0.999, 1e-8);

      // Timed WASM run
      const t0 = performance.now();
      wasm.adamOptimise(SURFACE_IDS.rosenbrock, -1, 1, numSteps, 0.01, 0.9, 0.999, 1e-8);
      const wasmMs = performance.now() - t0;

      // Timed JS run
      const jsMs = jsAdamBenchmark(numSteps);

      const wasmPerStepUs = (wasmMs / numSteps) * 1000;
      const jsPerStepUs = (jsMs / numSteps) * 1000;

      setWasmResult({
        wasmMs,
        wasmPerStepUs,
        jsMs,
        jsPerStepUs,
        speedup: jsMs / wasmMs,
        numSteps,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "WASM benchmark failed");
    } finally {
      setWasmLoading(false);
    }
  }

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
        Benchmarks
      </h2>

      {error && <p className="text-ctp-red text-xs mb-2">{error}</p>}

      {/* Server-side benchmark */}
      <div className="mb-4">
        <p className="text-xs text-ctp-overlay1 mb-2">
          Server-side: C (ctypes) vs JAX (JIT) Adam update step.
        </p>
        <button
          onClick={runServerBenchmark}
          disabled={serverLoading}
          className="bg-ctp-surface0 hover:bg-ctp-surface1 disabled:text-ctp-overlay0 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors mb-2"
        >
          {serverLoading ? "Running..." : "Run server benchmark"}
        </button>
        {serverResult && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="border border-ctp-surface1 rounded p-2">
              <div className="text-ctp-overlay1 mb-1">C (ctypes)</div>
              <div className="text-lg font-mono text-ctp-text">
                {serverResult.c_total_ms.toFixed(1)} ms
              </div>
              <div className="text-ctp-overlay1">
                {serverResult.c_per_step_us.toFixed(2)} &micro;s/step
              </div>
            </div>
            <div className="border border-ctp-surface1 rounded p-2">
              <div className="text-ctp-overlay1 mb-1">JAX (JIT)</div>
              <div className="text-lg font-mono text-ctp-text">
                {serverResult.jax_total_ms.toFixed(1)} ms
              </div>
              <div className="text-ctp-overlay1">
                {serverResult.jax_per_step_us.toFixed(2)} &micro;s/step
              </div>
            </div>
            <div className="col-span-2 border border-ctp-surface1 rounded p-2 text-center">
              <div className="text-ctp-overlay1 mb-1">
                {serverResult.speedup >= 1 ? "C is faster by" : "JAX is faster by"}
              </div>
              <div className="text-xl font-mono text-ctp-text">
                {serverResult.speedup >= 1
                  ? `${serverResult.speedup.toFixed(1)}x`
                  : `${(1 / serverResult.speedup).toFixed(1)}x`}
              </div>
              <div className="text-ctp-overlay0 mt-1">
                {serverResult.num_steps.toLocaleString()} steps, {serverResult.param_dim}D
              </div>
            </div>
          </div>
        )}
      </div>

      <hr className="border-ctp-surface1 mb-4" />

      {/* Client-side WASM benchmark */}
      <div>
        <p className="text-xs text-ctp-overlay1 mb-2">
          Client-side: WASM (C compiled) vs plain JavaScript Adam.
        </p>
        <button
          onClick={runWasmBenchmark}
          disabled={wasmLoading}
          className="bg-ctp-surface0 hover:bg-ctp-surface1 disabled:text-ctp-overlay0 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors mb-2"
        >
          {wasmLoading ? "Running..." : "Run WASM benchmark"}
        </button>
        {wasmResult && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="border border-ctp-surface1 rounded p-2">
              <div className="text-ctp-overlay1 mb-1">WASM (C)</div>
              <div className="text-lg font-mono text-ctp-text">
                {wasmResult.wasmMs.toFixed(1)} ms
              </div>
              <div className="text-ctp-overlay1">
                {wasmResult.wasmPerStepUs.toFixed(2)} &micro;s/step
              </div>
            </div>
            <div className="border border-ctp-surface1 rounded p-2">
              <div className="text-ctp-overlay1 mb-1">JavaScript</div>
              <div className="text-lg font-mono text-ctp-text">
                {wasmResult.jsMs.toFixed(1)} ms
              </div>
              <div className="text-ctp-overlay1">
                {wasmResult.jsPerStepUs.toFixed(2)} &micro;s/step
              </div>
            </div>
            <div className="col-span-2 border border-ctp-surface1 rounded p-2 text-center">
              <div className="text-ctp-overlay1 mb-1">
                {wasmResult.speedup >= 1 ? "WASM is faster by" : "JS is faster by"}
              </div>
              <div className="text-xl font-mono text-ctp-text">
                {wasmResult.speedup >= 1
                  ? `${wasmResult.speedup.toFixed(1)}x`
                  : `${(1 / wasmResult.speedup).toFixed(1)}x`}
              </div>
              <div className="text-ctp-overlay0 mt-1">
                {wasmResult.numSteps.toLocaleString()} steps, in-browser
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
