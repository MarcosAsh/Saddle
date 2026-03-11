"use client";

import { useState } from "react";
import { fetchBenchmark, type BenchmarkResponse } from "@/lib/api";

export default function BenchmarkPanel() {
  const [result, setResult] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runBenchmark() {
    setLoading(true);
    setError(null);
    try {
      const r = await fetchBenchmark(10000, 2);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Benchmark failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="border border-ctp-surface1 rounded-lg p-4 bg-ctp-mantle">
      <h2 className="text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider mb-3">
        C vs JAX Benchmark
      </h2>
      <p className="text-xs text-ctp-overlay1 mb-3">
        Compares the Adam update step implemented in C (via ctypes) against
        JAX&apos;s JIT-compiled version. Both receive identical pre-generated
        gradients, isolating the pure update arithmetic.
      </p>

      <button
        onClick={runBenchmark}
        disabled={loading}
        className="bg-ctp-surface0 hover:bg-ctp-surface1 disabled:text-ctp-overlay0 text-ctp-text text-xs font-medium py-1.5 px-3 rounded border border-ctp-surface1 transition-colors mb-3"
      >
        {loading ? "Running 10k steps..." : "Run benchmark"}
      </button>

      {error && <p className="text-ctp-red text-xs mb-2">{error}</p>}

      {result && (
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="border border-ctp-surface1 rounded p-3">
            <div className="text-ctp-overlay1 mb-1">C (ctypes)</div>
            <div className="text-lg font-mono text-ctp-text">
              {result.c_total_ms.toFixed(1)} ms
            </div>
            <div className="text-ctp-overlay1">
              {result.c_per_step_us.toFixed(2)} &micro;s/step
            </div>
          </div>
          <div className="border border-ctp-surface1 rounded p-3">
            <div className="text-ctp-overlay1 mb-1">JAX (JIT)</div>
            <div className="text-lg font-mono text-ctp-text">
              {result.jax_total_ms.toFixed(1)} ms
            </div>
            <div className="text-ctp-overlay1">
              {result.jax_per_step_us.toFixed(2)} &micro;s/step
            </div>
          </div>
          <div className="col-span-2 border border-ctp-surface1 rounded p-3 text-center">
            <div className="text-ctp-overlay1 mb-1">
              {result.speedup >= 1 ? "C is faster by" : "JAX is faster by"}
            </div>
            <div className="text-xl font-mono text-ctp-text">
              {result.speedup >= 1
                ? `${result.speedup.toFixed(1)}x`
                : `${(1 / result.speedup).toFixed(1)}x`}
            </div>
            <div className="text-ctp-overlay0 mt-1">
              {result.num_steps.toLocaleString()} steps, {result.param_dim}D
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
