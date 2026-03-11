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
    <div className="border border-neutral-200 rounded-lg p-4">
      <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider mb-3">
        C vs JAX Benchmark
      </h2>
      <p className="text-xs text-neutral-400 mb-3">
        Compares the Adam update step implemented in C (via ctypes) against
        JAX&apos;s JIT-compiled version. Both receive identical pre-generated
        gradients, isolating the pure update arithmetic.
      </p>

      <button
        onClick={runBenchmark}
        disabled={loading}
        className="bg-white hover:bg-neutral-50 disabled:text-neutral-300 text-black text-xs font-medium py-1.5 px-3 rounded border border-neutral-200 transition-colors mb-3"
      >
        {loading ? "Running 10k steps..." : "Run benchmark"}
      </button>

      {error && <p className="text-red-600 text-xs mb-2">{error}</p>}

      {result && (
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="border border-neutral-200 rounded p-3">
            <div className="text-neutral-400 mb-1">C (ctypes)</div>
            <div className="text-lg font-mono text-black">
              {result.c_total_ms.toFixed(1)} ms
            </div>
            <div className="text-neutral-400">
              {result.c_per_step_us.toFixed(2)} &micro;s/step
            </div>
          </div>
          <div className="border border-neutral-200 rounded p-3">
            <div className="text-neutral-400 mb-1">JAX (JIT)</div>
            <div className="text-lg font-mono text-black">
              {result.jax_total_ms.toFixed(1)} ms
            </div>
            <div className="text-neutral-400">
              {result.jax_per_step_us.toFixed(2)} &micro;s/step
            </div>
          </div>
          <div className="col-span-2 border border-neutral-200 rounded p-3 text-center">
            <div className="text-neutral-400 mb-1">
              {result.speedup >= 1 ? "C is faster by" : "JAX is faster by"}
            </div>
            <div className="text-xl font-mono text-black">
              {result.speedup >= 1
                ? `${result.speedup.toFixed(1)}x`
                : `${(1 / result.speedup).toFixed(1)}x`}
            </div>
            <div className="text-neutral-300 mt-1">
              {result.num_steps.toLocaleString()} steps, {result.param_dim}D
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
