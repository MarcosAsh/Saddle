"use client";

import type { SurfaceName, OptimiserName } from "@/lib/api";

const SURFACES: { value: SurfaceName; label: string }[] = [
  { value: "rosenbrock", label: "Rosenbrock" },
  { value: "beale", label: "Beale" },
  { value: "himmelblau", label: "Himmelblau" },
  { value: "bowl", label: "Bowl" },
];

const OPTIMISERS: { value: OptimiserName; label: string }[] = [
  { value: "sgd", label: "SGD + Momentum" },
  { value: "adam", label: "Adam" },
  { value: "adahessian", label: "AdaHessian" },
  { value: "c_adam", label: "Adam (C)" },
];

interface ControlsProps {
  surface: SurfaceName;
  setSurface: (s: SurfaceName) => void;
  optimiser1: OptimiserName;
  setOptimiser1: (o: OptimiserName) => void;
  optimiser2: OptimiserName | null;
  setOptimiser2: (o: OptimiserName | null) => void;
  lr: number;
  setLr: (v: number) => void;
  momentum: number;
  setMomentum: (v: number) => void;
  numSteps: number;
  setNumSteps: (v: number) => void;
  x0: number;
  setX0: (v: number) => void;
  y0: number;
  setY0: (v: number) => void;
  sideBySide: boolean;
  setSideBySide: (v: boolean) => void;
  onRun: () => void;
  running: boolean;
  animSpeed: number;
  setAnimSpeed: (v: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  display,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-neutral-500">{label}</span>
        <span className="text-black font-mono">{display ?? value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-black"
      />
    </div>
  );
}

export default function Controls(props: ControlsProps) {
  return (
    <div className="flex flex-col gap-4 p-4 border border-neutral-200 rounded-lg overflow-y-auto">
      <h2 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">
        Controls
      </h2>

      <div className="flex flex-col gap-1">
        <label className="text-xs text-neutral-500">Surface</label>
        <select
          value={props.surface}
          onChange={(e) => props.setSurface(e.target.value as SurfaceName)}
          className="bg-white text-black text-sm rounded px-2 py-1.5 border border-neutral-200 focus:border-black outline-none"
        >
          {SURFACES.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs text-neutral-500">Optimiser</label>
        <select
          value={props.optimiser1}
          onChange={(e) => props.setOptimiser1(e.target.value as OptimiserName)}
          className="bg-white text-black text-sm rounded px-2 py-1.5 border border-neutral-200 focus:border-black outline-none"
        >
          {OPTIMISERS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <label className="flex items-center gap-2 text-xs text-neutral-500 cursor-pointer">
        <input
          type="checkbox"
          checked={props.sideBySide}
          onChange={(e) => props.setSideBySide(e.target.checked)}
          className="accent-black"
        />
        Compare two optimisers
      </label>

      {props.sideBySide && (
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-500">Optimiser 2</label>
          <select
            value={props.optimiser2 ?? "adam"}
            onChange={(e) =>
              props.setOptimiser2(e.target.value as OptimiserName)
            }
            className="bg-white text-black text-sm rounded px-2 py-1.5 border border-neutral-200 focus:border-black outline-none"
          >
            {OPTIMISERS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
      )}

      <hr className="border-neutral-200" />

      <Slider
        label="Learning rate"
        value={Math.log10(props.lr)}
        min={-5}
        max={0}
        step={0.1}
        onChange={(v) => props.setLr(Math.pow(10, v))}
        display={props.lr.toExponential(1)}
      />

      <Slider
        label="Momentum"
        value={props.momentum}
        min={0}
        max={0.99}
        step={0.01}
        onChange={props.setMomentum}
        display={props.momentum.toFixed(2)}
      />

      <Slider
        label="Steps"
        value={props.numSteps}
        min={50}
        max={5000}
        step={50}
        onChange={props.setNumSteps}
      />

      <hr className="border-neutral-200" />

      <Slider
        label="x&#x2080;"
        value={props.x0}
        min={-5}
        max={5}
        step={0.1}
        onChange={props.setX0}
        display={props.x0.toFixed(1)}
      />

      <Slider
        label="y&#x2080;"
        value={props.y0}
        min={-5}
        max={5}
        step={0.1}
        onChange={props.setY0}
        display={props.y0.toFixed(1)}
      />

      <hr className="border-neutral-200" />

      <Slider
        label="Animation speed"
        value={props.animSpeed}
        min={1}
        max={50}
        step={1}
        onChange={props.setAnimSpeed}
        display={`${props.animSpeed} steps/frame`}
      />

      <button
        onClick={props.onRun}
        disabled={props.running}
        className="mt-2 bg-black hover:bg-neutral-800 disabled:bg-neutral-300 disabled:text-neutral-500 text-white text-sm font-medium py-2 px-4 rounded transition-colors"
      >
        {props.running ? "Running..." : "Run"}
      </button>
    </div>
  );
}
