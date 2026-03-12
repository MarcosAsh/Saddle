"use client";

import { useState } from "react";
import type { SurfaceName, OptimiserName, ScheduleName } from "@/lib/api";

const SURFACES: { value: SurfaceName; label: string }[] = [
  { value: "rosenbrock", label: "Rosenbrock" },
  { value: "beale", label: "Beale" },
  { value: "himmelblau", label: "Himmelblau" },
  { value: "bowl", label: "Bowl" },
  { value: "monkey_saddle", label: "Monkey Saddle" },
  { value: "custom", label: "Custom..." },
  { value: "nn_landscape", label: "Neural Net" },
];

const OPTIMISERS: { value: OptimiserName; label: string }[] = [
  { value: "sgd", label: "SGD + Momentum" },
  { value: "adam", label: "Adam" },
  { value: "adahessian", label: "AdaHessian" },
  { value: "c_adam", label: "Adam (C)" },
  { value: "rmsprop", label: "RMSprop" },
  { value: "lbfgs", label: "L-BFGS" },
];

const SCHEDULES: { value: ScheduleName; label: string }[] = [
  { value: "constant", label: "Constant" },
  { value: "cosine", label: "Cosine" },
  { value: "warmup_cosine", label: "Warmup + Cosine" },
  { value: "step_decay", label: "Step Decay" },
];

interface Preset {
  name: string;
  surface: SurfaceName;
  opt1: OptimiserName;
  opt2: OptimiserName;
  x0: number;
  y0: number;
  lr: number;
  numSteps: number;
  sideBySide: boolean;
}

const PRESETS: Preset[] = [
  {
    name: "Saddle Escape",
    surface: "monkey_saddle",
    opt1: "sgd",
    opt2: "adam",
    x0: 0.1,
    y0: 0.1,
    lr: 0.01,
    numSteps: 1000,
    sideBySide: true,
  },
  {
    name: "Rosenbrock Race",
    surface: "rosenbrock",
    opt1: "adam",
    opt2: "lbfgs",
    x0: -1,
    y0: 1,
    lr: 0.01,
    numSteps: 500,
    sideBySide: true,
  },
  {
    name: "Four Minima",
    surface: "himmelblau",
    opt1: "adam",
    opt2: "sgd",
    x0: -1,
    y0: -1,
    lr: 0.01,
    numSteps: 1000,
    sideBySide: true,
  },
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
  showGradients: boolean;
  setShowGradients: (v: boolean) => void;
  surfaceDescription: string | null;
  surfaceFormula: string | null;
  viewMode: "3d" | "contour";
  setViewMode: (v: "3d" | "contour") => void;
  schedule: ScheduleName;
  setSchedule: (s: ScheduleName) => void;
  warmupSteps: number;
  setWarmupSteps: (v: number) => void;
  batchSize: number | null;
  setBatchSize: (v: number | null) => void;
  noiseEnabled: boolean;
  setNoiseEnabled: (v: boolean) => void;
  customExpr: string;
  setCustomExpr: (v: string) => void;
  onAutoRun: () => void;
  nnSeed: number;
  setNnSeed: (v: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  display,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  display?: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-ctp-subtext0">{label}</span>
        <span className="text-ctp-text font-mono">{display ?? value}</span>
      </div>
      {hint && <p className="text-[10px] text-ctp-overlay0 leading-tight">{hint}</p>}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-ctp-mauve"
      />
    </div>
  );
}

export default function Controls(props: ControlsProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const isNNLandscape = props.surface === "nn_landscape";
  const isCustom = props.surface === "custom";

  const applyPreset = (preset: Preset) => {
    props.setSurface(preset.surface);
    props.setOptimiser1(preset.opt1);
    props.setOptimiser2(preset.opt2);
    props.setX0(preset.x0);
    props.setY0(preset.y0);
    props.setLr(preset.lr);
    props.setNumSteps(preset.numSteps);
    props.setSideBySide(preset.sideBySide);
    props.onAutoRun();
  };

  return (
    <div className="flex flex-col gap-4 p-4 border border-ctp-surface1 rounded-lg overflow-y-auto bg-ctp-mantle">
      {/* Mobile: compact header with toggle + run button */}
      <div className="flex items-center justify-between lg:hidden">
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="flex items-center gap-2 text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider"
        >
          <svg
            className={`w-3 h-3 transition-transform ${mobileOpen ? "rotate-90" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          Controls
        </button>
        <button
          onClick={props.onRun}
          disabled={props.running}
          className="bg-ctp-mauve hover:bg-ctp-lavender disabled:bg-ctp-surface1 disabled:text-ctp-overlay0 text-ctp-crust text-xs font-medium py-1.5 px-3 rounded transition-colors"
        >
          {props.running ? "Running..." : "Run"}
        </button>
      </div>

      {/* Desktop: always-visible header */}
      <h2 className="hidden lg:block text-xs font-semibold text-ctp-overlay1 uppercase tracking-wider">
        Controls
      </h2>

      {/* Controls body: hidden on mobile when collapsed, always visible on desktop */}
      <div className={`flex flex-col gap-4 ${mobileOpen ? "" : "hidden"} lg:flex`}>
        {/* Presets */}
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => applyPreset(preset)}
              disabled={props.running}
              className="text-[10px] px-2 py-1 rounded-full border border-ctp-surface2 text-ctp-subtext0 hover:border-ctp-mauve hover:text-ctp-text transition-colors disabled:opacity-50"
            >
              {preset.name}
            </button>
          ))}
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-ctp-subtext0">Surface</label>
          <select
            value={props.surface}
            onChange={(e) => props.setSurface(e.target.value as SurfaceName)}
            className="bg-ctp-surface0 text-ctp-text text-sm rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
          >
            {SURFACES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        {/* Custom surface expression input */}
        {isCustom && (
          <div className="flex flex-col gap-1">
            <label className="text-xs text-ctp-subtext0">Expression f(x, y)</label>
            <input
              type="text"
              value={props.customExpr}
              onChange={(e) => props.setCustomExpr(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") props.onRun();
              }}
              placeholder="sin(x)*cos(y)"
              className="bg-ctp-surface0 text-ctp-text text-sm font-mono rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
            />
            <p className="text-[10px] text-ctp-overlay0 leading-tight">
              Allowed: x, y, pi, e, +, -, *, /, **, sin, cos, exp, log, sqrt, abs, tan, tanh
            </p>
          </div>
        )}

        {/* Neural net surface controls */}
        {isNNLandscape && (
          <div className="flex flex-col gap-2">
            <p className="text-xs text-ctp-overlay1 leading-relaxed">
              Loss landscape of a small MLP (2-16-16-1) trained on a spiral dataset,
              projected onto two random directions around the optimum (Li et al. 2018).
            </p>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-ctp-subtext0">Random seed</label>
              <input
                type="number"
                value={props.nnSeed}
                onChange={(e) => props.setNnSeed(parseInt(e.target.value) || 42)}
                className="bg-ctp-surface0 text-ctp-text text-sm font-mono rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none w-20"
              />
            </div>
          </div>
        )}

        {/* Optimizer controls (hidden for nn_landscape) */}
        {!isNNLandscape && (
          <>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-ctp-subtext0">Optimiser</label>
              <select
                value={props.optimiser1}
                onChange={(e) => props.setOptimiser1(e.target.value as OptimiserName)}
                className="bg-ctp-surface0 text-ctp-text text-sm rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
              >
                {OPTIMISERS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Surface description */}
            {props.surfaceDescription && !isCustom && (
              <div className="text-xs text-ctp-overlay1 leading-relaxed">
                {props.surfaceFormula && (
                  <p className="font-mono text-ctp-subtext0 mb-1">{props.surfaceFormula}</p>
                )}
                <p>{props.surfaceDescription}</p>
              </div>
            )}

            <label className="flex items-center gap-2 text-xs text-ctp-subtext0 cursor-pointer">
              <input
                type="checkbox"
                checked={props.viewMode === "contour"}
                onChange={(e) => props.setViewMode(e.target.checked ? "contour" : "3d")}
                className="accent-ctp-mauve"
              />
              2D contour view
            </label>

            <label className="flex items-center gap-2 text-xs text-ctp-subtext0 cursor-pointer">
              <input
                type="checkbox"
                checked={props.showGradients}
                onChange={(e) => props.setShowGradients(e.target.checked)}
                className="accent-ctp-mauve"
              />
              Show gradient field
            </label>

            <label className="flex items-center gap-2 text-xs text-ctp-subtext0 cursor-pointer">
              <input
                type="checkbox"
                checked={props.sideBySide}
                onChange={(e) => props.setSideBySide(e.target.checked)}
                className="accent-ctp-mauve"
              />
              Compare two optimisers
            </label>

            {props.sideBySide && (
              <div className="flex flex-col gap-1">
                <label className="text-xs text-ctp-subtext0">Optimiser 2</label>
                <select
                  value={props.optimiser2 ?? "adam"}
                  onChange={(e) =>
                    props.setOptimiser2(e.target.value as OptimiserName)
                  }
                  className="bg-ctp-surface0 text-ctp-text text-sm rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
                >
                  {OPTIMISERS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <hr className="border-ctp-surface1" />

            <Slider
              label="Learning rate"
              value={Math.log10(props.lr)}
              min={-5}
              max={0}
              step={0.1}
              onChange={(v) => props.setLr(Math.pow(10, v))}
              display={props.lr.toExponential(1)}
              hint="Step size per update. Too high overshoots, too low crawls."
            />

            {/* LR Schedule */}
            <div className="flex flex-col gap-1">
              <label className="text-xs text-ctp-subtext0">LR Schedule</label>
              <select
                value={props.schedule}
                onChange={(e) => props.setSchedule(e.target.value as ScheduleName)}
                className="bg-ctp-surface0 text-ctp-text text-sm rounded px-2 py-1.5 border border-ctp-surface1 focus:border-ctp-mauve outline-none"
              >
                {SCHEDULES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            {props.schedule === "warmup_cosine" && (
              <Slider
                label="Warmup steps"
                value={props.warmupSteps}
                min={0}
                max={500}
                step={10}
                onChange={props.setWarmupSteps}
                hint="Linear ramp before cosine decay begins."
              />
            )}

            <Slider
              label="Momentum"
              value={props.momentum}
              min={0}
              max={0.99}
              step={0.01}
              onChange={props.setMomentum}
              display={props.momentum.toFixed(2)}
              hint="SGD velocity decay. Higher values carry more speed through valleys."
            />

            <Slider
              label="Steps"
              value={props.numSteps}
              min={50}
              max={5000}
              step={50}
              onChange={props.setNumSteps}
              hint="Total optimisation iterations to run."
            />

            <hr className="border-ctp-surface1" />

            {/* Stochastic gradients */}
            <label className="flex items-center gap-2 text-xs text-ctp-subtext0 cursor-pointer">
              <input
                type="checkbox"
                checked={props.noiseEnabled}
                onChange={(e) => {
                  props.setNoiseEnabled(e.target.checked);
                  if (!e.target.checked) props.setBatchSize(null);
                  else props.setBatchSize(32);
                }}
                className="accent-ctp-mauve"
              />
              Stochastic gradients
            </label>

            {props.noiseEnabled && props.batchSize !== null && (
              <Slider
                label="Batch size"
                value={Math.log2(props.batchSize)}
                min={0}
                max={8}
                step={1}
                onChange={(v) => props.setBatchSize(Math.pow(2, v))}
                display={String(props.batchSize)}
                hint="Simulates mini-batch noise. Smaller = more exploration."
              />
            )}

            <hr className="border-ctp-surface1" />

            <Slider
              label="x&#x2080;"
              value={props.x0}
              min={-5}
              max={5}
              step={0.1}
              onChange={props.setX0}
              display={props.x0.toFixed(1)}
              hint="Starting x coordinate on the surface."
            />

            <Slider
              label="y&#x2080;"
              value={props.y0}
              min={-5}
              max={5}
              step={0.1}
              onChange={props.setY0}
              display={props.y0.toFixed(1)}
              hint="Starting y coordinate on the surface."
            />

            <hr className="border-ctp-surface1" />

            <Slider
              label="Animation speed"
              value={props.animSpeed}
              min={1}
              max={50}
              step={1}
              onChange={props.setAnimSpeed}
              display={`${props.animSpeed} steps/frame`}
              hint="How many steps to advance per animation frame."
            />
          </>
        )}

        {/* Run button inside expanded controls on mobile */}
        <button
          onClick={props.onRun}
          disabled={props.running}
          className="mt-2 bg-ctp-mauve hover:bg-ctp-lavender disabled:bg-ctp-surface1 disabled:text-ctp-overlay0 text-ctp-crust text-sm font-medium py-2 px-4 rounded transition-colors lg:block"
        >
          {props.running ? "Running..." : isNNLandscape ? "Load Landscape" : "Run"}
        </button>
      </div>
    </div>
  );
}
