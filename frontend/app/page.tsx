"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Controls from "@/components/Controls";
import SurfacePlot from "@/components/SurfacePlot";
import ConvergencePlot from "@/components/ConvergencePlot";
import BenchmarkPanel from "@/components/BenchmarkPanel";
import {
  fetchSurface,
  fetchOptimise,
  fetchSurfaces,
  fetchGradientField,
  type SurfaceName,
  type OptimiserName,
  type SurfaceResponse,
  type TrajectoryPoint,
  type SurfaceInfo,
  type GradientFieldResponse,
} from "@/lib/api";

// --- URL hash helpers ---

const VALID_SURFACES = new Set<string>(["rosenbrock", "beale", "himmelblau", "bowl", "monkey_saddle"]);
const VALID_OPTIMISERS = new Set<string>(["sgd", "adam", "adahessian", "c_adam", "rmsprop", "lbfgs"]);

function parseHash(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const hash = window.location.hash.slice(1);
  const params: Record<string, string> = {};
  for (const part of hash.split("&")) {
    const [k, v] = part.split("=");
    if (k && v) params[decodeURIComponent(k)] = decodeURIComponent(v);
  }
  return params;
}

function writeHash(state: Record<string, string | number | boolean>) {
  const parts = Object.entries(state)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  window.history.replaceState(null, "", `#${parts}`);
}

const OPTIMISER_COLORS: Record<OptimiserName, string> = {
  sgd: "#f38ba8",
  adam: "#89b4fa",
  adahessian: "#cba6f7",
  c_adam: "#a6e3a1",
  rmsprop: "#f9e2af",
  lbfgs: "#94e2d5",
};

const OPTIMISER_LABELS: Record<OptimiserName, string> = {
  sgd: "SGD",
  adam: "Adam",
  adahessian: "AdaHessian",
  c_adam: "Adam (C)",
  rmsprop: "RMSprop",
  lbfgs: "L-BFGS",
};

interface TrajectoryState {
  points: TrajectoryPoint[];
  name: string;
  color: string;
  animIndex: number;
}

export default function Home() {
  // Read initial state from URL hash (if present)
  const [initFromHash] = useState(() => {
    const h = parseHash();
    return {
      surface: VALID_SURFACES.has(h.s ?? "") ? (h.s as SurfaceName) : "rosenbrock",
      opt1: VALID_OPTIMISERS.has(h.o1 ?? "") ? (h.o1 as OptimiserName) : "adam",
      opt2: h.o2 === "none" ? null : VALID_OPTIMISERS.has(h.o2 ?? "") ? (h.o2 as OptimiserName) : "sgd",
      lr: h.lr ? Number(h.lr) : 0.01,
      mom: h.mom ? Number(h.mom) : 0.9,
      steps: h.n ? Number(h.n) : 500,
      x0: h.x0 ? Number(h.x0) : -1.0,
      y0: h.y0 ? Number(h.y0) : 1.0,
      side: h.cmp !== "0",
      view: h.v === "2d" ? "contour" as const : "3d" as const,
      grad: h.gf === "1",
      speed: h.sp ? Number(h.sp) : 5,
    };
  });

  const [surface, setSurface] = useState<SurfaceName>(initFromHash.surface);
  const [optimiser1, setOptimiser1] = useState<OptimiserName>(initFromHash.opt1);
  const [optimiser2, setOptimiser2] = useState<OptimiserName | null>(initFromHash.opt2);
  const [lr, setLr] = useState(initFromHash.lr);
  const [momentum, setMomentum] = useState(initFromHash.mom);
  const [numSteps, setNumSteps] = useState(initFromHash.steps);
  const [x0, setX0] = useState(initFromHash.x0);
  const [y0, setY0] = useState(initFromHash.y0);
  const [sideBySide, setSideBySide] = useState(initFromHash.side);
  const [animSpeed, setAnimSpeed] = useState(initFromHash.speed);

  const [showGradients, setShowGradients] = useState(initFromHash.grad);
  const [viewMode, setViewMode] = useState<"3d" | "contour">(initFromHash.view);

  const [surfaceData, setSurfaceData] = useState<SurfaceResponse | null>(null);
  const [gradientField, setGradientField] = useState<GradientFieldResponse | null>(null);
  const [surfaceInfoMap, setSurfaceInfoMap] = useState<Record<string, SurfaceInfo>>({});
  const [trajectories, setTrajectories] = useState<TrajectoryState[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const animFrameRef = useRef<number | null>(null);
  const trajRef = useRef<TrajectoryState[]>([]);
  const animSpeedRef = useRef(animSpeed);
  const animateFnRef = useRef<() => void>(() => {});

  // Keep speed ref in sync
  useEffect(() => {
    animSpeedRef.current = animSpeed;
  }, [animSpeed]);

  // Define the animation function via ref to avoid circular deps
  useEffect(() => {
    animateFnRef.current = () => {
      let allDone = true;
      const speed = animSpeedRef.current;
      const updated = trajRef.current.map((t) => {
        if (t.animIndex < t.points.length - 1) {
          allDone = false;
          return {
            ...t,
            animIndex: Math.min(t.animIndex + speed, t.points.length - 1),
          };
        }
        return t;
      });
      trajRef.current = updated;
      setTrajectories([...updated]);

      if (!allDone) {
        animFrameRef.current = requestAnimationFrame(() => animateFnRef.current());
      } else {
        setRunning(false);
      }
    };
  });

  // Load surface info on mount
  useEffect(() => {
    fetchSurfaces()
      .then((infos) => {
        const map: Record<string, SurfaceInfo> = {};
        for (const info of infos) map[info.key] = info;
        setSurfaceInfoMap(map);
      })
      .catch(() => {});
  }, []);

  // Load surface when selection changes
  useEffect(() => {
    let cancelled = false;
    setSurfaceData(null);
    setGradientField(null);
    fetchSurface(surface, 150)
      .then((data) => {
        if (!cancelled) setSurfaceData(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [surface]);

  // Load gradient field when toggled on
  useEffect(() => {
    if (!showGradients) {
      setGradientField(null);
      return;
    }
    let cancelled = false;
    fetchGradientField(surface, 20)
      .then((data) => {
        if (!cancelled) setGradientField(data);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [surface, showGradients]);

  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  // Sync state to URL hash
  useEffect(() => {
    writeHash({
      s: surface,
      o1: optimiser1,
      o2: sideBySide && optimiser2 ? optimiser2 : "none",
      lr: lr,
      mom: momentum,
      n: numSteps,
      x0: x0,
      y0: y0,
      cmp: sideBySide ? "1" : "0",
      v: viewMode === "contour" ? "2d" : "3d",
      gf: showGradients ? "1" : "0",
      sp: animSpeed,
    });
  }, [surface, optimiser1, optimiser2, lr, momentum, numSteps, x0, y0, sideBySide, viewMode, showGradients, animSpeed]);

  // Click on contour plot to set starting point
  const handlePlotClick = useCallback((cx: number, cy: number) => {
    setX0(cx);
    setY0(cy);
  }, []);

  const handleRun = useCallback(async () => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setRunning(true);
    setError(null);
    setTrajectories([]);

    try {
      const optimisers: OptimiserName[] =
        sideBySide && optimiser2
          ? [optimiser1, optimiser2]
          : [optimiser1];

      const results = await Promise.all(
        optimisers.map((opt) =>
          fetchOptimise({
            surface,
            optimiser: opt,
            x0,
            y0,
            num_steps: numSteps,
            lr,
            momentum,
          })
        )
      );

      const newTrajs: TrajectoryState[] = results.map((res, i) => ({
        points: res.trajectory,
        name: OPTIMISER_LABELS[optimisers[i]],
        color: OPTIMISER_COLORS[optimisers[i]],
        animIndex: 0,
      }));

      trajRef.current = newTrajs;
      setTrajectories(newTrajs);
      animFrameRef.current = requestAnimationFrame(() => animateFnRef.current());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
      setRunning(false);
    }
  }, [
    surface,
    optimiser1,
    optimiser2,
    lr,
    momentum,
    numSteps,
    x0,
    y0,
    sideBySide,
  ]);

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-10 max-w-7xl mx-auto">
      <div className="mb-4 lg:mb-8 flex items-center gap-3 lg:gap-4">
        <img src="/logo.svg" alt="Saddle" className="h-10 lg:h-[4.2rem]" />
        <p className="text-xs lg:text-sm text-ctp-subtext0">
          Interactive optimiser visualisation
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-ctp-surface0 border border-ctp-red/30 rounded text-ctp-red text-sm">
          {error}
        </div>
      )}

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Controls */}
        <div className="w-full lg:w-60 shrink-0">
          <Controls
            surface={surface}
            setSurface={setSurface}
            optimiser1={optimiser1}
            setOptimiser1={setOptimiser1}
            optimiser2={optimiser2}
            setOptimiser2={setOptimiser2}
            lr={lr}
            setLr={setLr}
            momentum={momentum}
            setMomentum={setMomentum}
            numSteps={numSteps}
            setNumSteps={setNumSteps}
            x0={x0}
            setX0={setX0}
            y0={y0}
            setY0={setY0}
            sideBySide={sideBySide}
            setSideBySide={setSideBySide}
            onRun={handleRun}
            running={running}
            animSpeed={animSpeed}
            setAnimSpeed={setAnimSpeed}
            showGradients={showGradients}
            setShowGradients={setShowGradients}
            surfaceDescription={surfaceInfoMap[surface]?.description ?? null}
            surfaceFormula={surfaceInfoMap[surface]?.formula ?? null}
            viewMode={viewMode}
            setViewMode={setViewMode}
          />
        </div>

        {/* Plots */}
        <div className="flex-1 flex flex-col gap-4">
          <div className="border border-ctp-surface1 rounded-lg h-[350px] md:h-[500px] lg:h-[600px]">
            <SurfacePlot
              surface={surfaceData}
              trajectories={trajectories}
              gradientField={gradientField}
              title={surfaceInfoMap[surface]?.name ?? surface}
              viewMode={viewMode}
              onPointClick={handlePlotClick}
              startX={x0}
              startY={y0}
            />
          </div>

          {trajectories.length > 0 && (
            <div className="border border-ctp-surface1 rounded-lg h-[200px] md:h-[250px]">
              <ConvergencePlot trajectories={trajectories} />
            </div>
          )}

          <BenchmarkPanel />
        </div>
      </div>
    </main>
  );
}
