"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Controls from "@/components/Controls";
import SurfacePlot from "@/components/SurfacePlot";
import ConvergencePlot from "@/components/ConvergencePlot";
import BenchmarkPanel from "@/components/BenchmarkPanel";
import {
  fetchSurface,
  fetchOptimise,
  type SurfaceName,
  type OptimiserName,
  type SurfaceResponse,
  type TrajectoryPoint,
} from "@/lib/api";

const OPTIMISER_COLORS: Record<OptimiserName, string> = {
  sgd: "#e11d48",
  adam: "#2563eb",
  adahessian: "#7c3aed",
  c_adam: "#059669",
};

const OPTIMISER_LABELS: Record<OptimiserName, string> = {
  sgd: "SGD",
  adam: "Adam",
  adahessian: "AdaHessian",
  c_adam: "Adam (C)",
};

interface TrajectoryState {
  points: TrajectoryPoint[];
  name: string;
  color: string;
  animIndex: number;
}

export default function Home() {
  const [surface, setSurface] = useState<SurfaceName>("rosenbrock");
  const [optimiser1, setOptimiser1] = useState<OptimiserName>("adam");
  const [optimiser2, setOptimiser2] = useState<OptimiserName | null>("sgd");
  const [lr, setLr] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [numSteps, setNumSteps] = useState(500);
  const [x0, setX0] = useState(-1.0);
  const [y0, setY0] = useState(1.0);
  const [sideBySide, setSideBySide] = useState(true);
  const [animSpeed, setAnimSpeed] = useState(5);

  const [surfaceData, setSurfaceData] = useState<SurfaceResponse | null>(null);
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

  // Load surface when selection changes
  useEffect(() => {
    let cancelled = false;
    setSurfaceData(null);
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

  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
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
    <main className="min-h-screen p-6 md:p-10 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-black">Saddle</h1>
        <p className="text-sm text-neutral-400 mt-1">
          Interactive optimiser visualisation
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
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
          />
          <div className="mt-4">
            <BenchmarkPanel />
          </div>
        </div>

        {/* Plots */}
        <div className="flex-1 flex flex-col gap-4">
          <div className="border border-neutral-200 rounded-lg h-[500px] lg:h-[600px]">
            <SurfacePlot
              surface={surfaceData}
              trajectories={trajectories}
              title={`${surface.charAt(0).toUpperCase() + surface.slice(1)} surface`}
            />
          </div>

          {trajectories.length > 0 && (
            <div className="border border-neutral-200 rounded-lg h-[250px]">
              <ConvergencePlot trajectories={trajectories} />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
