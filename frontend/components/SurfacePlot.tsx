"use client";

import { useMemo } from "react";
import PlotlyChart from "./PlotlyChart";
import type { SurfaceResponse, TrajectoryPoint } from "@/lib/api";

interface SurfacePlotProps {
  surface: SurfaceResponse | null;
  trajectories: {
    points: TrajectoryPoint[];
    name: string;
    color: string;
    animIndex: number;
  }[];
  title?: string;
}

export default function SurfacePlot({
  surface,
  trajectories,
  title,
}: SurfacePlotProps) {
  const { xs, ys, zLog, zMin, zMax } = useMemo(() => {
    if (!surface) return { xs: [], ys: [], zLog: [], zMin: -10, zMax: 5 };
    const cols = surface.cols;
    const rows = surface.rows;
    const xArr = Array.from({ length: cols }, (_, j) =>
      surface.x_min + (j / (cols - 1)) * (surface.x_max - surface.x_min)
    );
    const yArr = Array.from({ length: rows }, (_, i) =>
      surface.y_min + (i / (rows - 1)) * (surface.y_max - surface.y_min)
    );
    let lo = Infinity;
    let hi = -Infinity;
    const zArr = surface.values.map((row) =>
      row.map((v) => {
        const l = Math.log10(Math.max(v, 1e-10));
        if (l < lo) lo = l;
        if (l > hi) hi = l;
        return l;
      })
    );
    return { xs: xArr, ys: yArr, zLog: zArr, zMin: lo, zMax: hi };
  }, [surface]);

  const data: Plotly.Data[] = useMemo(() => {
    const d: Plotly.Data[] = [];

    if (surface) {
      d.push({
        type: "surface",
        x: xs,
        y: ys,
        z: zLog,
        colorscale: "Viridis",
        showscale: false,
        opacity: 0.85,
        contours: {
          z: {
            show: true,
            usecolormap: true,
            project: { z: false },
            highlightcolor: "#fff",
            highlightwidth: 1,
          },
        },
        hoverinfo: "skip",
        lighting: {
          ambient: 0.6,
          diffuse: 0.5,
          specular: 0.2,
          roughness: 0.5,
        },
      } as Plotly.Data);
    }

    for (const traj of trajectories) {
      const visible = traj.points.slice(0, traj.animIndex + 1);
      if (visible.length === 0) continue;

      const trajZ = visible.map((p) =>
        Math.log10(Math.max(p.loss ?? 1e-10, 1e-10)) + 0.1
      );

      d.push({
        type: "scatter3d",
        mode: "lines",
        x: visible.map((p) => p.x),
        y: visible.map((p) => p.y),
        z: trajZ,
        name: traj.name,
        line: { color: traj.color, width: 6 },
      } as Plotly.Data);

      d.push({
        type: "scatter3d",
        mode: "markers",
        x: [visible[0].x],
        y: [visible[0].y],
        z: [trajZ[0]],
        marker: { size: 4, color: traj.color, symbol: "diamond" },
        showlegend: false,
        hoverinfo: "skip",
      } as Plotly.Data);

      if (visible.length > 1) {
        const last = visible[visible.length - 1];
        const lastZ = trajZ[trajZ.length - 1];
        d.push({
          type: "scatter3d",
          mode: "markers",
          x: [last.x],
          y: [last.y],
          z: [lastZ],
          marker: { size: 4, color: traj.color },
          showlegend: false,
          hoverinfo: "text",
          text: [
            `loss: ${last.loss != null ? last.loss.toExponential(3) : "N/A"}`,
          ],
        } as Plotly.Data);
      }
    }

    return d;
  }, [surface, xs, ys, zLog, trajectories]);

  return (
    <PlotlyChart
      data={data}
      layout={{
        title: title
          ? {
              text: title,
              font: {
                color: "#000",
                size: 14,
                family: "var(--font-geist-sans), sans-serif",
              },
            }
          : undefined,
        paper_bgcolor: "white",
        font: {
          color: "#525252",
          family: "var(--font-geist-sans), sans-serif",
        },
        uirevision: "stable",
        scene: {
          xaxis: {
            title: { text: "x", font: { size: 11 } },
            range: surface ? [surface.x_min, surface.x_max] : undefined,
            gridcolor: "#e5e5e5",
            backgroundcolor: "white",
            showbackground: true,
            tickfont: { color: "#737373", size: 10 },
          },
          yaxis: {
            title: { text: "y", font: { size: 11 } },
            range: surface ? [surface.y_min, surface.y_max] : undefined,
            gridcolor: "#e5e5e5",
            backgroundcolor: "white",
            showbackground: true,
            tickfont: { color: "#737373", size: 10 },
          },
          zaxis: {
            title: { text: "log\u2081\u2080(loss)", font: { size: 11 } },
            range: [zMin, zMax + 0.5],
            gridcolor: "#e5e5e5",
            backgroundcolor: "#fafafa",
            showbackground: true,
            tickfont: { color: "#737373", size: 10 },
          },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.0 },
            up: { x: 0, y: 0, z: 1 },
          },
          aspectmode: "cube",
        },
        legend: {
          x: 0,
          y: 1,
          bgcolor: "rgba(255,255,255,0.9)",
          bordercolor: "#e5e5e5",
          borderwidth: 1,
          font: { color: "#000" },
        },
        margin: { l: 0, r: 0, t: 40, b: 0 },
        autosize: true,
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
