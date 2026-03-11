"use client";

import { useMemo } from "react";
import PlotlyChart from "./PlotlyChart";
import type { SurfaceResponse, TrajectoryPoint, GradientFieldResponse } from "@/lib/api";

interface SurfacePlotProps {
  surface: SurfaceResponse | null;
  trajectories: {
    points: TrajectoryPoint[];
    name: string;
    color: string;
    animIndex: number;
  }[];
  gradientField: GradientFieldResponse | null;
  title?: string;
  viewMode: "3d" | "contour";
}

export default function SurfacePlot({
  surface,
  trajectories,
  gradientField,
  title,
  viewMode,
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

  // --- 3D traces ---
  const data3d: Plotly.Data[] = useMemo(() => {
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

    // Gradient field as 3D cones
    if (gradientField && surface) {
      const gf = gradientField;
      const px: number[] = [];
      const py: number[] = [];
      const pz: number[] = [];
      const ux: number[] = [];
      const uy: number[] = [];
      const uz: number[] = [];

      for (let i = 0; i < gf.y.length; i++) {
        for (let j = 0; j < gf.x.length; j++) {
          const gxi = gf.gx[i][j];
          const gyi = gf.gy[i][j];
          const mag = Math.sqrt(gxi * gxi + gyi * gyi);
          if (mag < 1e-10) continue;

          px.push(gf.x[j]);
          py.push(gf.y[i]);
          const sv = surface.values;
          const ci = Math.round(
            ((gf.y[i] - surface.y_min) / (surface.y_max - surface.y_min)) *
              (sv.length - 1)
          );
          const cj = Math.round(
            ((gf.x[j] - surface.x_min) / (surface.x_max - surface.x_min)) *
              (sv[0].length - 1)
          );
          const zVal =
            ci >= 0 && ci < sv.length && cj >= 0 && cj < sv[0].length
              ? Math.log10(Math.max(sv[ci][cj], 1e-10))
              : 0;
          pz.push(zVal + 0.15);

          ux.push(-gxi / mag);
          uy.push(-gyi / mag);
          uz.push(0);
        }
      }

      if (px.length > 0) {
        d.push({
          type: "cone",
          x: px,
          y: py,
          z: pz,
          u: ux,
          v: uy,
          w: uz,
          sizemode: "absolute",
          sizeref: 0.06,
          anchor: "tail",
          showscale: false,
          colorscale: [[0, "#ef4444"], [1, "#ef4444"]],
          opacity: 0.4,
          name: "Gradient",
          hoverinfo: "skip",
        } as Plotly.Data);
      }
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
  }, [surface, xs, ys, zLog, trajectories, gradientField]);

  // --- 2D contour traces ---
  const data2d: Plotly.Data[] = useMemo(() => {
    const d: Plotly.Data[] = [];

    if (surface) {
      d.push({
        type: "contour",
        x: xs,
        y: ys,
        z: zLog,
        colorscale: "Viridis",
        showscale: false,
        ncontours: 30,
        contours: {
          coloring: "heatmap",
        },
        hoverinfo: "skip",
        line: { width: 0.5, color: "rgba(255,255,255,0.15)" },
      } as Plotly.Data);
    }

    // Gradient field as line-segment arrows
    if (gradientField && surface) {
      const gf = gradientField;
      const ax: (number | null)[] = [];
      const ay: (number | null)[] = [];
      const xRange = surface.x_max - surface.x_min;
      const yRange = surface.y_max - surface.y_min;
      const scale = Math.min(xRange, yRange) / gf.x.length * 0.4;

      for (let i = 0; i < gf.y.length; i++) {
        for (let j = 0; j < gf.x.length; j++) {
          const gxi = gf.gx[i][j];
          const gyi = gf.gy[i][j];
          const mag = Math.sqrt(gxi * gxi + gyi * gyi);
          if (mag < 1e-10) continue;

          const x0 = gf.x[j];
          const y0 = gf.y[i];
          const dx = (-gxi / mag) * scale;
          const dy = (-gyi / mag) * scale;

          ax.push(x0, x0 + dx, null);
          ay.push(y0, y0 + dy, null);
        }
      }

      if (ax.length > 0) {
        d.push({
          type: "scatter",
          mode: "lines",
          x: ax,
          y: ay,
          line: { color: "rgba(239,68,68,0.5)", width: 1 },
          showlegend: false,
          hoverinfo: "skip",
        } as Plotly.Data);
      }
    }

    for (const traj of trajectories) {
      const visible = traj.points.slice(0, traj.animIndex + 1);
      if (visible.length === 0) continue;

      d.push({
        type: "scatter",
        mode: "lines",
        x: visible.map((p) => p.x),
        y: visible.map((p) => p.y),
        name: traj.name,
        line: { color: traj.color, width: 3 },
      } as Plotly.Data);

      // Start marker
      d.push({
        type: "scatter",
        mode: "markers",
        x: [visible[0].x],
        y: [visible[0].y],
        marker: { size: 8, color: traj.color, symbol: "diamond" },
        showlegend: false,
        hoverinfo: "skip",
      } as Plotly.Data);

      // Current position marker
      if (visible.length > 1) {
        const last = visible[visible.length - 1];
        d.push({
          type: "scatter",
          mode: "markers",
          x: [last.x],
          y: [last.y],
          marker: { size: 7, color: traj.color },
          showlegend: false,
          hoverinfo: "text",
          text: [
            `loss: ${last.loss != null ? last.loss.toExponential(3) : "N/A"}`,
          ],
        } as Plotly.Data);
      }
    }

    return d;
  }, [surface, xs, ys, zLog, trajectories, gradientField]);

  const titleConfig = title
    ? {
        text: title,
        font: {
          color: "#cdd6f4",
          size: 14,
          family: "var(--font-geist-sans), sans-serif",
        },
      }
    : undefined;

  const layout3d: Partial<Plotly.Layout> = {
    title: titleConfig,
    paper_bgcolor: "#1e1e2e",
    font: {
      color: "#a6adc8",
      family: "var(--font-geist-sans), sans-serif",
    },
    uirevision: "stable-3d",
    scene: {
      xaxis: {
        title: { text: "x", font: { size: 11 } },
        range: surface ? [surface.x_min, surface.x_max] : undefined,
        gridcolor: "#45475a",
        backgroundcolor: "#181825",
        showbackground: true,
        tickfont: { color: "#9399b2", size: 10 },
      },
      yaxis: {
        title: { text: "y", font: { size: 11 } },
        range: surface ? [surface.y_min, surface.y_max] : undefined,
        gridcolor: "#45475a",
        backgroundcolor: "#181825",
        showbackground: true,
        tickfont: { color: "#9399b2", size: 10 },
      },
      zaxis: {
        title: { text: "log\u2081\u2080(loss)", font: { size: 11 } },
        range: [zMin, zMax + 0.5],
        gridcolor: "#45475a",
        backgroundcolor: "#181825",
        showbackground: true,
        tickfont: { color: "#9399b2", size: 10 },
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
      bgcolor: "rgba(24,24,37,0.9)",
      bordercolor: "#45475a",
      borderwidth: 1,
      font: { color: "#cdd6f4" },
    },
    margin: { l: 0, r: 0, t: 40, b: 0 },
    autosize: true,
  };

  const layout2d: Partial<Plotly.Layout> = {
    title: titleConfig,
    paper_bgcolor: "#1e1e2e",
    plot_bgcolor: "#181825",
    font: {
      color: "#a6adc8",
      family: "var(--font-geist-sans), sans-serif",
    },
    uirevision: "stable-2d",
    xaxis: {
      title: { text: "x", font: { size: 11 } },
      range: surface ? [surface.x_min, surface.x_max] : undefined,
      gridcolor: "#45475a",
      tickfont: { color: "#9399b2", size: 10 },
      scaleanchor: "y",
      constrain: "domain",
    },
    yaxis: {
      title: { text: "y", font: { size: 11 } },
      range: surface ? [surface.y_min, surface.y_max] : undefined,
      gridcolor: "#45475a",
      tickfont: { color: "#9399b2", size: 10 },
      constrain: "domain",
    },
    legend: {
      x: 0,
      y: 1,
      bgcolor: "rgba(24,24,37,0.9)",
      bordercolor: "#45475a",
      borderwidth: 1,
      font: { color: "#cdd6f4" },
    },
    margin: { l: 50, r: 20, t: 40, b: 50 },
    autosize: true,
  };

  const is3d = viewMode === "3d";

  return (
    <PlotlyChart
      data={is3d ? data3d : data2d}
      layout={is3d ? layout3d : layout2d}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
