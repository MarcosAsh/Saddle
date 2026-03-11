"use client";

import PlotlyChart from "./PlotlyChart";
import type { TrajectoryPoint } from "@/lib/api";

interface ConvergencePlotProps {
  trajectories: {
    points: TrajectoryPoint[];
    name: string;
    color: string;
    animIndex: number;
  }[];
}

export default function ConvergencePlot({ trajectories }: ConvergencePlotProps) {
  const data: Plotly.Data[] = trajectories.map((traj) => {
    const visible = traj.points.slice(0, traj.animIndex + 1);
    return {
      type: "scatter",
      mode: "lines",
      x: visible.map((_, i) => i),
      y: visible.map((p) => Math.log10(Math.max(p.loss ?? 1e-20, 1e-20))),
      name: traj.name,
      line: { color: traj.color, width: 2 },
    } as Plotly.Data;
  });

  return (
    <PlotlyChart
      data={data}
      layout={{
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        font: { color: "#525252", size: 11, family: "var(--font-geist-sans), sans-serif" },
        xaxis: {
          title: { text: "Step", font: { size: 11, color: "#737373" } },
          gridcolor: "#e5e5e5",
          zerolinecolor: "#d4d4d4",
          linecolor: "#d4d4d4",
          tickfont: { color: "#737373" },
        },
        yaxis: {
          title: { text: "log\u2081\u2080(loss)", font: { size: 11, color: "#737373" } },
          gridcolor: "#e5e5e5",
          zerolinecolor: "#d4d4d4",
          linecolor: "#d4d4d4",
          tickfont: { color: "#737373" },
        },
        legend: {
          x: 1,
          xanchor: "right",
          y: 1,
          bgcolor: "rgba(255,255,255,0.9)",
          bordercolor: "#e5e5e5",
          borderwidth: 1,
          font: { color: "#000" },
        },
        margin: { l: 55, r: 15, t: 10, b: 45 },
        autosize: true,
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
