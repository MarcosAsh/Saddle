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
        paper_bgcolor: "#1e1e2e",
        plot_bgcolor: "#181825",
        font: { color: "#a6adc8", size: 11, family: "var(--font-geist-sans), sans-serif" },
        xaxis: {
          title: { text: "Step", font: { size: 11, color: "#9399b2" } },
          gridcolor: "#45475a",
          zerolinecolor: "#585b70",
          linecolor: "#585b70",
          tickfont: { color: "#9399b2" },
        },
        yaxis: {
          title: { text: "log\u2081\u2080(loss)", font: { size: 11, color: "#9399b2" } },
          gridcolor: "#45475a",
          zerolinecolor: "#585b70",
          linecolor: "#585b70",
          tickfont: { color: "#9399b2" },
        },
        legend: {
          x: 1,
          xanchor: "right",
          y: 1,
          bgcolor: "rgba(24,24,37,0.9)",
          bordercolor: "#45475a",
          borderwidth: 1,
          font: { color: "#cdd6f4" },
        },
        margin: { l: 55, r: 15, t: 10, b: 45 },
        autosize: true,
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
