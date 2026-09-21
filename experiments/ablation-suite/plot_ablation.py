#!/usr/bin/env python3
"""Render the Q-SPA optimization-step ablation with the unified chart style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import SimplePatchShadow, withStroke


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments/ablation-suite/raw/qspa-ablation.json"
OUTPUT = ROOT / "sections/04-evaluation/assets/qspa-ablation.svg"


def main() -> None:
    records = json.loads(INPUT.read_text(encoding="utf-8"))["records"]
    labels = [record["label"] for record in records]
    generation = np.array([record["generation_seconds"] for record in records], dtype=float)
    memory = np.array([record["peak_memory_gib"] for record in records], dtype=float)
    video_rate = 5.0 / generation
    x = np.arange(len(records), dtype=float)

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(13.8, 6.8), facecolor="white")
    axis.set_facecolor("white")
    colors = ["#8FA6B6", "#D28A4F", "#168A72", "#168A72"]
    bars = axis.bar(x, memory, width=0.48, color=colors, alpha=0.92, zorder=2)
    axis.set_ylabel("Peak GPU memory (GiB)", color="#344054", labelpad=9)
    axis.set_ylim(0, max(memory) * 1.30)
    axis.set_xticks(x, labels, fontsize=10.5)
    axis.tick_params(axis="x", length=0, pad=10)
    axis.tick_params(axis="y", colors="#667085", labelsize=10)
    axis.grid(axis="y", color="#E7EBEF", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    for index, (bar, value, record) in enumerate(zip(bars, memory, records, strict=True)):
        if index >= 2:
            bar.set_edgecolor("#075E4C")
            bar.set_linewidth(1.8)
            bar.set_path_effects([SimplePatchShadow(offset=(2, -2), alpha=0.3), withStroke(linewidth=2.5, foreground="#075E4C")])
        axis.text(bar.get_x() + bar.get_width() / 2, value + max(memory) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=10, color="#344054")

    rate_axis = axis.twinx()
    rate_axis.plot(x, video_rate, color="#C93F4B", linewidth=1.6, zorder=4)
    for point_x, value, index in zip(x, video_rate, range(len(records)), strict=True):
        marker = "*" if index >= 2 else "o"
        rate_axis.plot(point_x, value, marker=marker, color="#C93F4B", markersize=12 if marker == "*" else 5, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        rate_axis.text(point_x, value + max(video_rate) * 0.045, f"{value:.3f}", ha="center", va="bottom", fontsize=9.5, color="#C93F4B")
    rate_axis.set_ylim(0, max(video_rate) * 1.28)
    rate_axis.tick_params(axis="y", colors="#C93F4B", labelsize=10)
    rate_axis.spines["top"].set_visible(False)
    rate_axis.spines["right"].set_color("#C93F4B")
    rate_axis.set_ylabel("Generated video seconds / wall-clock second", color="#C93F4B", labelpad=9)
    axis.annotate("↓ lower is better", xy=(0.015, 0.97), xycoords="axes fraction", color="#344054", fontsize=10.5, ha="left", va="top")
    rate_axis.annotate("↑ higher is better", xy=(0.985, 0.97), xycoords="axes fraction", color="#C93F4B", fontsize=10.5, ha="right", va="top")

    handles = [
        Patch(facecolor="#8FA6B6", edgecolor="none", label="1 GPU"),
        Patch(facecolor="#D28A4F", edgecolor="none", label="4 GPUs"),
        Line2D([0], [0], color="#C93F4B", marker="o", linewidth=1.6, markersize=5, label="Video rate"),
        Line2D([0], [0], color="#C93F4B", marker="*", linewidth=1.6, markersize=10, label="Adapter video rate"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=10.5, columnspacing=1.5)
    fig.text(0.5, 0.065, "MiniMax-H3 Base / Turbo LoRA · 1344 × 768 · 124 frames · 5 s · T2AV · 50 Base steps / 8 Turbo steps · no CPU offload", ha="center", color="#69747D", fontsize=9.5)
    fig.subplots_adjust(left=0.075, right=0.94, top=0.94, bottom=0.22)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
