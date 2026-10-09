"""Plot the matched backend matrix using optional Matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = ("sparse", "icp", "hybrid")
COLORS = ("#3167a8", "#d47d20", "#8460a8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path)
    args = parser.parse_args()
    data = json.loads((args.comparison_dir / "comparison.json").read_text())
    ranges = data["ranges"]
    labels = [f"{row['source']['start']}–{row['source']['stop']}" for row in ranges]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    titles = (
        "Common 10-frame RPE translation",
        "Common 10-frame RPE rotation",
        "Full-clip endpoint translation drift",
        "End-to-end tracking throughput",
    )
    units = ("RMSE (mm)", "RMSE (degrees)", "Error (mm)", "Frames per second")
    x = np.arange(len(ranges))
    for panel, ax in enumerate(axes.flat):
        for offset, (name, color) in enumerate(zip(NAMES, COLORS)):
            values = []
            for row in ranges:
                if panel in (0, 1):
                    metric = "translation_m" if panel == 0 else "rotation_deg"
                    value = row["common_rpe"]["10"][name][metric]["rmse"]
                    value = (
                        None if value is None else value * (1000 if panel == 0 else 1)
                    )
                elif panel == 2:
                    segment = row["runs"][name]["evaluation"]["continuous_trajectory"]
                    value = (
                        None
                        if segment is None
                        else segment["endpoint_translation_error_m"]
                    )
                    value = None if value is None else value * 1000
                else:
                    value = row["runs"][name]["timing"]["frames_per_second"]
                values.append(np.nan if value is None else value)
            positions = x + (offset - 1) * 0.25
            ax.bar(positions, values, 0.23, color=color, label=name)
            for position, value in zip(positions, values):
                if not np.isfinite(value):
                    ax.text(
                        position,
                        0,
                        "×",
                        ha="center",
                        va="bottom",
                        color=color,
                        fontsize=15,
                    )
        ax.set(title=titles[panel], ylabel=units[panel], xticks=x, xticklabels=labels)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("SceneRecall RGB-D backends · scene0000_00", fontsize=16)
    fig.supxlabel(
        "Source frame ranges · × = full-clip drift unavailable after tracking loss"
    )
    for suffix in ("png", "svg"):
        fig.savefig(args.comparison_dir / f"backend_comparison.{suffix}", dpi=180)
    plt.close(fig)
    outcomes = [
        item
        for row in ranges
        for item in row["hybrid_refinement_outcomes"]
        if item["depth_rmse_initial_m"] is not None
        and item["depth_rmse_final_m"] is not None
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for dimension, ax in enumerate(axes):
        for boundary, color, label in (
            (False, "#3167a8", "Other targets"),
            (True, "#d47d20", "Target index divisible by 10"),
        ):
            selected = [
                item
                for item in outcomes
                if (item["source_target_index"] % 10 == 0) == boundary
            ]
            ax.scatter(
                [
                    (item["depth_rmse_final_m"] - item["depth_rmse_initial_m"]) * 1000
                    for item in selected
                ],
                [
                    (
                        item["final_error_m_deg"][dimension]
                        - item["initial_error_m_deg"][dimension]
                    )
                    * (1000 if dimension == 0 else 1)
                    for item in selected
                ],
                s=14,
                alpha=0.5,
                color=color,
                label=label,
                edgecolors="none",
            )
        ax.axhline(0, color="#555555", lw=0.8)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set(
            xlabel="Change in common held-out depth RMSE (mm)",
            ylabel="Change in reference error (mm)"
            if dimension == 0
            else "Change in reference error (degrees)",
            title="Translation" if dimension == 0 else "Rotation",
        )
        ax.text(
            0.03,
            0.95,
            "Above zero: pose error worsened",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle(
        "Hybrid refinement: lower depth residual can increase pose error", fontsize=14
    )
    for suffix in ("png", "svg"):
        fig.savefig(args.comparison_dir / f"refinement_tradeoff.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
