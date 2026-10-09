"""Export depth-experiment charts with optional Matplotlib, outside production.

Usage: uv run --with matplotlib python experiments/plot_depth_registration.py RUN
The registration experiment itself requires only the repository dependencies.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    periods = report["periodicity"]
    fig, axes = plt.subplots(
        len(periods),
        1,
        figsize=(10, 2.2 * len(periods)),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, period in zip(np.atleast_1d(axes), periods):
        values = [
            period["by_absolute_target_mod_10"][str(i)]["rotation_deg"]["rmse"]
            for i in range(10)
        ]
        ax.bar(range(10), values, color=["#bf4034"] + ["#397da5"] * 9)
        ax.set_title(Path(period["run"]).name, loc="left")
        ax.set_ylabel("rotation RMSE (°)")
        ax.set_ylim(0, 0.55)
        for x, value in enumerate(values):
            ax.text(x, value + 0.007, f"{value:.3f}", ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.15)
    np.atleast_1d(axes)[-1].set(
        xticks=range(10), xlabel="absolute target source-frame index modulo 10"
    )
    fig.savefig(args.run / "periodicity.png")
    fig.savefig(args.run / "periodicity.svg")
    plt.close(fig)

    suspects = [p for p in report["pairs"] if "weak_mode_heldout_profile" in p]
    fig, axes = plt.subplots(
        1, len(suspects), figsize=(12, 3.8), layout="constrained", squeeze=False
    )
    for ax, pair in zip(axes[0], suspects):
        profile = pair["weak_mode_heldout_profile"]
        amounts = np.array(profile["amounts_scaled"])
        residual = profile["common_rmse_mm"]
        ax.plot(amounts, residual, color="#397da5")
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.scatter([0], [residual[len(residual) // 2]], color="#bf4034", s=25, zorder=3)
        ax.set(
            title=f"{pair['source']} → {pair['target']}",
            xlabel="offset along weakest scaled pose direction",
            ylabel="held-out plane RMSE (mm)",
        )
        ax.grid(alpha=0.2)
        ax.ticklabel_format(axis="x", style="plain")
    fig.savefig(args.run / "weak_mode_profiles.png")
    fig.savefig(args.run / "weak_mode_profiles.svg")
    plt.close(fig)

    pairs = [p for p in report["pairs"] if p["target"] < 200]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = np.arange(len(pairs))
    for label, color, offset in [
        ("VO", "#777777", -0.24),
        ("reference", "#bf4034", 0),
        ("ICP from VO", "#397da5", 0.24),
    ]:
        values = [
            p["forward_scores"][label]["common_support"]["signed_point_to_plane_mm"][
                "rmse"
            ]
            for p in pairs
        ]
        axes[0].bar(x + offset, values, width=0.24, color=color, label=label)
    axes[0].set(
        ylabel="held-out plane RMSE (mm)",
        title="Same source samples, support shared by all four poses",
    )
    axes[0].legend(fontsize=8)
    for label, color in [
        ("error_to_reference_m_deg", "#bf4034"),
        ("error_to_vo_m_deg", "#777777"),
    ]:
        values = [p["registrations"]["forward"]["vo"][label][0] * 1000 for p in pairs]
        axes[1].plot(
            x,
            values,
            "o-",
            color=color,
            label="ICP distance to " + ("GT" if "reference" in label else "VO"),
        )
    axes[1].set(
        ylabel="translation difference (mm)",
        title="ICP agrees more closely with VO on the suspect pairs",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, [f"{p['source']}→{p['target']}" for p in pairs], rotation=50)
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(args.run / "registration_comparison.png")
    fig.savefig(args.run / "registration_comparison.svg")
    plt.close(fig)
    (args.run / "plot_metadata.json").write_text(
        json.dumps(
            {
                "matplotlib": matplotlib.__version__,
                "report_sha256": hashlib.sha256(
                    (args.run / "report.json").read_bytes()
                ).hexdigest(),
                "plot_script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
