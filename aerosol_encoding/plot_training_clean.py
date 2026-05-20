from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_LABELS = {
    "instrument_denoise_token_pretrain": "token pretrain",
    "deterministic_autoencode_warmup": "autoencode",
    "deterministic_denoise_autoencode": "denoise AE",
    "mask_one_mild": "mask-one mild",
    "mask_one_denoise": "mask-one denoise",
    "mask_one_refine": "mask-one refine",
    "ccn_hidden_refine": "CCN refine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clean aerosol encoder training diagnostics."
    )
    parser.add_argument("--history", required=True, help="history.csv from train.py")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Aerosol encoder training diagnostics")
    return parser.parse_args()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def add_stage_spans(axes: list[plt.Axes], history: pd.DataFrame) -> None:
    if "stage" not in history:
        return
    stage_groups = list(history.groupby("stage", sort=False))
    for index, (stage, group) in enumerate(stage_groups):
        start = float(group["epoch"].min())
        end = float(group["epoch"].max())
        color = "#eaf0f7" if index % 2 == 0 else "#f6efe7"
        for axis in axes:
            axis.axvspan(start, end, color=color, alpha=0.58, zorder=0)
            axis.axvline(start, color="0.70", linewidth=0.9, zorder=1)
        label = STAGE_LABELS.get(str(stage), str(stage).replace("_", " "))
        axes[0].text(
            0.5 * (start + end),
            0.965,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.25",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.5,
            },
        )


def main() -> None:
    args = parse_args()
    history = pd.read_csv(args.history)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    history["epoch"] = numeric(history, "epoch")
    history["train_loss"] = numeric(history, "train_loss")
    validation = (
        numeric(history, "validation_cross_loss")
        if "validation_cross_loss" in history
        else pd.Series(np.nan, index=history.index)
    )
    validation_points = history.loc[validation.notna(), ["epoch"]].copy()
    validation_points["validation_cross_loss"] = validation[validation.notna()]

    modality_columns = [
        column
        for column in history.columns
        if column.startswith("validation_cross_loss_")
        and column != "validation_cross_loss"
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.0, 10.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.05, 1.25]},
    )
    add_stage_spans(list(axes), history)

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        color="#1f77b4",
        linewidth=1.9,
        label="training objective",
    )
    axes[0].set_ylabel("train objective")
    axes[0].set_title(args.title)
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(
        0.01,
        0.08,
        "Training objective changes by stage; do not compare its scale directly to validation MSE.",
        transform=axes[0].transAxes,
        fontsize=8,
        color="0.30",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.0},
    )

    if not validation_points.empty:
        axes[1].plot(
            validation_points["epoch"],
            validation_points["validation_cross_loss"],
            color="#2ca02c",
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label="selected cross-prediction validation",
        )
        best_index = validation_points["validation_cross_loss"].idxmin()
        best_epoch = float(validation_points.loc[best_index, "epoch"])
        best_value = float(validation_points.loc[best_index, "validation_cross_loss"])
        latest_epoch = float(validation_points.iloc[-1]["epoch"])
        latest_value = float(validation_points.iloc[-1]["validation_cross_loss"])
        axes[1].scatter([best_epoch], [best_value], color="black", s=42, zorder=5)
        axes[1].scatter([latest_epoch], [latest_value], color="#d62728", s=42, zorder=5)
        axes[1].annotate(
            f"best epoch {int(best_epoch)}\n{best_value:.4f}",
            xy=(best_epoch, best_value),
            xytext=(8, -25),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.8},
        )
        axes[1].annotate(
            f"latest epoch {int(latest_epoch)}\n{latest_value:.4f}",
            xy=(latest_epoch, latest_value),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            color="#9a1f1f",
            arrowprops={"arrowstyle": "-", "color": "#d62728", "linewidth": 0.8},
        )
        y_min = float(validation_points["validation_cross_loss"].min())
        y_max = float(validation_points["validation_cross_loss"].max())
        padding = max(0.003, 0.12 * (y_max - y_min))
        axes[1].set_ylim(y_min - padding, y_max + padding)
    axes[1].set_ylabel("validation MSE")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(frameon=False, loc="upper right")

    for column in modality_columns:
        series = numeric(history, column)
        points = history.loc[series.notna(), ["epoch"]].copy()
        if points.empty:
            continue
        points[column] = series[series.notna()]
        label = column.replace("validation_cross_loss_", "")
        axes[2].plot(
            points["epoch"],
            points[column],
            linewidth=1.45,
            marker="o",
            markersize=3.0,
            label=label,
        )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("per-modality validation MSE")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, which="both", alpha=0.22)
    axes[2].legend(frameon=False, ncol=4, loc="upper center")

    for axis in axes:
        axis.set_xlim(float(history["epoch"].min()), float(history["epoch"].max()))

    fig.savefig(output, dpi=200)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
