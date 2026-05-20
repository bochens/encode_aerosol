from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import pandas as pd


STAGE_LABELS = {
    "deterministic_autoencode_warmup": "autoencode",
    "deterministic_denoise_autoencode": "denoise",
    "mixed_mild_mask": "mild mask",
    "leave_one_out_hidden_only": "leave-one",
    "leave_one_group_out_hidden_only": "group-out",
    "continue_leave_one_out_hidden_only": "extra leave-one",
    "continue_leave_one_group_out_hidden_only": "extra group-out",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aerosol encoder training curves.")
    parser.add_argument("--history", required=True, help="history.csv from train.py.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--title", default="Grouped aerosol encoder training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = pd.read_csv(args.history)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train", linewidth=2)

    if "validation_loss" in history:
        axes[0].plot(history["epoch"], history["validation_loss"], label="validation", linewidth=2)
        best_col = "validation_loss"
    else:
        best_col = "train_loss"
        if "validation_reconstruction_loss" in history:
            axes[0].plot(
                history["epoch"],
                history["validation_reconstruction_loss"],
                label="validation reconstruction",
                linewidth=2,
                marker="o",
                markersize=3.5,
            )
        if "validation_cross_loss" in history:
            axes[0].plot(
                history["epoch"],
                history["validation_cross_loss"],
                label="validation selected cross-prediction",
                linewidth=2,
                marker="o",
                markersize=3.5,
            )
            best_col = "validation_cross_loss"
        if "validation_strict_group_out_loss" in history:
            axes[0].plot(
                history["epoch"],
                history["validation_strict_group_out_loss"],
                label="validation strict all-sizing-hidden diagnostic",
                linewidth=1.5,
                linestyle="--",
                marker="s",
                markersize=3.5,
            )

    best_idx = history[best_col].idxmin()
    best_epoch = history.loc[best_idx, "epoch"]
    best_loss = history.loc[best_idx, best_col]
    axes[0].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    axes[0].scatter([best_epoch], [best_loss], color="black", s=35, zorder=5)
    axes[0].annotate(
        f"best selected validation\nepoch {int(best_epoch)}\n{best_loss:.3f}",
        xy=(best_epoch, best_loss),
        xytext=(8, 16),
        textcoords="offset points",
        fontsize=8,
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.8},
    )
    axes[0].set_ylabel("standardized MSE")
    axes[0].set_title(args.title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    stage_col = "stage" if "stage" in history else None
    if stage_col is not None:
        for stage_index, (stage, group) in enumerate(history.groupby(stage_col, sort=False)):
            start = group["epoch"].min()
            end = group["epoch"].max()
            shade = "0.86" if stage_index % 2 == 0 else "0.92"
            for axis in axes:
                axis.axvspan(start, end, color=shade, alpha=0.22, zorder=0)
                axis.axvline(start, color="0.75", linewidth=0.8, zorder=0)
            label = STAGE_LABELS.get(stage, str(stage).replace("_", " "))
            axes[0].text(
                (start + end) / 2,
                0.03,
                label,
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.25",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )

    modality_cols = [
        column
        for column in history.columns
        if column.startswith("validation_cross_loss_")
        and column != "validation_cross_loss"
    ]
    if not modality_cols:
        modality_cols = [
            column
            for column in history.columns
            if column.startswith("validation_loss_")
            and column != "validation_loss"
        ]

    for column in modality_cols:
        label = column.replace("validation_cross_loss_", "").replace("validation_loss_", "")
        axes[1].plot(
            history["epoch"],
            history[column],
            label=label,
            linewidth=1.25,
            marker="o",
            markersize=2.7,
        )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation standardized MSE")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False, ncol=2)

    fig.savefig(output, dpi=180)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
