from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .feature_store import load_feature_store
from .model import build_model_from_checkpoint, unpack_model_output

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whole-modality cross prediction.")
    parser.add_argument("--features", required=True, help="features.npz from build_features.")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint.pt.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize_from_checkpoint(matrix: np.ndarray, checkpoint: dict) -> tuple[np.ndarray, np.ndarray]:
    raw_indices = checkpoint["raw_feature_indices"]
    mean = checkpoint["mean"].astype(np.float32)
    std = checkpoint["std"].astype(np.float32)
    selected = matrix[:, raw_indices].astype(np.float32)
    normalized = (selected - mean) / std
    feature_mask = np.isfinite(normalized).astype(np.float32)
    normalized = np.where(feature_mask > 0, normalized, 0.0).astype(np.float32)
    valid_rows = feature_mask.any(axis=1)
    normalized = normalized[valid_rows]
    feature_mask = feature_mask[valid_rows]
    return normalized, feature_mask


def load_model(checkpoint: dict, device: str) -> torch.nn.Module:
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def split_modalities(
    batch_x: torch.Tensor,
    batch_mask: torch.Tensor,
    modality_indices: dict[str, list[int]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    x_by_modality = {}
    mask_by_modality = {}
    for modality, indices in modality_indices.items():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=batch_x.device)
        x_by_modality[modality] = batch_x.index_select(1, index_tensor)
        mask_by_modality[modality] = batch_mask.index_select(1, index_tensor)
    return x_by_modality, mask_by_modality


def evaluate_case(
    model: torch.nn.Module,
    x: np.ndarray,
    feature_mask: np.ndarray,
    row_indices: np.ndarray,
    checkpoint: dict,
    input_modalities: set[str],
    target_modality: str,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    dataset = TensorDataset(
        torch.from_numpy(x[row_indices].astype(np.float32)),
        torch.from_numpy(feature_mask[row_indices].astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    se_sum = 0.0
    baseline_se_sum = 0.0
    valid_count = 0.0
    valid_rows = 0

    with torch.no_grad():
        for batch_x, batch_mask in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            x_by_modality, mask_by_modality = split_modalities(
                batch_x, batch_mask, checkpoint["modality_indices"]
            )
            input_mask = {}
            any_input = torch.zeros(batch_x.shape[0], dtype=torch.bool, device=device)
            for modality, mask in mask_by_modality.items():
                observed = mask.sum(dim=1) > 0
                input_mask[modality] = observed & (modality in input_modalities)
                any_input |= input_mask[modality]

            target_mask = mask_by_modality[target_modality]
            usable_rows = any_input & (target_mask.sum(dim=1) > 0)
            if not torch.any(usable_rows):
                continue

            x_usable = {
                modality: values[usable_rows]
                for modality, values in x_by_modality.items()
            }
            mask_usable = {
                modality: values[usable_rows]
                for modality, values in mask_by_modality.items()
            }
            input_usable = {
                modality: values[usable_rows]
                for modality, values in input_mask.items()
            }
            _, decoded, _ = unpack_model_output(model(x_usable, mask_usable, input_usable))
            pred = decoded[target_modality]
            target = x_usable[target_modality]
            mask = mask_usable[target_modality]
            se_sum += float((((pred - target) ** 2) * mask).sum().cpu())
            baseline_se_sum += float(((target ** 2) * mask).sum().cpu())
            valid_count += float(mask.sum().cpu())
            valid_rows += int(usable_rows.sum().cpu())

    if valid_count <= 0:
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mean_baseline_mse": np.nan,
            "skill_vs_mean": np.nan,
            "valid_feature_values": 0.0,
            "valid_rows": 0.0,
        }

    mse = se_sum / valid_count
    baseline = baseline_se_sum / valid_count
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mean_baseline_mse": baseline,
        "skill_vs_mean": 1.0 - mse / baseline if baseline > 0 else np.nan,
        "valid_feature_values": valid_count,
        "valid_rows": float(valid_rows),
    }


def exclusion_group_for_target(checkpoint: dict, target: str) -> set[str]:
    for group in checkpoint.get("config", {}).get("cross_prediction_exclusion_groups", ()):
        group_set = set(group)
        if target in group_set:
            return group_set
    return set()


def plot_results(output: Path, all_other: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_frame = all_other[all_other["input_case"] == "all_other"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(plot_frame))
    ax.bar(x - 0.18, plot_frame["mse"], width=0.36, label="model")
    ax.bar(x + 0.18, plot_frame["mean_baseline_mse"], width=0.36, label="mean baseline")
    ax.set_xticks(x, plot_frame["target_modality"], rotation=25, ha="right")
    ax.set_ylabel("standardized MSE")
    ax.set_title("Leave-one-modality-out cross prediction")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output / "leave_one_out_cross_prediction.png", dpi=180)
    plt.close(fig)

    skill_frame = all_other.copy()
    skill_labels = [
        target if case == "all_other" else f"{target}\nstrict group"
        for target, case in zip(
            skill_frame["target_modality"],
            skill_frame["input_case"],
            strict=True,
        )
    ]
    colors = ["#2ca25f" if value >= 0 else "#de2d26" for value in skill_frame["skill_vs_mean"]]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(skill_frame))
    ax.bar(x, skill_frame["skill_vs_mean"], color=colors, width=0.72)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, skill_labels, rotation=35, ha="right")
    ax.set_ylabel("skill vs training mean")
    ax.set_title("Leave-one-out skill score")
    ax.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(skill_frame["skill_vs_mean"]):
        if np.isfinite(value):
            ax.text(
                index,
                value + (0.025 if value >= 0 else -0.035),
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    fig.savefig(output / "leave_one_out_skill.png", dpi=180)
    plt.close(fig)

    matrix = pairwise.pivot(index="target_modality", columns="source_modality", values="skill_vs_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    image = ax.imshow(matrix.values, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.set_title("Pairwise cross-prediction skill vs mean baseline")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.values[row, col]
            label = "" if np.isnan(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="1 - model MSE / baseline MSE")
    fig.savefig(output / "pairwise_cross_prediction_skill.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    matrix, _, _ = load_feature_store(args.features)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    x, feature_mask = normalize_from_checkpoint(matrix, checkpoint)
    model = load_model(checkpoint, args.device)

    if args.split == "all":
        row_indices = np.arange(x.shape[0])
    else:
        row_indices = np.asarray(checkpoint["splits"][args.split])

    target_modalities = tuple(checkpoint["target_modalities"])
    context_modalities = set(checkpoint.get("context_modalities", ()))
    rows = []
    for target in target_modalities:
        inputs = set(checkpoint["modality_indices"]) - {target}
        metrics = evaluate_case(
            model,
            x,
            feature_mask,
            row_indices,
            checkpoint,
            inputs,
            target,
            args.batch_size,
            args.device,
        )
        rows.append({"target_modality": target, "input_case": "all_other", **metrics})
        excluded_group = exclusion_group_for_target(checkpoint, target)
        if excluded_group:
            unrelated_inputs = set(checkpoint["modality_indices"]) - excluded_group
            metrics = evaluate_case(
                model,
                x,
                feature_mask,
                row_indices,
                checkpoint,
                unrelated_inputs,
                target,
                args.batch_size,
                args.device,
            )
            rows.append(
                {
                    "target_modality": target,
                    "input_case": "all_unrelated",
                    **metrics,
                }
            )
    all_other = pd.DataFrame(rows)
    all_other.to_csv(output / f"{args.split}_leave_one_out.csv", index=False)

    pairwise_rows = []
    for source in target_modalities:
        for target in target_modalities:
            if source == target:
                continue
            inputs = context_modalities | {source}
            metrics = evaluate_case(
                model,
                x,
                feature_mask,
                row_indices,
                checkpoint,
                inputs,
                target,
                args.batch_size,
                args.device,
            )
            pairwise_rows.append(
                {
                    "source_modality": source,
                    "target_modality": target,
                    "input_case": "met_plus_source",
                    **metrics,
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(output / f"{args.split}_pairwise.csv", index=False)
    plot_results(output, all_other, pairwise)

    print(f"wrote {output / f'{args.split}_leave_one_out.csv'}")
    print(f"wrote {output / f'{args.split}_pairwise.csv'}")
    print(f"wrote {output / 'leave_one_out_cross_prediction.png'}")
    print(f"wrote {output / 'leave_one_out_skill.png'}")
    print(f"wrote {output / 'pairwise_cross_prediction_skill.png'}")


if __name__ == "__main__":
    main()
