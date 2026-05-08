from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate_cross_prediction import evaluate_case, load_model
from .feature_store import load_feature_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate aerosol encoder skill by year, season, and proxy regime.")
    parser.add_argument("--features", required=True, help="features.npz from build_features.")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint.pt.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize_from_checkpoint_with_times(
    matrix: np.ndarray,
    times: np.ndarray,
    checkpoint: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_indices = checkpoint["raw_feature_indices"]
    mean = checkpoint["mean"].astype(np.float32)
    std = checkpoint["std"].astype(np.float32)
    selected = matrix[:, raw_indices].astype(np.float32)
    normalized = (selected - mean) / std
    feature_mask = np.isfinite(normalized).astype(np.float32)
    normalized = np.where(feature_mask > 0, normalized, 0.0).astype(np.float32)
    valid_rows = feature_mask.any(axis=1)
    return normalized[valid_rows], feature_mask[valid_rows], times[valid_rows]


def season_labels(times: np.ndarray) -> np.ndarray:
    months = pd.DatetimeIndex(times).month
    labels = np.full(len(months), "DJF", dtype=object)
    labels[np.isin(months, [3, 4, 5])] = "MAM"
    labels[np.isin(months, [6, 7, 8])] = "JJA"
    labels[np.isin(months, [9, 10, 11])] = "SON"
    return labels


def _proxy_from_features(
    x: np.ndarray,
    feature_mask: np.ndarray,
    feature_names: list[str],
    needles: tuple[str, ...],
) -> np.ndarray:
    indices = [
        index
        for index, feature_name in enumerate(feature_names)
        if any(needle in feature_name for needle in needles)
    ]
    if not indices:
        return np.full(x.shape[0], np.nan, dtype=np.float32)
    values = np.where(feature_mask[:, indices] > 0, x[:, indices], np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=1)


def regime_labels(
    x: np.ndarray,
    feature_mask: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    number_proxy = _proxy_from_features(
        x,
        feature_mask,
        feature_names,
        ("cpc_number", "total_N_conc"),
    )
    mass_optics_proxy = _proxy_from_features(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "total_V_conc", "_Dry_Neph3W"),
    )
    finite_number = np.isfinite(number_proxy)
    finite_mass = np.isfinite(mass_optics_proxy)
    if finite_number.sum() < 10 or finite_mass.sum() < 10:
        return np.full(x.shape[0], "unclassified", dtype=object)

    number_hi = np.nanpercentile(number_proxy, 67)
    number_lo = np.nanpercentile(number_proxy, 33)
    mass_hi = np.nanpercentile(mass_optics_proxy, 67)
    mass_lo = np.nanpercentile(mass_optics_proxy, 33)

    labels = np.full(x.shape[0], "moderate", dtype=object)
    high_number = number_proxy >= number_hi
    high_mass = mass_optics_proxy >= mass_hi
    low_number = number_proxy <= number_lo
    low_mass = mass_optics_proxy <= mass_lo
    labels[high_number & high_mass] = "mixed_high"
    labels[high_number & ~high_mass] = "number_dominated"
    labels[~high_number & high_mass] = "mass_optics_dominated"
    labels[low_number & low_mass] = "low_loading"
    labels[~(finite_number & finite_mass)] = "unclassified"
    return labels


def exclusion_group_for_target(checkpoint: dict, target: str) -> set[str]:
    for group in checkpoint.get("config", {}).get("cross_prediction_exclusion_groups", ()):
        group_set = set(group)
        if target in group_set:
            return group_set
    return set()


def evaluate_rows(
    model: torch.nn.Module,
    x: np.ndarray,
    feature_mask: np.ndarray,
    checkpoint: dict,
    row_indices: np.ndarray,
    batch_size: int,
    device: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    target_modalities = tuple(checkpoint["target_modalities"])
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
            batch_size,
            device,
        )
        rows.append({"target_modality": target, "input_case": "all_other", **metrics})
        excluded_group = exclusion_group_for_target(checkpoint, target)
        if excluded_group:
            metrics = evaluate_case(
                model,
                x,
                feature_mask,
                row_indices,
                checkpoint,
                set(checkpoint["modality_indices"]) - excluded_group,
                target,
                batch_size,
                device,
            )
            rows.append({"target_modality": target, "input_case": "all_unrelated", **metrics})
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    matrix, times, _ = load_feature_store(args.features)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    x, feature_mask, valid_times = normalize_from_checkpoint_with_times(matrix, times, checkpoint)
    model = load_model(checkpoint, args.device)

    if args.split == "all":
        base_indices = np.arange(x.shape[0])
    else:
        base_indices = np.asarray(checkpoint["splits"][args.split])

    datetime_index = pd.DatetimeIndex(valid_times)
    strata = {
        "year": datetime_index.year.astype(str).to_numpy(),
        "season": season_labels(valid_times),
        "regime": regime_labels(x, feature_mask, list(checkpoint["feature_names"])),
    }

    rows: list[dict[str, float | str]] = []
    for stratification, labels in strata.items():
        for label in sorted(pd.unique(labels)):
            row_indices = base_indices[labels[base_indices] == label]
            if row_indices.size == 0:
                continue
            for metrics in evaluate_rows(
                model,
                x,
                feature_mask,
                checkpoint,
                row_indices,
                args.batch_size,
                args.device,
            ):
                rows.append(
                    {
                        "split": args.split,
                        "stratification": stratification,
                        "stratum": str(label),
                        "rows_in_stratum": float(row_indices.size),
                        **metrics,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{args.split}_stratified_leave_one_out.csv", index=False)
    print(f"wrote {output / f'{args.split}_stratified_leave_one_out.csv'}")


if __name__ == "__main__":
    main()
