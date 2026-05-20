from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from aerosol_encoding.model import (
    ConditionalCCNActivationDecoder,
    CoordinateCCNActivationDecoder,
    build_model_from_checkpoint,
)
from aerosol_encoding.training_data import load_prepared_arrays

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")


TRADITIONAL_INPUT_MODALITIES = {
    "chemistry_acsm",
    "size_smps",
    "size_aps",
    "size_uhsas",
    "size_opc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare neural CCN retrieval against kappa-Kohler baseline on matched rows."
    )
    parser.add_argument("--prepared-arrays", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--baseline-name", default="kappa_kohler_mass_fraction")
    parser.add_argument(
        "--extra-baseline",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Additional baseline prediction CSV aligned by row/time/feature keys.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparison-name", default="matched_neural_vs_kappa")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument(
        "--neural-input-set",
        action="append",
        default=[],
        metavar="NAME=MODALITY[,MODALITY...]",
        help=(
            "Additional neural retrieval input set to evaluate. "
            "Example: neural_neph_only=optical_neph. May be repeated."
        ),
    )
    return parser.parse_args()


def _check_feature_space(arrays: Any, checkpoint: dict[str, Any]) -> None:
    checkpoint_features = list(checkpoint["feature_names"])
    if list(arrays.feature_names) != checkpoint_features:
        raise ValueError(
            "Prepared arrays feature names do not match the checkpoint feature names."
        )
    if arrays.x.shape[1] != len(checkpoint_features):
        raise ValueError(
            f"Prepared arrays have {arrays.x.shape[1]} features, but checkpoint has "
            f"{len(checkpoint_features)} features."
        )


def _split_modalities(
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


def _encode_rows(
    *,
    model: torch.nn.Module,
    arrays: Any,
    checkpoint: dict[str, Any],
    row_indices: np.ndarray,
    allowed_modalities: set[str],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    z_rows: list[np.ndarray] = []
    encoded_rows: list[np.ndarray] = []
    modality_indices = {
        modality: [int(index) for index in indices]
        for modality, indices in checkpoint["modality_indices"].items()
    }
    with torch.no_grad():
        for start in range(0, row_indices.size, batch_size):
            batch_rows = row_indices[start : start + batch_size]
            batch_x = torch.as_tensor(
                arrays.x[batch_rows].astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            batch_mask = torch.as_tensor(
                arrays.feature_mask[batch_rows].astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            x_by_modality, mask_by_modality = _split_modalities(
                batch_x,
                batch_mask,
                modality_indices,
            )
            input_mask = {}
            any_input = torch.zeros(batch_x.shape[0], dtype=torch.bool, device=device)
            for modality, mask in mask_by_modality.items():
                visible = (mask.sum(dim=1) > 0) & (modality in allowed_modalities)
                input_mask[modality] = visible
                any_input |= visible
            if not torch.all(any_input):
                keep = torch.nonzero(any_input, as_tuple=False).flatten()
                if keep.numel() == 0:
                    continue
                x_by_modality = {
                    modality: values.index_select(0, keep)
                    for modality, values in x_by_modality.items()
                }
                mask_by_modality = {
                    modality: values.index_select(0, keep)
                    for modality, values in mask_by_modality.items()
                }
                input_mask = {
                    modality: values.index_select(0, keep)
                    for modality, values in input_mask.items()
                }
                batch_rows = batch_rows[keep.detach().cpu().numpy()]
            z = model.encode(x_by_modality, mask_by_modality, input_mask)
            z_rows.append(z.detach().cpu().numpy().astype(np.float32))
            encoded_rows.append(batch_rows.astype(np.int64))
    if not z_rows:
        return np.empty((0, int(checkpoint["latent_dim"])), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.vstack(z_rows), np.concatenate(encoded_rows)


def _decode_observations(
    *,
    model: torch.nn.Module,
    z_by_row: np.ndarray,
    encoded_rows: np.ndarray,
    observations: pd.DataFrame,
    arrays: Any,
    checkpoint: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    row_to_position = {int(row): position for position, row in enumerate(encoded_rows)}
    positions = observations["row_index"].map(row_to_position).to_numpy()
    valid = pd.notna(positions)
    if not np.all(valid):
        missing = int((~valid).sum())
        raise ValueError(f"{missing} baseline observations could not be encoded by the neural model.")
    positions = positions.astype(np.int64)
    decoder = model.decoders["ccn_activation"] if "ccn_activation" in model.decoders else None
    if isinstance(decoder, ConditionalCCNActivationDecoder):
        return _decode_conditional_ccn_observations(
            model=model,
            decoder=decoder,
            z_by_row=z_by_row,
            positions=positions,
            observations=observations,
            arrays=arrays,
            checkpoint=checkpoint,
            batch_size=batch_size,
            device=device,
        )
    if not isinstance(decoder, CoordinateCCNActivationDecoder):
        raise TypeError(
            "ccn_activation decoder must be ConditionalCCNActivationDecoder "
            "or CoordinateCCNActivationDecoder"
        )

    supersaturation = observations["supersaturation_percent"].to_numpy(dtype=np.float32)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, positions.size, batch_size):
            batch_positions = positions[start : start + batch_size]
            batch_z = torch.as_tensor(
                z_by_row[batch_positions],
                dtype=torch.float32,
                device=device,
            )
            batch_ss = torch.as_tensor(
                supersaturation[start : start + batch_size, None],
                dtype=torch.float32,
                device=device,
            )
            prediction = model.decode_ccn_at_supersaturation(
                batch_z,
                batch_ss,
                physical=True,
            )
            predictions.append(prediction[:, 0].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(predictions)


def _decode_conditional_ccn_observations(
    *,
    model: torch.nn.Module,
    decoder: ConditionalCCNActivationDecoder,
    z_by_row: np.ndarray,
    positions: np.ndarray,
    observations: pd.DataFrame,
    arrays: Any,
    checkpoint: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ccn_global_indices = [int(index) for index in checkpoint["modality_indices"]["ccn_activation"]]
    global_to_local = {
        global_index: local_index
        for local_index, global_index in enumerate(ccn_global_indices)
    }
    n_ccn_global = observations["n_ccn_feature_index"].to_numpy(dtype=np.int64)
    observation_rows = observations["row_index"].to_numpy(dtype=np.int64)

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, positions.size, batch_size):
            stop = start + batch_size
            batch_positions = positions[start:stop]
            batch_z = torch.as_tensor(
                z_by_row[batch_positions],
                dtype=torch.float32,
                device=device,
            )
            batch_n_global = n_ccn_global[start:stop]
            batch_rows = observation_rows[start:stop]
            target_values = torch.as_tensor(
                arrays.x[batch_rows][:, ccn_global_indices].astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            target_mask = torch.as_tensor(
                arrays.feature_mask[batch_rows][:, ccn_global_indices].astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            local_n_indices: list[int] = []
            for n_global in batch_n_global:
                local_n_indices.append(global_to_local[int(n_global)])

            decoded = decoder(
                model.decoder_state(batch_z),
                target_values,
                target_mask,
            )
            local_index_tensor = torch.as_tensor(
                local_n_indices,
                dtype=torch.long,
                device=device,
            )
            row_index_tensor = torch.arange(
                decoded.shape[0],
                dtype=torch.long,
                device=device,
            )
            normalized_prediction = decoded[row_index_tensor, local_index_tensor]
            response_mean = torch.as_tensor(
                arrays.mean[batch_n_global],
                dtype=torch.float32,
                device=device,
            )
            response_std = torch.as_tensor(
                arrays.std[batch_n_global],
                dtype=torch.float32,
                device=device,
            )
            prediction_log1p = normalized_prediction * response_std + response_mean
            prediction_cm3 = torch.expm1(prediction_log1p).clamp_min(0.0)
            predictions.append(prediction_cm3.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(predictions)


def _parse_baseline_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Extra baseline must be NAME=CSV, got {spec!r}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Extra baseline has empty name: {spec!r}")
    return name, Path(path)


def _parse_neural_input_set(
    spec: str,
    available_modalities: set[str],
) -> tuple[str, set[str]]:
    if "=" not in spec:
        raise ValueError(f"Neural input set must be NAME=MODALITIES, got {spec!r}")
    name, raw_modalities = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Neural input set has empty name: {spec!r}")
    modalities = {
        modality.strip()
        for modality in raw_modalities.split(",")
        if modality.strip()
    }
    if not modalities:
        raise ValueError(f"Neural input set {name!r} has no modalities.")
    unknown = modalities - available_modalities
    if unknown:
        raise ValueError(
            f"Neural input set {name!r} has unknown modalities: {sorted(unknown)}"
        )
    if "ccn_activation" in modalities:
        raise ValueError(
            f"Neural input set {name!r} includes ccn_activation, which would leak the target."
        )
    return name, modalities


def _alignment_key(frame: pd.DataFrame) -> pd.MultiIndex:
    key_columns = ["row_index", "time_bin", "n_ccn_feature_index", "supersaturation_percent"]
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction file is missing alignment columns: {missing}")
    return pd.MultiIndex.from_frame(frame[key_columns])


def _aligned_prediction(
    base_frame: pd.DataFrame,
    prediction_path: Path,
) -> np.ndarray:
    other = pd.read_csv(prediction_path)
    base_key = _alignment_key(base_frame)
    other_key = _alignment_key(other)
    if other_key.has_duplicates:
        raise ValueError(f"{prediction_path} has duplicate row/time/feature prediction keys.")
    other = other.set_index(other_key)
    missing = base_key.difference(other.index)
    extra = other.index.difference(base_key)
    if len(missing) or len(extra):
        raise ValueError(
            f"{prediction_path} is not aligned with the primary baseline: "
            f"{len(missing)} missing keys, {len(extra)} extra keys."
        )
    aligned = other.loc[base_key, "predicted_ccn_cm3"].to_numpy(dtype=np.float64)
    observed = other.loc[base_key, "observed_ccn_cm3"].to_numpy(dtype=np.float64)
    if not np.allclose(observed, base_frame["observed_ccn_cm3"].to_numpy(dtype=np.float64)):
        raise ValueError(f"{prediction_path} observed CCN values do not match primary baseline.")
    return aligned


def _prediction_metrics(
    *,
    frame: pd.DataFrame,
    predicted_cm3: np.ndarray,
    arrays: Any,
) -> dict[str, float]:
    finite_prediction = np.isfinite(predicted_cm3)
    if not np.any(finite_prediction):
        return {
            "n": 0.0,
            "mae_cm3": float("nan"),
            "bias_cm3": float("nan"),
            "rmse_cm3": float("nan"),
            "log1p_rmse": float("nan"),
            "standardized_log1p_mse": float("nan"),
            "standardized_log1p_rmse": float("nan"),
            "mean_baseline_mse": float("nan"),
            "skill_vs_mean": float("nan"),
        }
    frame = frame.loc[finite_prediction].reset_index(drop=True)
    predicted_cm3 = predicted_cm3[finite_prediction]
    observed_cm3 = frame["observed_ccn_cm3"].to_numpy(dtype=np.float64)
    observed_log = frame["observed_log1p_ccn"].to_numpy(dtype=np.float64)
    predicted_cm3 = np.maximum(predicted_cm3.astype(np.float64), 0.0)
    predicted_log = np.log1p(predicted_cm3)
    feature_indices = frame["n_ccn_feature_index"].to_numpy(dtype=np.int64)
    feature_std = arrays.std[feature_indices].astype(np.float64)
    feature_mean = arrays.mean[feature_indices].astype(np.float64)
    standardized_error = (predicted_log - observed_log) / feature_std
    standardized_target = (observed_log - feature_mean) / feature_std
    physical_error = predicted_cm3 - observed_cm3
    mse = float(np.mean(standardized_error**2))
    mean_baseline_mse = float(np.mean(standardized_target**2))
    return {
        "n": float(frame.shape[0]),
        "mae_cm3": float(np.mean(np.abs(physical_error))),
        "bias_cm3": float(np.mean(physical_error)),
        "rmse_cm3": float(np.sqrt(np.mean(physical_error**2))),
        "log1p_rmse": float(np.sqrt(np.mean((predicted_log - observed_log) ** 2))),
        "standardized_log1p_mse": mse,
        "standardized_log1p_rmse": float(np.sqrt(mse)),
        "mean_baseline_mse": mean_baseline_mse,
        "skill_vs_mean": float(1.0 - mse / mean_baseline_mse),
    }


def _per_ss_bin(
    *,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    arrays: Any,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    bins = frame["supersaturation_percent"].round(2)
    for ss_bin in sorted(bins.unique()):
        subset = frame.loc[bins == ss_bin]
        for method, values in predictions.items():
            metrics = _prediction_metrics(
                frame=subset,
                predicted_cm3=values[subset.index.to_numpy()],
                arrays=arrays,
            )
            rows.append({"method": method, "supersaturation_bin": float(ss_bin), **metrics})
    return rows


def _plot_metrics(metrics_frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "kappa_kohler_mass_fraction": "kappa\nACSM+size",
        "random_forest_acsm_size": "RF\nACSM+size",
        "neural_all_non_ccn_inputs": "neural\nall non-CCN",
        "neural_acsm_plus_size_only": "neural\nACSM+size",
        "neural_neph_only": "neural\nneph only",
    }
    frame = metrics_frame.copy()
    frame["label"] = frame["method"].map(labels).fillna(frame["method"])
    palette = ["#7f8c8d", "#b35806", "#2c7fb8", "#41ab5d", "#756bb1", "#636363"]
    colors = palette[: frame.shape[0]]

    width = max(12.8, 2.35 * frame.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(width, 4.1), constrained_layout=True)
    x = np.arange(frame.shape[0])
    axes[0].bar(x, frame["standardized_log1p_mse"], color=colors, width=0.68)
    axes[0].set_xticks(x, frame["label"])
    axes[0].set_ylabel("standardized log1p MSE")
    axes[0].set_title("Lower is better")
    axes[0].grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(frame["standardized_log1p_mse"]):
        axes[0].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, frame["skill_vs_mean"], color=colors, width=0.68)
    axes[1].set_xticks(x, frame["label"])
    axes[1].set_ylabel("skill vs matched mean")
    axes[1].set_title("Higher is better")
    axes[1].set_ylim(0.0, max(1.0, float(frame["skill_vs_mean"].max()) + 0.08))
    axes[1].grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(frame["skill_vs_mean"]):
        axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Matched test CCN retrieval comparison", fontsize=13)
    fig.savefig(output / "matched_neural_comparison.png", dpi=180)
    fig.savefig(output / "matched_neural_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    arrays = load_prepared_arrays(args.prepared_arrays)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _check_feature_space(arrays, checkpoint)
    device = torch.device(args.device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    baseline = pd.read_csv(args.baseline_predictions)
    baseline = baseline.reset_index(drop=True)
    row_indices = np.asarray(sorted(baseline["row_index"].unique()), dtype=np.int64)

    modality_names = set(checkpoint["modality_indices"])
    all_other = modality_names - {"ccn_activation"}
    traditional = TRADITIONAL_INPUT_MODALITIES & modality_names
    custom_neural_input_sets = {}
    reserved_names = {
        args.baseline_name,
        "neural_all_non_ccn_inputs",
        "neural_acsm_plus_size_only",
    }
    for spec in args.neural_input_set:
        name, modalities = _parse_neural_input_set(spec, modality_names)
        if name in reserved_names or name in custom_neural_input_sets:
            raise ValueError(f"Duplicate or reserved neural input set name: {name!r}")
        custom_neural_input_sets[name] = modalities

    z_all, rows_all = _encode_rows(
        model=model,
        arrays=arrays,
        checkpoint=checkpoint,
        row_indices=row_indices,
        allowed_modalities=all_other,
        batch_size=args.batch_size,
        device=device,
    )
    z_traditional, rows_traditional = _encode_rows(
        model=model,
        arrays=arrays,
        checkpoint=checkpoint,
        row_indices=row_indices,
        allowed_modalities=traditional,
        batch_size=args.batch_size,
        device=device,
    )

    neural_all = _decode_observations(
        model=model,
        z_by_row=z_all,
        encoded_rows=rows_all,
        observations=baseline,
        arrays=arrays,
        checkpoint=checkpoint,
        batch_size=args.batch_size,
        device=device,
    )
    neural_traditional = _decode_observations(
        model=model,
        z_by_row=z_traditional,
        encoded_rows=rows_traditional,
        observations=baseline,
        arrays=arrays,
        checkpoint=checkpoint,
        batch_size=args.batch_size,
        device=device,
    )
    custom_neural_predictions: dict[str, np.ndarray] = {}
    custom_neural_encoded_rows: dict[str, int] = {}
    for name, modalities in custom_neural_input_sets.items():
        z_custom, rows_custom = _encode_rows(
            model=model,
            arrays=arrays,
            checkpoint=checkpoint,
            row_indices=row_indices,
            allowed_modalities=modalities,
            batch_size=args.batch_size,
            device=device,
        )
        custom_neural_encoded_rows[name] = int(rows_custom.size)
        prediction = np.full(baseline.shape[0], np.nan, dtype=np.float64)
        if rows_custom.size:
            encoded_observations = baseline[baseline["row_index"].isin(rows_custom)].copy()
            decoded = _decode_observations(
                model=model,
                z_by_row=z_custom,
                encoded_rows=rows_custom,
                observations=encoded_observations,
                arrays=arrays,
                checkpoint=checkpoint,
                batch_size=args.batch_size,
                device=device,
            )
            prediction[encoded_observations.index.to_numpy(dtype=np.int64)] = decoded
        custom_neural_predictions[name] = prediction
    baseline_predictions = {
        args.baseline_name: baseline["predicted_ccn_cm3"].to_numpy(dtype=np.float64)
    }
    for spec in args.extra_baseline:
        name, prediction_path = _parse_baseline_spec(spec)
        if name in baseline_predictions:
            raise ValueError(f"Duplicate baseline name {name!r}")
        baseline_predictions[name] = _aligned_prediction(baseline, prediction_path)

    metrics = {
        name: _prediction_metrics(
            frame=baseline,
            predicted_cm3=predicted,
            arrays=arrays,
        )
        for name, predicted in baseline_predictions.items()
    }
    metrics.update(
        {
        "neural_all_non_ccn_inputs": _prediction_metrics(
            frame=baseline,
            predicted_cm3=neural_all,
            arrays=arrays,
        ),
        "neural_acsm_plus_size_only": _prediction_metrics(
            frame=baseline,
            predicted_cm3=neural_traditional,
            arrays=arrays,
        ),
        **{
            name: _prediction_metrics(
                frame=baseline,
                predicted_cm3=predicted,
                arrays=arrays,
            )
            for name, predicted in custom_neural_predictions.items()
        },
        }
    )
    summary = {
        "prepared_arrays": args.prepared_arrays,
        "checkpoint": args.checkpoint,
        "baseline_predictions": args.baseline_predictions,
        "extra_baselines": args.extra_baseline,
        "matched_observations": int(baseline.shape[0]),
        "matched_unique_rows": int(row_indices.size),
        "traditional_input_modalities": sorted(traditional),
        "all_non_ccn_input_modalities": sorted(all_other),
        "custom_neural_input_modalities": {
            name: sorted(modalities)
            for name, modalities in custom_neural_input_sets.items()
        },
        "custom_neural_encoded_rows": custom_neural_encoded_rows,
        "metrics": metrics,
    }

    summary_path = output / f"{args.comparison_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    metrics_frame = pd.DataFrame(
        [{"method": method, **values} for method, values in metrics.items()]
    )
    metrics_path = output / f"{args.comparison_name}_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    _plot_metrics(metrics_frame, output)

    ss_frame = pd.DataFrame(
        _per_ss_bin(
            frame=baseline,
            predictions={
                **baseline_predictions,
                "neural_all_non_ccn_inputs": neural_all,
                "neural_acsm_plus_size_only": neural_traditional,
                **custom_neural_predictions,
            },
            arrays=arrays,
        )
    )
    ss_path = output / f"{args.comparison_name}_by_supersaturation.csv"
    ss_frame.to_csv(ss_path, index=False)

    if args.write_predictions:
        prediction_frame = baseline.copy()
        for name, predicted in baseline_predictions.items():
            prediction_frame[f"{name}_predicted_ccn_cm3"] = predicted
        prediction_frame["neural_all_non_ccn_predicted_ccn_cm3"] = neural_all
        prediction_frame["neural_acsm_plus_size_predicted_ccn_cm3"] = neural_traditional
        for name, predicted in custom_neural_predictions.items():
            prediction_frame[f"{name}_predicted_ccn_cm3"] = predicted
        prediction_frame.to_csv(output / f"{args.comparison_name}_predictions.csv", index=False)

    print(f"wrote {summary_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {ss_path}")
    print(f"wrote {output / 'matched_neural_comparison.png'}")
    print(metrics_frame.to_string(index=False))


if __name__ == "__main__":
    main()
