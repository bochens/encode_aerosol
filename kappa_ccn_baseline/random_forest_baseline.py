from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from aerosol_encoding.training_data import PreparedArrays, load_prepared_arrays

from .metrics import ccn_prediction_metrics, write_rows_csv
from .prepared_data import (
    ACSM_VARIABLES,
    build_acsm_indices,
    build_ccn_indices,
    ccn_observations_from_row,
    transformed_row,
)
from .size_merge import build_spectrum_indices, merged_row_spectrum


ACSM_FEATURE_ORDER = tuple(ACSM_VARIABLES)


@dataclass(frozen=True)
class RFDataset:
    x: np.ndarray
    y_log1p_ccn: np.ndarray
    metadata: list[dict[str, Any]]
    sampled_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random-forest CCN baseline from ACSM and size spectra."
    )
    parser.add_argument("--prepared-arrays", required=True, help="prepared_arrays.npz path.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--train-split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--eval-split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-train-samples", type=int, default=100000)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--limit-train-rows", type=int, default=None)
    parser.add_argument("--limit-eval-rows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=_parse_max_features, default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=20260517)
    parser.add_argument(
        "--merge-transition-decades",
        type=float,
        default=0.08,
        help="Error-function taper width for instrument overlap in log10(Dp).",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Write the fitted sklearn pipeline to random_forest_ccn_model.joblib.",
    )
    parser.add_argument(
        "--allow-missing-acsm",
        action="store_true",
        help="Allow ACSM predictor gaps to be filled by the training-set imputer.",
    )
    return parser.parse_args()


def _parse_max_features(value: str) -> str | int | float | None:
    if value in {"sqrt", "log2"}:
        return value
    if value.lower() in {"none", "null"}:
        return None
    try:
        parsed_float = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--max-features must be sqrt, log2, none, an int, or a float"
        ) from exc
    if parsed_float.is_integer() and parsed_float >= 1.0:
        return int(parsed_float)
    return parsed_float


def _row_indices(arrays: PreparedArrays, split: str, limit_rows: int | None) -> np.ndarray:
    indices = np.asarray(arrays.splits[split], dtype=np.int64)
    if limit_rows is not None:
        indices = indices[: max(int(limit_rows), 0)]
    return indices


def _diameter_grid(spectrum_indices) -> np.ndarray:
    grids = [index.diameter_nm for index in spectrum_indices.values()]
    if not grids:
        return np.asarray([], dtype=np.float64)
    grid = np.unique(np.concatenate(grids))
    grid.sort()
    return grid


def _feature_names(diameter_grid: np.ndarray) -> list[str]:
    names = ["supersaturation_percent"]
    names.extend(f"acsm_log1p_{name}" for name in ACSM_FEATURE_ORDER)
    names.extend(
        f"merged_log1p_dndlogdp_{diameter_nm:.6g}_nm"
        for diameter_nm in diameter_grid
    )
    return names


def _base_predictor_vector(
    values: np.ndarray,
    mask: np.ndarray,
    acsm_indices: dict[str, int],
    spectrum_indices,
    diameter_grid: np.ndarray,
    transition_decades: float,
) -> tuple[np.ndarray, int, int]:
    vector = np.full(1 + len(ACSM_FEATURE_ORDER) + diameter_grid.size, np.nan, dtype=np.float32)
    valid_acsm = 0
    for offset, key in enumerate(ACSM_FEATURE_ORDER, start=1):
        index = acsm_indices[key]
        if mask[index] and np.isfinite(values[index]):
            vector[offset] = np.float32(values[index])
            valid_acsm += 1

    diameter_nm, dndlogdp, _ = merged_row_spectrum(
        values,
        mask,
        spectrum_indices,
        transition_decades=transition_decades,
    )
    finite = np.isfinite(diameter_nm) & np.isfinite(dndlogdp) & (dndlogdp >= 0.0)
    if np.any(finite):
        positions = np.searchsorted(diameter_grid, diameter_nm[finite])
        log_values = np.log1p(dndlogdp[finite]).astype(np.float32)
        vector[1 + len(ACSM_FEATURE_ORDER) + positions] = log_values
    return vector, int(finite.sum()), valid_acsm


def build_rf_dataset(
    arrays: PreparedArrays,
    split: str,
    max_samples: int | None,
    limit_rows: int | None,
    random_state: int,
    shuffle_rows: bool,
    include_metadata: bool,
    transition_decades: float,
    require_complete_acsm: bool,
) -> tuple[RFDataset, list[str], np.ndarray]:
    acsm_indices = build_acsm_indices(arrays.feature_names)
    ccn_indices = build_ccn_indices(arrays.feature_names)
    spectrum_indices = build_spectrum_indices(arrays.feature_names)
    if not ccn_indices:
        raise ValueError("No CCN observation features found.")
    if not spectrum_indices:
        raise ValueError("No size-distribution dN/dlogDp features found.")

    diameter_grid = _diameter_grid(spectrum_indices)
    feature_names = _feature_names(diameter_grid)
    row_indices = _row_indices(arrays, split, limit_rows)
    rng = np.random.default_rng(random_state)
    if shuffle_rows:
        row_indices = rng.permutation(row_indices)

    features: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []
    sampled_rows = 0

    for row_index in row_indices:
        values, mask = transformed_row(arrays, int(row_index))
        observations = ccn_observations_from_row(values, mask, ccn_indices)
        if not observations:
            continue
        base_vector, valid_size_bins, valid_acsm = _base_predictor_vector(
            values,
            mask,
            acsm_indices,
            spectrum_indices,
            diameter_grid,
            transition_decades,
        )
        if valid_size_bins == 0:
            continue
        if require_complete_acsm and valid_acsm < len(ACSM_FEATURE_ORDER):
            continue

        sampled_rows += 1
        for observation in observations:
            sample = base_vector.copy()
            sample[0] = np.float32(observation["supersaturation_percent"])
            features.append(sample)
            targets.append(float(observation["observed_log1p_ccn"]))
            if include_metadata:
                n_ccn_index = int(observation["n_ccn_index"])
                metadata.append(
                    {
                        "row_index": int(row_index),
                        "time": str(arrays.times[int(row_index)]),
                        "time_bin": int(observation["time_bin"]),
                        "n_ccn_feature_index": n_ccn_index,
                        "supersaturation_percent": observation["supersaturation_percent"],
                        "observed_log1p_ccn": observation["observed_log1p_ccn"],
                        "observed_ccn_cm3": observation["observed_ccn_cm3"],
                        "target_std": float(arrays.std[n_ccn_index]),
                        "valid_size_bins": valid_size_bins,
                        "valid_acsm_features": valid_acsm,
                    }
                )
            if max_samples is not None and len(features) >= max_samples:
                break
        if max_samples is not None and len(features) >= max_samples:
            break

    if not features:
        raise ValueError(f"No usable RF samples found for split {split!r}.")

    x = np.vstack(features).astype(np.float32, copy=False)
    y = np.asarray(targets, dtype=np.float32)
    if shuffle_rows and x.shape[0] > 1:
        order = rng.permutation(x.shape[0])
        x = x[order]
        y = y[order]
        if include_metadata:
            metadata = [metadata[int(index)] for index in order]
    return RFDataset(x=x, y_log1p_ccn=y, metadata=metadata, sampled_rows=sampled_rows), feature_names, diameter_grid


def _fit_model(args: argparse.Namespace) -> Pipeline:
    forest = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        criterion="squared_error",
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("forest", forest),
        ]
    )


def _prediction_rows(
    eval_dataset: RFDataset,
    predicted_log1p: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata, prediction_log in zip(
        eval_dataset.metadata,
        predicted_log1p,
        strict=True,
    ):
        predicted_log = float(prediction_log)
        predicted_ccn = float(max(np.expm1(predicted_log), 0.0))
        observed_log = float(metadata["observed_log1p_ccn"])
        observed_ccn = float(metadata["observed_ccn_cm3"])
        log1p_error = predicted_log - observed_log
        target_std = float(metadata["target_std"])
        rows.append(
            {
                "row_index": metadata["row_index"],
                "time": metadata["time"],
                "time_bin": metadata["time_bin"],
                "n_ccn_feature_index": metadata["n_ccn_feature_index"],
                "supersaturation_percent": metadata["supersaturation_percent"],
                "predicted_ccn_cm3": predicted_ccn,
                "observed_ccn_cm3": observed_ccn,
                "error_cm3": predicted_ccn - observed_ccn,
                "predicted_log1p_ccn": predicted_log,
                "observed_log1p_ccn": observed_log,
                "log1p_error": log1p_error,
                "standardized_log1p_error": log1p_error / target_std,
                "valid_size_bins": metadata["valid_size_bins"],
                "valid_acsm_features": metadata["valid_acsm_features"],
            }
        )
    return rows


def _write_feature_importance(
    path: Path,
    model: Pipeline,
    feature_names: list[str],
) -> None:
    importances = np.asarray(model.named_steps["forest"].feature_importances_, dtype=np.float64)
    order = np.argsort(importances)[::-1]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature", "importance"])
        writer.writeheader()
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature": feature_names[int(index)],
                    "importance": float(importances[int(index)]),
                }
            )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    arrays = load_prepared_arrays(args.prepared_arrays)
    train_dataset, feature_names, diameter_grid = build_rf_dataset(
        arrays,
        split=args.train_split,
        max_samples=args.max_train_samples,
        limit_rows=args.limit_train_rows,
        random_state=args.random_state,
        shuffle_rows=True,
        include_metadata=False,
        transition_decades=args.merge_transition_decades,
        require_complete_acsm=not args.allow_missing_acsm,
    )
    eval_dataset, eval_feature_names, eval_diameter_grid = build_rf_dataset(
        arrays,
        split=args.eval_split,
        max_samples=args.max_eval_samples,
        limit_rows=args.limit_eval_rows,
        random_state=args.random_state,
        shuffle_rows=False,
        include_metadata=True,
        transition_decades=args.merge_transition_decades,
        require_complete_acsm=not args.allow_missing_acsm,
    )
    if feature_names != eval_feature_names or not np.array_equal(diameter_grid, eval_diameter_grid):
        raise ValueError("Train/eval RF feature schemas do not match.")

    model = _fit_model(args)
    model.fit(train_dataset.x, train_dataset.y_log1p_ccn)
    predicted_log1p = model.predict(eval_dataset.x)
    rows = _prediction_rows(eval_dataset, predicted_log1p)
    metrics = ccn_prediction_metrics(rows)

    prediction_path = output / f"{args.eval_split}_random_forest_ccn_predictions.csv"
    write_rows_csv(prediction_path, rows)
    importance_path = output / "random_forest_feature_importance.csv"
    _write_feature_importance(importance_path, model, feature_names)

    if args.save_model:
        model_path = output / "random_forest_ccn_model.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "diameter_grid_nm": diameter_grid,
                "args": vars(args),
            },
            model_path,
        )

    summary = {
        "prepared_arrays": args.prepared_arrays,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_samples": int(train_dataset.x.shape[0]),
        "train_sampled_rows": int(train_dataset.sampled_rows),
        "eval_samples": int(eval_dataset.x.shape[0]),
        "eval_sampled_rows": int(eval_dataset.sampled_rows),
        "feature_count": int(train_dataset.x.shape[1]),
        "diameter_bin_count": int(diameter_grid.size),
        "diameter_min_nm": float(diameter_grid.min()),
        "diameter_max_nm": float(diameter_grid.max()),
        "merge_transition_decades": args.merge_transition_decades,
        "require_complete_acsm": not args.allow_missing_acsm,
        "random_forest": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "random_state": args.random_state,
        },
        "metrics": metrics,
        "outputs": {
            "predictions": str(prediction_path),
            "feature_importance": str(importance_path),
        },
    }
    summary_path = output / f"{args.eval_split}_random_forest_ccn_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"wrote {prediction_path}")
    print(f"wrote {importance_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
