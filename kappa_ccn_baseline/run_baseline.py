from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aerosol_encoding.training_data import load_prepared_arrays

from .chemistry import ACSMKappaRecipe, acsm_component_fractions, kappa_from_acsm_masses
from .kohler import geometric_mean, geometric_std, predict_ccn_concentration
from .metrics import ccn_prediction_metrics, write_rows_csv
from .prepared_data import (
    acsm_masses_from_row,
    build_acsm_indices,
    build_ccn_indices,
    ccn_observations_from_row,
    transformed_row,
)
from .size_merge import build_spectrum_indices, merged_row_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict CCN from ACSM-derived kappa and merged size distributions."
    )
    parser.add_argument("--prepared-arrays", required=True, help="prepared_arrays.npz path.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test", "all"],
        help="Prepared-array split to evaluate.",
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional smoke-test row cap.")
    parser.add_argument("--kappa-organic", type=float, default=0.12)
    parser.add_argument("--kappa-inorganic", type=float, default=0.63)
    parser.add_argument("--fraction-basis", choices=["mass", "volume"], default="mass")
    parser.add_argument("--organic-density", type=float, default=1.4)
    parser.add_argument("--inorganic-density", type=float, default=1.75)
    parser.add_argument(
        "--merge-transition-decades",
        type=float,
        default=0.08,
        help="Error-function taper width for instrument overlap in log10(Dp).",
    )
    return parser.parse_args()


def _row_indices(arrays, split: str, limit_rows: int | None) -> np.ndarray:
    if split == "all":
        indices = np.arange(arrays.x.shape[0], dtype=np.int64)
    else:
        indices = np.asarray(arrays.splits[split], dtype=np.int64)
    if limit_rows is not None:
        indices = indices[: max(int(limit_rows), 0)]
    return indices


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    arrays = load_prepared_arrays(args.prepared_arrays)
    acsm_indices = build_acsm_indices(arrays.feature_names)
    ccn_indices = build_ccn_indices(arrays.feature_names)
    spectrum_indices = build_spectrum_indices(arrays.feature_names)
    if not spectrum_indices:
        raise ValueError("No size-distribution dN/dlogDp features found.")
    if not ccn_indices:
        raise ValueError("No CCN observation features found.")

    recipe = ACSMKappaRecipe(
        kappa_organic=args.kappa_organic,
        kappa_inorganic=args.kappa_inorganic,
        fraction_basis=args.fraction_basis,
        organic_density_g_cm3=args.organic_density,
        inorganic_density_g_cm3=args.inorganic_density,
    )
    rows: list[dict[str, Any]] = []
    kappa_values: list[float] = []
    split_indices = _row_indices(arrays, args.split, args.limit_rows)

    for row_index in split_indices:
        values, mask = transformed_row(arrays, int(row_index))
        masses = acsm_masses_from_row(values, mask, acsm_indices)
        kappa = kappa_from_acsm_masses(recipe=recipe, **masses)
        if not np.isfinite(kappa):
            continue
        organic_fraction, inorganic_fraction = acsm_component_fractions(recipe=recipe, **masses)
        diameter_nm, dndlogdp, merge_weight = merged_row_spectrum(
            values,
            mask,
            spectrum_indices,
            transition_decades=args.merge_transition_decades,
        )
        if diameter_nm.size < 2 or not np.any(np.isfinite(dndlogdp)):
            continue
        observations = ccn_observations_from_row(values, mask, ccn_indices)
        if not observations:
            continue

        kappa_values.append(kappa)
        for observation in observations:
            predicted, dcrit_nm = predict_ccn_concentration(
                diameter_nm,
                dndlogdp,
                kappa,
                observation["supersaturation_percent"],
            )
            if not np.isfinite(predicted):
                continue
            observed = observation["observed_ccn_cm3"]
            n_ccn_index = int(observation["n_ccn_index"])
            predicted_log1p = float(np.log1p(predicted))
            log1p_error = predicted_log1p - observation["observed_log1p_ccn"]
            standardized_log1p_error = log1p_error / float(arrays.std[n_ccn_index])
            rows.append(
                {
                    "row_index": int(row_index),
                    "time": str(arrays.times[int(row_index)]),
                    "time_bin": int(observation["time_bin"]),
                    "n_ccn_feature_index": n_ccn_index,
                    "supersaturation_percent": observation["supersaturation_percent"],
                    "kappa_chem": kappa,
                    "organic_fraction": organic_fraction,
                    "inorganic_fraction": inorganic_fraction,
                    "fraction_basis": recipe.fraction_basis,
                    "critical_diameter_nm": dcrit_nm,
                    "predicted_ccn_cm3": predicted,
                    "observed_ccn_cm3": observed,
                    "error_cm3": predicted - observed,
                    "predicted_log1p_ccn": predicted_log1p,
                    "observed_log1p_ccn": observation["observed_log1p_ccn"],
                    "log1p_error": log1p_error,
                    "standardized_log1p_error": standardized_log1p_error,
                    "merged_min_diameter_nm": float(np.nanmin(diameter_nm)),
                    "merged_max_diameter_nm": float(np.nanmax(diameter_nm)),
                    "merged_valid_bins": int(np.isfinite(dndlogdp).sum()),
                    "merge_weight_sum": float(np.nansum(merge_weight)),
                }
            )

    prediction_path = output / f"{args.split}_kappa_ccn_predictions.csv"
    write_rows_csv(prediction_path, rows)
    summary = {
        "prepared_arrays": args.prepared_arrays,
        "split": args.split,
        "evaluated_rows": int(split_indices.size),
        "prediction_rows": int(len(rows)),
        "recipe": {
            "kappa_organic": recipe.kappa_organic,
            "kappa_inorganic": recipe.kappa_inorganic,
            "fraction_basis": recipe.fraction_basis,
            "organic_density_g_cm3": recipe.organic_density_g_cm3,
            "inorganic_density_g_cm3": recipe.inorganic_density_g_cm3,
            "merge_transition_decades": args.merge_transition_decades,
        },
        "metrics": ccn_prediction_metrics(rows),
        "kappa_geometric_mean": geometric_mean(np.asarray(kappa_values, dtype=np.float64)),
        "kappa_geometric_std": geometric_std(np.asarray(kappa_values, dtype=np.float64)),
        "instrument_spectra": {
            modality: {
                "diameter_min_nm": float(index.diameter_nm.min()),
                "diameter_max_nm": float(index.diameter_nm.max()),
                "diameter_bins": int(index.diameter_nm.size),
            }
            for modality, index in spectrum_indices.items()
        },
    }
    summary_path = output / f"{args.split}_kappa_ccn_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"wrote {prediction_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
