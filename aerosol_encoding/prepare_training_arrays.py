from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .feature_store import load_feature_store
from .training_data import prepare_arrays, save_prepared_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and cache aerosol training arrays for low-RAM training runs."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--features", required=True, help="features.npz from build_features.")
    parser.add_argument("--output", required=True, help="Output prepared_arrays.npz path.")
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional JSON summary path. Defaults to output path with .json suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    matrix, times, metadata = load_feature_store(args.features)
    arrays = prepare_arrays(
        matrix=matrix,
        times=times,
        metadata=metadata,
        min_feature_coverage=config.min_feature_coverage,
        min_feature_std=config.min_feature_std,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        split_strategy=config.split_strategy,
        feature_coverage_basis=config.feature_coverage_basis,
    )
    save_prepared_arrays(args.output, arrays)

    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "x_shape": list(arrays.x.shape),
                "x_dtype": str(arrays.x.dtype),
                "feature_mask_shape": list(arrays.feature_mask.shape),
                "feature_mask_dtype": str(arrays.feature_mask.dtype),
                "times_shape": list(arrays.times.shape),
                "splits": {key: int(len(value)) for key, value in arrays.splits.items()},
                "modality_dims": {
                    modality: int(len(indices))
                    for modality, indices in arrays.modality_indices.items()
                },
                "n_feature_names": int(len(arrays.feature_names)),
            },
            handle,
            indent=2,
        )
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
