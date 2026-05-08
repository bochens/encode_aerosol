from __future__ import annotations

import argparse

from .config import load_config
from .feature_store import build_feature_frame, save_feature_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly ARM SGP aerosol feature store.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--start", required=True, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--output", required=True, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frame, modality_columns = build_feature_frame(config, args.start, args.end)
    npz_path = save_feature_store(
        frame=frame,
        modality_columns=modality_columns,
        config=config,
        output_dir=args.output,
        start=args.start,
        end=args.end,
    )
    print(f"wrote {npz_path}")
    print(f"rows={len(frame)} features={len(frame.columns)}")


if __name__ == "__main__":
    main()

