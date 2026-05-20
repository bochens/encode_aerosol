from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .inference import AerosolCCNRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer CCN concentration at requested supersaturations."
    )
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint path.")
    parser.add_argument(
        "--input",
        help=(
            "CSV with processed feature columns matching checkpoint feature names. "
            "Not required when --write-template is used."
        ),
    )
    parser.add_argument("--output", help="Output CSV for CCN predictions.")
    parser.add_argument(
        "--supersaturation",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.4, 0.8],
        help="Supersaturation values in percent, e.g. 0.2 for 0.2%%.",
    )
    parser.add_argument(
        "--input-modalities",
        nargs="*",
        help="Optional list of modalities allowed as inputs.",
    )
    parser.add_argument(
        "--use-ccn-input",
        action="store_true",
        help="Allow ccn_activation features as inputs. Off by default to avoid target leakage.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--write-template",
        help="Write a CSV listing expected feature names, roles, and training statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = AerosolCCNRetriever(args.checkpoint, device=args.device)

    if args.write_template:
        template_path = Path(args.write_template)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        retriever.feature_template().to_csv(template_path, index=False)
        print(f"wrote {template_path}")
        if args.input is None:
            return

    if args.input is None:
        raise ValueError("--input is required unless only --write-template is requested.")
    if args.output is None:
        raise ValueError("--output is required when --input is provided.")

    features = pd.read_csv(args.input)
    predictions = retriever.predict_ccn(
        features,
        args.supersaturation,
        input_modalities=args.input_modalities,
        use_ccn_input=args.use_ccn_input,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
