from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .feature_store import load_feature_store
from .model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latent aerosol encodings.")
    parser.add_argument("--features", required=True, help="features.npz from build_features.")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint.pt.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def split_modalities(
    batch_x: torch.Tensor,
    batch_mask: torch.Tensor,
    modality_indices: dict[str, list[int]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    x_by_modality = {}
    mask_by_modality = {}
    input_mask = {}
    for modality, indices in modality_indices.items():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=batch_x.device)
        x_mod = batch_x.index_select(1, index_tensor)
        mask_mod = batch_mask.index_select(1, index_tensor)
        x_by_modality[modality] = x_mod
        mask_by_modality[modality] = mask_mod
        input_mask[modality] = mask_mod.sum(dim=1) > 0
    return x_by_modality, mask_by_modality, input_mask


def main() -> None:
    args = parse_args()
    matrix, times, _ = load_feature_store(args.features)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

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
    times = times[valid_rows]

    model = build_model_from_checkpoint(checkpoint).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = TensorDataset(torch.from_numpy(normalized), torch.from_numpy(feature_mask))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    latents: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    modality_names = list(checkpoint["modality_indices"].keys())

    with torch.no_grad():
        for batch_x, batch_mask in loader:
            batch_x = batch_x.to(args.device)
            batch_mask = batch_mask.to(args.device)
            x_by_modality, mask_by_modality, input_mask = split_modalities(
                batch_x,
                batch_mask,
                checkpoint["modality_indices"],
            )
            z = model.encode(x_by_modality, mask_by_modality, input_mask)
            latents.append(z.cpu().numpy())
            observed_rows.append(
                np.column_stack([
                    input_mask[modality].cpu().numpy()
                    for modality in modality_names
                ])
            )

    z_all = np.concatenate(latents, axis=0)
    observed = np.concatenate(observed_rows, axis=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "encodings.npz",
        times=times,
        z=z_all,
        modality_observed=observed,
        modality_names=np.array(modality_names),
        latent_blocks=np.array(list(checkpoint.get("latent_blocks", {}).keys())),
        latent_block_dims=np.array(list(checkpoint.get("latent_blocks", {}).values())),
    )

    frame = pd.DataFrame(
        z_all,
        columns=[f"z_{idx:02d}" for idx in range(z_all.shape[1])],
    )
    frame.insert(0, "time", times.astype("datetime64[ns]").astype(str))
    for idx, modality in enumerate(modality_names):
        frame[f"observed_{modality}"] = observed[:, idx].astype(bool)
    frame.to_csv(output / "encodings.csv", index=False)
    print(f"wrote {output / 'encodings.npz'}")
    print(f"wrote {output / 'encodings.csv'}")


if __name__ == "__main__":
    main()
