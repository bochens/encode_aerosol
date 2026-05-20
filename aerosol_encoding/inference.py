from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .loss_masks import feature_role
from .model import build_model_from_checkpoint


@dataclass(frozen=True)
class InferenceBatch:
    x_by_modality: dict[str, torch.Tensor]
    feature_mask_by_modality: dict[str, torch.Tensor]
    input_modality_mask: dict[str, torch.Tensor]


class AerosolCCNRetriever:
    """Checkpoint-backed CCN retrieval from partial aerosol instrument inputs.

    Input values must already be in the same feature space as the training
    feature table: temporal windows, size-grid interpolation, angle encoding,
    and log1p transforms must match the selected checkpoint features.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.checkpoint: dict[str, Any] = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model = build_model_from_checkpoint(self.checkpoint).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()

        self.feature_names = list(self.checkpoint["feature_names"])
        self.feature_to_index = {
            feature_name: index for index, feature_name in enumerate(self.feature_names)
        }
        self.mean = np.asarray(self.checkpoint["mean"], dtype=np.float32)
        self.std = np.asarray(self.checkpoint["std"], dtype=np.float32)
        self.modality_indices = {
            modality: [int(index) for index in indices]
            for modality, indices in self.checkpoint["modality_indices"].items()
        }
        self.modality_names = tuple(self.modality_indices)

    def feature_template(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for modality, indices in self.modality_indices.items():
            for local_index, feature_index in enumerate(indices):
                feature_name = self.feature_names[feature_index]
                rows.append(
                    {
                        "modality": modality,
                        "local_index": local_index,
                        "feature_name": feature_name,
                        "role": feature_role(modality, feature_name),
                        "training_mean": float(self.mean[feature_index]),
                        "training_std": float(self.std[feature_index]),
                    }
                )
        return pd.DataFrame(rows)

    def _coerce_feature_frame(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        elif isinstance(features, pd.Series):
            frame = features.to_frame().T
        else:
            frame = pd.DataFrame([dict(features)])
        frame = frame.reset_index(drop=True)
        recognized = [column for column in frame.columns if column in self.feature_to_index]
        if not recognized:
            raise ValueError(
                "No input columns match checkpoint feature names. Use "
                "feature_template() or --write-template to inspect expected columns."
            )
        return frame

    def _normalized_matrix(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.full((len(frame), len(self.feature_names)), np.nan, dtype=np.float32)
        for column in frame.columns:
            index = self.feature_to_index.get(str(column))
            if index is None:
                continue
            values[:, index] = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float32,
            )
        feature_mask = np.isfinite(values)
        normalized = (values - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        normalized = np.where(feature_mask, normalized, 0.0).astype(np.float32)
        return normalized, feature_mask.astype(np.float32)

    def _batch(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame,
        *,
        input_modalities: Iterable[str] | None = None,
        use_ccn_input: bool = False,
    ) -> InferenceBatch:
        frame = self._coerce_feature_frame(features)
        normalized, feature_mask = self._normalized_matrix(frame)
        x = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(feature_mask, dtype=torch.float32, device=self.device)
        allowed = set(input_modalities) if input_modalities is not None else set(self.modality_names)
        unknown = sorted(allowed - set(self.modality_names))
        if unknown:
            raise ValueError(f"Unknown input modalities: {unknown}")
        if not use_ccn_input:
            allowed.discard("ccn_activation")

        x_by_modality: dict[str, torch.Tensor] = {}
        feature_mask_by_modality: dict[str, torch.Tensor] = {}
        input_modality_mask: dict[str, torch.Tensor] = {}
        any_visible = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        for modality, indices in self.modality_indices.items():
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
            modality_x = x.index_select(1, index_tensor)
            modality_mask = mask.index_select(1, index_tensor)
            visible = (modality_mask.sum(dim=1) > 0) & (modality in allowed)
            x_by_modality[modality] = modality_x
            feature_mask_by_modality[modality] = modality_mask
            input_modality_mask[modality] = visible
            any_visible |= visible
        if not torch.all(any_visible):
            bad_rows = torch.nonzero(~any_visible, as_tuple=False).flatten().cpu().tolist()
            raise ValueError(
                "At least one non-CCN input modality must be visible for every row. "
                f"Rows without visible inputs: {bad_rows[:10]}"
            )
        return InferenceBatch(
            x_by_modality=x_by_modality,
            feature_mask_by_modality=feature_mask_by_modality,
            input_modality_mask=input_modality_mask,
        )

    @torch.no_grad()
    def encode(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame,
        *,
        input_modalities: Iterable[str] | None = None,
        use_ccn_input: bool = False,
    ) -> torch.Tensor:
        batch = self._batch(
            features,
            input_modalities=input_modalities,
            use_ccn_input=use_ccn_input,
        )
        return self.model.encode(
            batch.x_by_modality,
            batch.feature_mask_by_modality,
            batch.input_modality_mask,
        )

    @torch.no_grad()
    def predict_ccn(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame,
        supersaturation_percent: float | Sequence[float] | np.ndarray,
        *,
        input_modalities: Iterable[str] | None = None,
        use_ccn_input: bool = False,
    ) -> pd.DataFrame:
        z = self.encode(
            features,
            input_modalities=input_modalities,
            use_ccn_input=use_ccn_input,
        )
        ss = np.atleast_1d(np.asarray(supersaturation_percent, dtype=np.float32))
        if np.any(~np.isfinite(ss)):
            raise ValueError("Supersaturation values must be finite.")
        ss_tensor = torch.as_tensor(ss, dtype=torch.float32, device=self.device)
        prediction = self.model.decode_ccn_at_supersaturation(
            z,
            ss_tensor,
            physical=True,
        )
        rows: list[dict[str, float | int]] = []
        prediction_np = prediction.detach().cpu().numpy()
        for row_index in range(prediction_np.shape[0]):
            for ss_index, ss_value in enumerate(ss):
                rows.append(
                    {
                        "row": row_index,
                        "supersaturation_percent": float(ss_value),
                        "predicted_N_CCN": float(prediction_np[row_index, ss_index]),
                    }
                )
        return pd.DataFrame(rows)
