from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


SIZING_MODALITIES = ("size_smps", "size_aps", "size_uhsas", "size_opc")
TIME_BIN_RE = re.compile(r"^(?P<base>.+)__time_bin_(?P<bin>\d+)$")


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("MLP depth must be at least 1")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth - 1):
        layers.extend([nn.Linear(current, hidden_dim), nn.GELU()])
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class GroupedMaskedAutoencoder(nn.Module):
    def __init__(
        self,
        modality_dims: Mapping[str, int],
        target_modalities: tuple[str, ...],
        hidden_dim: int,
        latent_dim: int,
        encoder_depth: int,
        decoder_depth: int,
    ) -> None:
        super().__init__()
        self.modality_names = tuple(modality_dims.keys())
        self.modality_to_index = {
            name: index
            for index, name in enumerate(self.modality_names)
        }
        self.target_modalities = tuple(target_modalities)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict(
            {
                name: make_mlp(dim * 2, hidden_dim, hidden_dim, encoder_depth)
                for name, dim in modality_dims.items()
            }
        )
        self.modality_embeddings = nn.Parameter(
            torch.zeros(len(self.modality_names), hidden_dim)
        )
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        self.gate_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim + 1, 1)
                for name in self.modality_names
            }
        )
        self.attention_head = nn.Linear(hidden_dim, 1)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoders = nn.ModuleDict(
            {
                name: make_mlp(latent_dim, hidden_dim, modality_dims[name], decoder_depth)
                for name in self.target_modalities
            }
        )

    def encode(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        encoded_modalities: list[torch.Tensor] = []
        attention_scores: list[torch.Tensor] = []
        modality_masks: list[torch.Tensor] = []
        for name in self.modality_names:
            x = x_by_modality[name]
            feature_mask = feature_mask_by_modality[name]
            modality_mask = input_modality_mask[name].to(dtype=x.dtype).unsqueeze(-1)

            observed_fraction = feature_mask.mean(dim=-1, keepdim=True)
            modality_embedding = self.modality_embeddings[self.modality_to_index[name]]
            encoded = self.encoders[name](torch.cat([x * feature_mask, feature_mask], dim=-1))
            encoded = encoded + modality_embedding.unsqueeze(0)

            gate = torch.sigmoid(self.gate_heads[name](torch.cat([encoded, observed_fraction], dim=-1)))
            score = self.attention_head(torch.tanh(encoded)) + torch.log(gate.clamp_min(1e-6))
            score = score.masked_fill(modality_mask <= 0, -1e9)
            encoded_modalities.append(encoded)
            attention_scores.append(score)
            modality_masks.append(modality_mask)

        if not encoded_modalities:
            raise RuntimeError("Model has no modalities")

        masks = torch.cat(modality_masks, dim=-1)
        any_visible = masks.sum(dim=1, keepdim=True) > 0
        if not torch.all(any_visible):
            raise RuntimeError("At least one row had no visible modalities")

        encoded_stack = torch.stack(encoded_modalities, dim=1)
        score_stack = torch.cat(attention_scores, dim=-1)
        weights = torch.softmax(score_stack, dim=1).unsqueeze(-1)
        pooled = (encoded_stack * weights).sum(dim=1)
        return self.latent_head(pooled)

    def forward(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(x_by_modality, feature_mask_by_modality, input_modality_mask)
        decoded = {name: self.decoders[name](z) for name in self.target_modalities}
        return z, decoded


def _infer_sequence_indices(modality: str, feature_names: tuple[str, ...]) -> tuple[int, ...]:
    if any(TIME_BIN_RE.match(feature_name) for feature_name in feature_names):
        return ()
    if any("__dN_dlogDp__" in feature_name for feature_name in feature_names):
        return tuple(
            index
            for index, feature_name in enumerate(feature_names)
            if "__dN_dlogDp__" in feature_name
        )
    if modality == "optical_neph":
        return tuple(
            index
            for index, feature_name in enumerate(feature_names)
            if "__Bs_" in feature_name or "__Bbs_" in feature_name
        )
    return ()


def _infer_temporal_indices(feature_names: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    by_bin: dict[int, dict[str, int]] = {}
    channel_order: list[str] = []
    for index, feature_name in enumerate(feature_names):
        match = TIME_BIN_RE.match(feature_name)
        if match is None:
            continue
        channel = match.group("base")
        time_bin = int(match.group("bin"))
        by_bin.setdefault(time_bin, {})[channel] = index
        if time_bin == 0:
            channel_order.append(channel)

    if not by_bin:
        return ()
    if not channel_order:
        first_bin = min(by_bin)
        channel_order = list(by_bin[first_bin])
    expected_bins = tuple(range(max(by_bin) + 1))
    if tuple(sorted(by_bin)) != expected_bins:
        raise ValueError(
            "Temporal feature bins must be contiguous starting at zero; got "
            f"{sorted(by_bin)}"
        )
    rows: list[tuple[int, ...]] = []
    for time_bin in expected_bins:
        channels = by_bin[time_bin]
        missing = [channel for channel in channel_order if channel not in channels]
        if missing:
            raise ValueError(
                f"Temporal bin {time_bin} is missing channels: {missing[:5]}"
            )
        rows.append(tuple(channels[channel] for channel in channel_order))
    return tuple(rows)


def _time_position_features(n_steps: int, fourier_frequencies: int) -> torch.Tensor:
    if n_steps <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    if n_steps == 1:
        scaled = torch.zeros((1, 1), dtype=torch.float32)
    else:
        position = torch.linspace(-1.0, 1.0, steps=n_steps, dtype=torch.float32).unsqueeze(-1)
        scaled = position
    features = [scaled]
    for frequency in range(1, fourier_frequencies + 1):
        angle = scaled * (math.pi * frequency)
        features.extend([torch.sin(angle), torch.cos(angle)])
    return torch.cat(features, dim=-1)


DIAMETER_LABEL_RE = re.compile(
    r"^(?P<coordinate>.+)_(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
DIAMETER_IN_FEATURE_RE = re.compile(
    r"(?P<coordinate>diameter_mobility|diameter_optical|"
    r"diameter_aerodynamic|diameter_midpoint|diameter_common_nm)_"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

DIAMETER_UNITS = {
    "diameter_mobility": "nm",
    "diameter_optical": "nm",
    "diameter_aerodynamic": "um",
    "diameter_midpoint": "um",
    "diameter_common_nm": "nm",
}


def _diameter_nm_from_feature_name(feature_name: str) -> float | None:
    if "__dN_dlogDp__" not in feature_name:
        return None
    match = DIAMETER_IN_FEATURE_RE.search(feature_name)
    if match is None:
        return None
    coordinate = match.group("coordinate")
    value = float(match.group("value"))
    unit = DIAMETER_UNITS.get(coordinate)
    if unit == "nm":
        return value
    if unit == "um":
        return value * 1000.0
    return None


def _diameter_position_features(
    feature_names: tuple[str, ...],
    sequence_indices: tuple[int, ...],
    fourier_frequencies: int,
) -> torch.Tensor:
    if not sequence_indices:
        return torch.empty((0, 0), dtype=torch.float32)
    diameters = [
        _diameter_nm_from_feature_name(feature_names[index])
        for index in sequence_indices
    ]
    if any(diameter is None or diameter <= 0 for diameter in diameters):
        return torch.empty((len(sequence_indices), 0), dtype=torch.float32)

    log_dp = torch.log10(torch.as_tensor(diameters, dtype=torch.float32))
    center = 0.5 * (log_dp.max() + log_dp.min())
    half_range = 0.5 * (log_dp.max() - log_dp.min()).clamp_min(1e-6)
    scaled = ((log_dp - center) / half_range).unsqueeze(-1)
    features = [scaled]
    for frequency in range(1, fourier_frequencies + 1):
        angle = scaled * (math.pi * frequency)
        features.extend([torch.sin(angle), torch.cos(angle)])
    return torch.cat(features, dim=-1)


class StructuredModalityEncoder(nn.Module):
    def __init__(
        self,
        modality: str,
        input_dim: int,
        feature_names: tuple[str, ...],
        hidden_dim: int,
        depth: int,
        sequence_encoder_type: str = "conv",
        sequence_fourier_frequencies: int = 0,
        sequence_transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        if len(feature_names) != input_dim:
            raise ValueError(
                f"Feature-name count for {modality} does not match input dim: "
                f"{len(feature_names)} != {input_dim}"
            )

        temporal_indices = _infer_temporal_indices(feature_names)
        temporal_flat_indices = {
            index
            for row in temporal_indices
            for index in row
        }
        sequence_indices = _infer_sequence_indices(modality, feature_names)
        scalar_indices = tuple(
            index
            for index in range(input_dim)
            if index not in set(sequence_indices) and index not in temporal_flat_indices
        )
        self.modality = modality
        self.input_dim = input_dim
        self.temporal_step_count = len(temporal_indices)
        self.temporal_channel_count = len(temporal_indices[0]) if temporal_indices else 0
        self.sequence_feature_count = len(sequence_indices)
        self.scalar_feature_count = len(scalar_indices)
        self.sequence_encoder_type = sequence_encoder_type

        if temporal_indices:
            temporal_index_tensor = torch.as_tensor(temporal_indices, dtype=torch.long)
        else:
            temporal_index_tensor = torch.empty((0, 0), dtype=torch.long)
        self.register_buffer(
            "temporal_indices",
            temporal_index_tensor,
            persistent=False,
        )
        self.register_buffer(
            "sequence_indices",
            torch.as_tensor(sequence_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "scalar_indices",
            torch.as_tensor(scalar_indices, dtype=torch.long),
            persistent=False,
        )

        temporal_position_features = _time_position_features(
            self.temporal_step_count,
            fourier_frequencies=sequence_fourier_frequencies,
        )
        self.register_buffer(
            "temporal_position_features",
            temporal_position_features,
            persistent=False,
        )
        if temporal_indices:
            temporal_position_dim = int(temporal_position_features.shape[-1])
            self.temporal_input_projection = nn.Sequential(
                nn.Linear(self.temporal_channel_count * 2 + temporal_position_dim, hidden_dim),
                nn.GELU(),
            )
            self.temporal_encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.temporal_output_norm = nn.LayerNorm(hidden_dim)
        else:
            self.temporal_input_projection = None
            self.temporal_encoder = None
            self.temporal_output_norm = None

        position_features = _diameter_position_features(
            feature_names,
            sequence_indices,
            fourier_frequencies=sequence_fourier_frequencies,
        )
        self.register_buffer("sequence_position_features", position_features, persistent=False)
        position_dim = int(position_features.shape[-1])
        sequence_is_diameter = position_dim > 0
        effective_sequence_encoder_type = (
            sequence_encoder_type
            if sequence_is_diameter
            else "conv"
        )

        sequence_channels = max(16, hidden_dim // 2)
        if sequence_indices:
            if effective_sequence_encoder_type == "diameter_transformer":
                self.sequence_input_projection = nn.Linear(2 + position_dim, hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=sequence_transformer_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.05,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.sequence_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=1,
                    norm=nn.LayerNorm(hidden_dim),
                    enable_nested_tensor=False,
                )
                self.sequence_projection = nn.Identity()
            else:
                conv_input_channels = 2 + (
                    position_dim
                    if effective_sequence_encoder_type == "diameter_fourier_conv"
                    else 0
                )
                self.sequence_input_projection = None
                self.sequence_encoder = nn.Sequential(
                    nn.Conv1d(conv_input_channels, sequence_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(sequence_channels, sequence_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                )
                self.sequence_projection = nn.Linear(sequence_channels, hidden_dim)
        else:
            self.sequence_encoder = None
            self.sequence_projection = None
            self.sequence_input_projection = None

        if scalar_indices:
            self.scalar_encoder = make_mlp(
                len(scalar_indices) * 2,
                hidden_dim,
                hidden_dim,
                depth,
            )
        else:
            self.scalar_encoder = None

        self.coverage_projection = nn.Linear(1, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _temporal_token(
        self,
        x: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        flat_indices = self.temporal_indices.reshape(-1)
        values = x.index_select(1, flat_indices).reshape(
            x.shape[0],
            self.temporal_step_count,
            self.temporal_channel_count,
        )
        masks = feature_mask.index_select(1, flat_indices).reshape_as(values)
        position = self.temporal_position_features.to(
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(0).expand(x.shape[0], -1, -1)
        step_input = torch.cat([values * masks, masks, position], dim=-1)
        projected = self.temporal_input_projection(step_input)  # type: ignore[misc]
        encoded, _ = self.temporal_encoder(projected)  # type: ignore[misc]
        step_mask = (masks.sum(dim=-1) > 0).to(dtype=x.dtype).unsqueeze(-1)
        weight_sum = step_mask.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * step_mask).sum(dim=1) / weight_sum
        return self.temporal_output_norm(pooled)  # type: ignore[misc]

    def _masked_sequence_mean(
        self,
        sequence_features: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.sequence_encoder_type == "diameter_transformer" and self.sequence_position_features.numel() > 0:
            position = self.sequence_position_features.to(
                dtype=sequence_features.dtype,
                device=sequence_features.device,
            ).unsqueeze(0).expand(sequence_features.shape[0], -1, -1)
            token_input = torch.cat(
                [
                    sequence_features[:, 0, :].unsqueeze(-1),
                    sequence_features[:, 1, :].unsqueeze(-1),
                    position,
                ],
                dim=-1,
            )
            projected = self.sequence_input_projection(token_input)  # type: ignore[misc]
            padding_mask = sequence_mask <= 0
            all_missing = padding_mask.all(dim=1, keepdim=True)
            padding_mask = torch.where(
                all_missing,
                torch.zeros_like(padding_mask),
                padding_mask,
            )
            encoded_tokens = self.sequence_encoder(  # type: ignore[misc]
                projected,
                src_key_padding_mask=padding_mask,
            )
            weights = sequence_mask.unsqueeze(-1)
            weight_sum = weights.sum(dim=1).clamp_min(1.0)
            return (encoded_tokens * weights).sum(dim=1) / weight_sum

        if self.sequence_position_features.numel() > 0 and self.sequence_encoder_type == "diameter_fourier_conv":
            position_channels = self.sequence_position_features.to(
                dtype=sequence_features.dtype,
                device=sequence_features.device,
            ).transpose(0, 1).unsqueeze(0).expand(sequence_features.shape[0], -1, -1)
            sequence_features = torch.cat([sequence_features, position_channels], dim=1)
        encoded = self.sequence_encoder(sequence_features)  # type: ignore[misc]
        weights = sequence_mask.unsqueeze(1)
        weight_sum = weights.sum(dim=-1).clamp_min(1.0)
        return (encoded * weights).sum(dim=-1) / weight_sum

    def forward(self, x: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        token = x.new_zeros(batch_size, self.output_norm.normalized_shape[0])

        if self.temporal_step_count > 0:
            token = token + self._temporal_token(x, feature_mask)

        if self.sequence_feature_count > 0:
            sequence_values = x.index_select(1, self.sequence_indices)
            sequence_mask = feature_mask.index_select(1, self.sequence_indices)
            sequence_input = torch.stack(
                [sequence_values * sequence_mask, sequence_mask],
                dim=1,
            )
            sequence_token = self._masked_sequence_mean(sequence_input, sequence_mask)
            token = token + self.sequence_projection(sequence_token)  # type: ignore[misc]

        if self.scalar_feature_count > 0:
            scalar_values = x.index_select(1, self.scalar_indices)
            scalar_mask = feature_mask.index_select(1, self.scalar_indices)
            scalar_input = torch.cat([scalar_values * scalar_mask, scalar_mask], dim=-1)
            token = token + self.scalar_encoder(scalar_input)  # type: ignore[misc]

        observed_fraction = feature_mask.mean(dim=-1, keepdim=True)
        token = token + self.coverage_projection(observed_fraction)
        return self.output_norm(token)


def _feature_indices_containing(
    feature_names: tuple[str, ...],
    needles: tuple[str, ...],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, feature_name in enumerate(feature_names)
        if any(needle in feature_name for needle in needles)
    )


class ConditionalCCNActivationDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        feature_names: tuple[str, ...],
        depth: int,
    ) -> None:
        super().__init__()
        n_ccn_indices = _feature_indices_containing(feature_names, ("__N_CCN",))
        supersaturation_indices = _feature_indices_containing(
            feature_names,
            ("__supersaturation_calculated", "__supersaturation_set_point"),
        )
        if not n_ccn_indices:
            raise ValueError("Conditional CCN decoder requires an N_CCN feature")
        if not supersaturation_indices:
            raise ValueError("Conditional CCN decoder requires supersaturation features")

        auxiliary_indices = tuple(
            index
            for index in range(output_dim)
            if index not in set(n_ccn_indices) | set(supersaturation_indices)
        )
        self.output_dim = output_dim
        self.register_buffer(
            "n_ccn_indices",
            torch.as_tensor(n_ccn_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "supersaturation_indices",
            torch.as_tensor(supersaturation_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "auxiliary_indices",
            torch.as_tensor(auxiliary_indices, dtype=torch.long),
            persistent=False,
        )
        self.n_ccn_head = make_mlp(
            latent_dim + len(supersaturation_indices) * 2,
            hidden_dim,
            len(n_ccn_indices),
            depth,
        )
        self.auxiliary_head = (
            make_mlp(latent_dim, hidden_dim, len(auxiliary_indices), depth)
            if auxiliary_indices
            else None
        )

    def forward(
        self,
        z: torch.Tensor,
        target_values: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = z.new_zeros(z.shape[0], self.output_dim)
        if target_values is None or target_mask is None:
            ss_values = z.new_zeros(z.shape[0], self.supersaturation_indices.numel())
            ss_mask = z.new_zeros(z.shape[0], self.supersaturation_indices.numel())
        else:
            ss_values = target_values.index_select(1, self.supersaturation_indices)
            ss_mask = target_mask.index_select(1, self.supersaturation_indices)
            output[:, self.supersaturation_indices] = ss_values * ss_mask

        ccn_input = torch.cat([z, ss_values * ss_mask, ss_mask], dim=-1)
        output[:, self.n_ccn_indices] = self.n_ccn_head(ccn_input)
        if self.auxiliary_head is not None and self.auxiliary_indices.numel() > 0:
            output[:, self.auxiliary_indices] = self.auxiliary_head(z)
        return output


def _identity_feature_stats(output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(output_dim, dtype=torch.float32), torch.ones(output_dim, dtype=torch.float32)


def _feature_suffix(feature_name: str, variable_name: str) -> str:
    marker = f"__{variable_name}"
    if marker not in feature_name:
        raise ValueError(f"Feature {feature_name!r} does not contain variable {variable_name!r}")
    return feature_name.split(marker, maxsplit=1)[1]


def _variable_indices_by_suffix(
    feature_names: tuple[str, ...],
    variable_name: str,
    *,
    mean_only: bool = False,
) -> dict[str, int]:
    output: dict[str, int] = {}
    marker = f"__{variable_name}"
    for index, feature_name in enumerate(feature_names):
        if marker not in feature_name:
            continue
        if mean_only and "__stat_" in feature_name:
            continue
        output[_feature_suffix(feature_name, variable_name)] = index
    return output


def _scalar_coordinate_features(values: torch.Tensor, frequencies: int = 4) -> torch.Tensor:
    if values.ndim in {1, 2}:
        values = values.unsqueeze(-1)
    features = [values]
    for frequency in range(1, frequencies + 1):
        angle = values * (math.pi * frequency)
        features.extend([torch.sin(angle), torch.cos(angle)])
    return torch.cat(features, dim=-1)


class CoordinateCCNActivationDecoder(nn.Module):
    """Predict N_CCN from aerosol latent state queried at supersaturation."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        feature_names: tuple[str, ...],
        depth: int,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        coordinate_frequencies: int = 4,
    ) -> None:
        super().__init__()
        n_ccn_by_suffix = _variable_indices_by_suffix(
            feature_names,
            "N_CCN",
            mean_only=True,
        )
        ss_calculated_by_suffix = _variable_indices_by_suffix(
            feature_names,
            "supersaturation_calculated",
            mean_only=True,
        )
        ss_setpoint_by_suffix = _variable_indices_by_suffix(
            feature_names,
            "supersaturation_set_point",
            mean_only=True,
        )
        if not n_ccn_by_suffix:
            raise ValueError("Coordinate CCN decoder requires mean N_CCN features")

        n_ccn_indices: list[int] = []
        ss_calculated_indices: list[int] = []
        ss_setpoint_indices: list[int] = []
        for suffix, n_ccn_index in n_ccn_by_suffix.items():
            calculated_index = ss_calculated_by_suffix.get(suffix)
            setpoint_index = ss_setpoint_by_suffix.get(suffix)
            if calculated_index is None and setpoint_index is None:
                raise ValueError(
                    "Coordinate CCN decoder could not pair N_CCN feature "
                    f"{feature_names[n_ccn_index]!r} with a supersaturation coordinate"
                )
            n_ccn_indices.append(n_ccn_index)
            ss_calculated_indices.append(
                calculated_index if calculated_index is not None else setpoint_index
            )
            ss_setpoint_indices.append(
                setpoint_index if setpoint_index is not None else calculated_index
            )

        if feature_mean is None or feature_std is None:
            mean, std = _identity_feature_stats(output_dim)
        else:
            mean = feature_mean.detach().to(dtype=torch.float32).clone()
            std = feature_std.detach().to(dtype=torch.float32).clone()

        coordinate_indices = ss_calculated_indices + ss_setpoint_indices
        ss_mean = mean[coordinate_indices]
        ss_std = std[coordinate_indices].clamp_min(1e-6)
        center = ss_mean.mean()
        scale = torch.sqrt(((ss_mean - center) ** 2 + ss_std ** 2).mean()).clamp_min(1e-6)

        self.output_dim = output_dim
        self.coordinate_frequencies = coordinate_frequencies
        self.register_buffer("feature_mean", mean, persistent=True)
        self.register_buffer("feature_std", std, persistent=True)
        self.register_buffer("n_ccn_indices", torch.as_tensor(n_ccn_indices, dtype=torch.long), persistent=False)
        self.register_buffer(
            "supersaturation_calculated_indices",
            torch.as_tensor(ss_calculated_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "supersaturation_setpoint_indices",
            torch.as_tensor(ss_setpoint_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("ss_center", center.reshape(1), persistent=True)
        self.register_buffer("ss_scale", scale.reshape(1), persistent=True)
        coordinate_dim = 1 + 2 * coordinate_frequencies
        self.response_head = make_mlp(
            latent_dim + coordinate_dim + 1,
            hidden_dim,
            1,
            depth,
        )

    def _physical_values(self, normalized: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        mean = self.feature_mean.index_select(0, indices).to(normalized.device)
        std = self.feature_std.index_select(0, indices).to(normalized.device)
        return normalized.index_select(1, indices) * std + mean

    def coordinate_features(self, supersaturation_percent: torch.Tensor) -> torch.Tensor:
        scaled = (supersaturation_percent - self.ss_center.to(supersaturation_percent.device)) / self.ss_scale.to(supersaturation_percent.device)
        return _scalar_coordinate_features(scaled, frequencies=self.coordinate_frequencies)

    def forward(
        self,
        z: torch.Tensor,
        target_values: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = z.new_zeros(z.shape[0], self.output_dim)
        if target_values is None or target_mask is None:
            ss_values = z.new_zeros(
                z.shape[0],
                self.supersaturation_calculated_indices.numel(),
            )
            ss_mask = z.new_zeros(
                z.shape[0],
                self.supersaturation_calculated_indices.numel(),
            )
        else:
            calculated_values = self._physical_values(
                target_values,
                self.supersaturation_calculated_indices,
            )
            calculated_mask = target_mask.index_select(
                1,
                self.supersaturation_calculated_indices,
            ).to(dtype=z.dtype)
            setpoint_values = self._physical_values(
                target_values,
                self.supersaturation_setpoint_indices,
            )
            setpoint_mask = target_mask.index_select(
                1,
                self.supersaturation_setpoint_indices,
            ).to(dtype=z.dtype)
            use_calculated = calculated_mask > 0
            ss_values = torch.where(use_calculated, calculated_values, setpoint_values)
            ss_mask = torch.maximum(calculated_mask, setpoint_mask)

        prediction = self.decode_at_supersaturation(z, ss_values, ss_mask)
        response_mean = self.feature_mean.index_select(0, self.n_ccn_indices).to(z.device)
        response_std = self.feature_std.index_select(0, self.n_ccn_indices).to(z.device)
        output[:, self.n_ccn_indices] = (prediction - response_mean) / response_std.clamp_min(1e-6)
        return output

    def decode_at_supersaturation(
        self,
        z: torch.Tensor,
        supersaturation_percent: torch.Tensor,
        coordinate_mask: torch.Tensor | None = None,
        physical: bool = False,
    ) -> torch.Tensor:
        if supersaturation_percent.ndim == 0:
            supersaturation_percent = supersaturation_percent.reshape(1)
        if supersaturation_percent.ndim == 1:
            supersaturation_percent = supersaturation_percent.unsqueeze(0).expand(
                z.shape[0],
                -1,
            )
        if coordinate_mask is None:
            coordinate_mask = torch.ones_like(supersaturation_percent)
        coord = self.coordinate_features(supersaturation_percent.to(dtype=z.dtype))
        query_count = supersaturation_percent.shape[1]
        z_expanded = z.unsqueeze(1).expand(-1, query_count, -1)
        query = torch.cat(
            [z_expanded, coord, coordinate_mask.to(dtype=z.dtype).unsqueeze(-1)],
            dim=-1,
        )
        prediction = self.response_head(query.reshape(-1, query.shape[-1]))
        transformed = prediction.reshape(z.shape[0], query_count) * coordinate_mask.to(dtype=z.dtype)
        if physical:
            return torch.expm1(transformed).clamp_min(0.0)
        return transformed


class CoordinateSizeDistributionDecoder(nn.Module):
    """Predict one instrument's dN/dlogDp as a queryable function of log-diameter.

    A separate instance is created for SMPS, APS, UHSAS, and OPC. The decoder
    does not receive an instrument-id coordinate; choosing the module is the
    instrument-specific routing decision.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        feature_names: tuple[str, ...],
        depth: int,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        coordinate_frequencies: int = 6,
    ) -> None:
        super().__init__()
        spectral: list[tuple[int, float]] = []
        for index, feature_name in enumerate(feature_names):
            diameter_nm = _diameter_nm_from_feature_name(feature_name)
            if diameter_nm is None:
                continue
            spectral.append((index, diameter_nm))
        if not spectral:
            raise ValueError("Coordinate size decoder requires dN_dlogDp diameter features")

        spectral_indices = [index for index, _ in spectral]
        diameter_nm = torch.as_tensor([diameter for _, diameter in spectral], dtype=torch.float32)
        log_dp = torch.log10(diameter_nm)
        center = 0.5 * (log_dp.max() + log_dp.min())
        scale = (0.5 * (log_dp.max() - log_dp.min())).clamp_min(1e-6)
        if feature_mean is None or feature_std is None:
            mean, std = _identity_feature_stats(output_dim)
        else:
            mean = feature_mean.detach().to(dtype=torch.float32).clone()
            std = feature_std.detach().to(dtype=torch.float32).clone()

        self.output_dim = output_dim
        self.coordinate_frequencies = coordinate_frequencies
        self.register_buffer("feature_mean", mean, persistent=True)
        self.register_buffer("feature_std", std, persistent=True)
        self.register_buffer("spectral_indices", torch.as_tensor(spectral_indices, dtype=torch.long), persistent=False)
        self.register_buffer("diameter_nm", diameter_nm, persistent=True)
        self.register_buffer("log_dp_center", center.reshape(1), persistent=True)
        self.register_buffer("log_dp_scale", scale.reshape(1), persistent=True)
        coordinate_dim = 1 + 2 * coordinate_frequencies
        self.response_head = make_mlp(
            latent_dim + coordinate_dim,
            hidden_dim,
            1,
            depth,
        )

    def coordinate_features(self, diameter_nm: torch.Tensor) -> torch.Tensor:
        log_dp = torch.log10(diameter_nm.clamp_min(1e-6))
        scaled = (log_dp - self.log_dp_center.to(diameter_nm.device)) / self.log_dp_scale.to(diameter_nm.device)
        return _scalar_coordinate_features(scaled, frequencies=self.coordinate_frequencies)

    def forward(
        self,
        z: torch.Tensor,
        target_values: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = z.new_zeros(z.shape[0], self.output_dim)
        prediction = self.decode_at_diameter(z, self.diameter_nm.to(z.device))
        response_mean = self.feature_mean.index_select(0, self.spectral_indices).to(z.device)
        response_std = self.feature_std.index_select(0, self.spectral_indices).to(z.device)
        output[:, self.spectral_indices] = (prediction - response_mean) / response_std.clamp_min(1e-6)
        return output

    def decode_at_diameter(
        self,
        z: torch.Tensor,
        diameter_nm: torch.Tensor,
        physical: bool = False,
    ) -> torch.Tensor:
        if diameter_nm.ndim == 0:
            diameter_nm = diameter_nm.reshape(1)
        coord = self.coordinate_features(diameter_nm.to(dtype=z.dtype, device=z.device))
        query_count = coord.shape[0]
        z_expanded = z.unsqueeze(1).expand(-1, query_count, -1)
        coord_expanded = coord.unsqueeze(0).expand(z.shape[0], -1, -1)
        query = torch.cat([z_expanded, coord_expanded], dim=-1)
        prediction = self.response_head(query.reshape(-1, query.shape[-1]))
        transformed = prediction.reshape(z.shape[0], query_count)
        if physical:
            return torch.expm1(transformed).clamp_min(0.0)
        return transformed


NEPH_CHANNELS = {"B": 0, "G": 1, "R": 2}
NEPH_KIND = {"Bs": 0, "Bbs": 1}
NEPH_STATE = {"Dry": 0, "Wet": 1}
NEPH_RESPONSE_RE = re.compile(r"__(?P<kind>Bbs|Bs)_(?P<channel>[BGR])_(?P<state>Dry|Wet)_Neph3W")


class CoordinateNephelometerDecoder(nn.Module):
    """Predict scattering/backscattering conditioned on RH and channel identity."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        feature_names: tuple[str, ...],
        depth: int,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        coordinate_frequencies: int = 4,
    ) -> None:
        super().__init__()
        if feature_mean is None or feature_std is None:
            mean, std = _identity_feature_stats(output_dim)
        else:
            mean = feature_mean.detach().to(dtype=torch.float32).clone()
            std = feature_std.detach().to(dtype=torch.float32).clone()

        response_indices: list[int] = []
        rh_indices: list[int] = []
        categorical_rows: list[list[float]] = []
        for index, feature_name in enumerate(feature_names):
            match = NEPH_RESPONSE_RE.search(feature_name)
            if match is None:
                continue
            variable = match.group(0).removeprefix("__")
            suffix = _feature_suffix(feature_name, variable)
            state = match.group("state")
            rh_variable = f"RH_Neph_{state}"
            rh_feature = feature_name.replace(variable, rh_variable)
            if rh_feature not in feature_names:
                # Same suffix, but robust to stream-specific prefixes.
                rh_feature = next(
                    (
                        candidate
                        for candidate in feature_names
                        if f"__{rh_variable}" in candidate
                        and _feature_suffix(candidate, rh_variable) == suffix
                    ),
                    "",
                )
            if not rh_feature:
                raise ValueError(
                    f"Coordinate nephelometer decoder could not pair {feature_name!r} "
                    f"with {rh_variable}"
                )
            rh_index = feature_names.index(rh_feature)
            response_indices.append(index)
            rh_indices.append(rh_index)
            row = [0.0] * (len(NEPH_CHANNELS) + len(NEPH_KIND) + len(NEPH_STATE))
            row[NEPH_CHANNELS[match.group("channel")]] = 1.0
            row[len(NEPH_CHANNELS) + NEPH_KIND[match.group("kind")]] = 1.0
            row[len(NEPH_CHANNELS) + len(NEPH_KIND) + NEPH_STATE[state]] = 1.0
            categorical_rows.append(row)

        if not response_indices:
            raise ValueError("Coordinate nephelometer decoder requires Bs/Bbs response features")

        rh_mean = mean[rh_indices]
        rh_std = std[rh_indices].clamp_min(1e-6)
        center = rh_mean.mean()
        scale = torch.sqrt(((rh_mean - center) ** 2 + rh_std ** 2).mean()).clamp_min(1e-6)

        self.output_dim = output_dim
        self.coordinate_frequencies = coordinate_frequencies
        self.register_buffer("feature_mean", mean, persistent=True)
        self.register_buffer("feature_std", std, persistent=True)
        self.register_buffer("response_indices", torch.as_tensor(response_indices, dtype=torch.long), persistent=False)
        self.register_buffer("rh_indices", torch.as_tensor(rh_indices, dtype=torch.long), persistent=False)
        self.register_buffer("categorical_features", torch.as_tensor(categorical_rows, dtype=torch.float32), persistent=True)
        self.register_buffer("rh_center", center.reshape(1), persistent=True)
        self.register_buffer("rh_scale", scale.reshape(1), persistent=True)
        coordinate_dim = 1 + 2 * coordinate_frequencies
        categorical_dim = self.categorical_features.shape[1]
        self.response_head = make_mlp(
            latent_dim + coordinate_dim + categorical_dim + 1,
            hidden_dim,
            1,
            depth,
        )

    def _physical_values(self, normalized: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        mean = self.feature_mean.index_select(0, indices).to(normalized.device)
        std = self.feature_std.index_select(0, indices).to(normalized.device)
        return normalized.index_select(1, indices) * std + mean

    def coordinate_features(self, rh_percent: torch.Tensor) -> torch.Tensor:
        scaled = (rh_percent - self.rh_center.to(rh_percent.device)) / self.rh_scale.to(rh_percent.device)
        return _scalar_coordinate_features(scaled, frequencies=self.coordinate_frequencies)

    def forward(
        self,
        z: torch.Tensor,
        target_values: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = z.new_zeros(z.shape[0], self.output_dim)
        if target_values is None or target_mask is None:
            rh_values = z.new_zeros(z.shape[0], self.rh_indices.numel())
            rh_mask = z.new_zeros(z.shape[0], self.rh_indices.numel())
        else:
            rh_values = self._physical_values(target_values, self.rh_indices)
            rh_mask = target_mask.index_select(1, self.rh_indices).to(dtype=z.dtype)
        prediction = self.decode_at_observed_channels(z, rh_values, rh_mask)
        response_mean = self.feature_mean.index_select(0, self.response_indices).to(z.device)
        response_std = self.feature_std.index_select(0, self.response_indices).to(z.device)
        output[:, self.response_indices] = (prediction - response_mean) / response_std.clamp_min(1e-6)
        return output

    def decode_at_observed_channels(
        self,
        z: torch.Tensor,
        rh_percent: torch.Tensor,
        coordinate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coordinate_mask is None:
            coordinate_mask = torch.ones_like(rh_percent)
        coord = self.coordinate_features(rh_percent.to(dtype=z.dtype))
        categorical = self.categorical_features.to(dtype=z.dtype, device=z.device)
        query_count = rh_percent.shape[1]
        z_expanded = z.unsqueeze(1).expand(-1, query_count, -1)
        categorical_expanded = categorical.unsqueeze(0).expand(z.shape[0], -1, -1)
        query = torch.cat(
            [
                z_expanded,
                coord,
                categorical_expanded,
                coordinate_mask.to(dtype=z.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        prediction = self.response_head(query.reshape(-1, query.shape[-1]))
        return prediction.reshape(z.shape[0], query_count) * coordinate_mask.to(dtype=z.dtype)


class StructuredTransformerAutoencoder(nn.Module):
    model_type = "structured_transformer_autoencoder"

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        target_modalities: tuple[str, ...],
        hidden_dim: int,
        latent_dim: int,
        encoder_depth: int,
        decoder_depth: int,
        feature_names_by_modality: Mapping[str, tuple[str, ...]] | None = None,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        sequence_encoder_type: str = "conv",
        sequence_fourier_frequencies: int = 0,
        sequence_transformer_heads: int = 4,
        conditional_ccn_decoder: bool = False,
        coordinate_decoders: Mapping[str, bool] | None = None,
        feature_mean_by_modality: Mapping[str, torch.Tensor] | None = None,
        feature_std_by_modality: Mapping[str, torch.Tensor] | None = None,
        instrument_pretraining: bool = False,
        sizing_crosstalk_layers: int = 0,
        sizing_crosstalk_heads: int = 4,
        decoder_expansion_depth: int = 0,
        transformer_ff_multiplier: float = 4.0,
        latent_head_hidden_dim: int | None = None,
        decoder_expansion_hidden_dim: int | None = None,
        legacy_latent_head: bool = False,
    ) -> None:
        super().__init__()
        if sizing_crosstalk_layers < 0:
            raise ValueError("sizing_crosstalk_layers must be nonnegative")
        if decoder_expansion_depth < 0:
            raise ValueError("decoder_expansion_depth must be nonnegative")
        if transformer_ff_multiplier < 1.0:
            raise ValueError("transformer_ff_multiplier must be at least 1.0")
        latent_head_hidden_dim = latent_head_hidden_dim or hidden_dim
        decoder_expansion_hidden_dim = decoder_expansion_hidden_dim or hidden_dim
        if latent_head_hidden_dim < latent_dim:
            raise ValueError("latent_head_hidden_dim must be at least latent_dim")
        if decoder_expansion_hidden_dim < hidden_dim:
            raise ValueError("decoder_expansion_hidden_dim must be at least hidden_dim")
        if sizing_crosstalk_layers > 0 and hidden_dim % sizing_crosstalk_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by sizing_crosstalk_heads: "
                f"{hidden_dim} % {sizing_crosstalk_heads} != 0"
            )
        self.modality_names = tuple(modality_dims.keys())
        self.modality_to_index = {
            name: index
            for index, name in enumerate(self.modality_names)
        }
        self.target_modalities = tuple(target_modalities)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.sequence_encoder_type = sequence_encoder_type
        self.sequence_fourier_frequencies = sequence_fourier_frequencies
        self.sequence_transformer_heads = sequence_transformer_heads
        self.conditional_ccn_decoder = conditional_ccn_decoder
        self.coordinate_decoders = dict(coordinate_decoders or {})
        self.instrument_pretraining = instrument_pretraining
        self.sizing_crosstalk_layers = sizing_crosstalk_layers
        self.sizing_crosstalk_heads = sizing_crosstalk_heads
        self.decoder_expansion_depth = decoder_expansion_depth
        self.transformer_ff_multiplier = float(transformer_ff_multiplier)
        self.transformer_ff_dim = int(round(hidden_dim * transformer_ff_multiplier))
        self.latent_head_hidden_dim = int(latent_head_hidden_dim)
        self.decoder_expansion_hidden_dim = int(decoder_expansion_hidden_dim)
        self.legacy_latent_head = bool(legacy_latent_head)
        self.sizing_modalities = tuple(
            name for name in self.modality_names if name in SIZING_MODALITIES
        )

        feature_names_by_modality = feature_names_by_modality or {}
        feature_mean_by_modality = feature_mean_by_modality or {}
        feature_std_by_modality = feature_std_by_modality or {}
        self.encoders = nn.ModuleDict()
        for name, dim in modality_dims.items():
            names = tuple(feature_names_by_modality.get(name, ()))
            if not names:
                names = tuple(f"{name}__feature_{index}" for index in range(dim))
            self.encoders[name] = StructuredModalityEncoder(
                modality=name,
                input_dim=dim,
                feature_names=names,
                hidden_dim=hidden_dim,
                depth=encoder_depth,
                sequence_encoder_type=sequence_encoder_type,
                sequence_fourier_frequencies=sequence_fourier_frequencies,
                sequence_transformer_heads=sequence_transformer_heads,
            )

        self.modality_embeddings = nn.Parameter(
            torch.zeros(len(self.modality_names), hidden_dim)
        )
        self.latent_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_query, mean=0.0, std=0.02)

        if sizing_crosstalk_layers > 0 and len(self.sizing_modalities) > 1:
            sizing_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=sizing_crosstalk_heads,
                dim_feedforward=self.transformer_ff_dim,
                dropout=0.05,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sizing_crosstalk = nn.TransformerEncoder(
                sizing_layer,
                num_layers=sizing_crosstalk_layers,
                norm=nn.LayerNorm(hidden_dim),
                enable_nested_tensor=False,
            )
        else:
            self.sizing_crosstalk = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=self.transformer_ff_dim,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        if self.legacy_latent_head:
            self.latent_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.latent_head_hidden_dim),
                nn.GELU(),
                nn.Linear(self.latent_head_hidden_dim, latent_dim),
            )
        else:
            self.latent_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.latent_head_hidden_dim),
                nn.GELU(),
                nn.Linear(self.latent_head_hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        if decoder_expansion_depth > 0:
            self.decoder_expander = make_mlp(
                latent_dim,
                self.decoder_expansion_hidden_dim,
                hidden_dim,
                decoder_expansion_depth,
            )
            self.decoder_expansion_norm = nn.LayerNorm(hidden_dim)
            decoder_input_dim = hidden_dim
        else:
            self.decoder_expander = nn.Identity()
            self.decoder_expansion_norm = nn.Identity()
            decoder_input_dim = latent_dim

        self.decoders = nn.ModuleDict()
        for name in self.target_modalities:
            names = tuple(feature_names_by_modality.get(name, ()))
            feature_mean = feature_mean_by_modality.get(name)
            feature_std = feature_std_by_modality.get(name)
            if self.coordinate_decoders.get("ccn_activation", False) and name == "ccn_activation":
                self.decoders[name] = CoordinateCCNActivationDecoder(
                    latent_dim=decoder_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=modality_dims[name],
                    feature_names=names,
                    depth=decoder_depth,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                )
            elif self.coordinate_decoders.get("size_spectra", False) and name in SIZING_MODALITIES:
                self.decoders[name] = CoordinateSizeDistributionDecoder(
                    latent_dim=decoder_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=modality_dims[name],
                    feature_names=names,
                    depth=decoder_depth,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                )
            elif self.coordinate_decoders.get("optical_neph", False) and name == "optical_neph":
                self.decoders[name] = CoordinateNephelometerDecoder(
                    latent_dim=decoder_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=modality_dims[name],
                    feature_names=names,
                    depth=decoder_depth,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                )
            elif conditional_ccn_decoder and name == "ccn_activation":
                self.decoders[name] = ConditionalCCNActivationDecoder(
                    latent_dim=decoder_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=modality_dims[name],
                    feature_names=names,
                    depth=decoder_depth,
                )
            else:
                self.decoders[name] = make_mlp(
                    decoder_input_dim,
                    hidden_dim,
                    modality_dims[name],
                    decoder_depth,
                )
        self.pretrain_decoders = (
            nn.ModuleDict(
                {
                    name: make_mlp(hidden_dim, hidden_dim, dim, decoder_depth)
                    for name, dim in modality_dims.items()
                }
            )
            if instrument_pretraining
            else None
        )

    def _modality_tokens(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        tokens: dict[str, torch.Tensor] = {}
        for name in self.modality_names:
            token = self.encoders[name](x_by_modality[name], feature_mask_by_modality[name])
            embedding = self.modality_embeddings[self.modality_to_index[name]]
            tokens[name] = token + embedding.unsqueeze(0)
        return tokens

    def _apply_sizing_crosstalk(
        self,
        modality_tokens: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.sizing_crosstalk is None:
            return dict(modality_tokens)

        sizing_tokens = torch.stack(
            [modality_tokens[name] for name in self.sizing_modalities],
            dim=1,
        )
        sizing_visibility = torch.stack(
            [input_modality_mask[name] for name in self.sizing_modalities],
            dim=1,
        )
        padding_mask = ~sizing_visibility

        all_sizing_hidden = padding_mask.all(dim=1)
        if torch.any(all_sizing_hidden):
            padding_mask = padding_mask.clone()
            padding_mask[all_sizing_hidden] = False

        crossed_tokens = self.sizing_crosstalk(
            sizing_tokens,
            src_key_padding_mask=padding_mask,
        )

        updated_tokens = dict(modality_tokens)
        for index, name in enumerate(self.sizing_modalities):
            visible = sizing_visibility[:, index].unsqueeze(-1)
            updated_tokens[name] = torch.where(
                visible,
                crossed_tokens[:, index, :],
                modality_tokens[name],
            )
        return updated_tokens

    def _fused_state(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        modality_tokens = self._modality_tokens(x_by_modality, feature_mask_by_modality)
        modality_tokens = self._apply_sizing_crosstalk(
            modality_tokens,
            input_modality_mask,
        )
        visibility = torch.stack(
            [input_modality_mask[name] for name in self.modality_names],
            dim=1,
        )
        if not torch.all(visibility.any(dim=1)):
            raise RuntimeError("At least one row had no visible modalities")

        modality_stack = torch.stack(
            [modality_tokens[name] for name in self.modality_names],
            dim=1,
        )
        latent_query = self.latent_query.expand(modality_stack.shape[0], -1, -1)
        transformer_tokens = torch.cat([modality_stack, latent_query], dim=1)
        latent_padding = torch.zeros(
            visibility.shape[0],
            1,
            dtype=torch.bool,
            device=visibility.device,
        )
        padding_mask = torch.cat([~visibility, latent_padding], dim=1)
        transformed = self.transformer(
            transformer_tokens,
            src_key_padding_mask=padding_mask,
        )
        return transformed[:, -1, :]

    def encode(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.latent_head(
            self._fused_state(
                x_by_modality,
                feature_mask_by_modality,
                input_modality_mask,
            )
        )

    def pretrain_decode_modalities(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.pretrain_decoders is None:
            raise RuntimeError("instrument pretraining decoders are disabled for this model")
        tokens = self._modality_tokens(x_by_modality, feature_mask_by_modality)
        return {
            name: self.pretrain_decoders[name](tokens[name])
            for name in self.modality_names
        }

    def _decode_targets(
        self,
        z: torch.Tensor,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        decoder_input = self.decoder_state(z)
        decoded: dict[str, torch.Tensor] = {}
        for name in self.target_modalities:
            decoder = self.decoders[name]
            if isinstance(
                decoder,
                (
                    ConditionalCCNActivationDecoder,
                    CoordinateCCNActivationDecoder,
                    CoordinateNephelometerDecoder,
                ),
            ):
                decoded[name] = decoder(
                    decoder_input,
                    x_by_modality.get(name),
                    feature_mask_by_modality.get(name),
                )
            else:
                decoded[name] = decoder(decoder_input)
        return decoded

    def decoder_state(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_expansion_norm(self.decoder_expander(z))

    def decode_ccn_at_supersaturation(
        self,
        z: torch.Tensor,
        supersaturation_percent: torch.Tensor,
        physical: bool = True,
    ) -> torch.Tensor:
        decoder = (
            self.decoders["ccn_activation"]
            if "ccn_activation" in self.decoders
            else None
        )
        if not isinstance(decoder, CoordinateCCNActivationDecoder):
            raise TypeError("ccn_activation is not using CoordinateCCNActivationDecoder")
        return decoder.decode_at_supersaturation(
            self.decoder_state(z),
            supersaturation_percent,
            physical=physical,
        )

    def decode_size_at_diameter(
        self,
        z: torch.Tensor,
        modality: str,
        diameter_nm: torch.Tensor,
        physical: bool = True,
    ) -> torch.Tensor:
        """Query a specific sizing decoder at one or more diameters.

        The `modality` argument chooses a separate decoder head, for example
        `size_smps` or `size_aps`. It is not embedded as a coordinate.
        """
        decoder = self.decoders[modality] if modality in self.decoders else None
        if not isinstance(decoder, CoordinateSizeDistributionDecoder):
            raise TypeError(f"{modality} is not using CoordinateSizeDistributionDecoder")
        return decoder.decode_at_diameter(
            self.decoder_state(z),
            diameter_nm,
            physical=physical,
        )

    def forward(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(
            x_by_modality,
            feature_mask_by_modality,
            input_modality_mask,
        )
        decoded = self._decode_targets(
            z,
            x_by_modality,
            feature_mask_by_modality,
        )
        return z, decoded


class StructuredTransformerVAE(StructuredTransformerAutoencoder):
    model_type = "structured_transformer_vae"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.latent_head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.latent_head_hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 2),
        )

    def _latent_distribution(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.latent_head(
            self._fused_state(
                x_by_modality,
                feature_mask_by_modality,
                input_modality_mask,
            )
        )
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=6.0)

    def _sample_latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _kl_metrics(self, mu: torch.Tensor, logvar: torch.Tensor) -> dict[str, torch.Tensor]:
        elementwise = 0.5 * (mu.square() + logvar.exp() - logvar - 1.0)
        return {
            "kl": elementwise.sum(dim=-1).mean() / self.latent_dim,
            "latent_mu_abs": mu.abs().mean(),
            "latent_logvar_mean": logvar.mean(),
        }

    def encode(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        mu, _ = self._latent_distribution(
            x_by_modality,
            feature_mask_by_modality,
            input_modality_mask,
        )
        return mu

    def forward(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        mu, logvar = self._latent_distribution(
            x_by_modality,
            feature_mask_by_modality,
            input_modality_mask,
        )
        z = self._sample_latent(mu, logvar)
        decoded = self._decode_targets(
            z,
            x_by_modality,
            feature_mask_by_modality,
        )
        return z, decoded, self._kl_metrics(mu, logvar)


DEFAULT_LATENT_BLOCKS: dict[str, int] = {
    "global": 32,
    "size": 32,
    "composition": 24,
    "hygroscopic": 16,
    "optical": 16,
    "nuisance": 8,
}


DEFAULT_BLOCK_MODALITY_MAP: dict[str, tuple[str, ...]] = {
    "global": (
        "met_context",
        "chemistry_acsm",
        "size_distribution",
        "cpc_number",
        "ccn_activation",
        "optical_neph",
    ),
    "size": (
        "size_distribution",
        "cpc_number",
        "ccn_activation",
        "optical_neph",
    ),
    "composition": (
        "chemistry_acsm",
        "optical_neph",
    ),
    "hygroscopic": (
        "ccn_activation",
        "optical_neph",
        "chemistry_acsm",
        "size_distribution",
    ),
    "optical": (
        "optical_neph",
        "size_distribution",
        "chemistry_acsm",
    ),
    "nuisance": (
        "met_context",
        "chemistry_acsm",
        "size_distribution",
        "cpc_number",
        "ccn_activation",
        "optical_neph",
    ),
}


class StructuredHierarchicalPoETransformerVAE(nn.Module):
    model_type = "hierarchical_poe_transformer_vae"

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        target_modalities: tuple[str, ...],
        hidden_dim: int,
        latent_blocks: Mapping[str, int] | None,
        encoder_depth: int,
        decoder_depth: int,
        feature_names_by_modality: Mapping[str, tuple[str, ...]] | None = None,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        block_modality_map: Mapping[str, tuple[str, ...]] | None = None,
        sequence_encoder_type: str = "conv",
        sequence_fourier_frequencies: int = 0,
        sequence_transformer_heads: int = 4,
        conditional_ccn_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.modality_names = tuple(modality_dims.keys())
        self.modality_to_index = {
            name: index
            for index, name in enumerate(self.modality_names)
        }
        self.target_modalities = tuple(target_modalities)
        self.hidden_dim = hidden_dim
        self.latent_blocks = dict(latent_blocks or DEFAULT_LATENT_BLOCKS)
        if not self.latent_blocks:
            raise ValueError("Structured PoE-VAE requires at least one latent block")
        self.block_names = tuple(self.latent_blocks.keys())
        self.latent_dim = sum(self.latent_blocks.values())
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.sequence_encoder_type = sequence_encoder_type
        self.sequence_fourier_frequencies = sequence_fourier_frequencies
        self.sequence_transformer_heads = sequence_transformer_heads
        self.conditional_ccn_decoder = conditional_ccn_decoder
        self.block_modality_map = {
            block: tuple(block_modality_map.get(block, DEFAULT_BLOCK_MODALITY_MAP.get(block, ())))
            if block_modality_map is not None
            else DEFAULT_BLOCK_MODALITY_MAP.get(block, self.modality_names)
            for block in self.block_names
        }

        feature_names_by_modality = feature_names_by_modality or {}
        self.encoders = nn.ModuleDict()
        for name, dim in modality_dims.items():
            names = tuple(feature_names_by_modality.get(name, ()))
            if not names:
                names = tuple(f"{name}__feature_{index}" for index in range(dim))
            self.encoders[name] = StructuredModalityEncoder(
                modality=name,
                input_dim=dim,
                feature_names=names,
                hidden_dim=hidden_dim,
                depth=encoder_depth,
                sequence_encoder_type=sequence_encoder_type,
                sequence_fourier_frequencies=sequence_fourier_frequencies,
                sequence_transformer_heads=sequence_transformer_heads,
            )

        self.modality_embeddings = nn.Parameter(
            torch.zeros(len(self.modality_names), hidden_dim)
        )
        self.block_embeddings = nn.Parameter(
            torch.zeros(len(self.block_names), hidden_dim)
        )
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.block_embeddings, mean=0.0, std=0.02)

        self.expert_heads = nn.ModuleDict()
        for modality in self.modality_names:
            heads = nn.ModuleDict()
            for block, dim in self.latent_blocks.items():
                if modality in self.block_modality_map.get(block, ()):
                    heads[block] = nn.Linear(hidden_dim, dim * 2)
            self.expert_heads[modality] = heads

        self.block_projectors = nn.ModuleDict(
            {
                block: nn.Linear(dim, hidden_dim)
                for block, dim in self.latent_blocks.items()
            }
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.context_to_latent_delta = make_mlp(
            hidden_dim * len(self.block_names),
            hidden_dim,
            self.latent_dim,
            depth=2,
        )
        self.decoders = nn.ModuleDict()
        for name in self.target_modalities:
            if conditional_ccn_decoder and name == "ccn_activation":
                self.decoders[name] = ConditionalCCNActivationDecoder(
                    latent_dim=self.latent_dim,
                    hidden_dim=hidden_dim,
                    output_dim=modality_dims[name],
                    feature_names=tuple(feature_names_by_modality.get(name, ())),
                    depth=decoder_depth,
                )
            else:
                self.decoders[name] = make_mlp(
                    self.latent_dim,
                    hidden_dim,
                    modality_dims[name],
                    decoder_depth,
                )

    def _modality_tokens(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        tokens: dict[str, torch.Tensor] = {}
        for name in self.modality_names:
            token = self.encoders[name](x_by_modality[name], feature_mask_by_modality[name])
            embedding = self.modality_embeddings[self.modality_to_index[name]]
            tokens[name] = token + embedding.unsqueeze(0)
        return tokens

    def _combine_block_experts(
        self,
        block: str,
        modality_tokens: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_token = next(iter(modality_tokens.values()))
        batch_size = first_token.shape[0]
        block_dim = self.latent_blocks[block]
        precision = first_token.new_ones(batch_size, block_dim)
        weighted_mean = first_token.new_zeros(batch_size, block_dim)

        for modality in self.modality_names:
            head = self.expert_heads[modality]
            if block not in head:
                continue
            mean, logvar = head[block](modality_tokens[modality]).chunk(2, dim=-1)
            visible = input_modality_mask[modality].to(dtype=mean.dtype).unsqueeze(-1)
            expert_precision = torch.exp(-logvar.clamp(min=-8.0, max=8.0)) * visible
            precision = precision + expert_precision
            weighted_mean = weighted_mean + mean * expert_precision

        variance = precision.reciprocal()
        mean = weighted_mean * variance
        logvar = torch.log(variance.clamp_min(1e-8))
        return mean, logvar

    def _posterior(
        self,
        modality_tokens: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        means: dict[str, torch.Tensor] = {}
        logvars: dict[str, torch.Tensor] = {}
        for block in self.block_names:
            means[block], logvars[block] = self._combine_block_experts(
                block,
                modality_tokens,
                input_modality_mask,
            )
        return means, logvars

    def _sample_blocks(
        self,
        means: Mapping[str, torch.Tensor],
        logvars: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        z_blocks: dict[str, torch.Tensor] = {}
        for block in self.block_names:
            if self.training:
                std = torch.exp(0.5 * logvars[block])
                z_blocks[block] = means[block] + torch.randn_like(std) * std
            else:
                z_blocks[block] = means[block]
        return z_blocks

    def _context_adjusted_latent(
        self,
        modality_tokens: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
        z_blocks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        z = torch.cat([z_blocks[block] for block in self.block_names], dim=-1)
        visibility = torch.stack(
            [input_modality_mask[name] for name in self.modality_names],
            dim=1,
        )
        modality_stack = torch.stack(
            [modality_tokens[name] for name in self.modality_names],
            dim=1,
        )
        block_tokens = torch.stack(
            [
                self.block_projectors[block](z_blocks[block])
                + self.block_embeddings[index].unsqueeze(0)
                for index, block in enumerate(self.block_names)
            ],
            dim=1,
        )
        transformer_tokens = torch.cat([modality_stack, block_tokens], dim=1)
        block_padding = torch.zeros(
            visibility.shape[0],
            len(self.block_names),
            dtype=torch.bool,
            device=visibility.device,
        )
        padding_mask = torch.cat([~visibility, block_padding], dim=1)
        transformed = self.transformer(
            transformer_tokens,
            src_key_padding_mask=padding_mask,
        )
        transformed_blocks = transformed[:, len(self.modality_names):, :]
        context_delta = self.context_to_latent_delta(transformed_blocks.flatten(start_dim=1))
        return z + context_delta

    def _kl_metrics(
        self,
        means: Mapping[str, torch.Tensor],
        logvars: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        total = next(iter(means.values())).new_zeros(())
        for block in self.block_names:
            mean = means[block]
            logvar = logvars[block]
            kl_sum = 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).sum(dim=-1).mean()
            metrics[f"kl_{block}"] = kl_sum / mean.shape[-1]
            total = total + kl_sum
        metrics["kl"] = total / self.latent_dim
        return metrics

    def _decode_targets(
        self,
        decode_z: torch.Tensor,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        decoded: dict[str, torch.Tensor] = {}
        for name in self.target_modalities:
            decoder = self.decoders[name]
            if isinstance(
                decoder,
                (
                    ConditionalCCNActivationDecoder,
                    CoordinateCCNActivationDecoder,
                    CoordinateNephelometerDecoder,
                ),
            ):
                decoded[name] = decoder(
                    decode_z,
                    x_by_modality.get(name),
                    feature_mask_by_modality.get(name),
                )
            else:
                decoded[name] = decoder(decode_z)
        return decoded

    def encode(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        modality_tokens = self._modality_tokens(x_by_modality, feature_mask_by_modality)
        any_visible = torch.stack(
            [input_modality_mask[name] for name in self.modality_names],
            dim=1,
        ).any(dim=1)
        if not torch.all(any_visible):
            raise RuntimeError("At least one row had no visible modalities")
        means, _ = self._posterior(modality_tokens, input_modality_mask)
        return self._context_adjusted_latent(
            modality_tokens,
            input_modality_mask,
            means,
        )

    def forward(
        self,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
        input_modality_mask: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        modality_tokens = self._modality_tokens(x_by_modality, feature_mask_by_modality)
        visibility = torch.stack(
            [input_modality_mask[name] for name in self.modality_names],
            dim=1,
        )
        if not torch.all(visibility.any(dim=1)):
            raise RuntimeError("At least one row had no visible modalities")

        means, logvars = self._posterior(modality_tokens, input_modality_mask)
        z_blocks = self._sample_blocks(means, logvars)
        decode_z = self._context_adjusted_latent(
            modality_tokens,
            input_modality_mask,
            z_blocks,
        )
        decoded = self._decode_targets(
            decode_z,
            x_by_modality,
            feature_mask_by_modality,
        )
        return decode_z, decoded, self._kl_metrics(means, logvars)


def unpack_model_output(
    output: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not isinstance(output, tuple):
        raise TypeError(f"Expected model output tuple, got {type(output)!r}")
    if len(output) == 2:
        z, decoded = output
        return z, decoded, {}
    if len(output) == 3:
        z, decoded, diagnostics = output
        return z, decoded, diagnostics
    raise ValueError(f"Expected model output length 2 or 3, got {len(output)}")


def feature_names_by_modality(
    feature_names: tuple[str, ...] | list[str],
    modality_indices: Mapping[str, list[int]],
) -> dict[str, tuple[str, ...]]:
    return {
        modality: tuple(feature_names[index] for index in indices)
        for modality, indices in modality_indices.items()
    }


def build_aerosol_model(
    model_type: str,
    modality_dims: Mapping[str, int],
    target_modalities: tuple[str, ...],
    hidden_dim: int,
    latent_dim: int,
    encoder_depth: int,
    decoder_depth: int,
    feature_names_by_modality_map: Mapping[str, tuple[str, ...]] | None = None,
    latent_blocks: Mapping[str, int] | None = None,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    block_modality_map: Mapping[str, tuple[str, ...]] | None = None,
    sequence_encoder_type: str = "conv",
    sequence_fourier_frequencies: int = 0,
    sequence_transformer_heads: int = 4,
    conditional_ccn_decoder: bool = False,
    coordinate_decoders: Mapping[str, bool] | None = None,
    feature_mean_by_modality_map: Mapping[str, torch.Tensor] | None = None,
    feature_std_by_modality_map: Mapping[str, torch.Tensor] | None = None,
    instrument_pretraining: bool = False,
    sizing_crosstalk_layers: int = 0,
    sizing_crosstalk_heads: int = 4,
    decoder_expansion_depth: int = 0,
    transformer_ff_multiplier: float = 4.0,
    latent_head_hidden_dim: int | None = None,
    decoder_expansion_hidden_dim: int | None = None,
    legacy_latent_head: bool = False,
) -> nn.Module:
    if model_type in {"grouped_masked_autoencoder", "grouped_autoencoder"}:
        return GroupedMaskedAutoencoder(
            modality_dims=modality_dims,
            target_modalities=target_modalities,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
        )
    if model_type == "hierarchical_poe_transformer_vae":
        return StructuredHierarchicalPoETransformerVAE(
            modality_dims=modality_dims,
            target_modalities=target_modalities,
            hidden_dim=hidden_dim,
            latent_blocks=latent_blocks,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            feature_names_by_modality=feature_names_by_modality_map,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            block_modality_map=block_modality_map,
            sequence_encoder_type=sequence_encoder_type,
            sequence_fourier_frequencies=sequence_fourier_frequencies,
            sequence_transformer_heads=sequence_transformer_heads,
            conditional_ccn_decoder=conditional_ccn_decoder,
        )
    if model_type == "structured_transformer_autoencoder":
        return StructuredTransformerAutoencoder(
            modality_dims=modality_dims,
            target_modalities=target_modalities,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            feature_names_by_modality=feature_names_by_modality_map,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            sequence_encoder_type=sequence_encoder_type,
            sequence_fourier_frequencies=sequence_fourier_frequencies,
            sequence_transformer_heads=sequence_transformer_heads,
            conditional_ccn_decoder=conditional_ccn_decoder,
            coordinate_decoders=coordinate_decoders,
            feature_mean_by_modality=feature_mean_by_modality_map,
            feature_std_by_modality=feature_std_by_modality_map,
            instrument_pretraining=instrument_pretraining,
            sizing_crosstalk_layers=sizing_crosstalk_layers,
            sizing_crosstalk_heads=sizing_crosstalk_heads,
            decoder_expansion_depth=decoder_expansion_depth,
            transformer_ff_multiplier=transformer_ff_multiplier,
            latent_head_hidden_dim=latent_head_hidden_dim,
            decoder_expansion_hidden_dim=decoder_expansion_hidden_dim,
            legacy_latent_head=legacy_latent_head,
        )
    if model_type == "structured_transformer_vae":
        return StructuredTransformerVAE(
            modality_dims=modality_dims,
            target_modalities=target_modalities,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            encoder_depth=encoder_depth,
            decoder_depth=decoder_depth,
            feature_names_by_modality=feature_names_by_modality_map,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            sequence_encoder_type=sequence_encoder_type,
            sequence_fourier_frequencies=sequence_fourier_frequencies,
            sequence_transformer_heads=sequence_transformer_heads,
            conditional_ccn_decoder=conditional_ccn_decoder,
            coordinate_decoders=coordinate_decoders,
            feature_mean_by_modality=feature_mean_by_modality_map,
            feature_std_by_modality=feature_std_by_modality_map,
            instrument_pretraining=instrument_pretraining,
            sizing_crosstalk_layers=sizing_crosstalk_layers,
            sizing_crosstalk_heads=sizing_crosstalk_heads,
            decoder_expansion_depth=decoder_expansion_depth,
            transformer_ff_multiplier=transformer_ff_multiplier,
            latent_head_hidden_dim=latent_head_hidden_dim,
            decoder_expansion_hidden_dim=decoder_expansion_hidden_dim,
            legacy_latent_head=legacy_latent_head,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def build_model_from_checkpoint(checkpoint: Mapping[str, Any]) -> nn.Module:
    config = checkpoint.get("config", {})
    model_type = str(checkpoint.get("model_type", config.get("model_type", "grouped_masked_autoencoder")))
    modality_indices = checkpoint.get("modality_indices", {})
    feature_names = checkpoint.get("feature_names", ())
    mean = checkpoint.get("mean")
    std = checkpoint.get("std")
    feature_mean_by_modality_map = None
    feature_std_by_modality_map = None
    if mean is not None and std is not None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        std_tensor = torch.as_tensor(std, dtype=torch.float32)
        feature_mean_by_modality_map = {
            modality: mean_tensor[indices]
            for modality, indices in modality_indices.items()
        }
        feature_std_by_modality_map = {
            modality: std_tensor[indices]
            for modality, indices in modality_indices.items()
        }
    model_state = checkpoint.get("model_state", {})
    legacy_latent_head = bool(checkpoint.get("legacy_latent_head", False))
    if (
        model_type == "structured_transformer_autoencoder"
        and "latent_head.5.weight" not in model_state
        and "latent_head.3.weight" in model_state
    ):
        legacy_latent_head = tuple(model_state["latent_head.3.weight"].shape)[0] == int(
            checkpoint["latent_dim"]
        )
    return build_aerosol_model(
        model_type=model_type,
        modality_dims=checkpoint["modality_dims"],
        target_modalities=tuple(checkpoint["target_modalities"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        encoder_depth=int(config.get("encoder_depth", 2)),
        decoder_depth=int(config.get("decoder_depth", 2)),
        feature_names_by_modality_map=feature_names_by_modality(feature_names, modality_indices),
        latent_blocks=checkpoint.get("latent_blocks", config.get("latent_blocks")),
        transformer_layers=int(config.get("transformer_layers", checkpoint.get("transformer_layers", 2))),
        transformer_heads=int(config.get("transformer_heads", checkpoint.get("transformer_heads", 4))),
        transformer_ff_multiplier=float(
            config.get(
                "transformer_ff_multiplier",
                checkpoint.get("transformer_ff_multiplier", 4.0),
            )
        ),
        latent_head_hidden_dim=int(
            config.get(
                "latent_head_hidden_dim",
                checkpoint.get("latent_head_hidden_dim", checkpoint.get("hidden_dim", 128)),
            )
        ),
        block_modality_map=checkpoint.get("block_modality_map", config.get("block_modality_map")),
        sequence_encoder_type=str(config.get("sequence_encoder_type", checkpoint.get("sequence_encoder_type", "conv"))),
        sequence_fourier_frequencies=int(
            config.get("sequence_fourier_frequencies", checkpoint.get("sequence_fourier_frequencies", 0))
        ),
        sequence_transformer_heads=int(
            config.get(
                "sequence_transformer_heads",
                checkpoint.get("sequence_transformer_heads", config.get("transformer_heads", 4)),
            )
        ),
        conditional_ccn_decoder=bool(
            config.get("conditional_ccn_decoder", checkpoint.get("conditional_ccn_decoder", False))
        ),
        coordinate_decoders=dict(
            config.get("coordinate_decoders", checkpoint.get("coordinate_decoders", {}))
        ),
        feature_mean_by_modality_map=feature_mean_by_modality_map,
        feature_std_by_modality_map=feature_std_by_modality_map,
        instrument_pretraining=bool(
            config.get("instrument_pretraining", checkpoint.get("instrument_pretraining", False))
        ),
        sizing_crosstalk_layers=int(
            config.get(
                "sizing_crosstalk_layers",
                checkpoint.get("sizing_crosstalk_layers", 0),
            )
        ),
        sizing_crosstalk_heads=int(
            config.get(
                "sizing_crosstalk_heads",
                checkpoint.get(
                    "sizing_crosstalk_heads",
                    config.get("transformer_heads", checkpoint.get("transformer_heads", 4)),
                ),
            )
        ),
        decoder_expansion_depth=int(
            config.get(
                "decoder_expansion_depth",
                checkpoint.get("decoder_expansion_depth", 0),
            )
        ),
        decoder_expansion_hidden_dim=int(
            config.get(
                "decoder_expansion_hidden_dim",
                checkpoint.get("decoder_expansion_hidden_dim", checkpoint.get("hidden_dim", 128)),
            )
        ),
        legacy_latent_head=legacy_latent_head,
    )
