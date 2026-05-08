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
    coordinate_label = feature_name.rsplit("__", maxsplit=1)[-1]
    match = DIAMETER_LABEL_RE.match(coordinate_label)
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
        sizing_crosstalk_layers: int = 0,
        sizing_crosstalk_heads: int = 4,
        decoder_expansion_depth: int = 0,
    ) -> None:
        super().__init__()
        if sizing_crosstalk_layers < 0:
            raise ValueError("sizing_crosstalk_layers must be nonnegative")
        if decoder_expansion_depth < 0:
            raise ValueError("decoder_expansion_depth must be nonnegative")
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
        self.sizing_crosstalk_layers = sizing_crosstalk_layers
        self.sizing_crosstalk_heads = sizing_crosstalk_heads
        self.decoder_expansion_depth = decoder_expansion_depth
        self.sizing_modalities = tuple(
            name for name in self.modality_names if name in SIZING_MODALITIES
        )

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
        self.latent_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_query, mean=0.0, std=0.02)

        if sizing_crosstalk_layers > 0 and len(self.sizing_modalities) > 1:
            sizing_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=sizing_crosstalk_heads,
                dim_feedforward=hidden_dim * 4,
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
        self.latent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        if decoder_expansion_depth > 0:
            self.decoder_expander = make_mlp(
                latent_dim,
                hidden_dim,
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
            if conditional_ccn_decoder and name == "ccn_activation":
                self.decoders[name] = ConditionalCCNActivationDecoder(
                    latent_dim=decoder_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=modality_dims[name],
                    feature_names=tuple(feature_names_by_modality.get(name, ())),
                    depth=decoder_depth,
                )
            else:
                self.decoders[name] = make_mlp(
                    decoder_input_dim,
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

    def _decode_targets(
        self,
        z: torch.Tensor,
        x_by_modality: Mapping[str, torch.Tensor],
        feature_mask_by_modality: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        decoder_input = self.decoder_expansion_norm(self.decoder_expander(z))
        decoded: dict[str, torch.Tensor] = {}
        for name in self.target_modalities:
            decoder = self.decoders[name]
            if isinstance(decoder, ConditionalCCNActivationDecoder):
                decoded[name] = decoder(
                    decoder_input,
                    x_by_modality.get(name),
                    feature_mask_by_modality.get(name),
                )
            else:
                decoded[name] = decoder(decoder_input)
        return decoded

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
            nn.Linear(self.hidden_dim, self.hidden_dim),
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
            if isinstance(decoder, ConditionalCCNActivationDecoder):
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
    sizing_crosstalk_layers: int = 0,
    sizing_crosstalk_heads: int = 4,
    decoder_expansion_depth: int = 0,
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
            sizing_crosstalk_layers=sizing_crosstalk_layers,
            sizing_crosstalk_heads=sizing_crosstalk_heads,
            decoder_expansion_depth=decoder_expansion_depth,
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
            sizing_crosstalk_layers=sizing_crosstalk_layers,
            sizing_crosstalk_heads=sizing_crosstalk_heads,
            decoder_expansion_depth=decoder_expansion_depth,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def build_model_from_checkpoint(checkpoint: Mapping[str, Any]) -> nn.Module:
    config = checkpoint.get("config", {})
    model_type = str(checkpoint.get("model_type", config.get("model_type", "grouped_masked_autoencoder")))
    modality_indices = checkpoint.get("modality_indices", {})
    feature_names = checkpoint.get("feature_names", ())
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
    )
