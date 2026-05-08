from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VariableSpec:
    name: str
    transform: str = "identity"


@dataclass(frozen=True)
class StreamSpec:
    name: str
    path: str
    variables: tuple[VariableSpec, ...]
    required: bool = True


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    role: str
    always_input: bool
    streams: tuple[StreamSpec, ...]


@dataclass(frozen=True)
class SizeGridSpec:
    enabled: bool = False
    min_diameter_nm: float = 3.0
    max_diameter_nm: float = 30000.0
    bins: int = 160
    interpolation: str = "linear_log_diameter"
    diameter_units: dict[str, str] | None = None


@dataclass(frozen=True)
class TemporalWindowSpec:
    enabled: bool = False
    default_step: str = "30min"
    default_stats: tuple[str, ...] = ("mean",)
    time_position_frequencies: int = 4
    modality_steps: dict[str, str] | None = None
    modality_stats: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    freq: str
    split_strategy: str
    feature_coverage_basis: str
    model_type: str
    latent_dim: int
    hidden_dim: int
    encoder_depth: int
    decoder_depth: int
    transformer_layers: int
    transformer_heads: int
    sequence_encoder_type: str
    sequence_fourier_frequencies: int
    sequence_transformer_heads: int
    conditional_ccn_decoder: bool
    sizing_crosstalk_layers: int
    sizing_crosstalk_heads: int
    decoder_expansion_depth: int
    latent_blocks: dict[str, int]
    block_modality_map: dict[str, tuple[str, ...]]
    size_grid: SizeGridSpec
    temporal_windows: TemporalWindowSpec
    batch_size: int
    learning_rate: float
    learning_rate_schedule: str
    min_learning_rate: float
    weight_decay: float
    input_mask_probability: float
    latent_l2_weight: float
    kl_weight: float
    closure_loss_weights: dict[str, float]
    min_feature_coverage: float
    min_feature_std: float
    validation_fraction: float
    test_fraction: float
    validation_interval: int
    reconstruction_validation_interval: int
    diagnostic_validation_interval: int
    seed: int
    training_stages: tuple[dict[str, Any], ...]
    cross_prediction_selection_mode: str
    cross_prediction_exclusion_groups: tuple[tuple[str, ...], ...]
    modalities: tuple[ModalitySpec, ...]

    @property
    def target_modalities(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.modalities if m.role == "target")

    @property
    def context_modalities(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.modalities if m.role == "context")

    @property
    def always_input_modalities(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.modalities if m.always_input)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {config_path} did not parse to a mapping")

    modalities: list[ModalitySpec] = []
    for modality_name, modality_raw in raw["modalities"].items():
        streams: list[StreamSpec] = []
        for stream_raw in modality_raw["streams"]:
            variables = tuple(
                VariableSpec(
                    name=variable_raw["name"],
                    transform=variable_raw.get("transform", "identity"),
                )
                for variable_raw in stream_raw["variables"]
            )
            streams.append(
                StreamSpec(
                    name=stream_raw["name"],
                    path=stream_raw["path"],
                    variables=variables,
                    required=bool(stream_raw.get("required", True)),
                )
            )
        modalities.append(
            ModalitySpec(
                name=modality_name,
                role=modality_raw.get("role", "target"),
                always_input=bool(modality_raw.get("always_input", False)),
                streams=tuple(streams),
            )
        )

    latent_blocks = {
        str(name): int(dim)
        for name, dim in dict(raw.get("latent_blocks", {})).items()
    }
    block_modality_map = {
        str(block): tuple(str(modality) for modality in modalities)
        for block, modalities in dict(raw.get("block_modality_map", {})).items()
    }
    model_type = str(raw.get("model_type", "grouped_masked_autoencoder"))
    if model_type not in {
        "grouped_masked_autoencoder",
        "grouped_autoencoder",
        "hierarchical_poe_transformer_vae",
        "structured_transformer_autoencoder",
        "structured_transformer_vae",
    }:
        raise ValueError(f"Unknown model_type: {model_type}")
    latent_dim = int(raw.get("latent_dim", 32))
    hidden_dim = int(raw.get("hidden_dim", 128))
    transformer_heads = int(raw.get("transformer_heads", 4))
    if model_type in {"structured_transformer_autoencoder", "structured_transformer_vae"}:
        if latent_blocks:
            raise ValueError(
                f"{model_type} uses one global latent vector; "
                "remove latent_blocks from the config"
            )
        if block_modality_map:
            raise ValueError(
                f"{model_type} does not use block_modality_map; "
                "remove block_modality_map from the config"
            )
    split_strategy = str(raw.get("split_strategy", "chronological"))
    if split_strategy not in {"chronological", "calendar_day_hash", "calendar_month_hash"}:
        raise ValueError(
            "split_strategy must be chronological, calendar_day_hash, or calendar_month_hash; "
            f"got {split_strategy!r}"
        )
    feature_coverage_basis = str(raw.get("feature_coverage_basis", "train"))
    if feature_coverage_basis not in {"train", "all"}:
        raise ValueError(
            "feature_coverage_basis must be 'train' or 'all', "
            f"got {feature_coverage_basis!r}"
        )
    sequence_encoder_type = str(raw.get("sequence_encoder_type", "conv"))
    if sequence_encoder_type not in {"conv", "diameter_fourier_conv", "diameter_transformer"}:
        raise ValueError(
            "sequence_encoder_type must be conv, diameter_fourier_conv, or diameter_transformer; "
            f"got {sequence_encoder_type!r}"
        )
    sequence_transformer_heads = int(raw.get("sequence_transformer_heads", transformer_heads))
    if hidden_dim % sequence_transformer_heads != 0:
        raise ValueError(
            "hidden_dim must be divisible by sequence_transformer_heads: "
            f"{hidden_dim} % {sequence_transformer_heads} != 0"
        )
    sizing_crosstalk_layers = int(raw.get("sizing_crosstalk_layers", 0))
    if sizing_crosstalk_layers < 0:
        raise ValueError("sizing_crosstalk_layers must be nonnegative")
    sizing_crosstalk_heads = int(raw.get("sizing_crosstalk_heads", transformer_heads))
    if sizing_crosstalk_layers > 0 and hidden_dim % sizing_crosstalk_heads != 0:
        raise ValueError(
            "hidden_dim must be divisible by sizing_crosstalk_heads: "
            f"{hidden_dim} % {sizing_crosstalk_heads} != 0"
        )
    decoder_expansion_depth = int(raw.get("decoder_expansion_depth", 0))
    if decoder_expansion_depth < 0:
        raise ValueError("decoder_expansion_depth must be nonnegative")
    if latent_blocks and sum(latent_blocks.values()) != latent_dim:
        raise ValueError(
            "latent_dim must equal the sum of latent_blocks for hierarchical models: "
            f"{latent_dim} != {sum(latent_blocks.values())}"
        )
    if model_type in {
        "hierarchical_poe_transformer_vae",
        "structured_transformer_autoencoder",
        "structured_transformer_vae",
    } and hidden_dim % transformer_heads != 0:
        raise ValueError(
            f"hidden_dim must be divisible by transformer_heads: {hidden_dim} % {transformer_heads} != 0"
        )
    learning_rate_schedule = str(raw.get("learning_rate_schedule", "constant"))
    if learning_rate_schedule not in {"constant", "cosine"}:
        raise ValueError(
            "learning_rate_schedule must be 'constant' or 'cosine', "
            f"got {learning_rate_schedule!r}"
        )
    modality_names = {modality.name for modality in modalities}
    unknown_block_modalities = sorted(
        {
            modality
            for block_modalities in block_modality_map.values()
            for modality in block_modalities
            if modality not in modality_names
        }
    )
    if unknown_block_modalities:
        raise ValueError(
            "block_modality_map references modalities not defined in config: "
            + ", ".join(unknown_block_modalities)
        )
    size_grid_raw = raw.get("size_grid", {}) or {}
    if not isinstance(size_grid_raw, dict):
        raise ValueError("size_grid must be a mapping when provided")
    size_grid = SizeGridSpec(
        enabled=bool(size_grid_raw.get("enabled", False)),
        min_diameter_nm=float(size_grid_raw.get("min_diameter_nm", 3.0)),
        max_diameter_nm=float(size_grid_raw.get("max_diameter_nm", 30000.0)),
        bins=int(size_grid_raw.get("bins", 160)),
        interpolation=str(size_grid_raw.get("interpolation", "linear_log_diameter")),
        diameter_units={
            str(name): str(unit)
            for name, unit in dict(size_grid_raw.get("diameter_units", {})).items()
        } or None,
    )
    if size_grid.enabled:
        if size_grid.bins < 2:
            raise ValueError("size_grid.bins must be at least 2")
        if not (size_grid.min_diameter_nm > 0 and size_grid.max_diameter_nm > size_grid.min_diameter_nm):
            raise ValueError("size_grid diameter bounds must satisfy 0 < min < max")
        if size_grid.interpolation != "linear_log_diameter":
            raise ValueError(
                "Only size_grid.interpolation='linear_log_diameter' is currently implemented"
            )
    temporal_raw = raw.get("temporal_windows", {}) or {}
    if not isinstance(temporal_raw, dict):
        raise ValueError("temporal_windows must be a mapping when provided")
    allowed_temporal_stats = {"mean", "std", "min", "max"}
    temporal_default_stats = tuple(
        str(stat) for stat in temporal_raw.get("default_stats", ("mean",))
    )
    unknown_default_stats = sorted(set(temporal_default_stats) - allowed_temporal_stats)
    if unknown_default_stats:
        raise ValueError(
            "temporal_windows.default_stats contains unsupported statistics: "
            + ", ".join(unknown_default_stats)
        )
    temporal_modality_stats = {
        str(modality): tuple(str(stat) for stat in stats)
        for modality, stats in dict(temporal_raw.get("modality_stats", {})).items()
    }
    for modality, stats in temporal_modality_stats.items():
        unknown_stats = sorted(set(stats) - allowed_temporal_stats)
        if unknown_stats:
            raise ValueError(
                f"temporal_windows.modality_stats for {modality} contains unsupported "
                "statistics: " + ", ".join(unknown_stats)
            )
    temporal_windows = TemporalWindowSpec(
        enabled=bool(temporal_raw.get("enabled", False)),
        default_step=str(temporal_raw.get("default_step", raw.get("freq", "1h"))),
        default_stats=temporal_default_stats,
        time_position_frequencies=int(temporal_raw.get("time_position_frequencies", 4)),
        modality_steps={
            str(modality): str(step)
            for modality, step in dict(temporal_raw.get("modality_steps", {})).items()
        } or None,
        modality_stats=temporal_modality_stats or None,
    )
    closure_loss_weights = {
        str(name): float(weight)
        for name, weight in dict(raw.get("closure_loss_weights", {})).items()
        if float(weight) != 0.0
    }
    cross_prediction_selection_mode = str(
        raw.get("cross_prediction_selection_mode", "leave_one_out")
    )
    if cross_prediction_selection_mode not in {"leave_one_out", "leave_one_out_unrelated"}:
        raise ValueError(
            "cross_prediction_selection_mode must be 'leave_one_out' or "
            "'leave_one_out_unrelated'"
        )

    return ExperimentConfig(
        data_root=Path(raw["data_root"]),
        freq=str(raw.get("freq", "1h")),
        split_strategy=split_strategy,
        feature_coverage_basis=feature_coverage_basis,
        model_type=model_type,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        encoder_depth=int(raw.get("encoder_depth", 2)),
        decoder_depth=int(raw.get("decoder_depth", 2)),
        transformer_layers=int(raw.get("transformer_layers", 2)),
        transformer_heads=transformer_heads,
        sequence_encoder_type=sequence_encoder_type,
        sequence_fourier_frequencies=int(raw.get("sequence_fourier_frequencies", 6)),
        sequence_transformer_heads=sequence_transformer_heads,
        conditional_ccn_decoder=bool(raw.get("conditional_ccn_decoder", False)),
        sizing_crosstalk_layers=sizing_crosstalk_layers,
        sizing_crosstalk_heads=sizing_crosstalk_heads,
        decoder_expansion_depth=decoder_expansion_depth,
        latent_blocks=latent_blocks,
        block_modality_map=block_modality_map,
        size_grid=size_grid,
        temporal_windows=temporal_windows,
        batch_size=int(raw.get("batch_size", 128)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        learning_rate_schedule=learning_rate_schedule,
        min_learning_rate=float(raw.get("min_learning_rate", 0.0)),
        weight_decay=float(raw.get("weight_decay", 1e-6)),
        input_mask_probability=float(raw.get("input_mask_probability", 0.35)),
        latent_l2_weight=float(raw.get("latent_l2_weight", 1e-6)),
        kl_weight=float(raw.get("kl_weight", 0.0)),
        closure_loss_weights=closure_loss_weights,
        min_feature_coverage=float(raw.get("min_feature_coverage", 0.05)),
        min_feature_std=float(raw.get("min_feature_std", 1e-12)),
        validation_fraction=float(raw.get("validation_fraction", 0.15)),
        test_fraction=float(raw.get("test_fraction", 0.15)),
        validation_interval=int(raw.get("validation_interval", 1)),
        reconstruction_validation_interval=int(
            raw.get("reconstruction_validation_interval", 1)
        ),
        diagnostic_validation_interval=int(raw.get("diagnostic_validation_interval", 1)),
        seed=int(raw.get("seed", 42)),
        training_stages=tuple(raw.get("training_stages", ())),
        cross_prediction_selection_mode=cross_prediction_selection_mode,
        cross_prediction_exclusion_groups=tuple(
            tuple(str(modality) for modality in group)
            for group in raw.get("cross_prediction_exclusion_groups", ())
        ),
        modalities=tuple(modalities),
    )


def config_to_metadata(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "data_root": str(config.data_root),
        "freq": config.freq,
        "split_strategy": config.split_strategy,
        "feature_coverage_basis": config.feature_coverage_basis,
        "model_type": config.model_type,
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "encoder_depth": config.encoder_depth,
        "decoder_depth": config.decoder_depth,
        "transformer_layers": config.transformer_layers,
        "transformer_heads": config.transformer_heads,
        "sequence_encoder_type": config.sequence_encoder_type,
        "sequence_fourier_frequencies": config.sequence_fourier_frequencies,
        "sequence_transformer_heads": config.sequence_transformer_heads,
        "conditional_ccn_decoder": config.conditional_ccn_decoder,
        "sizing_crosstalk_layers": config.sizing_crosstalk_layers,
        "sizing_crosstalk_heads": config.sizing_crosstalk_heads,
        "decoder_expansion_depth": config.decoder_expansion_depth,
        "latent_blocks": dict(config.latent_blocks),
        "block_modality_map": {
            block: list(modalities)
            for block, modalities in config.block_modality_map.items()
        },
        "size_grid": {
            "enabled": config.size_grid.enabled,
            "min_diameter_nm": config.size_grid.min_diameter_nm,
            "max_diameter_nm": config.size_grid.max_diameter_nm,
            "bins": config.size_grid.bins,
            "interpolation": config.size_grid.interpolation,
            "diameter_units": dict(config.size_grid.diameter_units or {}),
        },
        "temporal_windows": {
            "enabled": config.temporal_windows.enabled,
            "default_step": config.temporal_windows.default_step,
            "default_stats": list(config.temporal_windows.default_stats),
            "time_position_frequencies": config.temporal_windows.time_position_frequencies,
            "modality_steps": dict(config.temporal_windows.modality_steps or {}),
            "modality_stats": {
                modality: list(stats)
                for modality, stats in (config.temporal_windows.modality_stats or {}).items()
            },
        },
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "learning_rate_schedule": config.learning_rate_schedule,
        "min_learning_rate": config.min_learning_rate,
        "weight_decay": config.weight_decay,
        "input_mask_probability": config.input_mask_probability,
        "latent_l2_weight": config.latent_l2_weight,
        "kl_weight": config.kl_weight,
        "closure_loss_weights": dict(config.closure_loss_weights),
        "min_feature_coverage": config.min_feature_coverage,
        "min_feature_std": config.min_feature_std,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "validation_interval": config.validation_interval,
        "reconstruction_validation_interval": config.reconstruction_validation_interval,
        "diagnostic_validation_interval": config.diagnostic_validation_interval,
        "seed": config.seed,
        "training_stages": list(config.training_stages),
        "cross_prediction_selection_mode": config.cross_prediction_selection_mode,
        "cross_prediction_exclusion_groups": [
            list(group)
            for group in config.cross_prediction_exclusion_groups
        ],
        "modalities": {
            modality.name: {
                "role": modality.role,
                "always_input": modality.always_input,
                "streams": [
                    {
                        "name": stream.name,
                        "path": stream.path,
                        "required": stream.required,
                        "variables": [
                            {"name": variable.name, "transform": variable.transform}
                            for variable in stream.variables
                        ],
                    }
                    for stream in modality.streams
                ],
            }
            for modality in config.modalities
        },
    }
