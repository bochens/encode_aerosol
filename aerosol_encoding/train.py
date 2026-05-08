from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .closure import AerosolClosureLosses
from .config import config_to_metadata, load_config
from .feature_store import load_feature_store
from .model import build_aerosol_model, feature_names_by_modality, unpack_model_output
from .training_data import AerosolDataset, PreparedArrays, prepare_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped masked aerosol autoencoder.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--features", required=True, help="features.npz from build_features.")
    parser.add_argument("--output", required=True, help="Run output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Preserve configured stages but stop after this many total epochs.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume from a checkpoint saved by this training script.",
    )
    parser.add_argument(
        "--resume-with-fresh-optimizer",
        action="store_true",
        help=(
            "Resume legacy checkpoints that lack optimizer/scheduler state by "
            "restoring model weights and history, then initializing a fresh optimizer."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Torch device.")
    return parser.parse_args()


def split_modalities(
    batch_x: torch.Tensor,
    batch_mask: torch.Tensor,
    modality_indices: dict[str, list[int]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    x_by_modality = {}
    mask_by_modality = {}
    for modality, indices in modality_indices.items():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=batch_x.device)
        x_by_modality[modality] = batch_x.index_select(1, index_tensor)
        mask_by_modality[modality] = batch_mask.index_select(1, index_tensor)
    return x_by_modality, mask_by_modality


def make_input_modality_mask(
    feature_mask_by_modality: dict[str, torch.Tensor],
    target_modalities: tuple[str, ...],
    always_input_modalities: tuple[str, ...],
    cross_prediction_exclusion_groups: tuple[tuple[str, ...], ...],
    mode: str,
    mask_probability: float,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    device = next(iter(feature_mask_by_modality.values())).device
    batch_size = next(iter(feature_mask_by_modality.values())).shape[0]

    for modality, feature_mask in feature_mask_by_modality.items():
        observed = feature_mask.sum(dim=1) > 0
        if modality in always_input_modalities or modality not in target_modalities:
            output[modality] = observed
        elif mode == "autoencode":
            output[modality] = observed
        elif mode == "random_mask":
            keep = torch.rand(batch_size, device=device) >= mask_probability
            output[modality] = observed & keep
        elif mode in {"leave_one_out", "leave_one_out_unrelated", "leave_one_group_member_out"}:
            output[modality] = observed
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    if mode in {"leave_one_out", "leave_one_out_unrelated", "leave_one_group_member_out"}:
        target_candidates = [
            modality
            for modality in target_modalities
            if modality in output
        ]
        candidate_in_group = None
        if mode == "leave_one_group_member_out":
            grouped_candidates = {
                modality
                for group in cross_prediction_exclusion_groups
                for modality in group
            }
            if not any(modality in grouped_candidates for modality in target_candidates):
                raise ValueError(
                    "leave_one_group_member_out requires at least one target modality "
                    "from cross_prediction_exclusion_groups"
                )
            candidate_in_group = torch.as_tensor(
                [modality in grouped_candidates for modality in target_candidates],
                dtype=torch.bool,
                device=device,
            )
        random_scores = torch.rand(batch_size, len(target_candidates), device=device)
        observed_targets = torch.stack(
            [feature_mask_by_modality[modality].sum(dim=1) > 0 for modality in target_candidates],
            dim=1,
        )
        if mode == "leave_one_group_member_out":
            assert candidate_in_group is not None
            grouped_observed = observed_targets & candidate_in_group.unsqueeze(0)
            fallback_rows = ~grouped_observed.any(dim=1)
            grouped_scores = random_scores.masked_fill(~grouped_observed, -1.0)
            fallback_scores = random_scores.masked_fill(~observed_targets, -1.0)
            random_scores = torch.where(
                fallback_rows.unsqueeze(1),
                fallback_scores,
                grouped_scores,
            )
        else:
            random_scores = random_scores.masked_fill(~observed_targets, -1.0)
        hidden_index = random_scores.argmax(dim=1)
        for index, modality in enumerate(target_candidates):
            hide = hidden_index == index
            hidden_modalities = {modality}
            if mode == "leave_one_out_unrelated":
                for group in cross_prediction_exclusion_groups:
                    if modality in group:
                        hidden_modalities = set(group)
                        break
            for hidden_modality in hidden_modalities:
                if hidden_modality in output:
                    output[hidden_modality] = output[hidden_modality] & ~hide

    any_input = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for mask in output.values():
        any_input |= mask
    if torch.all(any_input):
        return output

    observed_target_modalities = [
        modality
        for modality in target_modalities
        if modality in output
    ]
    for row in torch.nonzero(~any_input, as_tuple=False).flatten():
        for modality in observed_target_modalities:
            if feature_mask_by_modality[modality][row].sum() > 0:
                output[modality][row] = True
                break
    return output


def diagnostic_cross_prediction_modes(
    selection_mode: str,
    cross_prediction_exclusion_groups: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, str], ...]:
    modes: list[tuple[str, str]] = []
    if selection_mode != "leave_one_out":
        modes.append(("single_leave_one_out", "leave_one_out"))
    if cross_prediction_exclusion_groups and selection_mode != "leave_one_out_unrelated":
        modes.append(("strict_group_out", "leave_one_out_unrelated"))
    return tuple(modes)


def corrupt_inputs(
    x_by_modality: dict[str, torch.Tensor],
    mask_by_modality: dict[str, torch.Tensor],
    feature_dropout_probability: float,
    feature_noise_std: float,
    training: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not training or (feature_dropout_probability <= 0 and feature_noise_std <= 0):
        return dict(x_by_modality), dict(mask_by_modality)

    input_x: dict[str, torch.Tensor] = {}
    input_mask: dict[str, torch.Tensor] = {}
    for modality, values in x_by_modality.items():
        mask = mask_by_modality[modality].clone()
        if feature_dropout_probability > 0:
            keep = torch.rand_like(mask) >= feature_dropout_probability
            mask = mask * keep.to(dtype=mask.dtype)
        corrupted = values
        if feature_noise_std > 0:
            corrupted = corrupted + torch.randn_like(values) * feature_noise_std * mask
        input_x[modality] = corrupted
        input_mask[modality] = mask
    return input_x, input_mask


def masked_loss(
    decoded: dict[str, torch.Tensor],
    x_by_modality: dict[str, torch.Tensor],
    mask_by_modality: dict[str, torch.Tensor],
    target_modalities: tuple[str, ...],
    loss_row_masks: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    for modality in target_modalities:
        target = x_by_modality[modality]
        mask = mask_by_modality[modality]
        if loss_row_masks is not None:
            row_mask = loss_row_masks.get(modality)
            if row_mask is not None:
                mask = mask * row_mask.to(dtype=mask.dtype).unsqueeze(-1)
        valid = mask.sum()
        if valid <= 0:
            continue
        loss = (((decoded[modality] - target) ** 2) * mask).sum() / valid
        losses.append(loss)
        metrics[f"loss_{modality}"] = float(loss.detach().cpu())

    if not losses:
        raise RuntimeError("Batch has no finite target values")
    total = torch.stack(losses).mean()
    metrics["loss"] = float(total.detach().cpu())
    return total, metrics


def make_loss_row_masks(
    mask_by_modality: dict[str, torch.Tensor],
    input_modality_mask: dict[str, torch.Tensor],
    target_modalities: tuple[str, ...],
    loss_mode: str,
) -> dict[str, torch.Tensor] | None:
    if loss_mode == "all":
        return None
    if loss_mode != "hidden_only":
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    return {
        modality: (mask_by_modality[modality].sum(dim=1) > 0) & ~input_modality_mask[modality]
        for modality in target_modalities
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    arrays: PreparedArrays,
    target_modalities: tuple[str, ...],
    always_input_modalities: tuple[str, ...],
    cross_prediction_exclusion_groups: tuple[tuple[str, ...], ...],
    stage_mode: str,
    loss_mode: str,
    mask_probability: float,
    feature_dropout_probability: float,
    feature_noise_std: float,
    latent_l2_weight: float,
    kl_weight: float,
    closure_losses: AerosolClosureLosses | None,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    count = 0

    for batch_x, batch_feature_mask in loader:
        batch_x = batch_x.to(device)
        batch_feature_mask = batch_feature_mask.to(device)
        x_by_modality, mask_by_modality = split_modalities(
            batch_x, batch_feature_mask, arrays.modality_indices
        )
        input_x_by_modality, input_mask_by_modality = corrupt_inputs(
            x_by_modality,
            mask_by_modality,
            feature_dropout_probability=feature_dropout_probability,
            feature_noise_std=feature_noise_std,
            training=training,
        )
        input_mask = make_input_modality_mask(
            input_mask_by_modality,
            target_modalities=target_modalities,
            always_input_modalities=always_input_modalities,
            cross_prediction_exclusion_groups=cross_prediction_exclusion_groups,
            mode=stage_mode,
            mask_probability=mask_probability,
        )
        loss_row_masks = make_loss_row_masks(
            mask_by_modality,
            input_mask,
            target_modalities,
            loss_mode,
        )

        with torch.set_grad_enabled(training):
            z, decoded, diagnostics = unpack_model_output(
                model(input_x_by_modality, input_mask_by_modality, input_mask)
            )
            loss, metrics = masked_loss(
                decoded,
                x_by_modality,
                mask_by_modality,
                target_modalities,
                loss_row_masks=loss_row_masks,
            )
            if loss_mode == "hidden_only":
                metrics["hidden_target_loss"] = metrics["loss"]
            else:
                metrics["reconstruction_loss"] = metrics["loss"]
            if kl_weight > 0 and "kl" in diagnostics:
                kl = diagnostics["kl"]
                loss = loss + kl_weight * kl
                for key, value in diagnostics.items():
                    metrics[key] = float(value.detach().cpu())
                metrics["kl_weight"] = kl_weight
            if latent_l2_weight > 0:
                latent_l2 = (z ** 2).mean()
                loss = loss + latent_l2_weight * latent_l2
                metrics["latent_l2"] = float(latent_l2.detach().cpu())
            if closure_losses is not None and closure_losses.enabled:
                closure_metrics = closure_losses(
                    decoded,
                    x_by_modality,
                    mask_by_modality,
                    input_mask,
                )
                for key, value in closure_metrics.items():
                    raw_name = key.removeprefix("closure_")
                    weight = closure_losses.weights[raw_name]
                    loss = loss + weight * value
                    metrics[key] = float(value.detach().cpu())
                    metrics[f"{key}_weighted"] = float((weight * value).detach().cpu())
            metrics["loss"] = float(loss.detach().cpu())
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        batch_n = batch_x.shape[0]
        count += batch_n
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * batch_n

    return {key: value / max(count, 1) for key, value in sums.items()}


def target_exclusion_group(
    target: str,
    cross_prediction_exclusion_groups: tuple[tuple[str, ...], ...],
) -> set[str]:
    for group in cross_prediction_exclusion_groups:
        group_set = set(group)
        if target in group_set:
            return group_set
    return {target}


def run_cross_prediction_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    arrays: PreparedArrays,
    target_modalities: tuple[str, ...],
    always_input_modalities: tuple[str, ...],
    cross_prediction_exclusion_groups: tuple[tuple[str, ...], ...],
    stage_mode: str,
    device: torch.device,
) -> dict[str, float]:
    if stage_mode not in {"leave_one_out", "leave_one_out_unrelated"}:
        raise ValueError(f"Cross-prediction evaluation does not support mode: {stage_mode}")

    model.eval()
    se_sums = {modality: 0.0 for modality in target_modalities}
    valid_counts = {modality: 0.0 for modality in target_modalities}

    with torch.no_grad():
        for batch_x, batch_feature_mask in loader:
            batch_x = batch_x.to(device)
            batch_feature_mask = batch_feature_mask.to(device)
            x_by_modality, mask_by_modality = split_modalities(
                batch_x, batch_feature_mask, arrays.modality_indices
            )
            observed_by_modality = {
                modality: mask.sum(dim=1) > 0
                for modality, mask in mask_by_modality.items()
            }

            for target in target_modalities:
                if target not in mask_by_modality:
                    continue
                hidden_modalities = {target}
                if stage_mode == "leave_one_out_unrelated":
                    hidden_modalities = target_exclusion_group(
                        target,
                        cross_prediction_exclusion_groups,
                    )

                input_mask: dict[str, torch.Tensor] = {}
                any_input = torch.zeros(batch_x.shape[0], dtype=torch.bool, device=device)
                for modality, observed in observed_by_modality.items():
                    is_hidden = modality in hidden_modalities and modality not in always_input_modalities
                    input_mask[modality] = (
                        torch.zeros_like(observed)
                        if is_hidden
                        else observed
                    )
                    any_input |= input_mask[modality]

                target_rows = observed_by_modality[target] & any_input
                if not torch.any(target_rows):
                    continue

                row_indices = torch.nonzero(target_rows, as_tuple=False).flatten()
                case_x_by_modality = {
                    modality: values.index_select(0, row_indices)
                    for modality, values in x_by_modality.items()
                }
                case_mask_by_modality = {
                    modality: values.index_select(0, row_indices)
                    for modality, values in mask_by_modality.items()
                }
                case_input_mask = {
                    modality: values.index_select(0, row_indices)
                    for modality, values in input_mask.items()
                }
                _, decoded, _ = unpack_model_output(
                    model(case_x_by_modality, case_mask_by_modality, case_input_mask)
                )
                effective_mask = case_mask_by_modality[target]
                valid = effective_mask.sum()
                if valid <= 0:
                    continue
                se = (
                    ((decoded[target] - case_x_by_modality[target]) ** 2)
                    * effective_mask
                ).sum()
                se_sums[target] += float(se.cpu())
                valid_counts[target] += float(valid.cpu())

    metrics: dict[str, float] = {}
    modality_losses: list[float] = []
    for modality in target_modalities:
        if valid_counts[modality] <= 0:
            continue
        loss = se_sums[modality] / valid_counts[modality]
        metrics[f"loss_{modality}"] = loss
        modality_losses.append(loss)
    if not modality_losses:
        raise RuntimeError("Cross-prediction evaluation found no finite target values")
    metrics["loss"] = float(np.mean(modality_losses))
    metrics["cross_prediction_loss"] = metrics["loss"]
    return metrics


def build_training_stages(
    config,
    cli_epochs: int | None,
    max_epochs: int | None = None,
) -> list[dict[str, float | int | str]]:
    if cli_epochs is not None:
        if max_epochs is not None:
            raise ValueError("--epochs and --max-epochs cannot be used together")
        return [
            {
                "name": "cli_random_mask",
                "mode": "random_mask",
                "epochs": int(cli_epochs),
                "input_mask_probability": float(config.input_mask_probability),
                "feature_dropout_probability": 0.0,
                "feature_noise_std": 0.0,
            }
        ]

    if config.training_stages:
        stages = [dict(stage) for stage in config.training_stages]
    else:
        stages = [
            {
                "name": "random_mask",
                "mode": "random_mask",
                "epochs": 100,
                "input_mask_probability": float(config.input_mask_probability),
                "feature_dropout_probability": 0.0,
                "feature_noise_std": 0.0,
            }
        ]

    if max_epochs is None:
        return stages
    if max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    truncated: list[dict[str, float | int | str]] = []
    remaining = int(max_epochs)
    for stage in stages:
        stage_epochs = int(stage.get("epochs", 1))
        if remaining <= 0:
            break
        clipped = dict(stage)
        clipped["epochs"] = min(stage_epochs, remaining)
        truncated.append(clipped)
        remaining -= int(clipped["epochs"])
    return truncated


def checkpoint_payload(
    model: torch.nn.Module,
    arrays: PreparedArrays,
    config_metadata: dict[str, Any],
    target_modalities: tuple[str, ...],
    context_modalities: tuple[str, ...],
    always_input_modalities: tuple[str, ...],
    history: list[dict[str, float]],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    global_epoch: int | None = None,
    best_validation: float | None = None,
) -> dict[str, Any]:
    modality_dims = {
        modality: len(indices)
        for modality, indices in arrays.modality_indices.items()
    }
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "modality_dims": modality_dims,
        "target_modalities": target_modalities,
        "context_modalities": context_modalities,
        "always_input_modalities": always_input_modalities,
        "modality_indices": arrays.modality_indices,
        "feature_names": arrays.feature_names,
        "raw_feature_indices": arrays.raw_feature_indices,
        "mean": arrays.mean,
        "std": arrays.std,
        "splits": arrays.splits,
        "dropped_features": arrays.dropped_features,
        "config": config_metadata,
        "history": history,
        "model_type": getattr(model, "model_type", "grouped_masked_autoencoder"),
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "latent_blocks": getattr(model, "latent_blocks", {}),
        "transformer_layers": getattr(model, "transformer_layers", None),
        "transformer_heads": getattr(model, "transformer_heads", None),
        "block_modality_map": getattr(model, "block_modality_map", {}),
        "sequence_encoder_type": getattr(model, "sequence_encoder_type", None),
        "sequence_fourier_frequencies": getattr(model, "sequence_fourier_frequencies", None),
        "sequence_transformer_heads": getattr(model, "sequence_transformer_heads", None),
        "conditional_ccn_decoder": getattr(model, "conditional_ccn_decoder", False),
        "sizing_crosstalk_layers": getattr(model, "sizing_crosstalk_layers", 0),
        "sizing_crosstalk_heads": getattr(model, "sizing_crosstalk_heads", None),
        "decoder_expansion_depth": getattr(model, "decoder_expansion_depth", 0),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if global_epoch is not None:
        payload["global_epoch"] = int(global_epoch)
    if best_validation is not None:
        payload["best_validation"] = float(best_validation)
    return payload


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def should_run_interval(epoch: int, total_epochs: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return epoch == 1 or epoch == total_epochs or epoch % interval == 0


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    arrays: PreparedArrays,
    target_modalities: tuple[str, ...],
) -> None:
    expected_model_type = getattr(model, "model_type", "grouped_masked_autoencoder")
    if checkpoint.get("model_type") != expected_model_type:
        raise ValueError(
            f"Checkpoint model_type={checkpoint.get('model_type')!r} does not match "
            f"current model_type={expected_model_type!r}"
        )
    checkpoint_dims = checkpoint.get("modality_dims")
    current_dims = {
        modality: len(indices)
        for modality, indices in arrays.modality_indices.items()
    }
    if checkpoint_dims != current_dims:
        raise ValueError("Checkpoint modality dimensions do not match the current features.")
    if list(checkpoint.get("feature_names", [])) != list(arrays.feature_names):
        raise ValueError("Checkpoint feature names do not match the current features.")
    if tuple(checkpoint.get("target_modalities", ())) != tuple(target_modalities):
        raise ValueError("Checkpoint target modalities do not match the current config.")


def best_validation_from_history(history: list[dict[str, float]]) -> float:
    values = [
        float(row["validation_cross_loss"])
        for row in history
        if "validation_cross_loss" in row and row["validation_cross_loss"] != ""
    ]
    if not values:
        raise ValueError("Cannot resume because history has no validation_cross_loss values.")
    return min(values)


def set_cosine_scheduler_epoch(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
) -> None:
    if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise TypeError(
            "Fresh-optimizer resume without scheduler_state is only implemented for "
            "CosineAnnealingLR."
        )
    for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
        lr = scheduler.eta_min + (base_lr - scheduler.eta_min) * (
            1.0 + math.cos(math.pi * epoch / scheduler.T_max)
        ) / 2.0
        group["lr"] = lr
    scheduler.last_epoch = epoch
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


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

    target_modalities = tuple(
        modality
        for modality in config.target_modalities
        if modality in arrays.modality_indices
    )
    always_input_modalities = tuple(
        modality
        for modality in config.always_input_modalities
        if modality in arrays.modality_indices
    )
    context_modalities = tuple(
        modality
        for modality in config.context_modalities
        if modality in arrays.modality_indices
    )
    modality_dims = {
        modality: len(indices)
        for modality, indices in arrays.modality_indices.items()
    }

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(args.device)
    feature_name_map = feature_names_by_modality(arrays.feature_names, arrays.modality_indices)
    model = build_aerosol_model(
        model_type=config.model_type,
        modality_dims=modality_dims,
        target_modalities=target_modalities,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        encoder_depth=config.encoder_depth,
        decoder_depth=config.decoder_depth,
        feature_names_by_modality_map=feature_name_map,
        latent_blocks=config.latent_blocks,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        block_modality_map=config.block_modality_map,
        sequence_encoder_type=config.sequence_encoder_type,
        sequence_fourier_frequencies=config.sequence_fourier_frequencies,
        sequence_transformer_heads=config.sequence_transformer_heads,
        conditional_ccn_decoder=config.conditional_ccn_decoder,
        sizing_crosstalk_layers=config.sizing_crosstalk_layers,
        sizing_crosstalk_heads=config.sizing_crosstalk_heads,
        decoder_expansion_depth=config.decoder_expansion_depth,
    ).to(device)
    config_metadata = config_to_metadata(config)
    mean_by_modality = {
        modality: torch.as_tensor(arrays.mean[indices], dtype=torch.float32, device=device)
        for modality, indices in arrays.modality_indices.items()
    }
    std_by_modality = {
        modality: torch.as_tensor(arrays.std[indices], dtype=torch.float32, device=device)
        for modality, indices in arrays.modality_indices.items()
    }
    closure_losses = (
        AerosolClosureLosses(
            feature_names_by_modality=feature_name_map,
            mean_by_modality=mean_by_modality,
            std_by_modality=std_by_modality,
            weights=config.closure_loss_weights,
        ).to(device)
        if config.closure_loss_weights
        else None
    )

    train_loader = DataLoader(
        AerosolDataset(arrays, arrays.splits["train"]),
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        AerosolDataset(arrays, arrays.splits["validation"]),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        AerosolDataset(arrays, arrays.splits["test"]),
        batch_size=config.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    stages = build_training_stages(config, args.epochs, args.max_epochs)
    total_epochs = sum(int(stage.get("epochs", 1)) for stage in stages)
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if config.learning_rate_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs, 1),
            eta_min=config.min_learning_rate,
        )
    validation_cross_mode = config.cross_prediction_selection_mode
    diagnostic_validation_modes = diagnostic_cross_prediction_modes(
        validation_cross_mode,
        config.cross_prediction_exclusion_groups,
    )
    if config.validation_interval <= 0:
        raise ValueError("validation_interval must be positive so a best checkpoint can be selected")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_validation = float("inf")
    global_epoch = 0
    resume_completed_epochs = 0
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(
            args.resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        validate_resume_checkpoint(resume_checkpoint, model, arrays, target_modalities)
        model.load_state_dict(resume_checkpoint["model_state"])
        history = list(resume_checkpoint.get("history", []))
        if not history:
            raise ValueError("Cannot resume from a checkpoint with empty history.")
        global_epoch = int(resume_checkpoint.get("global_epoch", history[-1]["epoch"]))
        resume_completed_epochs = global_epoch
        if global_epoch >= total_epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {global_epoch}, "
                f"but this run only has {total_epochs} configured epochs."
            )
        best_validation = float(
            resume_checkpoint.get("best_validation", best_validation_from_history(history))
        )
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        elif not args.resume_with_fresh_optimizer:
            raise ValueError(
                "Resume checkpoint lacks optimizer_state. Re-run with "
                "--resume-with-fresh-optimizer only when intentionally resuming a "
                "legacy checkpoint with a fresh optimizer."
            )
        if scheduler is not None:
            if "scheduler_state" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
            elif args.resume_with_fresh_optimizer:
                set_cosine_scheduler_epoch(optimizer, scheduler, global_epoch)
            else:
                raise ValueError(
                    "Resume checkpoint lacks scheduler_state. Re-run with "
                    "--resume-with-fresh-optimizer only when intentionally resuming a "
                    "legacy checkpoint with a fresh scheduler."
                )
        print(
            f"resuming from epoch={global_epoch} best_validation={best_validation:.5f} "
            f"fresh_optimizer={args.resume_with_fresh_optimizer and 'optimizer_state' not in resume_checkpoint}",
            flush=True,
        )

    completed_in_prior_stages = resume_completed_epochs
    for stage in stages:
        stage_name = str(stage.get("name", "stage"))
        stage_mode = str(stage.get("mode", "random_mask"))
        stage_epochs = int(stage.get("epochs", 1))
        if completed_in_prior_stages >= stage_epochs:
            completed_in_prior_stages -= stage_epochs
            continue
        start_stage_epoch = completed_in_prior_stages + 1
        completed_in_prior_stages = 0
        stage_mask_probability = float(
            stage.get("input_mask_probability", config.input_mask_probability)
        )
        stage_feature_dropout = float(stage.get("feature_dropout_probability", 0.0))
        stage_feature_noise = float(stage.get("feature_noise_std", 0.0))
        stage_loss_mode = str(stage.get("loss_mode", "all"))

        for stage_epoch in range(start_stage_epoch, stage_epochs + 1):
            global_epoch += 1
            train_metrics = run_epoch(
                model,
                train_loader,
                arrays,
                target_modalities,
                always_input_modalities,
                config.cross_prediction_exclusion_groups,
                stage_mode=stage_mode,
                loss_mode=stage_loss_mode,
                mask_probability=stage_mask_probability,
                feature_dropout_probability=stage_feature_dropout,
                feature_noise_std=stage_feature_noise,
                latent_l2_weight=config.latent_l2_weight,
                kl_weight=float(stage.get("kl_weight", config.kl_weight)),
                closure_losses=closure_losses,
                device=device,
                optimizer=optimizer,
            )
            run_cross_validation = should_run_interval(
                global_epoch,
                total_epochs,
                config.validation_interval,
            )
            run_reconstruction_validation = should_run_interval(
                global_epoch,
                total_epochs,
                config.reconstruction_validation_interval,
            )
            run_diagnostic_validation = should_run_interval(
                global_epoch,
                total_epochs,
                config.diagnostic_validation_interval,
            )

            validation_reconstruction_metrics: dict[str, float] = {}
            if run_reconstruction_validation:
                validation_reconstruction_metrics = run_epoch(
                    model,
                    validation_loader,
                    arrays,
                    target_modalities,
                    always_input_modalities,
                    config.cross_prediction_exclusion_groups,
                    stage_mode="autoencode",
                    loss_mode="all",
                    mask_probability=0.0,
                    feature_dropout_probability=0.0,
                    feature_noise_std=0.0,
                    latent_l2_weight=0.0,
                    kl_weight=0.0,
                    closure_losses=None,
                    device=device,
                    optimizer=None,
                )
            validation_cross_metrics: dict[str, float] = {}
            if run_cross_validation:
                validation_cross_metrics = run_cross_prediction_epoch(
                    model,
                    validation_loader,
                    arrays,
                    target_modalities,
                    always_input_modalities,
                    config.cross_prediction_exclusion_groups,
                    stage_mode=validation_cross_mode,
                    device=device,
                )
            diagnostic_validation_metrics: dict[str, dict[str, float]] = {}
            if run_diagnostic_validation:
                diagnostic_validation_metrics = {
                    label: run_cross_prediction_epoch(
                        model,
                        validation_loader,
                        arrays,
                        target_modalities,
                        always_input_modalities,
                        config.cross_prediction_exclusion_groups,
                        stage_mode=mode,
                        device=device,
                    )
                    for label, mode in diagnostic_validation_modes
                }
            row = {
                "epoch": float(global_epoch),
                "stage_epoch": float(stage_epoch),
                "stage": stage_name,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_reconstruction_{key}": value for key, value in validation_reconstruction_metrics.items()}
            )
            row.update({f"validation_cross_{key}": value for key, value in validation_cross_metrics.items()})
            for label, metrics in diagnostic_validation_metrics.items():
                row.update({f"validation_{label}_{key}": value for key, value in metrics.items()})
            diagnostic_text = " ".join(
                f"val_{label}={metrics['loss']:.5f}"
                for label, metrics in diagnostic_validation_metrics.items()
            )
            validation_text_parts = []
            if validation_reconstruction_metrics:
                validation_text_parts.append(
                    f"val_recon={validation_reconstruction_metrics['loss']:.5f}"
                )
            else:
                validation_text_parts.append("val_recon=skipped")
            if validation_cross_metrics:
                validation_text_parts.append(
                    f"val_cross={validation_cross_metrics['loss']:.5f}"
                )
            else:
                validation_text_parts.append("val_cross=skipped")
            if diagnostic_text:
                validation_text_parts.append(diagnostic_text)
            history.append(row)
            write_history(output / "history.csv", history)
            print(
                f"epoch={global_epoch} stage={stage_name} mode={stage_mode} "
                f"loss_mode={stage_loss_mode} "
                f"train_loss={train_metrics['loss']:.5f} "
                f"{' '.join(validation_text_parts)}",
                flush=True,
            )

            if validation_cross_metrics and validation_cross_metrics["loss"] < best_validation:
                best_validation = validation_cross_metrics["loss"]
            if scheduler is not None:
                scheduler.step()
            payload = checkpoint_payload(
                model,
                arrays,
                config_metadata,
                target_modalities,
                context_modalities,
                always_input_modalities,
                history,
                optimizer=optimizer,
                scheduler=scheduler,
                global_epoch=global_epoch,
                best_validation=best_validation,
            )
            if validation_cross_metrics and validation_cross_metrics["loss"] <= best_validation:
                torch.save(payload, output / "checkpoint.pt")
            torch.save(payload, output / "last_checkpoint.pt")

    best_checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = run_cross_prediction_epoch(
        model,
        test_loader,
        arrays,
        target_modalities,
        always_input_modalities,
        config.cross_prediction_exclusion_groups,
        stage_mode=validation_cross_mode,
        device=device,
    )
    test_metrics["selection_mode"] = validation_cross_mode
    with (output / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    for label, mode in diagnostic_validation_modes:
        diagnostic_test_metrics = run_cross_prediction_epoch(
            model,
            test_loader,
            arrays,
            target_modalities,
            always_input_modalities,
            config.cross_prediction_exclusion_groups,
            stage_mode=mode,
            device=device,
        )
        diagnostic_test_metrics["selection_mode"] = mode
        with (output / f"test_metrics_{label}.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostic_test_metrics, handle, indent=2)
    write_history(output / "history.csv", history)

    with (output / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_names": arrays.feature_names,
                "modality_indices": arrays.modality_indices,
                "dropped_features": arrays.dropped_features,
                "splits": {key: value.tolist() for key, value in arrays.splits.items()},
            },
            handle,
            indent=2,
        )
    print(f"wrote {output / 'checkpoint.pt'}")
    print(f"test_loss={test_metrics['loss']:.5f}")


if __name__ == "__main__":
    main()
