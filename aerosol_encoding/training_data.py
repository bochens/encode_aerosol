from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TIME_BIN_RE = re.compile(r"^(?P<base>.+)__time_bin_(?P<bin>\d+)$")


@dataclass(frozen=True)
class PreparedArrays:
    x: np.ndarray
    feature_mask: np.ndarray
    times: np.ndarray
    modality_indices: dict[str, list[int]]
    feature_names: list[str]
    raw_feature_indices: list[int]
    mean: np.ndarray
    std: np.ndarray
    splits: dict[str, np.ndarray]
    dropped_features: dict[str, list[str]]


class AerosolDataset(Dataset):
    def __init__(self, arrays: PreparedArrays, indices: np.ndarray) -> None:
        self.x = torch.from_numpy(arrays.x[indices].astype(np.float32))
        self.feature_mask = torch.from_numpy(arrays.feature_mask[indices].astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.feature_mask[index]


def chronological_splits(
    n_rows: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, np.ndarray]:
    if n_rows < 10:
        raise ValueError(f"Need at least 10 rows for train/validation/test split, got {n_rows}")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 0.8:
        raise ValueError("Validation/test fractions must be nonnegative and leave at least 20% training data")

    test_n = max(1, int(round(n_rows * test_fraction)))
    validation_n = max(1, int(round(n_rows * validation_fraction)))
    train_n = n_rows - validation_n - test_n
    if train_n < 1:
        raise ValueError("Split leaves no training rows")

    return {
        "train": np.arange(0, train_n),
        "validation": np.arange(train_n, train_n + validation_n),
        "test": np.arange(train_n + validation_n, n_rows),
    }


def _fractional_assignments(
    scores: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = scores < test_fraction
    validation = (scores >= test_fraction) & (scores < test_fraction + validation_fraction)
    train = ~(test | validation)
    return train, validation, test


def _stable_unit_score(label: str) -> float:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / float(2**64 - 1)


def _row_hash_splits(
    n_rows: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, np.ndarray]:
    scores = np.asarray(
        [_stable_unit_score(f"row:{index}") for index in range(n_rows)],
        dtype=np.float64,
    )
    train, validation, test = _fractional_assignments(
        scores,
        validation_fraction,
        test_fraction,
    )
    if not (train.any() and validation.any() and test.any()):
        order = np.argsort(scores)
        test_n = max(1, int(round(n_rows * test_fraction)))
        validation_n = max(1, int(round(n_rows * validation_fraction)))
        train_n = n_rows - validation_n - test_n
        if train_n < 1:
            raise ValueError("Split leaves no training rows")
        train = np.zeros(n_rows, dtype=bool)
        validation = np.zeros(n_rows, dtype=bool)
        test = np.zeros(n_rows, dtype=bool)
        test[order[:test_n]] = True
        validation[order[test_n:test_n + validation_n]] = True
        train[order[test_n + validation_n:]] = True
    return {
        "train": np.flatnonzero(train),
        "validation": np.flatnonzero(validation),
        "test": np.flatnonzero(test),
    }


def calendar_hash_splits(
    times: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    calendar_unit: str,
) -> dict[str, np.ndarray]:
    if times.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows for train/validation/test split, got {times.shape[0]}")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 0.8:
        raise ValueError("Validation/test fractions must be nonnegative and leave at least 20% training data")

    datetime_index = pd.DatetimeIndex(times)
    if calendar_unit == "day":
        labels = datetime_index.strftime("%Y-%m-%d")
    elif calendar_unit == "month":
        labels = datetime_index.strftime("%Y-%m")
    else:
        raise ValueError(f"Unsupported calendar hash unit: {calendar_unit}")
    unique_scores = {
        label: _stable_unit_score(f"{calendar_unit}:{label}")
        for label in pd.Index(labels).unique()
    }
    scores = np.asarray([unique_scores[label] for label in labels], dtype=np.float64)
    train, validation, test = _fractional_assignments(
        scores,
        validation_fraction,
        test_fraction,
    )
    if not (train.any() and validation.any() and test.any()):
        return _row_hash_splits(times.shape[0], validation_fraction, test_fraction)
    return {
        "train": np.flatnonzero(train),
        "validation": np.flatnonzero(validation),
        "test": np.flatnonzero(test),
    }


def make_splits(
    times: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    split_strategy: str,
) -> dict[str, np.ndarray]:
    if split_strategy == "chronological":
        return chronological_splits(times.shape[0], validation_fraction, test_fraction)
    if split_strategy == "calendar_day_hash":
        return calendar_hash_splits(times, validation_fraction, test_fraction, "day")
    if split_strategy == "calendar_month_hash":
        return calendar_hash_splits(times, validation_fraction, test_fraction, "month")
    raise ValueError(f"Unknown split_strategy: {split_strategy}")


def prepare_arrays(
    matrix: np.ndarray,
    times: np.ndarray,
    metadata: dict[str, Any],
    min_feature_coverage: float,
    min_feature_std: float,
    validation_fraction: float,
    test_fraction: float,
    split_strategy: str = "chronological",
    feature_coverage_basis: str = "train",
) -> PreparedArrays:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got {matrix.shape}")
    if feature_coverage_basis not in {"train", "all"}:
        raise ValueError(
            f"feature_coverage_basis must be 'train' or 'all', got {feature_coverage_basis!r}"
        )

    splits = make_splits(times, validation_fraction, test_fraction, split_strategy)
    train_indices = splits["train"]
    coverage_indices = train_indices if feature_coverage_basis == "train" else np.arange(matrix.shape[0])
    feature_names = list(metadata["feature_names"])
    raw_modality_indices: dict[str, list[int]] = {
        modality: list(indices)
        for modality, indices in metadata["modality_indices"].items()
    }

    selected_raw_indices: list[int] = []
    selected_feature_names: list[str] = []
    selected_modality_indices: dict[str, list[int]] = {}
    dropped_features: dict[str, list[str]] = {}

    for modality, raw_indices in raw_modality_indices.items():
        selected_modality_indices[modality] = []
        dropped_features[modality] = []
        coverage_values = matrix[np.ix_(coverage_indices, raw_indices)]
        finite = np.isfinite(coverage_values)
        coverage = finite.mean(axis=0)
        std = np.full(len(raw_indices), np.nan, dtype=np.float64)
        covered = coverage > 0.0
        if np.any(covered):
            with np.errstate(invalid="ignore"):
                train_values = matrix[np.ix_(train_indices, raw_indices)]
                std[covered] = np.nanstd(train_values[:, covered], axis=0)

        for local_idx, raw_idx in enumerate(raw_indices):
            keep = (
                coverage[local_idx] >= min_feature_coverage
                and np.isfinite(std[local_idx])
                and std[local_idx] > min_feature_std
            )
            if keep:
                selected_modality_indices[modality].append(len(selected_raw_indices))
                selected_raw_indices.append(raw_idx)
                selected_feature_names.append(feature_names[raw_idx])
            else:
                dropped_features[modality].append(feature_names[raw_idx])

        if not selected_modality_indices[modality]:
            raise ValueError(
                f"All features were dropped for modality {modality}. "
                f"Check date range, QC masks, and min_feature_coverage={min_feature_coverage}."
            )

    (
        selected_raw_indices,
        selected_feature_names,
        selected_modality_indices,
        dropped_features,
    ) = drop_incomplete_temporal_channels(
        selected_raw_indices,
        selected_feature_names,
        selected_modality_indices,
        dropped_features,
    )

    selected = matrix[:, selected_raw_indices].astype(np.float32)
    train_selected = selected[train_indices]
    mean = np.nanmean(train_selected, axis=0).astype(np.float32)
    std = np.nanstd(train_selected, axis=0).astype(np.float32)

    bad_norm = ~np.isfinite(mean) | ~np.isfinite(std) | (std <= 1e-12)
    if np.any(bad_norm):
        bad_names = [selected_feature_names[idx] for idx in np.flatnonzero(bad_norm)]
        raise ValueError(f"Selected features have invalid normalization statistics: {bad_names[:10]}")

    normalized = (selected - mean) / std
    feature_mask = np.isfinite(normalized)
    normalized = np.where(feature_mask, normalized, 0.0).astype(np.float32)
    valid_rows = feature_mask.any(axis=1)
    if not np.any(valid_rows):
        raise ValueError("No rows have any finite selected features")

    normalized = normalized[valid_rows]
    feature_mask = feature_mask[valid_rows]
    times = times[valid_rows]
    splits = make_splits(times, validation_fraction, test_fraction, split_strategy)

    return PreparedArrays(
        x=normalized,
        feature_mask=feature_mask.astype(np.float32),
        times=times,
        modality_indices=selected_modality_indices,
        feature_names=selected_feature_names,
        raw_feature_indices=selected_raw_indices,
        mean=mean,
        std=std,
        splits=splits,
        dropped_features=dropped_features,
    )


def drop_incomplete_temporal_channels(
    selected_raw_indices: list[int],
    selected_feature_names: list[str],
    selected_modality_indices: dict[str, list[int]],
    dropped_features: dict[str, list[str]],
) -> tuple[list[int], list[str], dict[str, list[int]], dict[str, list[str]]]:
    keep = np.ones(len(selected_feature_names), dtype=bool)
    for modality, selected_indices in selected_modality_indices.items():
        temporal_by_channel: dict[str, dict[int, int]] = {}
        for selected_index in selected_indices:
            match = TIME_BIN_RE.match(selected_feature_names[selected_index])
            if match is None:
                continue
            temporal_by_channel.setdefault(match.group("base"), {})[
                int(match.group("bin"))
            ] = selected_index
        if not temporal_by_channel:
            continue
        all_bins = {
            time_bin
            for bins in temporal_by_channel.values()
            for time_bin in bins
        }
        expected_bins = set(range(max(all_bins) + 1)) if all_bins else set()
        for channel, bins in temporal_by_channel.items():
            if set(bins) == expected_bins:
                continue
            for selected_index in bins.values():
                keep[selected_index] = False
                dropped_features.setdefault(modality, []).append(
                    selected_feature_names[selected_index]
                )

    old_to_new: dict[int, int] = {}
    new_raw_indices: list[int] = []
    new_feature_names: list[str] = []
    for old_index, should_keep in enumerate(keep):
        if not should_keep:
            continue
        old_to_new[old_index] = len(new_raw_indices)
        new_raw_indices.append(selected_raw_indices[old_index])
        new_feature_names.append(selected_feature_names[old_index])

    new_modality_indices: dict[str, list[int]] = {}
    for modality, selected_indices in selected_modality_indices.items():
        new_indices = [
            old_to_new[index]
            for index in selected_indices
            if index in old_to_new
        ]
        if not new_indices:
            raise ValueError(
                f"All selected features were dropped for modality {modality} "
                "while enforcing complete temporal channels"
            )
        new_modality_indices[modality] = new_indices
    return new_raw_indices, new_feature_names, new_modality_indices, dropped_features
