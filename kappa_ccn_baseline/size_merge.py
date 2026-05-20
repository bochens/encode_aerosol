from __future__ import annotations

from dataclasses import dataclass
import re
from math import erf

import numpy as np


SIZE_MODALITIES = ("size_smps", "size_uhsas", "size_opc", "size_aps")

SPECTRUM_RE = re.compile(
    r"^(?P<modality>size_[^_]+)__diameter_grid__dN_dlogDp"
    r"__diameter_common_nm_(?P<diameter>[0-9.eE+-]+)"
    r"__time_bin_(?P<time_bin>\d+)$"
)


@dataclass(frozen=True)
class InstrumentSpectrumIndex:
    modality: str
    diameter_nm: np.ndarray
    feature_indices_by_diameter: tuple[np.ndarray, ...]


def build_spectrum_indices(
    feature_names: list[str] | tuple[str, ...],
    modalities: tuple[str, ...] = SIZE_MODALITIES,
) -> dict[str, InstrumentSpectrumIndex]:
    grouped: dict[str, dict[float, list[int]]] = {
        modality: {}
        for modality in modalities
    }
    for index, feature_name in enumerate(feature_names):
        match = SPECTRUM_RE.match(feature_name)
        if match is None:
            continue
        modality = match.group("modality")
        if modality not in grouped:
            continue
        diameter = float(match.group("diameter"))
        grouped[modality].setdefault(diameter, []).append(index)

    output: dict[str, InstrumentSpectrumIndex] = {}
    for modality, by_diameter in grouped.items():
        if not by_diameter:
            continue
        diameters = np.asarray(sorted(by_diameter), dtype=np.float64)
        feature_groups = tuple(
            np.asarray(sorted(by_diameter[float(diameter)]), dtype=np.int64)
            for diameter in diameters
        )
        output[modality] = InstrumentSpectrumIndex(
            modality=modality,
            diameter_nm=diameters,
            feature_indices_by_diameter=feature_groups,
        )
    return output


def instrument_log1p_spectrum(
    transformed_row: np.ndarray,
    feature_mask_row: np.ndarray,
    index: InstrumentSpectrumIndex,
) -> np.ndarray:
    """Average one instrument across its time bins in log1p(dN/dlogDp) space."""

    values = np.full(index.diameter_nm.shape, np.nan, dtype=np.float64)
    for diameter_index, feature_indices in enumerate(index.feature_indices_by_diameter):
        mask = feature_mask_row[feature_indices].astype(bool)
        if not np.any(mask):
            continue
        candidates = transformed_row[feature_indices][mask]
        candidates = candidates[np.isfinite(candidates)]
        if candidates.size == 0:
            continue
        values[diameter_index] = float(np.nanmean(candidates))
    return values


def _erf_window(
    log_grid: np.ndarray,
    log_min: float,
    log_max: float,
    transition_decades: float,
) -> np.ndarray:
    scale = max(float(transition_decades), 1.0e-6)
    lower = 0.5 * (1.0 + np.asarray([erf((x - log_min) / scale) for x in log_grid]))
    upper = 0.5 * (1.0 + np.asarray([erf((log_max - x) / scale) for x in log_grid]))
    return lower * upper


def merge_log1p_spectra(
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    transition_decades: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge instruments on a common diameter grid.

    Inputs are per-instrument pairs of diameter_nm and log1p(dN/dlogDp).
    Overlapping regions are averaged in log space with an error-function
    taper so instrument edges do not create sharp steps.
    """

    available_diameters = [
        diameters[np.isfinite(log_values)]
        for diameters, log_values in spectra.values()
        if np.any(np.isfinite(log_values))
    ]
    if not available_diameters:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    diameter_grid = np.unique(np.concatenate(available_diameters))
    diameter_grid.sort()
    log_grid = np.log10(diameter_grid)
    weighted_sum = np.zeros_like(diameter_grid, dtype=np.float64)
    weight_sum = np.zeros_like(diameter_grid, dtype=np.float64)

    for diameters, log_values in spectra.values():
        finite = np.isfinite(diameters) & np.isfinite(log_values)
        if finite.sum() == 0:
            continue
        valid_diameters = diameters[finite]
        valid_logs = log_values[finite]
        mapped = np.full_like(diameter_grid, np.nan, dtype=np.float64)
        positions = np.searchsorted(diameter_grid, valid_diameters)
        mapped[positions] = valid_logs
        log_min = float(np.log10(valid_diameters.min()))
        log_max = float(np.log10(valid_diameters.max()))
        weights = _erf_window(log_grid, log_min, log_max, transition_decades)
        weights *= np.isfinite(mapped)
        weighted_sum += np.nan_to_num(mapped, nan=0.0) * weights
        weight_sum += weights

    merged_log = np.full_like(diameter_grid, np.nan, dtype=np.float64)
    valid = weight_sum > 0.0
    merged_log[valid] = weighted_sum[valid] / weight_sum[valid]
    merged_dndlogdp = np.expm1(merged_log)
    merged_dndlogdp[~np.isfinite(merged_dndlogdp)] = np.nan
    merged_dndlogdp = np.where(merged_dndlogdp >= 0.0, merged_dndlogdp, np.nan)
    return diameter_grid, merged_dndlogdp, weight_sum


def merged_row_spectrum(
    transformed_row: np.ndarray,
    feature_mask_row: np.ndarray,
    indices: dict[str, InstrumentSpectrumIndex],
    transition_decades: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra = {
        modality: (
            index.diameter_nm,
            instrument_log1p_spectrum(transformed_row, feature_mask_row, index),
        )
        for modality, index in indices.items()
    }
    return merge_log1p_spectra(spectra, transition_decades=transition_decades)
