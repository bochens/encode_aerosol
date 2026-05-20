from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from aerosol_encoding.training_data import PreparedArrays


CCN_RE = re.compile(
    r"^ccn_activation__.+__(?P<variable>N_CCN|supersaturation_calculated|supersaturation_set_point)"
    r"__time_bin_(?P<time_bin>\d+)$"
)


ACSM_VARIABLES = {
    "organic_mass": "total_organics_CDCE",
    "sulfate_mass": "sulfate_CDCE",
    "ammonium_mass": "ammonium_CDCE",
    "nitrate_mass": "nitrate_CDCE",
    "chloride_mass": "chloride_CDCE",
}


@dataclass(frozen=True)
class CCNObservationIndex:
    time_bin: int
    n_ccn_index: int | None = None
    supersaturation_calculated_index: int | None = None
    supersaturation_set_point_index: int | None = None


def transformed_row(arrays: PreparedArrays, row_index: int) -> tuple[np.ndarray, np.ndarray]:
    values = arrays.x[row_index].astype(np.float64) * arrays.std + arrays.mean
    mask = arrays.feature_mask[row_index].astype(bool)
    values = values.astype(np.float64, copy=True)
    values[~mask] = np.nan
    return values, mask


def _first_feature_index(feature_names: list[str], variable_name: str) -> int:
    needle = f"__{variable_name}__time_bin_000"
    for index, feature_name in enumerate(feature_names):
        if needle in feature_name:
            return index
    raise ValueError(f"Could not find ACSM feature {variable_name!r}")


def build_acsm_indices(feature_names: list[str]) -> dict[str, int]:
    return {
        key: _first_feature_index(feature_names, variable)
        for key, variable in ACSM_VARIABLES.items()
    }


def acsm_masses_from_row(
    transformed_values: np.ndarray,
    feature_mask: np.ndarray,
    acsm_indices: dict[str, int],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, index in acsm_indices.items():
        if not feature_mask[index] or not np.isfinite(transformed_values[index]):
            output[key] = float("nan")
        else:
            output[key] = float(max(np.expm1(transformed_values[index]), 0.0))
    return output


def build_ccn_indices(feature_names: list[str]) -> tuple[CCNObservationIndex, ...]:
    by_time: dict[int, dict[str, int]] = {}
    for index, feature_name in enumerate(feature_names):
        match = CCN_RE.match(feature_name)
        if match is None:
            continue
        time_bin = int(match.group("time_bin"))
        variable = match.group("variable")
        by_time.setdefault(time_bin, {})[variable] = index

    output = []
    for time_bin in sorted(by_time):
        row = by_time[time_bin]
        if "N_CCN" not in row:
            continue
        output.append(
            CCNObservationIndex(
                time_bin=time_bin,
                n_ccn_index=row.get("N_CCN"),
                supersaturation_calculated_index=row.get("supersaturation_calculated"),
                supersaturation_set_point_index=row.get("supersaturation_set_point"),
            )
        )
    return tuple(output)


def ccn_observations_from_row(
    transformed_values: np.ndarray,
    feature_mask: np.ndarray,
    ccn_indices: tuple[CCNObservationIndex, ...],
) -> list[dict[str, float]]:
    observations: list[dict[str, float]] = []
    for index in ccn_indices:
        if index.n_ccn_index is None:
            continue
        if not feature_mask[index.n_ccn_index]:
            continue
        n_log = transformed_values[index.n_ccn_index]
        if not np.isfinite(n_log):
            continue

        ss_index = index.supersaturation_calculated_index
        if ss_index is None or not feature_mask[ss_index]:
            ss_index = index.supersaturation_set_point_index
        if ss_index is None or not feature_mask[ss_index]:
            continue
        ss_percent = transformed_values[ss_index]
        if not np.isfinite(ss_percent) or ss_percent <= 0.0:
            continue

        observations.append(
            {
                "n_ccn_index": float(index.n_ccn_index),
                "time_bin": float(index.time_bin),
                "observed_log1p_ccn": float(n_log),
                "observed_ccn_cm3": float(max(np.expm1(n_log), 0.0)),
                "supersaturation_percent": float(ss_percent),
            }
        )
    return observations
