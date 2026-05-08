from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .arm_netcdf import aggregate_stream
from .config import ExperimentConfig, SizeGridSpec, config_to_metadata


DIAMETER_LABEL_RE = re.compile(
    r"^(?P<coordinate>.+)_(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)

DEFAULT_DIAMETER_UNITS = {
    "diameter_mobility": "nm",
    "diameter_optical": "nm",
    "diameter_aerodynamic": "um",
    "diameter_midpoint": "um",
    "diameter_common_nm": "nm",
}


def build_feature_frame(
    config: ExperimentConfig,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"End date {end} is before start date {start}")
    if config.temporal_windows.enabled:
        return build_temporal_feature_frame(config, start, end)

    modality_columns: dict[str, list[str]] = {}
    modality_frames: list[pd.DataFrame] = []

    for modality in config.modalities:
        stream_frames = [
            frame
            for stream in modality.streams
            for frame in [
                _aggregate_optional_stream(config, modality.name, stream, start_ts, end_ts)
            ]
            if not frame.empty
        ]
        if not stream_frames:
            raise FileNotFoundError(
                f"No required or optional streams with files for modality {modality.name} "
                f"from {start} to {end}"
            )
        modality_frame = pd.concat(stream_frames, axis=1).sort_index()
        modality_frame = _apply_size_grid_to_modality(
            modality_frame,
            modality_name=modality.name,
            size_grid=config.size_grid,
        )
        modality_columns[modality.name] = list(modality_frame.columns)
        modality_frames.append(modality_frame)

    frame = pd.concat(modality_frames, axis=1).sort_index()
    frame = frame.loc[(frame.index >= start_ts) & (frame.index < end_ts + pd.Timedelta(days=1))]
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame, modality_columns


def _diameter_nm_from_column(column: str, size_grid: SizeGridSpec) -> tuple[float, str] | None:
    if "__dN_dlogDp__" not in column:
        return None
    coordinate_label = column.rsplit("__", maxsplit=1)[-1]
    match = DIAMETER_LABEL_RE.match(coordinate_label)
    if match is None:
        raise ValueError(f"Could not parse diameter coordinate from feature column {column!r}")

    coordinate = match.group("coordinate")
    value = float(match.group("value"))
    units = dict(DEFAULT_DIAMETER_UNITS)
    units.update(size_grid.diameter_units or {})
    unit = units.get(coordinate)
    if unit is None:
        raise ValueError(
            f"No diameter unit configured for coordinate {coordinate!r} in column {column!r}"
        )
    if unit == "nm":
        return value, coordinate
    if unit == "um":
        return value * 1000.0, coordinate
    raise ValueError(
        f"Unsupported diameter unit {unit!r} for coordinate {coordinate!r}; use 'nm' or 'um'"
    )


def _stat_from_column(column: str) -> str:
    match = re.search(r"__stat_([^_]+)", column)
    if match is None:
        return "mean"
    return match.group(1)


def _apply_size_grid_to_modality(
    frame: pd.DataFrame,
    modality_name: str,
    size_grid: SizeGridSpec,
) -> pd.DataFrame:
    if not size_grid.enabled:
        return frame

    parsed_columns: list[tuple[str, float, str]] = []
    native_coordinates: set[str] = set()
    for column in frame.columns:
        parsed = _diameter_nm_from_column(column, size_grid)
        if parsed is None:
            continue
        diameter_nm, coordinate = parsed
        parsed_columns.append((column, diameter_nm, _stat_from_column(column)))
        native_coordinates.add(coordinate)

    if not parsed_columns:
        return frame

    grid = np.geomspace(
        size_grid.min_diameter_nm,
        size_grid.max_diameter_nm,
        size_grid.bins,
    )
    log_grid = np.log10(grid)
    grid_frames: list[pd.DataFrame] = []
    source_columns: list[str] = []
    for stat in sorted({stat for _, _, stat in parsed_columns}):
        diameter_to_columns: dict[float, list[str]] = {}
        for column, diameter, column_stat in parsed_columns:
            if column_stat != stat:
                continue
            diameter_to_columns.setdefault(diameter, []).append(column)
        source_diameters = np.asarray(sorted(diameter_to_columns), dtype=np.float64)
        if np.any(source_diameters <= 0):
            raise ValueError(f"{modality_name} has nonpositive diameter coordinates")
        source_columns.extend(
            column
            for diameter in source_diameters
            for column in diameter_to_columns[float(diameter)]
        )
        source_values = np.full((frame.shape[0], source_diameters.shape[0]), np.nan, dtype=np.float64)
        for diameter_index, diameter in enumerate(source_diameters):
            columns = diameter_to_columns[float(diameter)]
            values = frame[columns].to_numpy(dtype=np.float64, copy=True)
            finite = np.isfinite(values)
            counts = finite.sum(axis=1)
            sums = np.where(finite, values, 0.0).sum(axis=1)
            source_values[counts > 0, diameter_index] = sums[counts > 0] / counts[counts > 0]
        gridded = np.full((source_values.shape[0], grid.shape[0]), np.nan, dtype=np.float64)
        log_source = np.log10(source_diameters)
        for row_index in range(source_values.shape[0]):
            finite = np.isfinite(source_values[row_index])
            if finite.sum() < 2:
                continue
            gridded[row_index] = np.interp(
                log_grid,
                log_source[finite],
                source_values[row_index, finite],
                left=np.nan,
                right=np.nan,
            )
        stat_label = "" if stat == "mean" else f"__stat_{stat}"
        grid_columns = [
            f"{modality_name}__diameter_grid__dN_dlogDp{stat_label}__diameter_common_nm_{diameter:.6g}"
            for diameter in grid
        ]
        grid_frames.append(pd.DataFrame(gridded, index=frame.index, columns=grid_columns))

    scalar_frame = frame.drop(columns=source_columns)
    output = pd.concat([*grid_frames, scalar_frame], axis=1)
    output.attrs["native_diameter_coordinates"] = sorted(native_coordinates)
    return output


def _temporal_step_for_modality(config: ExperimentConfig, modality_name: str) -> str:
    steps = config.temporal_windows.modality_steps or {}
    return steps.get(modality_name, config.temporal_windows.default_step)


def _temporal_stats_for_modality(config: ExperimentConfig, modality_name: str) -> tuple[str, ...]:
    stats = config.temporal_windows.modality_stats or {}
    return stats.get(modality_name, config.temporal_windows.default_stats)


def _windowed_modality_frame(
    modality_frame: pd.DataFrame,
    anchors: pd.DatetimeIndex,
    anchor_freq: str,
    step_freq: str,
) -> pd.DataFrame:
    anchor_delta = pd.Timedelta(anchor_freq)
    step_delta = pd.Timedelta(step_freq)
    if step_delta <= pd.Timedelta(0):
        raise ValueError(f"Temporal step must be positive, got {step_freq!r}")
    if anchor_delta < step_delta:
        raise ValueError(
            f"Temporal step {step_freq!r} cannot be longer than anchor frequency {anchor_freq!r}"
        )
    ratio = anchor_delta / step_delta
    n_steps = int(round(float(ratio)))
    if abs(float(ratio) - n_steps) > 1e-9:
        raise ValueError(
            f"Temporal step {step_freq!r} must evenly divide anchor frequency {anchor_freq!r}"
        )

    step_frames: list[pd.DataFrame] = []
    for step_index in range(n_steps):
        lookup_times = anchors + step_index * step_delta
        step_frame = modality_frame.reindex(lookup_times)
        step_frame.index = anchors
        step_frame = step_frame.rename(
            columns={
                column: f"{column}__time_bin_{step_index:03d}"
                for column in modality_frame.columns
            }
        )
        step_frames.append(step_frame)
    return pd.concat(step_frames, axis=1)


def build_temporal_feature_frame(
    config: ExperimentConfig,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    end_exclusive = end_ts + pd.Timedelta(days=1)
    anchors = pd.date_range(start_ts, end_exclusive, freq=config.freq, inclusive="left")
    if anchors.empty:
        raise ValueError(f"No temporal anchors from {start} to {end} at {config.freq}")

    modality_columns: dict[str, list[str]] = {}
    modality_frames: list[pd.DataFrame] = []

    for modality in config.modalities:
        step_freq = _temporal_step_for_modality(config, modality.name)
        stats = _temporal_stats_for_modality(config, modality.name)
        stream_frames = [
            frame
            for stream in modality.streams
            for frame in [
                _aggregate_optional_stream(
                    config,
                    modality.name,
                    stream,
                    start_ts,
                    end_ts,
                    freq=step_freq,
                    stats=stats,
                )
            ]
            if not frame.empty
        ]
        if not stream_frames:
            raise FileNotFoundError(
                f"No required or optional streams with files for modality {modality.name} "
                f"from {start} to {end}"
            )
        modality_frame = pd.concat(stream_frames, axis=1).sort_index()
        modality_frame = _apply_size_grid_to_modality(
            modality_frame,
            modality_name=modality.name,
            size_grid=config.size_grid,
        )
        modality_frame = _windowed_modality_frame(
            modality_frame,
            anchors=anchors,
            anchor_freq=config.freq,
            step_freq=step_freq,
        )
        modality_columns[modality.name] = list(modality_frame.columns)
        modality_frames.append(modality_frame)

    frame = pd.concat(modality_frames, axis=1).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame, modality_columns


def _aggregate_optional_stream(
    config: ExperimentConfig,
    modality_name: str,
    stream,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    freq: str | None = None,
    stats: tuple[str, ...] = ("mean",),
) -> pd.DataFrame:
    try:
        return aggregate_stream(
            data_root=config.data_root,
            modality_name=modality_name,
            stream=stream,
            start=start_ts,
            end=end_ts,
            freq=freq or config.freq,
            stats=stats,
        )
    except FileNotFoundError:
        if stream.required:
            raise
        return pd.DataFrame()


def save_feature_store(
    frame: pd.DataFrame,
    modality_columns: dict[str, list[str]],
    config: ExperimentConfig,
    output_dir: str | Path,
    start: str,
    end: str,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_names = list(frame.columns)
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    modality_indices = {
        modality: [feature_to_index[column] for column in columns]
        for modality, columns in modality_columns.items()
    }
    matrix = frame.to_numpy(dtype=np.float32, copy=True)
    times = frame.index.values.astype("datetime64[ns]")

    npz_path = output / "features.npz"
    np.savez_compressed(npz_path, X=matrix, times=times)

    metadata: dict[str, Any] = config_to_metadata(config)
    metadata.update(
        {
            "start": start,
            "end": end,
            "n_rows": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "feature_names": feature_names,
            "modality_indices": modality_indices,
            "modality_columns": modality_columns,
        }
    )
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    coverage = frame.notna().mean().sort_values()
    coverage.to_csv(output / "feature_coverage.csv", header=["finite_fraction"])
    return npz_path


def load_feature_store(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    metadata_path = npz_path.with_name("metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with np.load(npz_path, allow_pickle=False) as payload:
        matrix = payload["X"].astype(np.float32)
        times = payload["times"]
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return matrix, times, metadata
