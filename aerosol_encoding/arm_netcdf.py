from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import StreamSpec, VariableSpec

DATE_RE = re.compile(r"\.(\d{8})\.\d{6}")


def file_date(path: Path) -> pd.Timestamp:
    match = DATE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse ARM date from {path}")
    return pd.Timestamp(match.group(1))


def stream_files(
    data_root: Path,
    stream: StreamSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    stream_dir = data_root / stream.path
    if not stream_dir.is_dir():
        raise FileNotFoundError(f"Missing stream directory: {stream_dir}")

    selected: list[Path] = []
    for suffix in ("*.nc", "*.cdf"):
        for path in stream_dir.glob(suffix):
            date = file_date(path)
            if start.normalize() <= date <= end.normalize():
                selected.append(path)
    selected.sort(key=file_date)
    return selected


def apply_transform(values: xr.DataArray, transform: str) -> xr.DataArray:
    if transform == "identity":
        return values
    if transform == "log1p_nonnegative":
        return np.log1p(values.where(values >= 0.0))
    if transform == "angle_degrees":
        radians = np.deg2rad(values)
        return xr.concat(
            [np.sin(radians), np.cos(radians)],
            dim=pd.Index(["sin", "cos"], name="angle_component"),
        ).transpose("time", "angle_component")
    raise ValueError(f"Unknown transform: {transform}")


def apply_quality_masks(dataset: xr.Dataset, variable: str, values: xr.DataArray) -> xr.DataArray:
    qc_name = f"qc_{variable}"
    if qc_name in dataset and dataset[qc_name].dims == values.dims:
        values = values.where(dataset[qc_name] == 0)

    attrs = dataset[variable].attrs
    if "valid_min" in attrs:
        values = values.where(values >= float(np.asarray(attrs["valid_min"]).min()))
    if "valid_max" in attrs:
        values = values.where(values <= float(np.asarray(attrs["valid_max"]).max()))
    return values


def _coordinate_label(values: xr.DataArray, dim: str, index: int) -> str:
    if dim in values.coords:
        raw = values.coords[dim].values[index]
        if np.issubdtype(np.asarray(raw).dtype, np.number):
            return f"{float(raw):.6g}"
        return str(raw)
    return str(index)


def _dataarray_to_frame(
    values: xr.DataArray,
    modality_name: str,
    stream_name: str,
    variable: VariableSpec,
    stat: str = "mean",
) -> pd.DataFrame:
    values = values.load()
    if "time" not in values.dims:
        raise ValueError(f"{stream_name}/{variable.name} has no time dimension")

    stat_label = "" if stat == "mean" else f"__stat_{stat}"
    if values.dims == ("time",):
        column = f"{modality_name}__{stream_name}__{variable.name}{stat_label}"
        return pd.DataFrame({column: values.values}, index=pd.DatetimeIndex(values["time"].values))

    non_time_dims = tuple(dim for dim in values.dims if dim != "time")
    if len(non_time_dims) != 1:
        raise ValueError(
            f"{stream_name}/{variable.name} has unsupported dimensions {values.dims}"
        )

    dim = non_time_dims[0]
    matrix = values.transpose("time", dim).values
    columns = [
        f"{modality_name}__{stream_name}__{variable.name}{stat_label}__{dim}_{_coordinate_label(values, dim, idx)}"
        for idx in range(matrix.shape[1])
    ]
    return pd.DataFrame(matrix, index=pd.DatetimeIndex(values["time"].values), columns=columns)


def _resample_stat(values: xr.DataArray, freq: str, stat: str) -> xr.DataArray:
    resampler = values.resample(time=freq)
    if stat == "mean":
        return resampler.mean(skipna=True)
    if stat == "std":
        mean = resampler.mean(skipna=True)
        square_mean = (values * values).resample(time=freq).mean(skipna=True)
        variance = (square_mean - mean * mean).clip(min=0.0)
        return np.sqrt(variance)
    if stat == "min":
        return resampler.min(skipna=True)
    if stat == "max":
        return resampler.max(skipna=True)
    raise ValueError(f"Unsupported temporal statistic: {stat}")


def aggregate_file(
    path: Path,
    modality_name: str,
    stream: StreamSpec,
    freq: str,
    stats: tuple[str, ...] = ("mean",),
) -> pd.DataFrame:
    with xr.open_dataset(path, engine="netcdf4") as dataset:
        frames: list[pd.DataFrame] = []
        for variable in stream.variables:
            if variable.name not in dataset:
                raise KeyError(f"{path} does not contain required variable {variable.name}")
            values = dataset[variable.name].astype("float64")
            values = apply_quality_masks(dataset, variable.name, values)
            values = apply_transform(values, variable.transform)
            for stat in stats:
                resampled = _resample_stat(values, freq, stat)
                frames.append(
                    _dataarray_to_frame(
                        resampled,
                        modality_name,
                        stream.name,
                        variable,
                        stat=stat,
                    )
                )

    if not frames:
        raise ValueError(f"No variables configured for {stream.name}")
    return pd.concat(frames, axis=1)


def aggregate_stream(
    data_root: Path,
    modality_name: str,
    stream: StreamSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
    stats: tuple[str, ...] = ("mean",),
) -> pd.DataFrame:
    files = stream_files(data_root, stream, start, end)
    if not files:
        raise FileNotFoundError(
            f"No files found for {stream.name} from {start.date()} to {end.date()}"
        )

    frames = [aggregate_file(path, modality_name, stream, freq, stats=stats) for path in files]
    frame = pd.concat(frames, axis=0)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.loc[(frame.index >= start) & (frame.index < end + pd.Timedelta(days=1))]
    return frame
