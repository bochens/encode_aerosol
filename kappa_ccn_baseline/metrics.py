from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def ccn_prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    observed = np.asarray([row["observed_ccn_cm3"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["predicted_ccn_cm3"] for row in rows], dtype=np.float64)
    standardized_error = np.asarray(
        [row["standardized_log1p_error"] for row in rows],
        dtype=np.float64,
    )
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if finite.sum() == 0:
        return {
            "n": 0.0,
            "mae_cm3": float("nan"),
            "bias_cm3": float("nan"),
            "rmse_cm3": float("nan"),
            "log1p_rmse": float("nan"),
            "standardized_log1p_mse": float("nan"),
            "standardized_log1p_rmse": float("nan"),
        }

    error = predicted[finite] - observed[finite]
    log_error = np.log1p(predicted[finite]) - np.log1p(observed[finite])
    standardized_finite = standardized_error[np.isfinite(standardized_error)]
    standardized_mse = (
        float(np.mean(standardized_finite**2))
        if standardized_finite.size
        else float("nan")
    )
    return {
        "n": float(finite.sum()),
        "mae_cm3": float(np.mean(np.abs(error))),
        "bias_cm3": float(np.mean(error)),
        "rmse_cm3": float(np.sqrt(np.mean(error**2))),
        "log1p_rmse": float(np.sqrt(np.mean(log_error**2))),
        "standardized_log1p_mse": standardized_mse,
        "standardized_log1p_rmse": float(np.sqrt(standardized_mse)),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
