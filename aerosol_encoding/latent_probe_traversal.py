from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .plot_latent_aerosol_types import aerosol_type_labels, ordered_types


DEFAULT_TARGETS = [
    "cpc_number_proxy",
    "smps_number_proxy",
    "uhsas_accum_proxy",
    "aps_coarse_proxy",
    "organic_proxy",
    "sulfate_proxy",
    "ammonium_proxy",
    "nitrate_proxy",
    "chloride_proxy",
    "acsm_volume_proxy",
    "dry_scattering_proxy",
]

DIAGNOSTIC_TARGETS = ["cdce_proxy", "observed_modality_count"]

PLOT_FAMILIES = {
    "size_optical": [
        "cpc_number_proxy",
        "smps_number_proxy",
        "uhsas_accum_proxy",
        "aps_coarse_proxy",
        "dry_scattering_proxy",
    ],
    "chemistry": [
        "organic_proxy",
        "sulfate_proxy",
        "ammonium_proxy",
        "nitrate_proxy",
        "chloride_proxy",
        "acsm_volume_proxy",
    ],
    "diagnostics": [
        "cdce_proxy",
        "observed_modality_count",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear probes and probe-based latent traversals for 64-D aerosol z."
    )
    parser.add_argument("--latent-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="test_no_ccn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-test-size", type=float, default=0.3)
    parser.add_argument("--max-scatter-points", type=int, default=5000)
    parser.add_argument("--traversal-sigma", type=float, default=3.0)
    parser.add_argument("--traversal-steps", type=int, default=101)
    return parser.parse_args()


def z_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("z_")]
    if len(columns) != 64:
        raise ValueError(f"Expected 64 z_* bottleneck columns, found {len(columns)}")
    return columns


def target_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in DEFAULT_TARGETS if column in frame.columns]
    columns.extend(column for column in DIAGNOSTIC_TARGETS if column in frame.columns)
    if not columns:
        raise ValueError("No proxy target columns found for linear probing.")
    return columns


def proxy_label(column: str) -> str:
    return column.removesuffix("_proxy").replace("_", " ")


def robust_zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    median = float(np.nanmedian(values))
    q25, q75 = np.nanpercentile(values, [25, 75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return (values - median) / scale, median, scale


def chronological_train_test_indices(n_rows: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 10:
        raise ValueError("Need at least 10 finite rows for a chronological probe split.")
    split = int(np.floor(n_rows * (1.0 - test_size)))
    split = min(max(split, 5), n_rows - 5)
    return np.arange(split), np.arange(split, n_rows)


def fit_single_probe(
    x_scaled: np.ndarray,
    y_raw: np.ndarray,
    split: str,
    seed: int,
    random_test_size: float,
) -> tuple[dict[str, float], RidgeCV, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(y_raw).astype(bool)
    x = x_scaled[finite]
    y, y_center, y_scale = robust_zscore(y_raw[finite].astype(np.float64))
    if len(y) < 50:
        raise ValueError("Need at least 50 finite target rows for linear probe.")
    if split == "chronological":
        train_idx, test_idx = chronological_train_test_indices(len(y), random_test_size)
    elif split == "random":
        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=random_test_size,
            random_state=seed,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    alphas = np.logspace(-4, 4, 25)
    model = RidgeCV(alphas=alphas)
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    spearman = spearmanr(y[test_idx], pred, nan_policy="omit").statistic
    metrics = {
        "split": split,
        "n_finite": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "target_center": y_center,
        "target_scale": y_scale,
        "alpha": float(model.alpha_),
        "r2": float(r2_score(y[test_idx], pred)),
        "rmse_robust_z": float(np.sqrt(mean_squared_error(y[test_idx], pred))),
        "spearman": float(spearman) if np.isfinite(spearman) else float("nan"),
    }
    return metrics, model, finite, test_idx, y


def fit_probe_table(
    frame: pd.DataFrame,
    x_scaled: np.ndarray,
    targets: list[str],
    seed: int,
    random_test_size: float,
) -> tuple[pd.DataFrame, dict[str, RidgeCV], pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    full_models: dict[str, RidgeCV] = {}
    coefficient_rows: dict[str, np.ndarray] = {}

    for target in targets:
        y_raw = frame[target].to_numpy(dtype=np.float64)
        for split in ("chronological", "random"):
            metrics, _, _, _, _ = fit_single_probe(
                x_scaled,
                y_raw,
                split=split,
                seed=seed,
                random_test_size=random_test_size,
            )
            rows.append({"target": target, **metrics})

        finite = np.isfinite(y_raw).astype(bool)
        y, _, _ = robust_zscore(y_raw[finite].astype(np.float64))
        model = RidgeCV(alphas=np.logspace(-4, 4, 25))
        model.fit(x_scaled[finite], y)
        full_models[target] = model
        coefficient_rows[target] = np.asarray(model.coef_, dtype=np.float64)

    coefficient_frame = pd.DataFrame(
        coefficient_rows,
        index=[f"zstd_{index:02d}" for index in range(x_scaled.shape[1])],
    ).T
    return pd.DataFrame(rows), full_models, coefficient_frame


def plot_probe_scores(metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    chronological = metrics[metrics["split"] == "chronological"].copy()
    chronological = chronological.sort_values("r2", ascending=True)
    random = metrics[metrics["split"] == "random"].set_index("target")
    y_positions = np.arange(len(chronological))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True, constrained_layout=True)
    axes[0].barh(y_positions - 0.16, chronological["r2"], height=0.32, label="chronological 70/30", color="#3182bd")
    axes[0].barh(
        y_positions + 0.16,
        random.loc[chronological["target"], "r2"],
        height=0.32,
        label="random 70/30",
        color="#9ecae1",
    )
    axes[0].axvline(0.0, color="#525252", linewidth=0.8)
    axes[0].set_xlabel("R2 on held-out probe split")
    axes[0].set_yticks(y_positions, [proxy_label(target) for target in chronological["target"]])
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="x", alpha=0.2)

    axes[1].barh(y_positions - 0.16, chronological["spearman"], height=0.32, color="#31a354")
    axes[1].barh(
        y_positions + 0.16,
        random.loc[chronological["target"], "spearman"],
        height=0.32,
        color="#a1d99b",
    )
    axes[1].axvline(0.0, color="#525252", linewidth=0.8)
    axes[1].set_xlabel("Spearman r on held-out probe split")
    axes[1].grid(True, axis="x", alpha=0.2)
    fig.suptitle("Linear probe readability from frozen 64-D aerosol bottleneck")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_coefficient_heatmap(coefficients: pd.DataFrame, output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    ordered_columns = (
        coefficients.abs()
        .max(axis=0)
        .sort_values(ascending=False)
        .index.tolist()
    )
    ordered = coefficients.loc[:, ordered_columns]

    fig, ax = plt.subplots(figsize=(15.5, 6.8), constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(ordered.to_numpy()), 98))
    vmax = max(vmax, 1e-6)
    image = ax.imshow(ordered.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(ordered_columns)), [name.replace("zstd_", "z") for name in ordered_columns], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(ordered.index)), [proxy_label(target) for target in ordered.index])
    ax.set_title("Linear-probe coefficients: which bottleneck dimensions carry each proxy")
    fig.colorbar(image, ax=ax, label="ridge coefficient on standardized z")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return ordered_columns


def pca_traversal_frame(
    x_scaled: np.ndarray,
    models: dict[str, RidgeCV],
    targets: list[str],
    sigma: float,
    steps: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    pca = PCA(n_components=8, random_state=0)
    pca.fit(x_scaled)
    center = np.nanmedian(x_scaled, axis=0)
    grid = np.linspace(-sigma, sigma, steps)
    rows: list[dict[str, float | str | int]] = []
    for pc_index, component in enumerate(pca.components_, start=1):
        for value in grid:
            point = center + value * component
            row: dict[str, float | str | int] = {"direction": f"PC{pc_index}", "coordinate": float(value)}
            for target in targets:
                row[target] = float(models[target].predict(point.reshape(1, -1))[0])
            rows.append(row)
    return pd.DataFrame(rows), pca.explained_variance_ratio_


def zdim_traversal_frame(
    x_scaled: np.ndarray,
    models: dict[str, RidgeCV],
    targets: list[str],
    ordered_z_columns: list[str],
    sigma: float,
    steps: int,
    n_dims: int = 12,
) -> pd.DataFrame:
    center = np.nanmedian(x_scaled, axis=0)
    grid = np.linspace(-sigma, sigma, steps)
    rows: list[dict[str, float | str | int]] = []
    for z_name in ordered_z_columns[:n_dims]:
        z_index = int(z_name.removeprefix("zstd_"))
        for value in grid:
            point = center.copy()
            point[z_index] = value
            row: dict[str, float | str | int] = {"direction": z_name.replace("zstd_", "z"), "coordinate": float(value)}
            for target in targets:
                row[target] = float(models[target].predict(point.reshape(1, -1))[0])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_traversal_grid(traversal: pd.DataFrame, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    targets = [column for column in traversal.columns if column not in {"direction", "coordinate"}]
    directions = list(dict.fromkeys(traversal["direction"].tolist()))
    ncols = 4
    nrows = int(np.ceil(len(directions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.0, 3.1 * nrows), sharex=True, sharey=True, constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)
    cmap = plt.get_cmap("tab20")
    colors = {target: cmap(index % 20) for index, target in enumerate(targets)}
    for ax, direction in zip(axes_array, directions, strict=False):
        subset = traversal[traversal["direction"] == direction]
        for target in targets:
            ax.plot(
                subset["coordinate"],
                subset[target],
                linewidth=1.4,
                color=colors[target],
                label=proxy_label(target),
            )
        ax.axhline(0.0, color="#525252", linewidth=0.7)
        ax.axvline(0.0, color="#525252", linewidth=0.7, linestyle=":")
        ax.set_title(direction)
        ax.grid(True, alpha=0.18)
    for ax in axes_array[len(directions):]:
        ax.axis("off")
    for ax in axes_array[-ncols:]:
        ax.set_xlabel("latent coordinate displacement, standardized units")
    for ax in axes_array[::ncols]:
        ax.set_ylabel("probe-predicted proxy z-score")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_traversal_families(
    traversal: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    stem: str,
    title_prefix: str,
) -> None:
    for family, wanted_targets in PLOT_FAMILIES.items():
        present_targets = [target for target in wanted_targets if target in traversal.columns]
        if not present_targets:
            continue
        columns = ["direction", "coordinate", *present_targets]
        plot_traversal_grid(
            traversal[columns],
            output_dir / f"{prefix}_{stem}_{family}.png",
            f"{title_prefix}: {family.replace('_', ' ')}",
        )


def plot_probe_scatter(
    frame: pd.DataFrame,
    x_scaled: np.ndarray,
    targets: list[str],
    models: dict[str, RidgeCV],
    output: Path,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    targets = targets[: min(12, len(targets))]
    ncols = 4
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.8, 3.15 * nrows), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, target in zip(axes_array, targets, strict=False):
        y_raw = frame[target].to_numpy(dtype=np.float64)
        finite = np.isfinite(y_raw)
        y, _, _ = robust_zscore(y_raw[finite])
        x = x_scaled[finite]
        pred = models[target].predict(x)
        if len(y) > max_points:
            idx = np.sort(rng.choice(len(y), size=max_points, replace=False))
            y = y[idx]
            pred = pred[idx]
        ax.scatter(y, pred, s=5, alpha=0.28, linewidths=0, color="#3182bd")
        low = float(np.nanpercentile(np.concatenate([y, pred]), 1))
        high = float(np.nanpercentile(np.concatenate([y, pred]), 99))
        ax.plot([low, high], [low, high], color="#de2d26", linewidth=1.0)
        ax.set_title(proxy_label(target))
        ax.set_xlabel("observed proxy z-score")
        ax.set_ylabel("linear probe prediction")
        ax.grid(True, alpha=0.18)
    for ax in axes_array[len(targets):]:
        ax.axis("off")
    fig.suptitle("Observed vs linear-probe predicted aerosol proxies")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fit_type_probe(
    frame: pd.DataFrame,
    x_scaled: np.ndarray,
    seed: int,
    random_test_size: float,
    output_png: Path,
    output_csv: Path,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    labels = aerosol_type_labels(frame).to_numpy()
    counts = pd.Series(labels).value_counts()
    keep = np.array([counts[label] >= 50 for label in labels])
    x = x_scaled[keep]
    y = labels[keep]
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=random_test_size,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    model = LogisticRegression(
        penalty="l2",
        C=0.25,
        class_weight="balanced",
        max_iter=2000,
        multi_class="auto",
        solver="lbfgs",
    )
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    labels_order = ordered_types(pd.Series(y))
    matrix = confusion_matrix(y[test_idx], pred, labels=labels_order, normalize="true")
    matrix_frame = pd.DataFrame(matrix, index=labels_order, columns=labels_order)
    matrix_frame.to_csv(output_csv)

    fig, ax = plt.subplots(figsize=(9.5, 7.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels_order)), labels_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels_order)), labels_order)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if value >= 0.05:
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted proxy type")
    ax.set_ylabel("proxy-defined type")
    ax.set_title("Linear classifier probe from 64-D z to proxy-defined aerosol type")
    fig.colorbar(image, ax=ax, label="row-normalized fraction")
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    return {
        "n_rows": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.latent_csv)
    columns = z_columns(frame)
    targets = target_columns(frame)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(frame[columns].to_numpy(dtype=np.float32))

    metrics, models, coefficients = fit_probe_table(
        frame,
        x_scaled,
        targets,
        args.seed,
        args.random_test_size,
    )
    metrics_path = output_dir / f"{args.prefix}_linear_probe_metrics.csv"
    coefficient_path = output_dir / f"{args.prefix}_linear_probe_coefficients.csv"
    metrics.to_csv(metrics_path, index=False)
    coefficients.to_csv(coefficient_path)

    plot_probe_scores(metrics, output_dir / f"{args.prefix}_linear_probe_scores.png")
    ordered_z_columns = plot_coefficient_heatmap(
        coefficients,
        output_dir / f"{args.prefix}_linear_probe_coefficient_heatmap.png",
    )
    plot_probe_scatter(
        frame,
        x_scaled,
        targets,
        models,
        output_dir / f"{args.prefix}_linear_probe_predicted_vs_observed.png",
        args.max_scatter_points,
        args.seed,
    )

    pc_traversal, explained = pca_traversal_frame(
        x_scaled,
        models,
        targets,
        args.traversal_sigma,
        args.traversal_steps,
    )
    pc_traversal_path = output_dir / f"{args.prefix}_pc_latent_traversal.csv"
    pc_traversal.to_csv(pc_traversal_path, index=False)
    plot_traversal_grid(
        pc_traversal[[column for column in pc_traversal.columns if column not in DIAGNOSTIC_TARGETS]],
        output_dir / f"{args.prefix}_pc_latent_traversal.png",
        "Probe-based traversal along major PCA directions of 64-D z",
    )
    plot_traversal_families(
        pc_traversal,
        output_dir,
        args.prefix,
        "pc_latent_traversal",
        "Probe-based traversal along major PCA directions",
    )

    zdim_traversal = zdim_traversal_frame(
        x_scaled,
        models,
        targets,
        ordered_z_columns,
        args.traversal_sigma,
        args.traversal_steps,
    )
    zdim_traversal_path = output_dir / f"{args.prefix}_zdim_latent_traversal.csv"
    zdim_traversal.to_csv(zdim_traversal_path, index=False)
    plot_traversal_grid(
        zdim_traversal[[column for column in zdim_traversal.columns if column not in DIAGNOSTIC_TARGETS]],
        output_dir / f"{args.prefix}_zdim_latent_traversal.png",
        "Probe-based traversal of individual high-information bottleneck coordinates",
    )
    plot_traversal_families(
        zdim_traversal,
        output_dir,
        args.prefix,
        "zdim_latent_traversal",
        "Probe-based traversal of individual bottleneck coordinates",
    )

    type_probe = fit_type_probe(
        frame,
        x_scaled,
        args.seed,
        args.random_test_size,
        output_dir / f"{args.prefix}_aerosol_type_linear_probe_confusion.png",
        output_dir / f"{args.prefix}_aerosol_type_linear_probe_confusion.csv",
    )

    summary = {
        "rows": int(len(frame)),
        "latent_dim": int(len(columns)),
        "targets": targets,
        "linear_probe": {
            "best_chronological_r2": metrics[metrics["split"] == "chronological"]
            .sort_values("r2", ascending=False)
            .head(6)[["target", "r2", "spearman"]]
            .to_dict(orient="records"),
            "worst_chronological_r2": metrics[metrics["split"] == "chronological"]
            .sort_values("r2", ascending=True)
            .head(6)[["target", "r2", "spearman"]]
            .to_dict(orient="records"),
        },
        "pca_traversal_explained_variance_first_8": [float(value) for value in explained],
        "aerosol_type_linear_probe": type_probe,
    }
    summary_path = output_dir / f"{args.prefix}_probe_traversal_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"wrote {metrics_path}")
    print(f"wrote {coefficient_path}")
    print(f"wrote {output_dir / f'{args.prefix}_linear_probe_scores.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_linear_probe_coefficient_heatmap.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_linear_probe_predicted_vs_observed.png'}")
    print(f"wrote {pc_traversal_path}")
    print(f"wrote {output_dir / f'{args.prefix}_pc_latent_traversal.png'}")
    for family in PLOT_FAMILIES:
        print(f"wrote {output_dir / f'{args.prefix}_pc_latent_traversal_{family}.png'}")
    print(f"wrote {zdim_traversal_path}")
    print(f"wrote {output_dir / f'{args.prefix}_zdim_latent_traversal.png'}")
    for family in PLOT_FAMILIES:
        print(f"wrote {output_dir / f'{args.prefix}_zdim_latent_traversal_{family}.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_linear_probe_confusion.png'}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
