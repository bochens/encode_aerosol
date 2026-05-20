from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROXY_COLUMNS = [
    "observed_modality_count",
    "cpc_number_proxy",
    "organic_percent_proxy",
    "sulfate_percent_proxy",
    "ammonium_percent_proxy",
    "nitrate_percent_proxy",
    "chloride_percent_proxy",
    "acsm_speciated_mass_conc_proxy",
    "acsm_volume_proxy",
    "cdce_proxy",
    "dry_scattering_proxy",
    "smps_number_proxy",
    "aps_coarse_proxy",
    "uhsas_accum_proxy",
]

ACSM_PERCENT_COLUMNS = [
    "organic_percent_proxy",
    "sulfate_percent_proxy",
    "ammonium_percent_proxy",
    "nitrate_percent_proxy",
    "chloride_percent_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot extra PCA diagnostics for saved latent z table.")
    parser.add_argument("--latent-pca-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="test_no_ccn")
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def proxy_label(name: str) -> str:
    if name.endswith("_percent_proxy"):
        return f"{name.removesuffix('_percent_proxy').replace('_', ' ')} (%)"
    return name.removesuffix("_proxy").replace("_", " ")


def output_stem(name: str) -> str:
    return (
        name.removesuffix("_percent_proxy")
        .removesuffix("_proxy")
        .replace("_", "-")
    )


def deterministic_sample(frame: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(frame), size=max_points, replace=False))
    return frame.iloc[indices].copy()


def compute_full_pca(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    z_columns = [column for column in frame.columns if column.startswith("z_")]
    if not z_columns:
        raise ValueError("No z_* latent columns found in latent PCA CSV.")
    z_scaled = StandardScaler().fit_transform(frame[z_columns].to_numpy(dtype=np.float32))
    pca = PCA(n_components=len(z_columns), random_state=0)
    pcs = pca.fit_transform(z_scaled)
    pc_frame = pd.DataFrame(
        pcs,
        columns=[f"PC{index + 1}" for index in range(pcs.shape[1])],
        index=frame.index,
    )
    return pc_frame, pca.explained_variance_ratio_


def plot_scree(explained: np.ndarray, output: Path) -> None:
    import matplotlib.pyplot as plt

    cumulative = np.cumsum(explained)
    x = np.arange(1, explained.size + 1)
    fig, ax1 = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    ax1.bar(x, explained * 100.0, color="#3182bd", alpha=0.78, label="individual PC")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance (%)")
    ax1.set_xlim(0.4, min(32, explained.size) + 0.6)
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative * 100.0, color="#de2d26", marker="o", markersize=3, linewidth=1.8)
    ax2.set_ylabel("cumulative explained variance (%)")
    ax2.set_ylim(0, 103)
    for threshold in [50, 70, 80, 90, 95]:
        component = int(np.searchsorted(cumulative, threshold / 100.0) + 1)
        ax2.axhline(threshold, color="#777777", linewidth=0.7, linestyle=":")
        ax2.text(
            min(32, explained.size) + 0.4,
            threshold,
            f"{threshold}%: PC{component}",
            va="center",
            ha="left",
            fontsize=8,
            color="#555555",
        )
    ax1.set_title("Bottleneck PCA scree and cumulative variance")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pc_proxy_heatmap(pc_frame: pd.DataFrame, frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    proxies = [column for column in PROXY_COLUMNS if column in frame.columns]
    corr_frame = pd.concat([pc_frame.iloc[:, :20], frame[proxies]], axis=1)
    corr = corr_frame.corr(method="spearman").loc[pc_frame.columns[:20], proxies]
    fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(proxies)), [proxy_label(column) for column in proxies], rotation=35, ha="right")
    ax.set_yticks(np.arange(20), pc_frame.columns[:20])
    ax.set_title("Spearman correlation: PCA directions vs aerosol proxies")
    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            value = corr.iat[row, col]
            if abs(value) >= 0.25:
                ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Spearman r")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_later_pc_pairs(
    pc_frame: pd.DataFrame,
    frame: pd.DataFrame,
    output: Path,
    max_points: int,
    seed: int,
    color_column: str,
) -> None:
    import matplotlib.pyplot as plt

    non_pc_frame = frame.drop(
        columns=[column for column in frame.columns if column.startswith("PC")],
        errors="ignore",
    )
    plot_frame = deterministic_sample(pd.concat([pc_frame, non_pc_frame], axis=1), max_points, seed)
    pairs = [("PC1", "PC2"), ("PC3", "PC4"), ("PC5", "PC6"), ("PC7", "PC8"), ("PC9", "PC10"), ("PC11", "PC12")]
    if color_column not in plot_frame:
        raise ValueError(f"Color column not found in latent table: {color_column}")
    values = plot_frame[color_column].to_numpy(dtype=np.float64)
    vmin, vmax = np.nanpercentile(values, [2, 98])
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    for ax, (x_name, y_name) in zip(axes.ravel(), pairs, strict=True):
        scatter = ax.scatter(
            plot_frame[x_name],
            plot_frame[y_name],
            c=values,
            cmap="viridis",
            s=5,
            alpha=0.42,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True, alpha=0.18)
    color_label = proxy_label(color_column)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.82, label=color_label)
    fig.suptitle(f"Later PCA score planes colored by {color_label}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_later_pc_pairs_for_all_proxies(
    pc_frame: pd.DataFrame,
    frame: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    max_points: int,
    seed: int,
) -> list[Path]:
    outputs: list[Path] = []
    for color_column in [column for column in PROXY_COLUMNS if column in frame.columns]:
        output = output_dir / f"{prefix}_later_pc_pairs_by_{output_stem(color_column)}.png"
        plot_later_pc_pairs(
            pc_frame,
            frame,
            output,
            max_points,
            seed,
            color_column,
        )
        outputs.append(output)
    return outputs


def plot_acsm_percent_later_pc_matrix(
    pc_frame: pd.DataFrame,
    frame: pd.DataFrame,
    output: Path,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    missing = [column for column in ACSM_PERCENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing ACSM percent columns: {missing}")
    plot_frame = deterministic_sample(pd.concat([pc_frame, frame[ACSM_PERCENT_COLUMNS]], axis=1), max_points, seed)
    pairs = [
        ("PC1", "PC2"),
        ("PC3", "PC4"),
        ("PC5", "PC6"),
        ("PC7", "PC8"),
        ("PC9", "PC10"),
        ("PC11", "PC12"),
    ]
    fig, axes = plt.subplots(
        len(ACSM_PERCENT_COLUMNS),
        len(pairs),
        figsize=(19.5, 13.2),
        constrained_layout=True,
        squeeze=False,
    )
    for row, color_column in enumerate(ACSM_PERCENT_COLUMNS):
        values = plot_frame[color_column].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        if valid.sum() < 10:
            continue
        vmin, vmax = np.nanpercentile(values, [2, 98])
        row_scatter = None
        for col, (x_name, y_name) in enumerate(pairs):
            ax = axes[row, col]
            row_scatter = ax.scatter(
                plot_frame[x_name],
                plot_frame[y_name],
                c=values,
                cmap="viridis",
                s=4,
                alpha=0.42,
                linewidths=0,
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(f"{x_name}-{y_name}", fontsize=10)
            if col == 0:
                ax.set_ylabel(proxy_label(color_column), fontsize=10)
            else:
                ax.set_yticklabels([])
            if row == len(ACSM_PERCENT_COLUMNS) - 1:
                ax.set_xlabel(x_name)
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.16)
        if row_scatter is not None:
            fig.colorbar(
                row_scatter,
                ax=axes[row, :].tolist(),
                shrink=0.72,
                label="ACSM species fraction (%)",
            )
    fig.suptitle("ACSM composition percentages across later PCA score planes", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_raw_z_heatmap(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    z_columns = [column for column in frame.columns if column.startswith("z_")]
    proxies = [column for column in PROXY_COLUMNS if column in frame.columns]
    corr = frame[z_columns + proxies].corr(method="spearman").loc[z_columns, proxies]
    ordering_score = corr.abs().max(axis=1)
    corr = corr.loc[ordering_score.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.6, 11.0), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(proxies)), [proxy_label(column) for column in proxies], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), corr.index, fontsize=7)
    ax.set_title("Raw 64-D bottleneck coordinates vs aerosol proxies")
    fig.colorbar(image, ax=ax, label="Spearman r")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.latent_pca_csv)
    pc_frame, explained = compute_full_pca(frame)
    plot_scree(explained, output_dir / f"{args.prefix}_pca_scree.png")
    plot_pc_proxy_heatmap(pc_frame, frame, output_dir / f"{args.prefix}_pc_proxy_heatmap.png")
    legacy_later_pc_output = output_dir / f"{args.prefix}_later_pc_pairs.png"
    plot_later_pc_pairs(
        pc_frame,
        frame,
        legacy_later_pc_output,
        args.max_points,
        args.seed,
        "dry_scattering_proxy" if "dry_scattering_proxy" in frame else "observed_modality_count",
    )
    later_pc_outputs = plot_later_pc_pairs_for_all_proxies(
        pc_frame,
        frame,
        output_dir,
        args.prefix,
        args.max_points,
        args.seed,
    )
    acsm_percent_matrix = output_dir / f"{args.prefix}_acsm_percent_later_pc_matrix.png"
    plot_acsm_percent_later_pc_matrix(
        pc_frame,
        frame,
        acsm_percent_matrix,
        args.max_points,
        args.seed,
    )
    plot_raw_z_heatmap(frame, output_dir / f"{args.prefix}_raw_z_proxy_heatmap.png")
    print(f"wrote {output_dir / f'{args.prefix}_pca_scree.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_pc_proxy_heatmap.png'}")
    print(f"wrote {legacy_later_pc_output}")
    for output in later_pc_outputs:
        print(f"wrote {output}")
    print(f"wrote {acsm_percent_matrix}")
    print(f"wrote {output_dir / f'{args.prefix}_raw_z_proxy_heatmap.png'}")


if __name__ == "__main__":
    main()
