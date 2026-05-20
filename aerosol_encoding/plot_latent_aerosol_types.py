from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROXY_COLUMNS = [
    "cpc_number_proxy",
    "smps_number_proxy",
    "uhsas_accum_proxy",
    "aps_coarse_proxy",
    "organic_percent_proxy",
    "sulfate_percent_proxy",
    "ammonium_percent_proxy",
    "nitrate_percent_proxy",
    "chloride_percent_proxy",
    "acsm_speciated_mass_conc_proxy",
    "acsm_volume_proxy",
    "dry_scattering_proxy",
]

ACSM_PERCENT_COLUMNS = [
    "organic_percent_proxy",
    "sulfate_percent_proxy",
    "ammonium_percent_proxy",
    "nitrate_percent_proxy",
    "chloride_percent_proxy",
]

TYPE_ORDER = [
    "low proxy / background",
    "small-number rich",
    "coarse enhanced",
    "accumulation / scattering",
    "organic rich",
    "sulfate rich",
    "nitrate rich",
    "chloride enhanced",
    "mixed high loading",
    "mixed / other",
]

TYPE_COLORS = {
    "low proxy / background": "#8c8c8c",
    "small-number rich": "#1f78b4",
    "coarse enhanced": "#b15928",
    "accumulation / scattering": "#33a02c",
    "organic rich": "#e31a1c",
    "sulfate rich": "#6a3d9a",
    "nitrate rich": "#fb9a99",
    "chloride enhanced": "#cab2d6",
    "mixed high loading": "#ff7f00",
    "mixed / other": "#bdbdbd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot proxy-defined aerosol type views of a saved 64-D bottleneck table."
    )
    parser.add_argument("--latent-pca-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="test_no_ccn")
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--tsne-points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def z_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("z_")]
    if len(columns) != 64:
        raise ValueError(f"Expected 64 bottleneck columns named z_*, found {len(columns)}.")
    return columns


def deterministic_sample(frame: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame.copy()
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(frame), size=max_points, replace=False))
    return frame.iloc[indices].copy()


def proxy_zscores(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PROXY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing proxy columns required for aerosol type labels: {missing}")
    proxies = frame[PROXY_COLUMNS].copy()
    z = pd.DataFrame(index=frame.index)
    for column in PROXY_COLUMNS:
        values = proxies[column].to_numpy(dtype=np.float64)
        median = np.nanmedian(values)
        q25, q75 = np.nanpercentile(values, [25, 75])
        scale = (q75 - q25) / 1.349
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = np.nanstd(values)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
        z[column] = (values - median) / scale
    return z


def proxy_label(name: str) -> str:
    return (
        name.removesuffix("_percent_proxy")
        .removesuffix("_proxy")
        .replace("_", " ")
    )


def aerosol_type_labels(frame: pd.DataFrame) -> pd.Series:
    """Create interpretable, proxy-defined aerosol types for latent-space viewing.

    These labels are diagnostics, not source-apportionment labels. They describe
    which observed aerosol proxy is unusually strong in a row.
    """

    proxy_z = proxy_zscores(frame)
    high = proxy_z >= 0.65
    low = proxy_z <= -0.45

    labels = pd.Series("mixed / other", index=frame.index, dtype=object)

    loading_columns = [
        "cpc_number_proxy",
        "smps_number_proxy",
        "uhsas_accum_proxy",
        "dry_scattering_proxy",
        "organic_percent_proxy",
        "sulfate_percent_proxy",
        "ammonium_percent_proxy",
        "nitrate_percent_proxy",
        "acsm_volume_proxy",
    ]
    low_loading = low[loading_columns].sum(axis=1) >= 4
    high_loading = high[loading_columns].sum(axis=1) >= 4

    labels[low_loading] = "low proxy / background"
    labels[high_loading] = "mixed high loading"

    small_number = (
        high["cpc_number_proxy"]
        & high["smps_number_proxy"]
        & ~high["dry_scattering_proxy"]
        & ~high["uhsas_accum_proxy"]
    )
    labels[small_number] = "small-number rich"

    coarse = high["aps_coarse_proxy"] & (
        (proxy_z["aps_coarse_proxy"] - proxy_z["smps_number_proxy"] > 0.55)
        | low["smps_number_proxy"]
    )
    labels[coarse] = "coarse enhanced"

    accumulation_scattering = (
        high["uhsas_accum_proxy"]
        & high["dry_scattering_proxy"]
        & ~(high["organic_percent_proxy"] | high["sulfate_percent_proxy"] | high["nitrate_percent_proxy"])
    )
    labels[accumulation_scattering] = "accumulation / scattering"

    organic_rich = high["organic_percent_proxy"] & (
        proxy_z["organic_percent_proxy"]
        - proxy_z[["sulfate_percent_proxy", "nitrate_percent_proxy"]].max(axis=1)
        > 0.45
    )
    sulfate_rich = high["sulfate_percent_proxy"] & (
        proxy_z["sulfate_percent_proxy"]
        - proxy_z[["organic_percent_proxy", "nitrate_percent_proxy"]].max(axis=1)
        > 0.45
    )
    nitrate_rich = high["nitrate_percent_proxy"] & (
        proxy_z["nitrate_percent_proxy"]
        - proxy_z[["organic_percent_proxy", "sulfate_percent_proxy"]].max(axis=1)
        > 0.45
    )
    chloride_enhanced = high["chloride_percent_proxy"] & (
        proxy_z["chloride_percent_proxy"]
        - proxy_z[["organic_percent_proxy", "sulfate_percent_proxy", "nitrate_percent_proxy"]].max(axis=1)
        > 0.35
    )
    labels[organic_rich] = "organic rich"
    labels[sulfate_rich] = "sulfate rich"
    labels[nitrate_rich] = "nitrate rich"
    labels[chloride_enhanced] = "chloride enhanced"

    return labels


def ordered_types(labels: pd.Series) -> list[str]:
    present = set(labels.dropna().unique())
    return [label for label in TYPE_ORDER if label in present] + sorted(present - set(TYPE_ORDER))


def add_type_labels(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["aerosol_type_proxy"] = aerosol_type_labels(output)
    return output


def compute_pca(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    columns = z_columns(frame)
    scaled = StandardScaler().fit_transform(frame[columns].to_numpy(dtype=np.float32))
    pca = PCA(n_components=12, random_state=0)
    pcs = pca.fit_transform(scaled)
    pc_frame = pd.DataFrame(
        pcs,
        columns=[f"PC{index + 1}" for index in range(pcs.shape[1])],
        index=frame.index,
    )
    return pc_frame, scaled, pca.explained_variance_ratio_


def plot_pca_pairs(
    frame: pd.DataFrame,
    pc_frame: pd.DataFrame,
    output: Path,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    plot_frame = deterministic_sample(
        pd.concat([pc_frame, frame[["aerosol_type_proxy"]]], axis=1),
        max_points,
        seed,
    )
    pairs = [("PC1", "PC2"), ("PC3", "PC4"), ("PC5", "PC6"), ("PC7", "PC8")]
    types = ordered_types(plot_frame["aerosol_type_proxy"])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    for ax, (x_name, y_name) in zip(axes.ravel(), pairs, strict=True):
        for label in types:
            subset = plot_frame[plot_frame["aerosol_type_proxy"] == label]
            ax.scatter(
                subset[x_name],
                subset[y_name],
                s=7,
                alpha=0.38,
                linewidths=0,
                color=TYPE_COLORS.get(label, "#525252"),
                label=f"{label} (n={len(subset)})",
            )
        centroids = plot_frame.groupby("aerosol_type_proxy")[[x_name, y_name]].median()
        for label, row in centroids.iterrows():
            ax.scatter(
                row[x_name],
                row[y_name],
                marker="X",
                s=90,
                color=TYPE_COLORS.get(label, "#525252"),
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True, alpha=0.18)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("64-D aerosol bottleneck PCA planes by proxy-defined aerosol type", fontsize=15)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_type_facets(
    frame: pd.DataFrame,
    pc_frame: pd.DataFrame,
    output: Path,
    x_name: str,
    y_name: str,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    plot_frame = deterministic_sample(
        pd.concat([pc_frame[[x_name, y_name]], frame[["aerosol_type_proxy"]]], axis=1),
        max_points,
        seed,
    )
    types = ordered_types(plot_frame["aerosol_type_proxy"])
    ncols = 4
    nrows = int(np.ceil(len(types) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14.5, 3.25 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(-1)
    x_values = plot_frame[x_name]
    y_values = plot_frame[y_name]
    x_pad = 0.04 * (x_values.max() - x_values.min())
    y_pad = 0.04 * (y_values.max() - y_values.min())
    for ax, label in zip(axes_array, types, strict=False):
        subset = plot_frame[plot_frame["aerosol_type_proxy"] == label]
        ax.scatter(
            x_values,
            y_values,
            s=4,
            color="#d9d9d9",
            alpha=0.18,
            linewidths=0,
        )
        ax.scatter(
            subset[x_name],
            subset[y_name],
            s=10,
            color=TYPE_COLORS.get(label, "#525252"),
            alpha=0.68,
            linewidths=0,
        )
        ax.scatter(
            subset[x_name].median(),
            subset[y_name].median(),
            marker="X",
            s=120,
            color=TYPE_COLORS.get(label, "#525252"),
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ax.set_title(f"{label}\nn={len(subset)}", fontsize=10)
        ax.grid(True, alpha=0.16)
        ax.set_xlim(x_values.min() - x_pad, x_values.max() + x_pad)
        ax.set_ylim(y_values.min() - y_pad, y_values.max() + y_pad)
    for ax in axes_array[len(types) :]:
        ax.axis("off")
    for ax in axes_array[-ncols:]:
        ax.set_xlabel(x_name)
    for ax in axes_array[::ncols]:
        ax.set_ylabel(y_name)
    fig.suptitle(f"Each proxy-defined aerosol type highlighted in {x_name}-{y_name} bottleneck PCA space")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_proxy_type_heatmap(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    proxy_z = proxy_zscores(frame)
    heatmap_frame = pd.concat([frame[["aerosol_type_proxy"]], proxy_z], axis=1)
    types = ordered_types(frame["aerosol_type_proxy"])
    summary = heatmap_frame.groupby("aerosol_type_proxy")[PROXY_COLUMNS].median().loc[types]
    counts = frame["aerosol_type_proxy"].value_counts().reindex(types).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    image = ax.imshow(summary.to_numpy(), cmap="RdBu_r", vmin=-1.8, vmax=1.8, aspect="auto")
    ax.set_xticks(
        np.arange(len(PROXY_COLUMNS)),
        [proxy_label(column) for column in PROXY_COLUMNS],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(np.arange(len(types)), [f"{label}\n(n={counts[label]})" for label in types])
    for row in range(summary.shape[0]):
        for col in range(summary.shape[1]):
            value = summary.iat[row, col]
            ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("What each proxy-defined aerosol type means")
    fig.colorbar(image, ax=ax, label="robust proxy z-score")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def plot_acsm_percent_by_type(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    missing = [column for column in ACSM_PERCENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing ACSM percent columns: {missing}")
    types = ordered_types(frame["aerosol_type_proxy"])
    summary = frame.groupby("aerosol_type_proxy")[ACSM_PERCENT_COLUMNS].median().loc[types]
    counts = frame["aerosol_type_proxy"].value_counts().reindex(types).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    image = ax.imshow(summary.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(
        np.arange(len(ACSM_PERCENT_COLUMNS)),
        [proxy_label(column) for column in ACSM_PERCENT_COLUMNS],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(np.arange(len(types)), [f"{label}\n(n={counts[label]})" for label in types])
    for row in range(summary.shape[0]):
        for col in range(summary.shape[1]):
            value = summary.iat[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0f}%", ha="center", va="center", fontsize=8)
    ax.set_title("Median ACSM composition by proxy-defined aerosol type")
    fig.colorbar(image, ax=ax, label="species fraction of ACSM speciated mass (%)")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def plot_z_type_heatmap(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    columns = z_columns(frame)
    z_scaled = pd.DataFrame(
        StandardScaler().fit_transform(frame[columns].to_numpy(dtype=np.float32)),
        columns=columns,
        index=frame.index,
    )
    heatmap_frame = pd.concat([frame[["aerosol_type_proxy"]], z_scaled], axis=1)
    types = ordered_types(frame["aerosol_type_proxy"])
    type_means = heatmap_frame.groupby("aerosol_type_proxy")[columns].mean().loc[types]
    order = type_means.abs().max(axis=0).sort_values(ascending=False).index
    ordered_means = type_means[order]
    counts = frame["aerosol_type_proxy"].value_counts().reindex(types).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(15.5, 5.8), constrained_layout=True)
    image = ax.imshow(ordered_means.to_numpy(), cmap="RdBu_r", vmin=-1.15, vmax=1.15, aspect="auto")
    ax.set_xticks(np.arange(len(order)), [name.replace("z_", "z") for name in order], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(types)), [f"{label} (n={counts[label]})" for label in types])
    ax.set_title("Mean standardized 64-D bottleneck coordinates by proxy-defined aerosol type")
    fig.colorbar(image, ax=ax, label="mean standardized z")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return ordered_means


def plot_tsne_types(
    frame: pd.DataFrame,
    scaled_z: np.ndarray,
    output: Path,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    if len(frame) > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(frame), size=max_points, replace=False))
        plot_frame = frame.iloc[indices].copy()
        z = scaled_z[indices]
    else:
        plot_frame = frame.copy()
        z = scaled_z

    perplexity = min(50, max(5, (len(plot_frame) - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_iter=1200,
    ).fit_transform(z)
    plot_frame["TSNE1"] = embedding[:, 0]
    plot_frame["TSNE2"] = embedding[:, 1]
    types = ordered_types(plot_frame["aerosol_type_proxy"])

    fig, ax = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    for label in types:
        subset = plot_frame[plot_frame["aerosol_type_proxy"] == label]
        ax.scatter(
            subset["TSNE1"],
            subset["TSNE2"],
            s=7,
            alpha=0.42,
            linewidths=0,
            color=TYPE_COLORS.get(label, "#525252"),
            label=f"{label} (n={len(subset)})",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Local structure of the 64-D bottleneck by proxy-defined aerosol type")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_kmeans_types(
    frame: pd.DataFrame,
    pc_frame: pd.DataFrame,
    scaled_z: np.ndarray,
    output: Path,
    seed: int,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    kmeans = KMeans(n_clusters=6, n_init=30, random_state=seed)
    clusters = kmeans.fit_predict(scaled_z)
    plot_frame = pd.concat([pc_frame[["PC1", "PC2", "PC3", "PC4"]], frame[["aerosol_type_proxy"]]], axis=1)
    plot_frame["latent_cluster"] = clusters
    cross = pd.crosstab(plot_frame["latent_cluster"], plot_frame["aerosol_type_proxy"], normalize="index")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    scatter = axes[0].scatter(
        plot_frame["PC1"],
        plot_frame["PC2"],
        c=plot_frame["latent_cluster"],
        cmap="tab10",
        s=5,
        alpha=0.35,
        linewidths=0,
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Unsupervised k-means clusters in 64-D z")
    axes[0].grid(True, alpha=0.18)
    fig.colorbar(scatter, ax=axes[0], label="latent cluster")

    types = ordered_types(frame["aerosol_type_proxy"])
    cross = cross.reindex(columns=types, fill_value=0.0)
    image = axes[1].imshow(cross.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.5, cross.to_numpy().max()))
    axes[1].set_xticks(np.arange(len(types)), types, rotation=35, ha="right")
    axes[1].set_yticks(np.arange(cross.shape[0]), [f"cluster {index}" for index in cross.index])
    axes[1].set_title("Cluster composition by proxy-defined type")
    for row in range(cross.shape[0]):
        for col in range(cross.shape[1]):
            value = cross.iat[row, col]
            if value >= 0.08:
                axes[1].text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], label="fraction of cluster")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return cross


def write_summary(
    frame: pd.DataFrame,
    explained: np.ndarray,
    scaled_z: np.ndarray,
    output: Path,
    seed: int,
) -> None:
    labels = frame["aerosol_type_proxy"].to_numpy()
    valid_labels = pd.Series(labels).value_counts()
    if valid_labels.size > 1 and len(frame) > 100:
        sample = deterministic_sample(frame.assign(_row=np.arange(len(frame))), 5000, seed)
        sampled_z = scaled_z[sample["_row"].to_numpy(dtype=int)]
        sampled_labels = sample["aerosol_type_proxy"].to_numpy()
        label_counts = pd.Series(sampled_labels).value_counts()
        keep = np.array([label_counts[label] >= 5 for label in sampled_labels])
        type_silhouette = float(silhouette_score(sampled_z[keep], sampled_labels[keep]))
    else:
        type_silhouette = float("nan")

    summary = {
        "n_rows": int(len(frame)),
        "aerosol_type_counts": frame["aerosol_type_proxy"].value_counts().to_dict(),
        "pc_explained_variance_first_12": [float(value) for value in explained],
        "pc_cumulative_first_12": [float(value) for value in np.cumsum(explained)],
        "aerosol_type_silhouette_z_space": type_silhouette,
    }
    with output.open("w") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.latent_pca_csv)
    frame = add_type_labels(frame)
    pc_frame, scaled_z, explained = compute_pca(frame)

    typed_csv = output_dir / f"{args.prefix}_latent_with_aerosol_type.csv"
    frame.to_csv(typed_csv, index=False)

    proxy_summary = plot_proxy_type_heatmap(
        frame,
        output_dir / f"{args.prefix}_aerosol_type_proxy_summary.png",
    )
    proxy_summary.to_csv(output_dir / f"{args.prefix}_aerosol_type_proxy_summary.csv")

    acsm_percent_summary = plot_acsm_percent_by_type(
        frame,
        output_dir / f"{args.prefix}_acsm_percent_by_aerosol_type.png",
    )
    acsm_percent_summary.to_csv(output_dir / f"{args.prefix}_acsm_percent_by_aerosol_type.csv")

    z_means = plot_z_type_heatmap(
        frame,
        output_dir / f"{args.prefix}_aerosol_type_64d_z_heatmap.png",
    )
    z_means.to_csv(output_dir / f"{args.prefix}_aerosol_type_64d_z_means.csv")

    plot_pca_pairs(
        frame,
        pc_frame,
        output_dir / f"{args.prefix}_aerosol_type_pca_pairs.png",
        args.max_points,
        args.seed,
    )
    plot_type_facets(
        frame,
        pc_frame,
        output_dir / f"{args.prefix}_aerosol_type_pc1_pc2_facets.png",
        "PC1",
        "PC2",
        args.max_points,
        args.seed,
    )
    plot_type_facets(
        frame,
        pc_frame,
        output_dir / f"{args.prefix}_aerosol_type_pc3_pc4_facets.png",
        "PC3",
        "PC4",
        args.max_points,
        args.seed,
    )
    plot_tsne_types(
        frame,
        scaled_z,
        output_dir / f"{args.prefix}_aerosol_type_tsne.png",
        args.tsne_points,
        args.seed,
    )
    cluster_composition = plot_kmeans_types(
        frame,
        pc_frame,
        scaled_z,
        output_dir / f"{args.prefix}_latent_kmeans_vs_aerosol_type.png",
        args.seed,
    )
    cluster_composition.to_csv(output_dir / f"{args.prefix}_latent_kmeans_vs_aerosol_type.csv")
    write_summary(
        frame,
        explained,
        scaled_z,
        output_dir / f"{args.prefix}_aerosol_type_summary.json",
        args.seed,
    )

    print(f"wrote {typed_csv}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_proxy_summary.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_64d_z_heatmap.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_pca_pairs.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_pc1_pc2_facets.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_pc3_pc4_facets.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_tsne.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_latent_kmeans_vs_aerosol_type.png'}")
    print(f"wrote {output_dir / f'{args.prefix}_aerosol_type_summary.json'}")


if __name__ == "__main__":
    main()
