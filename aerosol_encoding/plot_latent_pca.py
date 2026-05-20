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
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model_from_checkpoint
from .training_data import load_prepared_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project trained aerosol bottleneck encodings with PCA."
    )
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint.pt.")
    parser.add_argument(
        "--prepared-arrays",
        required=True,
        help="Prepared training arrays matching the checkpoint.",
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test", "all"],
        help="Rows used for PCA and plots.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--exclude-input-modality",
        action="append",
        default=[],
        help=(
            "Modality to hide before encoding latent z. May be repeated. "
            "Useful for inspecting retrieval latents such as all non-CCN inputs."
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help="Deterministic plotting subsample size; all rows are still written to CSV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_modalities(
    batch_x: torch.Tensor,
    batch_mask: torch.Tensor,
    modality_indices: dict[str, list[int]],
    excluded_modalities: set[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    excluded_modalities = excluded_modalities or set()
    x_by_modality = {}
    mask_by_modality = {}
    input_mask = {}
    for modality, indices in modality_indices.items():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=batch_x.device)
        x_modality = batch_x.index_select(1, index_tensor)
        mask_modality = batch_mask.index_select(1, index_tensor)
        x_by_modality[modality] = x_modality
        mask_by_modality[modality] = mask_modality
        input_mask[modality] = (mask_modality.sum(dim=1) > 0) & (
            modality not in excluded_modalities
        )
    return x_by_modality, mask_by_modality, input_mask


def season_labels(times: np.ndarray) -> np.ndarray:
    month = pd.DatetimeIndex(times).month.to_numpy()
    labels = np.full(month.shape, "DJF", dtype=object)
    labels[np.isin(month, [3, 4, 5])] = "MAM"
    labels[np.isin(month, [6, 7, 8])] = "JJA"
    labels[np.isin(month, [9, 10, 11])] = "SON"
    return labels


def masked_feature_mean(
    x: np.ndarray,
    feature_mask: np.ndarray,
    feature_names: list[str],
    contains: tuple[str, ...],
    excludes: tuple[str, ...] = (),
) -> np.ndarray:
    indices = [
        idx
        for idx, name in enumerate(feature_names)
        if all(fragment in name for fragment in contains)
        and not any(fragment in name for fragment in excludes)
    ]
    if not indices:
        raise ValueError(f"No features matched contains={contains} excludes={excludes}")
    values = x[:, indices]
    masks = feature_mask[:, indices].astype(np.float32)
    count = masks.sum(axis=1)
    total = (values * masks).sum(axis=1)
    output = np.full(x.shape[0], np.nan, dtype=np.float32)
    valid = count > 0
    output[valid] = total[valid] / count[valid]
    return output


ACSM_SPECIES_FEATURES = {
    "organic": "total_organics_CDCE",
    "sulfate": "sulfate_CDCE",
    "ammonium": "ammonium_CDCE",
    "nitrate": "nitrate_CDCE",
    "chloride": "chloride_CDCE",
}
ACSM_STANDARDIZED_PROXY_COLUMNS = {
    f"{species}_proxy" for species in ACSM_SPECIES_FEATURES
}
ACSM_SPECIES_MASS_CONC_PROXY_COLUMNS = {
    f"{species}_mass_conc_proxy" for species in ACSM_SPECIES_FEATURES
}
ACSM_PERCENT_PROXY_COLUMNS = [
    f"{species}_percent_proxy" for species in ACSM_SPECIES_FEATURES
]


def masked_log1p_physical_mean(
    x: np.ndarray,
    feature_mask: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_names: list[str],
    contains: tuple[str, ...],
) -> np.ndarray:
    """Invert prepared-array standardization and log1p for nonnegative features."""

    indices = [
        idx
        for idx, name in enumerate(feature_names)
        if all(fragment in name for fragment in contains)
    ]
    if not indices:
        raise ValueError(f"No features matched contains={contains}")
    index_array = np.asarray(indices, dtype=np.int64)
    transformed = x[:, index_array] * feature_std[index_array] + feature_mean[index_array]
    physical = np.expm1(transformed)
    physical = np.where(physical > 0.0, physical, 0.0)
    masks = feature_mask[:, index_array].astype(np.float32)
    count = masks.sum(axis=1)
    total = (physical * masks).sum(axis=1)
    output = np.full(x.shape[0], np.nan, dtype=np.float32)
    valid = count > 0
    output[valid] = total[valid] / count[valid]
    return output


def acsm_species_percentages(
    x: np.ndarray,
    feature_mask: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return ACSM species concentration and percent of summed speciated mass."""

    concentrations = pd.DataFrame(index=np.arange(x.shape[0]))
    for species, feature in ACSM_SPECIES_FEATURES.items():
        concentrations[f"{species}_mass_conc_proxy"] = masked_log1p_physical_mean(
            x,
            feature_mask,
            feature_mean,
            feature_std,
            feature_names,
            ("chemistry_acsm", feature),
        )

    species_columns = list(concentrations.columns)
    species_values = concentrations[species_columns].to_numpy(dtype=np.float64)
    valid_all = np.isfinite(species_values).all(axis=1)
    total_mass = np.nansum(species_values, axis=1)
    valid_percent = valid_all & (total_mass > 0.0)

    percentages = pd.DataFrame(index=concentrations.index)
    percentages["acsm_speciated_mass_conc_proxy"] = np.where(
        valid_percent,
        total_mass,
        np.nan,
    )
    for species in ACSM_SPECIES_FEATURES:
        values = concentrations[f"{species}_mass_conc_proxy"].to_numpy(dtype=np.float64)
        output = np.full(x.shape[0], np.nan, dtype=np.float32)
        output[valid_percent] = 100.0 * values[valid_percent] / total_mass[valid_percent]
        percentages[f"{species}_percent_proxy"] = output
    return pd.concat([concentrations, percentages], axis=1)


def quantile_flags(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    if valid.sum() < 10:
        return np.zeros_like(valid), np.zeros_like(valid)
    low, high = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return values <= low, values >= high


def proxy_regime_table(
    x: np.ndarray,
    feature_mask: np.ndarray,
    feature_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> pd.DataFrame:
    proxies = pd.DataFrame()
    proxies["cpc_number_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("cpc_number", "__concentration__"),
        ("__stat_std__", "__stat_min__", "__stat_max__"),
    )
    proxies["organic_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "total_organics_CDCE"),
    )
    proxies["sulfate_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "sulfate_CDCE"),
    )
    proxies["ammonium_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "ammonium_CDCE"),
    )
    proxies["nitrate_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "nitrate_CDCE"),
    )
    proxies["chloride_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "chloride_CDCE"),
    )
    proxies["acsm_volume_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "acsm_vol_conc"),
    )
    proxies["cdce_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("chemistry_acsm", "CDCE"),
    )
    proxies = pd.concat(
        [
            proxies,
            acsm_species_percentages(
                x,
                feature_mask,
                feature_mean,
                feature_std,
                feature_names,
            ),
        ],
        axis=1,
    )
    proxies["dry_scattering_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("optical_neph", "Dry_Neph3W"),
        ("RH_Neph_Dry", "T_Neph_Dry", "P_Neph_Dry"),
    )
    proxies["smps_number_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("size_smps", "dN_dlogDp"),
    )
    proxies["aps_coarse_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("size_aps", "dN_dlogDp"),
    )
    proxies["uhsas_accum_proxy"] = masked_feature_mean(
        x,
        feature_mask,
        feature_names,
        ("size_uhsas", "dN_dlogDp"),
    )

    _, high_cpc = quantile_flags(proxies["cpc_number_proxy"].to_numpy())
    _, high_organic = quantile_flags(proxies["organic_percent_proxy"].to_numpy())
    _, high_sulfate = quantile_flags(proxies["sulfate_percent_proxy"].to_numpy())
    _, high_nitrate = quantile_flags(proxies["nitrate_percent_proxy"].to_numpy())
    _, high_chloride = quantile_flags(proxies["chloride_percent_proxy"].to_numpy())
    _, high_scattering = quantile_flags(proxies["dry_scattering_proxy"].to_numpy())
    low_smps, _ = quantile_flags(proxies["smps_number_proxy"].to_numpy())
    _, high_coarse = quantile_flags(proxies["aps_coarse_proxy"].to_numpy())
    _, high_accum = quantile_flags(proxies["uhsas_accum_proxy"].to_numpy())

    regime = np.full(x.shape[0], "mixed/background", dtype=object)
    regime[high_scattering] = "high scattering"
    regime[high_accum & ~high_scattering] = "accumulation mode"
    regime[high_coarse & low_smps] = "coarse enhanced"
    regime[high_cpc & ~high_scattering] = "high number"
    regime[high_organic & ~high_sulfate] = "organic rich"
    regime[high_sulfate & ~high_organic] = "sulfate rich"
    regime[high_nitrate & ~(high_organic | high_sulfate)] = "nitrate rich"
    regime[high_chloride & ~(high_organic | high_sulfate | high_nitrate)] = "chloride enhanced"
    proxies["proxy_regime"] = regime
    return proxies


def encode_latents(
    checkpoint: dict,
    x: np.ndarray,
    feature_mask: np.ndarray,
    batch_size: int,
    device: str,
    excluded_modalities: set[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(feature_mask.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    observed_rows = []
    modalities = list(checkpoint["modality_indices"])

    with torch.no_grad():
        for batch_x, batch_mask in loader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            x_by_modality, mask_by_modality, input_mask = split_modalities(
                batch_x,
                batch_mask,
                checkpoint["modality_indices"],
                excluded_modalities=excluded_modalities,
            )
            z = model.encode(x_by_modality, mask_by_modality, input_mask)
            latents.append(z.cpu().numpy())
            observed_rows.append(
                np.column_stack(
                    [input_mask[modality].cpu().numpy() for modality in modalities]
                )
            )

    observed = pd.DataFrame(
        np.concatenate(observed_rows, axis=0),
        columns=[f"observed_{modality}" for modality in modalities],
    )
    observed["observed_modality_count"] = observed.sum(axis=1)
    return np.concatenate(latents, axis=0), observed


def deterministic_plot_indices(n_rows: int, max_points: int, seed: int) -> np.ndarray:
    if n_rows <= max_points:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False))


def plot_3d_categorical(
    frame: pd.DataFrame,
    label_column: str,
    output: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(9.4, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for idx, label in enumerate(sorted(frame[label_column].dropna().unique())):
        subset = frame[frame[label_column] == label]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            subset["PC3"],
            s=8,
            alpha=0.42,
            color=colors(idx % 10),
            label=str(label),
            linewidths=0,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
    )
    ax.view_init(elev=24, azim=-56)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_3d_continuous(
    frame: pd.DataFrame,
    value_column: str,
    output: Path,
    title: str,
    colorbar_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.0, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        frame["PC1"],
        frame["PC2"],
        frame["PC3"],
        c=frame[value_column],
        cmap="viridis",
        s=8,
        alpha=0.42,
        linewidths=0,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-56)
    fig.colorbar(scatter, ax=ax, shrink=0.75, label=colorbar_label)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pair_grid(
    frame: pd.DataFrame,
    label_column: str,
    output: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = sorted(str(label) for label in frame[label_column].dropna().unique())
    colors = dict(zip(labels, plt.get_cmap("tab10").colors[: len(labels)], strict=True))
    pairs = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)
    for ax, (x_name, y_name) in zip(axes, pairs, strict=True):
        for label in labels:
            subset = frame[frame[label_column].astype(str) == label]
            ax.scatter(
                subset[x_name],
                subset[y_name],
                s=6,
                alpha=0.32,
                color=colors[label],
                label=label,
                linewidths=0,
            )
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True, alpha=0.18)
    axes[-1].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        markerscale=2,
        fontsize=8,
    )
    fig.suptitle(title)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_proxy_gradient_grid(
    frame: pd.DataFrame,
    proxy_columns: list[str],
    output: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    n_columns = 4
    n_rows = int(np.ceil(len(proxy_columns) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(13.4, 3.8 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, column in zip(axes.ravel(), proxy_columns, strict=False):
        values = frame[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        if valid.sum() > 10:
            vmin, vmax = np.nanpercentile(values, [2, 98])
        else:
            vmin, vmax = np.nanmin(values), np.nanmax(values)
        scatter = ax.scatter(
            frame["PC1"],
            frame["PC2"],
            c=values,
            cmap="viridis",
            s=5,
            alpha=0.45,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(proxy_display_label(column))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.18)
        fig.colorbar(scatter, ax=ax, shrink=0.78, label=proxy_colorbar_label(column))
    for ax in axes.ravel()[len(proxy_columns) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def proxy_display_label(column: str) -> str:
    if column.endswith("_percent_proxy"):
        species = column.removesuffix("_percent_proxy").replace("_", " ")
        return f"{species} (%)"
    return column.removesuffix("_proxy").replace("_", " ")


def proxy_colorbar_label(column: str) -> str:
    if column.endswith("_percent_proxy"):
        return "ACSM species fraction (%)"
    if column == "acsm_speciated_mass_conc_proxy":
        return "ACSM summed species concentration"
    if column == "observed_modality_count":
        return "observed modality count"
    return "standardized proxy"


def clustering_summary(
    pcs: np.ndarray,
    frame: pd.DataFrame,
    seed: int,
    max_silhouette_rows: int = 6000,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    if pcs.shape[0] > max_silhouette_rows:
        rng = np.random.default_rng(seed)
        score_indices = np.sort(
            rng.choice(pcs.shape[0], size=max_silhouette_rows, replace=False)
        )
    else:
        score_indices = np.arange(pcs.shape[0])
    for k in [3, 4, 5, 6]:
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(pcs[:, :3])
        summary[f"kmeans_k{k}_silhouette_pc123"] = float(
            silhouette_score(pcs[score_indices, :3], labels[score_indices])
        )
    for column in ["season", "proxy_regime"]:
        labels = frame[column].astype(str).to_numpy()
        unique = np.unique(labels)
        if unique.size > 1 and min((labels == label).sum() for label in unique) > 1:
            summary[f"{column}_silhouette_pc123"] = float(
                silhouette_score(pcs[score_indices, :3], labels[score_indices])
            )
    summary["silhouette_rows"] = int(score_indices.size)
    return summary


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    arrays = load_prepared_arrays(args.prepared_arrays)
    if list(arrays.feature_names) != list(checkpoint["feature_names"]):
        raise ValueError("Prepared arrays feature_names do not match checkpoint feature_names")

    if args.split == "all":
        row_indices = np.arange(arrays.x.shape[0])
    else:
        row_indices = np.asarray(checkpoint["splits"][args.split], dtype=np.int64)
    x = arrays.x[row_indices]
    feature_mask = arrays.feature_mask[row_indices]
    times = arrays.times[row_indices]
    excluded_modalities = set(args.exclude_input_modality)
    unknown_excluded = excluded_modalities - set(checkpoint["modality_indices"])
    if unknown_excluded:
        raise ValueError(f"Unknown excluded modalities: {sorted(unknown_excluded)}")

    z, observed = encode_latents(
        checkpoint,
        x,
        feature_mask,
        args.batch_size,
        args.device,
        excluded_modalities=excluded_modalities,
    )
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    pca = PCA(n_components=3, random_state=args.seed)
    pcs = pca.fit_transform(z_scaled)

    frame = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"])
    frame.insert(0, "time", times.astype("datetime64[ns]").astype(str))
    frame["year"] = pd.DatetimeIndex(times).year.to_numpy()
    frame["season"] = season_labels(times)
    frame = pd.concat(
        [
            frame,
            observed.reset_index(drop=True),
            proxy_regime_table(
                x,
                feature_mask,
                arrays.feature_names,
                arrays.mean,
                arrays.std,
            ),
            pd.DataFrame(z, columns=[f"z_{idx:02d}" for idx in range(z.shape[1])]),
        ],
        axis=1,
    )
    frame.to_csv(output / f"{args.split}_latent_pca.csv", index=False)

    plot_indices = deterministic_plot_indices(len(frame), args.max_points, args.seed)
    plot_frame = frame.iloc[plot_indices].copy()
    plot_3d_categorical(
        plot_frame,
        "season",
        output / f"{args.split}_latent_pca_3d_season.png",
        f"{args.split} 64-D bottleneck PCA colored by season",
    )
    plot_3d_categorical(
        plot_frame,
        "proxy_regime",
        output / f"{args.split}_latent_pca_3d_proxy_regime.png",
        f"{args.split} 64-D bottleneck PCA colored by proxy regime",
    )
    plot_3d_continuous(
        plot_frame,
        "year",
        output / f"{args.split}_latent_pca_3d_year.png",
        f"{args.split} 64-D bottleneck PCA colored by year",
        "year",
    )
    plot_pair_grid(
        plot_frame,
        "season",
        output / f"{args.split}_latent_pca_pair_grid_season.png",
        f"{args.split} 64-D bottleneck PCA colored by season",
    )
    plot_pair_grid(
        plot_frame,
        "proxy_regime",
        output / f"{args.split}_latent_pca_pair_grid_proxy_regime.png",
        f"{args.split} 64-D bottleneck PCA colored by proxy regime",
    )

    proxy_columns = [
        column
        for column in frame.columns
        if column.endswith("_proxy")
    ]
    gradient_exclusions = (
        ACSM_STANDARDIZED_PROXY_COLUMNS
        | ACSM_SPECIES_MASS_CONC_PROXY_COLUMNS
    )
    gradient_proxy_columns = [
        column
        for column in proxy_columns
        if column not in gradient_exclusions
    ]
    gradient_columns = ["observed_modality_count", *gradient_proxy_columns]
    plot_proxy_gradient_grid(
        plot_frame,
        gradient_columns,
        output / f"{args.split}_latent_pca_proxy_gradients.png",
        f"{args.split} 64-D bottleneck PCA colored by continuous aerosol proxies",
    )
    correlations = frame[["PC1", "PC2", "PC3", *proxy_columns]].corr(
        method="spearman"
    ).loc[["PC1", "PC2", "PC3"], proxy_columns]
    correlations.to_csv(output / f"{args.split}_latent_pca_proxy_correlations.csv")

    summary = {
        "split": args.split,
        "rows": int(len(frame)),
        "latent_dim": int(z.shape[1]),
        "pca_input": "standardized latent dimensions",
        "excluded_input_modalities": sorted(excluded_modalities),
        "explained_variance_ratio": [
            float(value)
            for value in pca.explained_variance_ratio_
        ],
        "cumulative_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "season_counts": frame["season"].value_counts().sort_index().to_dict(),
        "proxy_regime_counts": frame["proxy_regime"].value_counts().sort_index().to_dict(),
        "observed_modality_count": {
            str(key): int(value)
            for key, value in frame["observed_modality_count"].value_counts().sort_index().items()
        },
        **clustering_summary(pcs, frame, args.seed),
    }
    with (output / f"{args.split}_latent_pca_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"wrote {output / f'{args.split}_latent_pca.csv'}")
    print(f"wrote {output / f'{args.split}_latent_pca_summary.json'}")
    print(f"wrote {output / f'{args.split}_latent_pca_3d_season.png'}")
    print(f"wrote {output / f'{args.split}_latent_pca_3d_proxy_regime.png'}")
    print(f"wrote {output / f'{args.split}_latent_pca_3d_year.png'}")
    print(f"wrote {output / f'{args.split}_latent_pca_pair_grid_season.png'}")
    print(f"wrote {output / f'{args.split}_latent_pca_pair_grid_proxy_regime.png'}")
    print(f"wrote {output / f'{args.split}_latent_pca_proxy_gradients.png'}")


if __name__ == "__main__":
    main()
