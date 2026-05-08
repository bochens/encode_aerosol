from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .config import load_config


COLORS = {
    "input": "#e8f1fb",
    "encoder": "#eee9fb",
    "token": "#fff3d8",
    "fusion": "#e9f5f1",
    "latent": "#fff0e6",
    "decoder": "#fbe8f0",
    "loss": "#f1f1f1",
    "edge": "#46515c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw paper-style block schematics for the aerosol transformer autoencoder."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG/PDF figures.")
    return parser.parse_args()


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: int = 10,
    linewidth: float = 1.2,
):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=color,
        edgecolor=COLORS["edge"],
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.2,
    )
    return box


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float = 0.0,
    linewidth: float = 1.25,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=COLORS["edge"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def setup_axis(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def draw_overall(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    fig, ax = setup_axis((15, 7.6))
    ax.text(
        0.5,
        0.965,
        "Deterministic multimodal transformer autoencoder",
        ha="center",
        va="top",
        fontsize=20,
        weight="bold",
    )

    input_text = (
        "Hourly multimodal inputs\n\n"
        "AOSMET context\nACSM chemistry\nSMPS / APS / UHSAS / OPC size spectra\n"
        "CPC number\nCCN activation\nDry + wet neph\n\n"
        "Each feature carries value + observed mask"
    )
    encoder_text = (
        "Structured modality encoders\n\n"
        "Scalar MLP encoders\nfor chemistry, met, CPC, CCN\n\n"
        "Diameter-bin transformer encoders\nfor SMPS, APS, UHSAS, OPC\n\n"
        "Neph wavelength encoder\nfor dry/wet scattering"
    )
    token_text = (
        "Tokenization\n\n"
        "one 192-D token per modality\n+ one learned 192-D latent query\n\n"
        "input visibility mask removes\nhidden instruments from attention"
    )
    fusion_text = (
        "Transformer fusion\n\n"
        f"{config.transformer_layers} encoder layers\n"
        f"{config.transformer_heads} attention heads\n"
        f"width {config.hidden_dim}\n\n"
        "latent query attends to\nvisible instrument tokens"
    )
    latent_text = f"Aerosol encoding\n\nz: {config.latent_dim} deterministic dimensions"
    decoder_text = (
        "Target decoders\n\n"
        "predict each instrument family:\nACSM, SMPS, APS, UHSAS, OPC,\nCPC, CCN, neph"
    )

    boxes = [
        ((0.035, 0.35), 0.17, 0.36, input_text, COLORS["input"]),
        ((0.245, 0.34), 0.18, 0.38, encoder_text, COLORS["encoder"]),
        ((0.465, 0.39), 0.16, 0.28, token_text, COLORS["token"]),
        ((0.665, 0.39), 0.16, 0.28, fusion_text, COLORS["fusion"]),
        ((0.86, 0.57), 0.12, 0.13, latent_text, COLORS["latent"]),
        ((0.86, 0.31), 0.12, 0.17, decoder_text, COLORS["decoder"]),
    ]
    for xy, width, height, text, color in boxes:
        add_box(ax, xy, width, height, text, color, fontsize=10)

    add_arrow(ax, (0.205, 0.53), (0.245, 0.53))
    add_arrow(ax, (0.425, 0.53), (0.465, 0.53))
    add_arrow(ax, (0.625, 0.53), (0.665, 0.53))
    add_arrow(ax, (0.825, 0.53), (0.86, 0.635))
    add_arrow(ax, (0.92, 0.57), (0.92, 0.48))

    ax.text(
        0.5,
        0.045,
        "Forward pass: visible instrument measurements -> modality tokens -> transformer-fused aerosol state -> predicted instrument outputs",
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.25",
    )
    save(fig, output)


def draw_fusion(output: Path) -> None:
    fig, ax = setup_axis((11.5, 7.0))
    ax.text(0.5, 0.95, "Transformer fusion block", ha="center", va="top", fontsize=19, weight="bold")

    add_box(ax, (0.06, 0.61), 0.22, 0.18, "Visible modality tokens\nT_met, T_acsm, ...\nshape: M x 192", COLORS["token"])
    add_box(ax, (0.06, 0.31), 0.22, 0.16, "Latent query token\nlearned parameter\nshape: 1 x 192", COLORS["token"])
    add_box(ax, (0.37, 0.54), 0.20, 0.18, "Concatenate tokens\n[M visible tokens\n+ 1 latent query]", COLORS["fusion"])
    add_box(ax, (0.37, 0.26), 0.20, 0.16, "Key-padding mask\nhidden modalities ignored\nlatent query kept", COLORS["loss"])
    add_box(ax, (0.66, 0.60), 0.22, 0.14, "Multi-head self-attention\n6 heads", COLORS["fusion"])
    add_box(ax, (0.66, 0.42), 0.22, 0.12, "Residual + LayerNorm", COLORS["fusion"])
    add_box(ax, (0.66, 0.27), 0.22, 0.12, "Feed-forward MLP\n192 -> 768 -> 192", COLORS["fusion"])
    add_box(ax, (0.66, 0.11), 0.22, 0.12, "Residual + LayerNorm\nrepeat x2 layers", COLORS["fusion"])
    add_box(ax, (0.38, 0.06), 0.20, 0.12, "Read out latent-query row\nMLP: 192 -> 192 -> 256", COLORS["latent"])

    add_arrow(ax, (0.28, 0.70), (0.37, 0.64))
    add_arrow(ax, (0.28, 0.39), (0.37, 0.60), rad=0.08)
    add_arrow(ax, (0.47, 0.54), (0.66, 0.67))
    add_arrow(ax, (0.47, 0.42), (0.66, 0.67), rad=0.12)
    add_arrow(ax, (0.77, 0.60), (0.77, 0.54))
    add_arrow(ax, (0.77, 0.42), (0.77, 0.39))
    add_arrow(ax, (0.77, 0.27), (0.77, 0.23))
    add_arrow(ax, (0.66, 0.17), (0.58, 0.12))

    ax.text(
        0.5,
        0.015,
        "Only the latent-query output becomes the aerosol encoding; hidden instrument tokens are present in memory but masked from attention evidence.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.25",
    )
    save(fig, output)


def draw_training(output: Path) -> None:
    fig, ax = setup_axis((14, 7.8))
    ax.text(0.5, 0.95, "Training objective and masking", ha="center", va="top", fontsize=19, weight="bold")

    add_box(ax, (0.04, 0.63), 0.18, 0.16, "Observed hourly row\nall available instruments\nstandardized values + masks", COLORS["input"], fontsize=9)
    add_box(ax, (0.30, 0.74), 0.18, 0.12, "Stage 1\nAutoencode\nsame visible inputs", COLORS["loss"], fontsize=9)
    add_box(ax, (0.30, 0.56), 0.18, 0.12, "Stage 2\nDenoise autoencode\nfeature dropout + noise", COLORS["loss"], fontsize=9)
    add_box(ax, (0.30, 0.38), 0.18, 0.12, "Stages 3-6\nMasked hidden-only\nhide target instruments", COLORS["loss"], fontsize=9)
    add_box(ax, (0.56, 0.54), 0.20, 0.16, "Same transformer autoencoder\nshared weights\nall stages", COLORS["fusion"], fontsize=10)
    add_box(ax, (0.82, 0.72), 0.14, 0.12, "Reconstruction loss\nvisible targets", COLORS["decoder"], fontsize=9)
    add_box(ax, (0.82, 0.52), 0.14, 0.12, "Cross-prediction loss\nhidden targets only", COLORS["decoder"], fontsize=9)
    add_box(ax, (0.82, 0.32), 0.14, 0.12, "Closure losses\nneph / CCN proxy\nphysical consistency", COLORS["decoder"], fontsize=9)

    for y in (0.80, 0.62, 0.44):
        add_arrow(ax, (0.22, 0.72), (0.30, y))
        add_arrow(ax, (0.48, y), (0.56, 0.62))
    add_arrow(ax, (0.76, 0.62), (0.82, 0.78))
    add_arrow(ax, (0.76, 0.61), (0.82, 0.58))
    add_arrow(ax, (0.76, 0.59), (0.82, 0.38))

    add_box(
        ax,
        (0.08, 0.08),
        0.37,
        0.12,
        "Strict sizing evaluation rule:\nwhen scoring SMPS, APS, UHSAS, or OPC,\nall sizing instruments are removed from the input.",
        "#fff7e6",
        fontsize=10,
    )
    add_box(
        ax,
        (0.52, 0.08),
        0.36,
        0.12,
        "Skill score:\n1 - model MSE / training-mean baseline MSE\npositive = better than mean baseline",
        "#eef7ee",
        fontsize=10,
    )
    save(fig, output)


def draw_sizing_crosstalk(output: Path) -> None:
    fig, ax = setup_axis((14, 7.8))
    ax.text(
        0.5,
        0.95,
        "Sizing-instrument crosstalk and masking",
        ha="center",
        va="top",
        fontsize=19,
        weight="bold",
    )

    add_box(
        ax,
        (0.05, 0.67),
        0.17,
        0.14,
        "SMPS token\nmobility diameter\nfine-mode spectrum",
        COLORS["token"],
        fontsize=9,
    )
    add_box(
        ax,
        (0.05, 0.49),
        0.17,
        0.14,
        "UHSAS token\noptical diameter\naccumulation-mode spectrum",
        COLORS["token"],
        fontsize=9,
    )
    add_box(
        ax,
        (0.05, 0.31),
        0.17,
        0.14,
        "APS token\naerodynamic diameter\ncoarse-mode spectrum",
        COLORS["token"],
        fontsize=9,
    )
    add_box(
        ax,
        (0.05, 0.13),
        0.17,
        0.14,
        "OPC token\noptical diameter\ncoarse/accumulation overlap",
        COLORS["token"],
        fontsize=9,
    )
    add_box(
        ax,
        (0.30, 0.30),
        0.22,
        0.32,
        "Transformer fusion\n\n"
        "visible sizing tokens\nattend to each other\n\n"
        "overlap differences can carry\n"
        "size, density, refractive-index,\n"
        "or calibration information",
        COLORS["fusion"],
        fontsize=8,
    )
    add_box(
        ax,
        (0.59, 0.70),
        0.34,
        0.14,
        "Case A: one sizing instrument hidden\nexample: SMPS hidden; APS, UHSAS, OPC visible\n\nAllows sizing crosstalk from overlapping instruments.",
        "#eef7ee",
        fontsize=9,
    )
    add_box(
        ax,
        (0.59, 0.48),
        0.34,
        0.14,
        "Case B: strict sizing group hidden\nexample: predict SMPS with SMPS, APS, UHSAS, OPC all hidden\n\nMeasures retrieval from non-sizing data only.",
        "#fff7e6",
        fontsize=9,
    )
    add_box(
        ax,
        (0.59, 0.26),
        0.34,
        0.14,
        "Case C: pairwise crosstalk test\nexample: UHSAS + context -> OPC\n\nQuantifies which instrument helps which target.",
        "#f3eefb",
        fontsize=9,
    )
    add_box(
        ax,
        (0.59, 0.06),
        0.34,
        0.13,
        "Current checkpoint selection uses Case B for sizing targets.\nThe saved skill CSV also reports Case A and pairwise tests.",
        COLORS["loss"],
        fontsize=9,
    )

    for y in (0.74, 0.56, 0.38, 0.20):
        add_arrow(ax, (0.22, y), (0.30, 0.46), rad=0.08)
    add_arrow(ax, (0.52, 0.52), (0.59, 0.77))
    add_arrow(ax, (0.52, 0.47), (0.59, 0.55))
    add_arrow(ax, (0.52, 0.42), (0.59, 0.33))

    ax.text(
        0.5,
        0.01,
        "The architecture supports sizing crosstalk; the evaluation protocol decides whether that crosstalk is allowed.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.25",
    )
    save(fig, output)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    draw_overall(config_path, output_dir / "aerosol_block_schematic_overall.png")
    draw_fusion(output_dir / "aerosol_block_schematic_fusion.png")
    draw_training(output_dir / "aerosol_block_schematic_training.png")
    draw_sizing_crosstalk(output_dir / "aerosol_block_schematic_sizing_crosstalk.png")
    print(f"wrote {output_dir / 'aerosol_block_schematic_overall.png'}")
    print(f"wrote {output_dir / 'aerosol_block_schematic_fusion.png'}")
    print(f"wrote {output_dir / 'aerosol_block_schematic_training.png'}")
    print(f"wrote {output_dir / 'aerosol_block_schematic_sizing_crosstalk.png'}")


if __name__ == "__main__":
    main()
