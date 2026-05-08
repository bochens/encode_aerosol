from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path
from typing import Mapping

import torch


SIZING_MODALITIES = ("size_smps", "size_aps", "size_uhsas", "size_opc")
TIME_BIN_RE = re.compile(r"^(?P<base>.+)__time_bin_(?P<bin>\d+)$")

DISPLAY_NAMES = {
    "met_context": "AOSMET",
    "chemistry_acsm": "ACSM-CDCE",
    "size_smps": "SMPS",
    "size_aps": "APS",
    "size_uhsas": "UHSAS",
    "size_opc": "OPC",
    "cpc_number": "CPC",
    "ccn_activation": "CCN",
    "optical_neph": "Dry/wet neph",
}

NODE_COLORS = {
    "input": ("#eaf2fb", "#2f5f8f"),
    "size": ("#fff1cc", "#9a6a00"),
    "encoder": ("#f0ecff", "#6f58c9"),
    "token": ("#eefcf7", "#27825f"),
    "fusion": ("#eaf5f5", "#3f6970"),
    "latent": ("#f9eadf", "#b45f26"),
    "decoder": ("#fcebf3", "#a64f7a"),
    "training": ("#f8fafc", "#64748b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render checkpoint-derived architecture schematics with Graphviz dot. "
            "The figures are generated from the trained checkpoint metadata, not "
            "from hand-positioned drawing coordinates."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt.")
    parser.add_argument(
        "--output-dir",
        default="docs/figures",
        help="Directory for PNG, SVG, PDF, and DOT outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="aerosol_encoder_graphviz",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def prepend_common_graphviz_paths() -> None:
    candidates = (
        str(Path(sys.executable).resolve().parent),
        "/opt/homebrew/bin",
        "/usr/local/bin",
    )
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for candidate in reversed(candidates):
        if candidate and candidate not in path_parts:
            path_parts.insert(0, candidate)
    os.environ["PATH"] = os.pathsep.join(path_parts)


def html_label(title: str, lines: tuple[str, ...], fill: str, border: str) -> str:
    body_rows = []
    for line in lines:
        body_rows.append(
            "<TR><TD BALIGN=\"CENTER\">"
            f"<FONT POINT-SIZE=\"10\">{html.escape(line)}</FONT>"
            "</TD></TR>"
        )
    return (
        "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
        f"COLOR=\"{border}\" CELLPADDING=\"7\">"
        f"<TR><TD BGCOLOR=\"{fill}\"><B>{html.escape(title)}</B></TD></TR>"
        f"{''.join(body_rows)}"
        "</TABLE>>"
    )


def add_node(
    graph,
    name: str,
    title: str,
    lines: tuple[str, ...],
    style: str,
) -> None:
    fill, border = NODE_COLORS[style]
    graph.node(name, label=html_label(title, lines, fill, border), shape="plain")


def add_rank(graph, names: tuple[str, ...]) -> None:
    with graph.subgraph() as rank:
        rank.attr(rank="same")
        for name in names:
            rank.node(name)


def temporal_shapes(checkpoint: Mapping[str, object]) -> dict[str, tuple[int, int]]:
    feature_names = checkpoint.get("feature_names", ())
    modality_indices = checkpoint.get("modality_indices", {})
    if not isinstance(feature_names, (list, tuple)) or not isinstance(modality_indices, dict):
        return {}
    output: dict[str, tuple[int, int]] = {}
    for modality, indices in modality_indices.items():
        bins: dict[int, set[str]] = {}
        for index in indices:
            feature_name = str(feature_names[int(index)])
            match = TIME_BIN_RE.match(feature_name)
            if match is None:
                continue
            bins.setdefault(int(match.group("bin")), set()).add(match.group("base"))
        if not bins:
            continue
        output[str(modality)] = (len(bins), max(len(channels) for channels in bins.values()))
    return output


def dim_label(modality: str, dim: int, temporal: Mapping[str, tuple[int, int]]) -> str:
    if modality in temporal:
        steps, channels = temporal[modality]
        return f"{dim} = {steps} x {channels}"
    return str(dim)


def modality_encoder_lines(
    modality: str,
    dim: int,
    hidden_dim: int,
    config: Mapping[str, object],
    temporal: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[str, ...]:
    temporal = temporal or {}
    if modality in temporal:
        steps, channels = temporal[modality]
        return (
            "temporal GRU encoder",
            f"{steps} time steps x {channels} channels",
            "value + observed mask + time position",
            f"{dim} features -> {hidden_dim}-D token",
        )
    if modality in SIZING_MODALITIES:
        return (
            "diameter-bin transformer encoder",
            "value + observed mask per bin",
            f"{dim} features -> {hidden_dim}-D token",
        )
    if modality == "optical_neph":
        return (
            "wavelength/RH structured encoder",
            "dry + humidified scattering",
            f"{dim} features -> {hidden_dim}-D token",
        )
    if modality == "ccn_activation" and bool(config.get("conditional_ccn_decoder", False)):
        return (
            "scalar context encoder",
            "decoder is supersaturation-conditioned",
            f"{dim} features -> {hidden_dim}-D token",
        )
    return (
        "scalar MLP encoder",
        "value + observed mask inputs",
        f"{dim} features -> {hidden_dim}-D token",
    )


def make_graph(name: str):
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise RuntimeError(
            "The Python graphviz package is required. Use the Research_DL "
            "environment where graphviz is installed."
        ) from exc

    graph = Digraph(name=name, engine="dot")
    graph.attr(
        bgcolor="white",
        rankdir="LR",
        splines="spline",
        nodesep="0.46",
        ranksep="0.72",
        margin="0.05",
        pad="0.08",
        outputorder="edgesfirst",
        concentrate="false",
    )
    graph.attr("node", fontname="Helvetica", fontsize="12")
    graph.attr(
        "edge",
        fontname="Helvetica",
        fontsize="9",
        color="#334155",
        arrowsize="0.65",
        penwidth="1.25",
    )
    return graph


def build_overview_graph(checkpoint: Mapping[str, object]):
    graph = make_graph("aerosol_encoder_checkpoint_overview")
    modality_dims = checkpoint["modality_dims"]
    if not isinstance(modality_dims, dict):
        raise TypeError("checkpoint modality_dims must be a dictionary")
    modalities = tuple(modality_dims.keys())
    target_modalities = tuple(checkpoint["target_modalities"])
    hidden_dim = int(checkpoint["hidden_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    model_type = str(checkpoint.get("model_type", config.get("model_type", "")))
    temporal = temporal_shapes(checkpoint)
    temporal = temporal_shapes(checkpoint)
    transformer_layers = int(config.get("transformer_layers", checkpoint.get("transformer_layers", 2)))
    transformer_heads = int(config.get("transformer_heads", checkpoint.get("transformer_heads", 4)))
    sizing_crosstalk_layers = int(
        config.get(
            "sizing_crosstalk_layers",
            checkpoint.get("sizing_crosstalk_layers", 0),
        )
    )
    sizing_crosstalk_heads = int(
        config.get(
            "sizing_crosstalk_heads",
            checkpoint.get("sizing_crosstalk_heads", transformer_heads),
        )
    )
    decoder_expansion_depth = int(
        config.get(
            "decoder_expansion_depth",
            checkpoint.get("decoder_expansion_depth", 0),
        )
    )
    sizing_modalities = tuple(
        modality for modality in SIZING_MODALITIES if modality in modality_dims
    )
    scalar_modalities = tuple(
        modality
        for modality in modalities
        if modality not in SIZING_MODALITIES and modality != "optical_neph"
    )
    target_labels = tuple(DISPLAY_NAMES.get(modality, modality) for modality in target_modalities)
    target_line_1 = ", ".join(target_labels[:4])
    target_line_2 = ", ".join(target_labels[4:])

    add_node(
        graph,
        "scalar_inputs",
        "Context, chemistry, number, CCN",
        (
            ", ".join(
                f"{DISPLAY_NAMES.get(modality, modality)}={dim_label(modality, int(modality_dims[modality]), temporal)}"
                for modality in scalar_modalities
            ),
            "standardized values plus observed masks",
            "temporal channels preserved where configured",
            f"{len(scalar_modalities)} modality families",
        ),
        "input",
    )
    add_node(
        graph,
        "size_inputs",
        "Sizing inputs kept separate",
        (
            ", ".join(
                f"{DISPLAY_NAMES[modality]}={dim_label(modality, int(modality_dims[modality]), temporal)}"
                for modality in sizing_modalities
            ),
            "shared log-Dp grid and time bins where configured",
            "instrument identity retained",
            "missing instruments are masked",
        ),
        "size",
    )
    add_node(
        graph,
        "neph_inputs",
        "Optical inputs",
        (
            f"Dry/wet neph={dim_label('optical_neph', int(modality_dims.get('optical_neph', 0)), temporal)}",
            "wavelength/RH/P/T structure retained",
            "available rows become one token",
        ),
        "input",
    )
    add_node(
        graph,
        "scalar_encoders",
        "Scalar MLP encoders",
        (
            "one encoder per non-sizing modality",
            "temporal GRU if time bins exist",
            "otherwise scalar MLP",
            f"value+mask -> {hidden_dim}-D token",
            "hidden modality tokens are key-padded",
        ),
        "encoder",
    )
    add_node(
        graph,
        "size_encoders",
        "Diameter-aware size encoders",
        (
            "temporal GRU over sub-window size states",
            "or diameter transformer for non-temporal runs",
            "SMPS, APS, UHSAS, OPC stay separate",
            f"each output is one {hidden_dim}-D token",
        ),
        "encoder",
    )
    add_node(
        graph,
        "neph_encoder",
        "Neph structured encoder",
        (
            "temporal GRU if time bins exist",
            "otherwise structured scattering encoder",
            "wavelength and RH context",
            f"output is one {hidden_dim}-D token",
        ),
        "encoder",
    )

    graph.edge("scalar_inputs", "scalar_encoders")
    graph.edge("size_inputs", "size_encoders")
    graph.edge("neph_inputs", "neph_encoder")

    size_output_node = "size_encoders"
    if sizing_crosstalk_layers > 0 and len(sizing_modalities) > 1:
        add_node(
            graph,
            "sizing_crosstalk",
            "Sizing crosstalk block",
            (
                f"{sizing_crosstalk_layers} transformer layer",
                f"{sizing_crosstalk_heads} attention heads, width {hidden_dim}",
                "visible sizing tokens attend to each other first",
                "hidden sizing target is key-padded",
            ),
            "fusion",
        )
        graph.edge("size_encoders", "sizing_crosstalk")
        size_output_node = "sizing_crosstalk"

    add_node(
        graph,
        "visible_bank",
        "Visible token bank",
        (
            f"{len(modalities)} possible instrument tokens",
            "hidden instruments are masked before fusion",
            "instrument identity embeddings stay attached",
        ),
        "token",
    )
    graph.edge("scalar_encoders", "visible_bank")
    graph.edge(size_output_node, "visible_bank")
    graph.edge("neph_encoder", "visible_bank")

    add_node(
        graph,
        "latent_query",
        "Learned latent query",
        (f"1 token x {hidden_dim}-D", "reads the fused aerosol state"),
        "fusion",
    )
    add_node(
        graph,
        "transformer",
        "Global transformer fusion",
        (
            f"{transformer_layers} encoder layers",
            f"{transformer_heads} attention heads",
            f"model width {hidden_dim}",
            "visible instrument tokens exchange information",
        ),
        "fusion",
    )
    add_node(
        graph,
        "latent",
        "Aerosol encoding z",
        (
            (f"{latent_dim}-D Gaussian mean + log-variance")
            if model_type == "structured_transformer_vae"
            else f"{latent_dim} deterministic dimensions",
            (
                "sampled during training; mean used for evaluation"
                if model_type == "structured_transformer_vae"
                else "no VAE sampling or KL loss"
            ),
            (
                "KL loss regularizes z toward N(0, I)"
                if model_type == "structured_transformer_vae"
                else "shared bottleneck used by all decoders"
            ),
        ),
        "latent",
    )
    if decoder_expansion_depth > 0:
        expansion_middle = (
            f"{decoder_expansion_depth - 1} hidden Linear {hidden_dim}->{hidden_dim} layers"
            if decoder_expansion_depth > 1
            else "direct projection to decoder state"
        )
        add_node(
            graph,
            "decoder_expansion",
            "Decoder expansion block",
            (
                f"{decoder_expansion_depth} linear layers",
                f"{latent_dim}-D z -> {hidden_dim}-D decoder state",
                f"first Linear {latent_dim}->{hidden_dim}",
                expansion_middle,
                "LayerNorm before target decoders",
            ),
            "decoder",
        )
        decoder_input_node = "decoder_expansion"
    else:
        decoder_input_node = "latent"
    add_node(
        graph,
        "decoder_bank",
        "Target decoders",
        (
            target_line_1,
            target_line_2,
            f"each target decoder has {hidden_dim}-D hidden width",
            "CCN head receives supersaturation when available",
        ),
        "decoder",
    )
    graph.edge("visible_bank", "transformer")
    graph.edge("latent_query", "transformer", style="dashed")
    graph.edge("transformer", "latent")
    if decoder_input_node != "latent":
        graph.edge("latent", decoder_input_node)
    graph.edge(decoder_input_node, "decoder_bank")

    add_node(
        graph,
        "loss",
        "Training loss",
        (
            "observed-feature MSE on hidden targets",
            "plus KL loss for VAE checkpoints" if model_type == "structured_transformer_vae" else "no KL term for deterministic checkpoints",
            "leave-one-out validation selects checkpoint",
            "strict all-sizing-hidden remains diagnostic",
        ),
        "training",
    )
    graph.edge("decoder_bank", "loss", style="dashed", color="#64748b")

    add_rank(graph, ("scalar_inputs", "size_inputs", "neph_inputs"))
    add_rank(graph, ("scalar_encoders", "size_encoders", "neph_encoder"))
    add_rank(graph, ("visible_bank", "latent_query"))
    return graph


def build_sizing_crosstalk_graph(checkpoint: Mapping[str, object]):
    graph = make_graph("aerosol_encoder_sizing_crosstalk")
    modality_dims = checkpoint["modality_dims"]
    if not isinstance(modality_dims, dict):
        raise TypeError("checkpoint modality_dims must be a dictionary")
    hidden_dim = int(checkpoint["hidden_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    model_type = str(checkpoint.get("model_type", config.get("model_type", "")))
    temporal = temporal_shapes(checkpoint)
    transformer_layers = int(config.get("transformer_layers", checkpoint.get("transformer_layers", 2)))
    transformer_heads = int(config.get("transformer_heads", checkpoint.get("transformer_heads", 4)))
    sizing_crosstalk_layers = int(
        config.get(
            "sizing_crosstalk_layers",
            checkpoint.get("sizing_crosstalk_layers", 0),
        )
    )
    sizing_crosstalk_heads = int(
        config.get(
            "sizing_crosstalk_heads",
            checkpoint.get("sizing_crosstalk_heads", transformer_heads),
        )
    )
    decoder_expansion_depth = int(
        config.get(
            "decoder_expansion_depth",
            checkpoint.get("decoder_expansion_depth", 0),
        )
    )

    add_node(
        graph,
        "nonsize_context",
        "Non-sizing context",
        (
            "AOSMET, ACSM, CPC, CCN, neph",
            "available instruments become tokens",
            "used in all-size-hidden diagnostic",
        ),
        "input",
    )
    token_nodes: list[str] = []
    for modality in SIZING_MODALITIES:
        if modality not in modality_dims:
            continue
        dim = int(modality_dims[modality])
        token = f"token_{modality}"
        token_nodes.append(token)
        add_node(
            graph,
            token,
            f"{DISPLAY_NAMES[modality]} token",
            (
                f"{dim_label(modality, dim, temporal)} input features",
                "temporal window retained" if modality in temporal else "diameter-bin features",
                f"separate {hidden_dim}-D modality token",
                "not merged with other sizing instruments",
            ),
            "size",
        )

    has_sizing_crosstalk = sizing_crosstalk_layers > 0 and len(token_nodes) > 1
    if has_sizing_crosstalk:
        add_node(
            graph,
            "sizing_crosstalk",
            "Sizing crosstalk transformer",
            (
                f"{sizing_crosstalk_layers} layer, {sizing_crosstalk_heads} heads",
                "runs before global fusion",
                "only visible sizing tokens contribute",
                "hidden target token is key-padded",
            ),
            "fusion",
        )
    add_node(
        graph,
        "visible_tokens",
        "Visible global token set",
        (
            (
                "crosstalk-adjusted sizing tokens"
                if has_sizing_crosstalk
                else "raw sizing tokens enter directly"
            ),
            "plus non-sizing context tokens",
            (
                "global transformer is first sizing-mixing block"
                if not has_sizing_crosstalk
                else "input mask controls what can speak"
            ),
        ),
        "token",
    )
    add_node(
        graph,
        "transformer",
        "Global transformer fusion",
        (
            f"{transformer_layers} layers, {transformer_heads} heads",
            "mixes sizing, chemistry, CCN, CPC, neph, met",
            "learned latent query reads fused state",
        ),
        "fusion",
    )
    add_node(
        graph,
        "latent",
        "Aerosol bottleneck",
        (
            f"z in R^{latent_dim}",
            (
                "VAE mean/logvar; sample in training"
                if model_type == "structured_transformer_vae"
                else "deterministic code"
            ),
            "used by every decoder",
        ),
        "latent",
    )
    if decoder_expansion_depth > 0:
        expansion_middle = (
            f"{decoder_expansion_depth - 1} hidden Linear layers"
            if decoder_expansion_depth > 1
            else "direct projection"
        )
        add_node(
            graph,
            "decoder_expansion",
            "Decoder expansion",
            (
                f"{decoder_expansion_depth} linear layers",
                f"{latent_dim}-D -> {hidden_dim}-D",
                expansion_middle,
                "then target-specific decoders",
            ),
            "decoder",
        )
        decoder_source = "decoder_expansion"
    else:
        decoder_source = "latent"
    add_node(
        graph,
        "one_hidden",
        "Single-size hidden test",
        (
            "hide one sizing target",
            "other sizing instruments remain visible",
            (
                "help comes through global transformer only"
                if not has_sizing_crosstalk
                else "answers: can sizing instruments help each other?"
            ),
        ),
        "training",
    )
    add_node(
        graph,
        "all_hidden",
        "Strict diagnostic",
        (
            "hide SMPS + APS + UHSAS + OPC together",
            "only non-sizing instruments remain visible",
            "answers: how much size is in chemistry/CCN/neph/met?",
        ),
        "training",
    )

    graph.edge("nonsize_context", "visible_tokens")
    for token in token_nodes:
        graph.edge(token, "sizing_crosstalk" if has_sizing_crosstalk else "visible_tokens")
    if has_sizing_crosstalk:
        graph.edge("sizing_crosstalk", "visible_tokens")
    graph.edge("visible_tokens", "transformer")
    graph.edge("transformer", "latent")
    if decoder_source != "latent":
        graph.edge("latent", decoder_source)
    graph.edge("one_hidden", "visible_tokens", style="dashed")
    graph.edge("all_hidden", "visible_tokens", style="dashed")

    decoder_nodes: list[str] = []
    for modality in SIZING_MODALITIES:
        if modality not in modality_dims:
            continue
        decoder = f"decoder_{modality}"
        decoder_nodes.append(decoder)
        add_node(
            graph,
            decoder,
            f"Predict {DISPLAY_NAMES[modality]}",
            (
                "decoded from shared z",
                "compared only where target is observed",
            ),
            "decoder",
        )
        graph.edge(decoder_source, decoder)

    add_rank(graph, tuple(token_nodes))
    add_rank(graph, ("one_hidden", "all_hidden"))
    add_rank(graph, tuple(decoder_nodes))
    return graph


def render_graph(graph, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_path = output_dir / f"{stem}.dot"
    dot_path.write_text(graph.source, encoding="utf-8")
    written = [dot_path]
    for fmt in ("png", "svg", "pdf"):
        rendered = Path(
            graph.render(
                filename=stem,
                directory=str(output_dir),
                format=fmt,
                cleanup=True,
            )
        )
        written.append(rendered)
    return written


def main() -> None:
    args = parse_args()
    prepend_common_graphviz_paths()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    modality_dims = checkpoint.get("modality_dims", {})
    if not isinstance(modality_dims, dict):
        modality_dims = {}
    sizing_crosstalk_layers = int(
        config.get(
            "sizing_crosstalk_layers",
            checkpoint.get("sizing_crosstalk_layers", 0),
        )
    )
    sizing_count = sum(1 for modality in SIZING_MODALITIES if modality in modality_dims)
    sizing_suffix = (
        "sizing_crosstalk"
        if sizing_crosstalk_layers > 0 and sizing_count > 1
        else "sizing_masking"
    )
    output_dir = Path(args.output_dir)
    written: list[Path] = []
    written.extend(
        render_graph(
            build_overview_graph(checkpoint),
            output_dir,
            f"{args.prefix}_overview",
        )
    )
    written.extend(
        render_graph(
            build_sizing_crosstalk_graph(checkpoint),
            output_dir,
            f"{args.prefix}_{sizing_suffix}",
        )
    )
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
