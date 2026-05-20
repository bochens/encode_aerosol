from __future__ import annotations

import argparse
import html
import os
import subprocess
from pathlib import Path

import torch


COLORS = {
    "input": ("#eaf2fb", "#2f5f8f"),
    "encoder": ("#f0ecff", "#6f58c9"),
    "token": ("#fff3d8", "#9a6a00"),
    "fusion": ("#e7f4f1", "#2d7c68"),
    "latent": ("#fff0e6", "#b45f26"),
    "decoder": ("#fcebf3", "#a64f7a"),
    "coordinate": ("#f8fafc", "#64748b"),
}


DISPLAY_NAMES = {
    "met_context": "AOSMET context",
    "chemistry_acsm": "ACSM chemistry",
    "size_smps": "SMPS size",
    "size_aps": "APS size",
    "size_uhsas": "UHSAS size",
    "size_opc": "OPC size",
    "cpc_number": "CPC number",
    "ccn_activation": "CCN activation",
    "optical_neph": "Dry/wet neph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a compact paper-quality aerosol encoder schematic with Graphviz."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="final_epoch790_paper_schematic")
    return parser.parse_args()


def esc(text: object) -> str:
    return html.escape(str(text))


def input_grid_label(title: str, rows: list[tuple[str, str]], fill: str, border: str) -> str:
    cells = []
    for left, right in rows:
        cells.append(
            f"<TD WIDTH=\"138\" HEIGHT=\"58\" BGCOLOR=\"{fill}\" COLOR=\"{border}\">"
            f"<B>{esc(left)}</B><BR/><FONT POINT-SIZE=\"11\">{esc(right)}</FONT>"
            "</TD>"
        )
    body = []
    for i in range(0, len(cells), 3):
        body.append("<TR>" + "".join(cells[i : i + 3]) + "</TR>")
    return (
        "<<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"8\" "
        f"COLOR=\"{border}\" CELLPADDING=\"5\">"
        f"<TR><TD COLSPAN=\"3\"><B>{esc(title)}</B></TD></TR>"
        f"{''.join(body)}"
        "</TABLE>>"
    )


def block_label(title: str, lines: list[str], fill: str, border: str) -> str:
    rows = []
    for line in lines:
        rows.append(f"<TR><TD>{esc(line)}</TD></TR>")
    return (
        "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
        f"COLOR=\"{border}\" CELLPADDING=\"8\">"
        f"<TR><TD BGCOLOR=\"{fill}\"><B>{esc(title)}</B></TD></TR>"
        f"{''.join(rows)}"
        "</TABLE>>"
    )


def dot_header() -> list[str]:
    return [
        "digraph G {",
        "  graph [rankdir=LR, bgcolor=white, margin=0.04, pad=0.08,",
        "         nodesep=0.42, ranksep=0.58, splines=ortho, outputorder=edgesfirst];",
        "  node [shape=plain, fontname=Helvetica, fontsize=12];",
        "  edge [fontname=Helvetica, fontsize=10, color=\"#334155\", arrowsize=0.72, penwidth=1.45];",
        "  labelloc=\"t\";",
        "  label=\"64-D coordinate-conditioned multimodal aerosol encoder\";",
        "  fontsize=22;",
        "  fontname=Helvetica;",
    ]


def add_node(lines: list[str], name: str, label: str) -> None:
    lines.append(f"  {name} [label={label}];")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    dims = checkpoint["modality_dims"]
    hidden_dim = int(checkpoint["hidden_dim"])
    latent_dim = int(checkpoint["latent_dim"])
    decoder_hidden = int(config.get("decoder_expansion_hidden_dim", hidden_dim))
    transformer_layers = int(config.get("transformer_layers", checkpoint.get("transformer_layers", 4)))
    transformer_heads = int(config.get("transformer_heads", checkpoint.get("transformer_heads", 8)))
    decoder_depth = int(config.get("decoder_depth", 3))

    dot = dot_header()
    input_rows = [
        (DISPLAY_NAMES.get(name, name), f"{int(dims[name]):,} features")
        for name in dims
    ]
    fill, border = COLORS["input"]
    add_node(dot, "inputs", input_grid_label("Processed 30-min feature windows", input_rows, fill, border))

    fill, border = COLORS["encoder"]
    add_node(
        dot,
        "encoders",
        block_label(
            "Structured modality encoders",
            [
                "value + observed mask for every feature",
                "temporal GRU where sub-window bins exist",
                "diameter-aware transformer for size spectra",
                f"each modality -> one {hidden_dim}-D token",
            ],
            fill,
            border,
        ),
    )

    fill, border = COLORS["token"]
    add_node(
        dot,
        "tokens",
        block_label(
            "Visible token bank",
            [
                "9 instrument/context tokens possible",
                f"+ one learned {hidden_dim}-D latent query",
                "hidden instruments are key-padded",
            ],
            fill,
            border,
        ),
    )

    fill, border = COLORS["fusion"]
    add_node(
        dot,
        "fusion",
        block_label(
            "Global transformer fusion",
            [
                f"{transformer_layers} layers, {transformer_heads} heads",
                f"token width {hidden_dim}",
                "latent query attends to visible tokens",
            ],
            fill,
            border,
        ),
    )

    fill, border = COLORS["latent"]
    dot.append(
        "  bottleneck [shape=box, style=\"rounded,filled\", "
        f"fillcolor=\"{fill}\", color=\"{border}\", penwidth=1.6, "
        "fixedsize=true, width=1.55, height=0.64, "
        f"label=< <B>Aerosol z</B><BR/><FONT POINT-SIZE=\"11\">{latent_dim}-D bottleneck</FONT> >];"
    )

    fill, border = COLORS["decoder"]
    add_node(
        dot,
        "expand",
        block_label(
            "Decoder expansion",
            [f"{latent_dim} -> {decoder_hidden} -> {hidden_dim}", "4-layer MLP + LayerNorm"],
            fill,
            border,
        ),
    )
    add_node(
        dot,
        "decoders",
        block_label(
            "Target decoders",
            [
                f"separate {decoder_depth}-layer heads",
                "ACSM, CPC, CCN, neph",
                "SMPS, APS, UHSAS, OPC",
            ],
            fill,
            border,
        ),
    )

    fill, border = COLORS["coordinate"]
    add_node(
        dot,
        "coords",
        block_label(
            "Query coordinates",
            ["CCN: supersaturation", "Size: log diameter", "Neph: channel + RH state"],
            fill,
            border,
        ),
    )

    dot.extend(
        [
            "  { rank=same; inputs }",
            "  { rank=same; encoders }",
            "  { rank=same; tokens }",
            "  { rank=same; fusion }",
            "  { rank=same; bottleneck }",
            "  { rank=same; expand }",
            "  { rank=same; decoders coords }",
            "  inputs -> encoders;",
            "  encoders -> tokens;",
            "  tokens -> fusion;",
            "  fusion -> bottleneck;",
            "  bottleneck -> expand;",
            "  expand -> decoders;",
            "  coords -> decoders [style=dashed, constraint=false, color=\"#64748b\", arrowsize=0.62];",
            "}",
        ]
    )

    dot_path = output_dir / f"{args.prefix}.dot"
    dot_path.write_text("\n".join(dot) + "\n", encoding="utf-8")
    for suffix, fmt in ((".png", "png"), (".pdf", "pdf"), (".svg", "svg")):
        output = output_dir / f"{args.prefix}{suffix}"
        subprocess.run(["dot", f"-T{fmt}", str(dot_path), "-o", str(output)], check=True)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
