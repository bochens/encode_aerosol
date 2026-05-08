from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn

from .model import build_model_from_checkpoint, unpack_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace and render the aerosol encoder with torchview + Graphviz."
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint.pt from train.py.")
    parser.add_argument("--output", required=True, help="Output path, usually .png.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--component-depth", type=int, default=3)
    parser.add_argument("--graph-dir", default="LR", choices=["LR", "TB", "RL", "BT"])
    parser.add_argument("--component-graph-dir", default="TB", choices=["LR", "TB", "RL", "BT"])
    parser.add_argument("--expand-nested", action="store_true")
    parser.add_argument("--show-functions", action="store_true")
    parser.add_argument("--show-inner-tensors", action="store_true")
    parser.add_argument("--roll", action="store_true", default=True)
    parser.add_argument(
        "--no-components",
        action="store_true",
        help="Only render the collapsed full-model overview.",
    )
    return parser.parse_args()


def prepend_executable_directory_to_path() -> None:
    executable_dir = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    if executable_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = executable_dir + os.pathsep + current_path


class TorchviewAerosolOverviewWrapper(nn.Module):
    def __init__(self, model: nn.Module, checkpoint: dict) -> None:
        super().__init__()
        self.model = model
        self.modality_names = tuple(checkpoint["modality_indices"].keys())
        self.target_modalities = tuple(checkpoint["target_modalities"])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        n_modalities = len(self.modality_names)
        if len(inputs) != n_modalities * 3:
            raise ValueError(
                f"Expected {n_modalities * 3} tensor inputs, got {len(inputs)}"
            )
        x_tensors = inputs[:n_modalities]
        feature_mask_tensors = inputs[n_modalities:2 * n_modalities]
        modality_mask_tensors = inputs[2 * n_modalities:]
        x_by_modality = {
            modality: tensor
            for modality, tensor in zip(self.modality_names, x_tensors, strict=True)
        }
        feature_mask_by_modality = {
            modality: tensor
            for modality, tensor in zip(self.modality_names, feature_mask_tensors, strict=True)
        }
        input_modality_mask = {
            modality: tensor.squeeze(-1).to(dtype=torch.bool)
            for modality, tensor in zip(self.modality_names, modality_mask_tensors, strict=True)
        }
        z, decoded, _ = unpack_model_output(
            self.model(x_by_modality, feature_mask_by_modality, input_modality_mask)
        )
        decoded_flat = torch.cat(
            [decoded[modality] for modality in self.target_modalities],
            dim=-1,
        )
        return torch.cat([z, decoded_flat], dim=-1)


def make_full_model_inputs(
    checkpoint: dict,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    modality_dims = checkpoint["modality_dims"]
    modality_names = tuple(checkpoint["modality_indices"].keys())
    for modality in modality_names:
        tensors.append(torch.zeros(
            batch_size,
            modality_dims[modality],
            dtype=torch.float32,
        ))
    for modality in modality_names:
        tensors.append(torch.ones(
            batch_size,
            modality_dims[modality],
            dtype=torch.float32,
        ))
    for _ in modality_names:
        tensors.append(torch.ones(batch_size, 1, dtype=torch.bool))
    return tuple(tensors)


def render_torchview_graph(
    module: torch.nn.Module,
    input_data,
    output: Path,
    graph_name: str,
    depth: int,
    graph_dir: str,
    args: argparse.Namespace,
) -> None:
    try:
        from torchview import draw_graph
    except ImportError as exc:
        raise RuntimeError(
            "torchview is required for architecture plotting. Install torchview and Graphviz "
            "in the Research_DL environment."
        ) from exc

    prepend_executable_directory_to_path()
    graph = draw_graph(
        module.eval(),
        input_data=input_data,
        graph_name=graph_name,
        depth=depth,
        device="cpu",
        mode="eval",
        strict=False,
        expand_nested=args.expand_nested,
        graph_dir=graph_dir,
        hide_module_functions=not args.show_functions,
        hide_inner_tensors=not args.show_inner_tensors,
        roll=args.roll,
        show_shapes=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    requested_format = output.suffix.lower().lstrip(".") or "png"
    rendered_path = Path(
        graph.visual_graph.render(
            filename=output.stem,
            directory=str(output.parent),
            format=requested_format,
            cleanup=True,
        )
    )
    if rendered_path != output and rendered_path.exists():
        output.write_bytes(rendered_path.read_bytes())

    for extra_format in ("svg", "pdf"):
        if extra_format == requested_format:
            continue
        graph.visual_graph.render(
            filename=output.with_suffix("").name,
            directory=str(output.parent),
            format=extra_format,
            cleanup=True,
        )
    output.with_suffix(".dot").write_text(graph.visual_graph.source, encoding="utf-8")


def suffix_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}{output.suffix}")


def render_graph(checkpoint_path: Path, output: Path, args: argparse.Namespace) -> list[Path]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    written = [output]
    render_torchview_graph(
        TorchviewAerosolOverviewWrapper(model, checkpoint),
        make_full_model_inputs(checkpoint, args.batch_size),
        output,
        graph_name="aerosol_encoder_overview",
        depth=args.depth,
        graph_dir=args.graph_dir,
        args=args,
    )
    if args.no_components:
        return written

    modality_dims = checkpoint["modality_dims"]
    component_specs = [
        (
            "diameter_size_encoder",
            model.encoders["size_smps"],
            (
                torch.zeros(args.batch_size, modality_dims["size_smps"], dtype=torch.float32),
                torch.ones(args.batch_size, modality_dims["size_smps"], dtype=torch.float32),
            ),
        ),
        (
            "optical_neph_encoder",
            model.encoders["optical_neph"],
            (
                torch.zeros(args.batch_size, modality_dims["optical_neph"], dtype=torch.float32),
                torch.ones(args.batch_size, modality_dims["optical_neph"], dtype=torch.float32),
            ),
        ),
        (
            "conditional_ccn_decoder",
            model.decoders["ccn_activation"],
            (
                torch.zeros(args.batch_size, checkpoint["latent_dim"], dtype=torch.float32),
                torch.zeros(args.batch_size, modality_dims["ccn_activation"], dtype=torch.float32),
                torch.ones(args.batch_size, modality_dims["ccn_activation"], dtype=torch.float32),
            ),
        ),
    ]
    for suffix, module, input_data in component_specs:
        component_output = suffix_path(output, suffix)
        render_torchview_graph(
            module,
            input_data,
            component_output,
            graph_name=f"aerosol_{suffix}",
            depth=args.component_depth,
            graph_dir=args.component_graph_dir,
            args=args,
        )
        written.append(component_output)
    return written


def main() -> None:
    args = parse_args()
    outputs = render_graph(Path(args.checkpoint), Path(args.output), args)
    for output in outputs:
        print(f"wrote {output}")
        print(f"wrote {output.with_suffix('.svg')}")
        print(f"wrote {output.with_suffix('.pdf')}")
        print(f"wrote {output.with_suffix('.dot')}")


if __name__ == "__main__":
    main()
