from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write model shape and parameter summary.")
    parser.add_argument("--checkpoint", required=True, help="checkpoint.pt from train.py.")
    parser.add_argument("--output", required=True, help="Output text path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    transformer_layers = int(config.get("transformer_layers", checkpoint.get("transformer_layers", 0) or 0))
    transformer_heads = int(config.get("transformer_heads", checkpoint.get("transformer_heads", 0) or 0))
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

    lines = [
        f"checkpoint: {args.checkpoint}",
        f"model_type: {checkpoint.get('model_type', config.get('model_type', 'grouped_masked_autoencoder'))}",
        f"latent_dim: {checkpoint['latent_dim']}",
        f"hidden_dim: {checkpoint['hidden_dim']}",
        f"global_transformer: layers={transformer_layers} heads={transformer_heads}",
        f"sizing_crosstalk_transformer: layers={sizing_crosstalk_layers} heads={sizing_crosstalk_heads}",
        f"decoder_expansion_depth: {decoder_expansion_depth}",
    ]
    model_type = str(checkpoint.get("model_type", config.get("model_type", "")))
    if checkpoint.get("latent_blocks"):
        lines.append(
            "latent_blocks: "
            + ", ".join(
                f"{name}={dim}"
                for name, dim in checkpoint["latent_blocks"].items()
            )
        )
    lines.extend(
        [
            "",
            "ARCHITECTURE BLOCKS",
            "1. Structured encoders produce one hidden_dim token per modality.",
            (
                "2. SMPS, APS, UHSAS, and OPC tokens go directly into global fusion as separate modality tokens."
                if sizing_crosstalk_layers <= 0
                else "2. SMPS, APS, UHSAS, and OPC tokens pass through the sizing-crosstalk transformer before global fusion."
            ),
            "3. The global transformer fuses all visible modality tokens plus one learned latent-query token.",
            (
                "4. The latent head maps the query output to Gaussian mean/log-variance; training samples z and adds KL loss."
                if model_type == "structured_transformer_vae"
                else "4. The latent head maps the query output to the deterministic aerosol bottleneck z."
            ),
            "5. The decoder-expansion block maps z back to hidden_dim before target-specific decoders.",
        ]
    )
    if decoder_expansion_depth > 0:
        expansion_layers = [
            f"Linear({checkpoint['latent_dim']} -> {checkpoint['hidden_dim']})"
        ]
        for _ in range(decoder_expansion_depth - 1):
            expansion_layers.extend(
                [
                    "GELU",
                    f"Linear({checkpoint['hidden_dim']} -> {checkpoint['hidden_dim']})",
                ]
            )
        expansion_layers.append("LayerNorm")
        lines.append("decoder_expansion_layers: " + ", ".join(expansion_layers))
    lines.extend(["", "MODALITY SHAPES"])
    for name, dim in checkpoint["modality_dims"].items():
        role = "target" if name in checkpoint["target_modalities"] else "context"
        encoders = getattr(model, "encoders", {})
        encoder = encoders[name] if name in encoders else None
        temporal = ""
        if encoder is not None and getattr(encoder, "temporal_step_count", 0) > 0:
            temporal = (
                f" temporal={encoder.temporal_step_count}x"
                f"{encoder.temporal_channel_count}"
            )
        lines.append(
            f"{name:18s} role={role:7s} features={dim:4d} "
            f"value_mask_input={2 * dim:4d}{temporal}"
        )

    lines.extend(["", "LAYER SHAPES"])
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            params = sum(parameter.numel() for parameter in module.parameters())
            lines.append(
                f"{name:45s} Linear({module.in_features} -> {module.out_features}) params={params}"
            )
        elif isinstance(module, torch.nn.Conv1d):
            params = sum(parameter.numel() for parameter in module.parameters())
            lines.append(
                f"{name:45s} Conv1d({module.in_channels} -> {module.out_channels}, kernel={module.kernel_size}) params={params}"
            )
        elif isinstance(module, torch.nn.GRU):
            params = sum(parameter.numel() for parameter in module.parameters())
            lines.append(
                f"{name:45s} GRU(input={module.input_size}, hidden={module.hidden_size}, "
                f"layers={module.num_layers}, batch_first={module.batch_first}) params={params}"
            )
        elif isinstance(module, torch.nn.LayerNorm):
            params = sum(parameter.numel() for parameter in module.parameters())
            lines.append(f"{name:45s} LayerNorm({module.normalized_shape}) params={params}")

    lines.extend(
        [
            "",
            f"total_parameters: {sum(parameter.numel() for parameter in model.parameters())}",
            f"trainable_parameters: {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
