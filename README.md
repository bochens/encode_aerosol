# Aerosol Encoding

Train multimodal transformer autoencoders for ARM SGP aerosol observations.

The current method note is [docs/method_paper.md](docs/method_paper.md). It now includes the 3600-epoch no-special-sizing-crosstalk VAE run. That VAE run is a negative result: it is cleaner architecturally, but it does not beat the deterministic transformer baseline and does not show useful cross-prediction grokking.

## Latest VAE No-Crosstalk Grokking Run

This run treats SMPS, APS, UHSAS, and OPC as separate modality tokens. There is no sizing-only crosstalk subnet; any sizing crosstalk must happen inside the global transformer.

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_gru_32_bottleneck_vae_no_sizing_crosstalk_grokking3600.yaml \
  --features artifacts/temporal_gru_30min_20220607_20220620/features/features.npz \
  --output artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600
```

| Output | Path |
|---|---|
| Selected checkpoint | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/checkpoint.pt` |
| Final checkpoint | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/last_checkpoint.pt` |
| Loss curve | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/loss_curve.png` |
| Network summary | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/network_summary.txt` |
| Full architecture schematic | `docs/figures/aerosol_encoder_temporal_gru_30min_vae32_no_sizing_crosstalk_grokking3600_graphviz_overview.png` |
| Sizing masking schematic | `docs/figures/aerosol_encoder_temporal_gru_30min_vae32_no_sizing_crosstalk_grokking3600_graphviz_sizing_masking.png` |
| Selected test metrics | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/test_metrics.json` |
| Selected skill scores | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/cross_prediction_test/test_leave_one_out.csv` |
| Final skill scores | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/cross_prediction_last_test/test_leave_one_out.csv` |
| Selected latent encodings | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/encodings_selected/encodings.csv` |
| Final latent encodings | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/encodings_last/encodings.csv` |

The selected checkpoint is still epoch 6 with validation selected cross-prediction MSE `0.524`. The final epoch has validation selected cross-prediction MSE `0.647`. More epochs improved some individual final-checkpoint targets, especially CCN, UHSAS, SMPS, and CPC, but it damaged chemistry and optical neph and did not improve the selected validation objective.

## 64-D VAE Early-Mask Diagnostic

The 64-D diagnostic keeps the same 96-D modality-token width but expands the VAE bottleneck and decoder expansion:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_gru_64_bottleneck_vae_no_sizing_crosstalk_earlymask.yaml \
  --features artifacts/temporal_gru_30min_20220607_20220620/features/features.npz \
  --output artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask
```

This diagnostic was stopped at epoch 310 after the mild-mask phase plateaued. The selected checkpoint is epoch 71 with validation cross-prediction MSE `0.528`; it does not beat the 32-D VAE selected validation MSE `0.524`, but it improves several test skill scores. The decoder expansion is `Linear(64->96), GELU, Linear(96->96), GELU, Linear(96->96), GELU, Linear(96->96), LayerNorm`.

| Output | Path |
|---|---|
| Config | `configs/sgp_e13_no_htdma_30min_temporal_gru_64_bottleneck_vae_no_sizing_crosstalk_earlymask.yaml` |
| Selected checkpoint | `artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask/checkpoint.pt` |
| Loss curve | `artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask/loss_curve.png` |
| Network summary | `artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask/network_summary.txt` |
| Skill scores | `artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask/cross_prediction_test/test_leave_one_out.csv` |
| Selected latent encodings | `artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask/encodings_selected/encodings.csv` |

## Deterministic Explicit-Crosstalk Baseline

Build the feature store:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.build_features \
  --config configs/sgp_e13_no_htdma_long_diameter_aware_32_bottleneck_96wide_3layer_long_crosstalk_transformer_autoencoder.yaml \
  --start 2022-06-07 \
  --end 2022-06-20 \
  --output artifacts/clean_transformer_autoencoder_20220607_20220620/features
```

Train the explicit sizing-crosstalk model:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_long_diameter_aware_32_bottleneck_96wide_3layer_long_crosstalk_transformer_autoencoder.yaml \
  --features artifacts/clean_transformer_autoencoder_20220607_20220620/features/features.npz \
  --output artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run
```

Generate plots and skill scores:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.plot_training \
  --history artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/history.csv \
  --output artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/loss_curve.png \
  --title "32-D 96-wide 3-layer sizing-crosstalk bottleneck transformer aerosol encoder, full run"

/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.evaluate_cross_prediction \
  --features artifacts/clean_transformer_autoencoder_20220607_20220620/features/features.npz \
  --checkpoint artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/checkpoint.pt \
  --output artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/cross_prediction_test \
  --split test
```

Generate schematics:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.plot_graphviz_architecture \
  --checkpoint artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/checkpoint.pt \
  --output-dir docs/figures \
  --prefix aerosol_encoder_32_96wide_3layer_full_main_crosstalk_graphviz
```

Export the learned aerosol encoding:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.export_encodings \
  --features artifacts/clean_transformer_autoencoder_20220607_20220620/features/features.npz \
  --checkpoint artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/checkpoint.pt \
  --output artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/encodings
```

## Main Outputs

| Output | Path |
|---|---|
| Method note | `docs/method_paper.md` |
| Architecture schematic | `docs/figures/aerosol_encoder_32_96wide_3layer_full_main_crosstalk_graphviz_overview.png` |
| Sizing crosstalk schematic | `docs/figures/aerosol_encoder_32_96wide_3layer_full_main_crosstalk_graphviz_sizing_crosstalk.png` |
| Checkpoint | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/checkpoint.pt` |
| Network summary | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/network_summary.txt` |
| Loss curve | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/loss_curve.png` |
| Selected test metrics | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/test_metrics.json` |
| Strict diagnostic metrics | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/run/test_metrics_strict_group_out.json` |
| Skill plot | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/cross_prediction_test/leave_one_out_skill.png` |
| Pairwise skill heatmap | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/cross_prediction_test/pairwise_cross_prediction_skill.png` |
| Latent encodings CSV | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/encodings/encodings.csv` |
| Latent encodings NPZ | `artifacts/bottleneck32_96wide_3layer_long_main_crosstalk_transformer_autoencoder_full_20220607_20220620/encodings/encodings.npz` |

## 30-Minute Temporal-GRU Pilot

Build the temporal feature store:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.build_features \
  --config configs/sgp_e13_no_htdma_30min_temporal_gru_32_bottleneck.yaml \
  --start 2022-06-07 \
  --end 2022-06-20 \
  --output artifacts/temporal_gru_30min_20220607_20220620/features
```

Run a stage-preserving pilot:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_gru_32_bottleneck.yaml \
  --features artifacts/temporal_gru_30min_20220607_20220620/features/features.npz \
  --output artifacts/temporal_gru_30min_20220607_20220620/run_staged30_std001 \
  --max-epochs 30
```

Pilot outputs:

The two schematics have different purposes. The temporal architecture schematic is the full model path. The sizing-crosstalk schematic is a zoom-in diagnostic for the sizing-mask tests; it is not a separate training network.

| Output | Path |
|---|---|
| Loss curve | `artifacts/temporal_gru_30min_20220607_20220620/run_staged30_std001/loss_curve.png` |
| Network summary | `artifacts/temporal_gru_30min_20220607_20220620/run_staged30_std001/network_summary.txt` |
| Temporal architecture schematic | `docs/figures/aerosol_encoder_temporal_gru_30min_std001_graphviz_overview.png` |
| Temporal sizing-crosstalk schematic | `docs/figures/aerosol_encoder_temporal_gru_30min_std001_graphviz_sizing_crosstalk.png` |
| Test metrics | `artifacts/temporal_gru_30min_20220607_20220620/run_staged30_std001/test_metrics.json` |
| Skill scores | `artifacts/temporal_gru_30min_20220607_20220620/run_staged30_std001/cross_prediction_test/test_leave_one_out.csv` |
