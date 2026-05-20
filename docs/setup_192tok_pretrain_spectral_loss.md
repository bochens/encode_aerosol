# Planned Run: 192-Token Instrument Pretraining With Size Spectral Losses

This setup is the next comparison against the baseline documented in:

`docs/paper_det64_96tok_no_vae_baseline.md`

## Purpose

The baseline used 96-dimensional modality tokens and a 64-dimensional aerosol bottleneck. This planned run keeps the 64-dimensional bottleneck but increases the modality-token width to 192 dimensions, adds per-instrument denoising token pretraining, and adds size-spectrum physical losses.

## Config

`configs/sgp_e13_no_htdma_30min_temporal_pretrain_192tok_64bottleneck_spectral_loss.yaml`

## Main Changes

| Component | Baseline | New setup |
|---|---:|---:|
| Modality token width | 96 | 192 |
| Aerosol bottleneck | 64 | 64 |
| Transformer heads | 4 | 8 |
| Instrument token pretraining | no | yes |
| VAE | no | no |
| Explicit sizing crosstalk block | no | no |
| Size spectral physical losses | no | yes |

## Training Schedule

| Stage | Epochs | Purpose |
|---|---:|---|
| instrument denoising token pretraining | 30 | train each instrument encoder to produce a useful 192-D token before global fusion |
| autoencode warmup | 20 | learn global reconstruction with all observed modalities |
| denoise autoencode | 20 | stabilize against feature dropout and noise |
| mild random mask | 50 | begin hidden-modality prediction |
| leave-one-out | 80 | train standard cross-instrument retrieval |
| leave-one-group-out | 50 | test hard sizing-hidden cases without a special sizing subnet |

Total configured epochs: 250.

## Losses

The model still uses observed-feature standardized MSE for decoded targets. The new size losses add:

| Loss | Weight | Meaning |
|---|---:|---|
| `log_spectrum` | 0.02 | MSE in unstandardized log1p dN/dlogDp space |
| `moment` | 0.03 | log moment loss for number, surface-area proxy, and volume proxy |
| `shape` | 0.02 | normalized spectral-shape loss across diameter bins |

The size spectral losses are computed separately for each available time-bin slice of each sizing instrument.

## Run Command

```bash
python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_pretrain_192tok_64bottleneck_spectral_loss.yaml \
  --features artifacts/temporal_gru_30min_20161129_20230421/features/features.npz \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --output artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps \
  --device mps
```
