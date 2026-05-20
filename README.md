# Encode Aerosol

Research code for learning compact multimodal atmospheric aerosol representations
from ARM Southern Great Plains E13 aerosol and AOS meteorology measurements.

The main model is a deterministic multimodal transformer autoencoder. It maps a
30-minute aerosol state into a 64-dimensional bottleneck and retrieves measured
instrument responses, including size spectra as functions of diameter and CCN
concentration as a function of supersaturation.

## Current Reference Run

The current reference method note is:

`docs/paper_det64_128tok_coordinate_final.md`

Reference checkpoint directory used in that note:

`artifacts/temporal_gru_30min_20161129_20230421/run_det64_128tok_huber_mask1_coordinate_mps_reset_20260513`

`artifacts/` is intentionally ignored by git. Trained weights, generated feature
arrays, figures, and prediction tables are not included in the repository.

## Data

The project expects ARM SGP E13 NetCDF files downloaded separately. Config files
use this portable placeholder:

`data/DOE_SGP`

Set that path to your local ARM data directory, or edit `data_root` in the
relevant YAML config.

The current feature audit is:

`docs/data_feature_role_report.md`

## Model Summary

The current reference run uses:

| Item | Value |
|---|---:|
| Data period | 2016-11-29 to 2023-04-21 |
| Sampling | 30 minutes |
| Feature matrix | 112,030 rows x 7,897 features |
| Modality token width | 128 |
| Bottleneck latent | 64 |
| Global fusion | 4-layer, 8-head transformer |
| VAE | no |
| Special sizing-only subnet | no |
| Coordinate decoders | CCN, sizing, optical neph |

The model treats SMPS, APS, UHSAS, and OPC as distinct modalities. Cross-talk
between instruments happens through the global transformer when modality tokens
are visible.

## Key Files

| File | Purpose |
|---|---|
| `aerosol_encoding/train.py` | training entry point |
| `aerosol_encoding/model.py` | multimodal autoencoder and coordinate decoders |
| `aerosol_encoding/prepare_training_arrays.py` | feature-array preparation |
| `aerosol_encoding/infer_ccn.py` | CCN inference CLI |
| `aerosol_encoding/plot_latent_pca.py` | latent PCA diagnostics |
| `kappa_ccn_baseline/` | kappa-Kohler and random-forest CCN baselines |
| `configs/sgp_e13_no_htdma_30min_temporal_pretrain_128tok_64bottleneck_huber_mask1_coordinate.yaml` | current reference config |
| `docs/paper_det64_128tok_coordinate_final.md` | current method paper and result summary |

## Installation

Create a Python environment with PyTorch, NumPy, pandas, scikit-learn, xarray,
netCDF4, matplotlib, PyYAML, and joblib. On Apple Silicon, use a PyTorch build
with MPS support for local GPU training.

Example:

```bash
python -m pip install numpy pandas scipy scikit-learn xarray netCDF4 matplotlib pyyaml joblib torch
```

## Reproduce Feature Arrays

```bash
python -m aerosol_encoding.prepare_training_arrays \
  --config configs/sgp_e13_no_htdma_30min_temporal_pretrain_128tok_64bottleneck_huber_mask1_coordinate.yaml \
  --output artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz
```

## Train

```bash
python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_pretrain_128tok_64bottleneck_huber_mask1_coordinate.yaml \
  --features artifacts/temporal_gru_30min_20161129_20230421/features/features.npz \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --output artifacts/temporal_gru_30min_20161129_20230421/run_det64_128tok_huber_mask1_coordinate_mps_reset_20260513 \
  --device mps
```

Use `--device cuda` on CUDA systems or `--device cpu` for small checks.

## Evaluate Baselines

Kappa-Kohler CCN baseline:

```bash
python -m kappa_ccn_baseline.run_baseline \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --split test \
  --fraction-basis mass \
  --output artifacts/kappa_ccn_baseline
```

Random-forest CCN baseline:

```bash
python -m kappa_ccn_baseline.random_forest_baseline \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --train-split train \
  --eval-split test \
  --max-train-samples 100000 \
  --n-estimators 100 \
  --min-samples-leaf 5 \
  --output artifacts/random_forest_ccn_baseline
```

## License

No open-source license has been selected yet. Add a license before making the
GitHub repository public.
