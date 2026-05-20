# Deterministic Multimodal Aerosol Encoding at ARM SGP

## Abstract

This run trains a deterministic multimodal transformer autoencoder to represent ambient aerosol state from ARM Southern Great Plains E13 measurements. The model uses 30-minute feature windows from 2016-11-29 through 2023-04-21. Each instrument family is treated as a separate modality, visible modalities are fused by a global transformer, and the fused state is compressed into a 64-dimensional aerosol encoding before all target instruments are decoded.

The current model is deterministic. SMPS, APS, UHSAS, and OPC are separate modality tokens, and information exchange among all instruments occurs through the global transformer.

The best checkpoint is epoch 150. Validation leave-one-out standardized MSE improved from 0.349 at epoch 120 to 0.330 at epoch 150. Held-out test leave-one-out MSE is 0.429, and strict all-sizing-hidden test MSE is 0.436.

## Data

| Item | Value |
|---|---:|
| Site/facility | ARM SGP E13 |
| Period | 2016-11-29 00:00 to 2023-04-21 23:30 |
| Time grid | 30 minutes |
| Raw feature matrix | 112,080 rows x 17,665 features |
| Selected matrix | 112,030 rows x 7,897 features |
| Train / validation / test rows | 78,573 / 16,031 / 17,426 |
| Split | deterministic calendar-month hash |

Selected dimensions by modality:

| Modality | Features | Role |
|---|---:|---|
| `met_context` | 420 | always-visible context |
| `chemistry_acsm` | 7 | target |
| `size_smps` | 450 | target |
| `size_aps` | 1,980 | target |
| `size_uhsas` | 1,590 | target |
| `size_opc` | 2,220 | target |
| `cpc_number` | 240 | target |
| `ccn_activation` | 180 | target |
| `optical_neph` | 810 | target |

## Network

![Current architecture](figures/current_architecture_tikz.png)

The architecture is:

1. Modality-specific structured encoders convert each observed instrument family into one 96-dimensional token.
2. A learned 96-dimensional query token is appended to the visible modality tokens. This query is a trained model parameter, not an instrument measurement.
3. A global transformer encoder, 3 layers and 4 heads, fuses the visible tokens.
4. The learned-query output row is projected to a 64-dimensional deterministic aerosol encoding.
5. A 4-layer decoder expansion maps 64 dimensions back to 96.
6. Target heads decode ACSM, SMPS, APS, UHSAS, OPC, CPC, CCN, and neph features.

The 64-dimensional vector is the bottleneck. The decoder does not receive the original 7,897-dimensional feature vector.

## Training

All features are standardized using the training split. The loss is observed-feature standardized MSE.

Training schedule:

| Epochs | Stage | Purpose |
|---:|---|---|
| 1-10 | autoencode warmup | learn basic reconstruction |
| 11-20 | denoise autoencode | stabilize against small feature dropout/noise |
| 21-50 | mild random mask | begin hidden-modality prediction |
| 51-90 | leave-one-out | predict one hidden target modality |
| 91-120 | all-sizing-hidden strict mask | train harder sizing diagnostic cases |
| 121-180 | extra leave-one-out | continue with validation every 10 epochs |
| 181-210 | extra strict mask | stopped because validation degraded |

During masked stages, the loss is applied only to hidden target modalities. This prevents the model from being rewarded for copying visible targets.

Soft closure terms were retained with small weights:

| Closure term | Weight |
|---|---:|
| size/composition to dry scattering | 0.03 |
| dry/wet neph humidification response | 0.03 |
| CCN activation ratio consistency | 0.02 |

## Validation And Test Metrics

![Validation and test MSE](figures/validation_and_test_mse.png)

Validation points:

| Epoch | Stage | Leave-one-out MSE | Strict MSE |
|---:|---|---:|---:|
| 1 | autoencode warmup | 0.522 | 0.544 |
| 100 | strict mask | 0.353 | not evaluated |
| 120 | strict mask | 0.349 | 0.353 |
| 130 | extra leave-one-out | 0.338 | not evaluated |
| 140 | extra leave-one-out | 0.334 | not evaluated |
| 150 | extra leave-one-out | **0.330** | **0.339** |
| 160 | extra leave-one-out | 0.333 | not evaluated |
| 170 | extra leave-one-out | 0.335 | not evaluated |
| 180 | extra leave-one-out | 0.339 | 0.347 |
| 190 | extra strict mask | 0.337 | not evaluated |
| 200 | extra strict mask | 0.352 | not evaluated |
| 210 | extra strict mask | 0.347 | 0.353 |

The best checkpoint is epoch 150. The later strict-mask continuation did not improve the selected validation metric, so training was stopped at epoch 210.

Held-out test metrics at the epoch-150 checkpoint:

| Test mode | Mean standardized MSE |
|---|---:|
| leave-one-out | 0.429 |
| strict all-sizing-hidden | 0.436 |

## Skill

Skill is:

`skill = 1 - model_MSE / training_mean_baseline_MSE`

Skill above 0 means the model beats predicting the training mean.

![Leave-one-out skill](figures/leave_one_out_skill.png)

Per-target leave-one-out test skill:

| Target | Test MSE | Mean baseline MSE | Skill |
|---|---:|---:|---:|
| `ccn_activation` | 0.094 | 0.362 | 0.739 |
| `size_smps` | 0.280 | 0.983 | 0.715 |
| `cpc_number` | 0.338 | 1.062 | 0.682 |
| `chemistry_acsm` | 0.509 | 1.263 | 0.597 |
| `size_opc` | 0.812 | 1.929 | 0.579 |
| `optical_neph` | 0.439 | 0.980 | 0.552 |
| `size_uhsas` | 0.666 | 1.283 | 0.481 |
| `size_aps` | 0.293 | 0.503 | 0.417 |

Strict all-sizing-hidden skill for sizing targets:

| Target | Strict MSE | Mean baseline MSE | Skill |
|---|---:|---:|---:|
| `size_smps` | 0.303 | 0.983 | 0.692 |
| `size_aps` | 0.272 | 0.503 | 0.458 |
| `size_uhsas` | 0.627 | 1.283 | 0.511 |
| `size_opc` | 0.906 | 1.929 | 0.531 |

## Bottleneck PCA Diagnostic

The epoch-150 checkpoint was used to encode the held-out test split with all observed modalities visible. PCA was then fit to the standardized 64-dimensional bottleneck coordinates. The first three PCs explain 42.5% of the standardized latent variance:

| PC | Explained variance |
|---|---:|
| PC1 | 17.4% |
| PC2 | 14.7% |
| PC3 | 10.4% |

![Test latent PCA by season](figures/test_latent_pca_3d_season.png)

![Test latent PCA by proxy regime](figures/test_latent_pca_3d_proxy_regime.png)

![Test latent PCA seasonal pair grid](figures/test_latent_pca_pair_grid_season.png)

![Test latent PCA proxy-regime pair grid](figures/test_latent_pca_pair_grid_proxy_regime.png)

The latent space is better described as a set of continuous gradients than as clean discrete clusters. Season has weak separation in the first three PCs, with a season silhouette score of -0.016. Simple proxy aerosol-regime labels also overlap strongly, with a proxy-regime silhouette score of -0.088. K-means on PC1-PC3 gives moderate geometric grouping, with silhouette 0.296 for k=3, but those groups are not the same as the simple seasonal or proxy-regime labels.

The strongest rank correlations between PCs and proxy variables are:

| Axis | Strongest proxy correlations |
|---|---|
| PC1 | dry scattering -0.82, organics -0.72, sulfate -0.69, UHSAS accumulation -0.68 |
| PC2 | SMPS number -0.69, CPC number -0.51, organics -0.58 |
| PC3 | CPC number -0.58, SMPS number -0.46, APS coarse +0.27 |

These signs are arbitrary because PCA axes can flip, but the magnitudes show that the bottleneck is not random. The first few latent axes are strongly tied to optical loading, chemical loading, and size/number proxies.

## Interpretation

The deterministic 64-dimensional bottleneck is restrictive relative to the 7,897 selected input features, but still preserves useful cross-instrument information. The extra leave-one-out continuation improved validation until epoch 150. The later stricter continuation did not improve the selected validation metric.

The strongest held-out skills are CCN activation, SMPS, CPC number, and ACSM chemistry. UHSAS and APS are harder, which is consistent with their different size ranges and measurement physics. OPC remains positive but benefits less when all other sizing instruments are hidden.

The bottleneck PCA supports the same interpretation: the model is learning continuous aerosol-state coordinates rather than hard categories. That is scientifically reasonable for ambient aerosol because source, aging, humidity, and instrument response tend to vary continuously.

## Limitations

This is not yet a final aerosol state representation.

1. The result is from one site/facility period.
2. Instrument missingness is structured by deployment and maintenance, not random.
3. The strict all-sizing-hidden test is a statistical diagnostic, not a causal physical closure.
4. The current model is deterministic and does not express uncertainty over multiple possible aerosol states.
5. Closure losses are weak consistency terms, not a replacement for explicit optical or hygroscopic growth calculations.

## Reproducibility

Run directory:

`artifacts/temporal_gru_30min_20161129_20230421/run_det64_no_vae_70_15_15_mps`

Important files:

| File | Meaning |
|---|---|
| `checkpoint.pt` | best checkpoint, epoch 150; trained weights are in `model_state` |
| `last_checkpoint.pt` | last checkpoint, epoch 210 |
| `history.csv` | training and validation history |
| `test_metrics.json` | leave-one-out test metrics |
| `test_metrics_strict_group_out.json` | strict test metrics |
| `cross_prediction_test/test_leave_one_out.csv` | per-target skill table |
| `latent_pca_test/test_latent_pca.csv` | held-out bottleneck PCA coordinates, labels, proxies, and 64-D z values |
| `latent_pca_test/test_latent_pca_summary.json` | PCA explained variance and clustering diagnostics |

The best checkpoint contains 1,777,949 trained parameters in the PyTorch state dictionary. It also stores normalization statistics, selected feature names, modality dimensions, split indices, optimizer state, scheduler state, and training history.

Continuation command:

```bash
python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_gru_64_bottleneck_no_vae_long_e13_70_15_15_extend240_val10.yaml \
  --features artifacts/temporal_gru_30min_20161129_20230421/features/features.npz \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --output artifacts/temporal_gru_30min_20161129_20230421/run_det64_no_vae_70_15_15_mps \
  --resume-checkpoint artifacts/temporal_gru_30min_20161129_20230421/run_det64_no_vae_70_15_15_mps/last_checkpoint.pt \
  --resume-with-fresh-optimizer \
  --device mps
```

Latent PCA command:

```bash
python -m aerosol_encoding.plot_latent_pca \
  --checkpoint artifacts/temporal_gru_30min_20161129_20230421/run_det64_no_vae_70_15_15_mps/checkpoint.pt \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --output artifacts/temporal_gru_30min_20161129_20230421/run_det64_no_vae_70_15_15_mps/latent_pca_test \
  --split test \
  --device cpu
```
