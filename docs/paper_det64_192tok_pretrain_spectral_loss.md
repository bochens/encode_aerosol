# Multimodal Aerosol State Encoding with Per-Instrument Token Pretraining and Size-Spectral Losses

This note documents one completed training run only:

`artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps`

The run trains a deterministic multimodal transformer autoencoder for ARM SGP E13 aerosol observations. Each instrument family is encoded as a separate modality token. The global aerosol state is a 64-dimensional bottleneck vector `z`, selected by leave-one-modality-out validation.

## Abstract

The goal is to learn a compact aerosol representation that can be used to reconstruct or cross-predict aerosol measurements from different instruments. The model uses 30-minute feature windows from AOSMET, ACSM-CDCE, SMPS, APS, UHSAS, OPC, CPC, CCN, and dry/wet nephelometer measurements. Each modality is first encoded to a 192-dimensional token, visible tokens are fused by a 3-layer transformer, and a learned latent-query token is projected to a deterministic 64-dimensional aerosol encoding. Decoders expand the 64-dimensional encoding back to 192 dimensions before predicting each target instrument family.

The best checkpoint was selected at epoch 190, before the final group-out phase ended. On the held-out test split, the selected checkpoint reached leave-one-out cross-prediction MSE `0.4043` and strict sizing-group-out MSE `0.4198`, both in standardized feature units. All target families have positive skill relative to a training-mean baseline.

## Data

Data are from ARM Southern Great Plains E13, excluding HTDMA. The feature matrix uses 30-minute rows from `2016-11-29 00:00` to `2023-04-21 23:30`.

| Split | Rows |
|---|---:|
| Train | 78,573 |
| Validation | 16,031 |
| Test | 17,426 |
| Total | 112,030 |

The selected training array contains 7,897 standardized features. Missing observations are carried explicitly as feature masks; missing values are not treated as zeros.

| Modality | Role | Feature dimensions | Temporal shape |
|---|---|---:|---|
| AOSMET | Context, always visible when observed | 420 | 30 x 14 |
| ACSM-CDCE | Target | 7 | 1 x 7 |
| SMPS | Target | 450 | 6 x 75 |
| APS | Target | 1,980 | 30 x 66 |
| UHSAS | Target | 1,590 | 30 x 53 |
| OPC | Target | 2,220 | 30 x 74 |
| CPC | Target | 240 | 30 x 8 |
| CCN | Target | 180 | 30 x 6 |
| Dry/wet neph | Target | 810 | 30 x 27 |

The sizing instruments are not merged into a single input. SMPS, APS, UHSAS, and OPC each retain their own modality identity. They share a log-diameter representation in the feature construction, but the network is still allowed to learn that these instruments represent different measurement physics.

## Architecture

![Architecture overview](figures/det64_192tok_pretrain_spectral_architecture_overview.png)

The model is a deterministic structured transformer autoencoder:

| Component | Setting |
|---|---:|
| Modality token width | 192 |
| Aerosol bottleneck dimension | 64 |
| Global transformer | 3 layers |
| Attention heads | 8 |
| Decoder expansion depth | 4 linear layers |
| VAE | no |
| Dedicated sizing-only crosstalk subnet | no |
| Trainable parameters | 7,286,166 |

The forward path is:

1. Each observed instrument family is encoded into one 192-dimensional modality token.
2. A learned 192-dimensional latent-query token is appended to the visible modality tokens.
3. A key-padding mask removes hidden modalities from the transformer's evidence.
4. The global transformer lets visible instrument tokens exchange information.
5. The output row corresponding to the latent query is projected to the 64-dimensional aerosol encoding `z`.
6. A 4-layer decoder expansion maps `z` back to a 192-dimensional decoder state.
7. Target-specific decoders predict ACSM, SMPS, APS, UHSAS, OPC, CPC, CCN, and neph features.

The 64-dimensional `z` is the actual bottleneck. The 192-dimensional token width and decoder expansion are not the aerosol encoding; they are internal workspace dimensions that let the network combine and decode information around the bottleneck.

## What Group-Out Training Means

![Sizing masking schematic](figures/det64_192tok_pretrain_spectral_architecture_sizing_masking.png)

Group-out training is a stricter version of leave-one-out masking.

In normal leave-one-out training, one target modality is hidden from the input and the model must predict it from the remaining visible modalities. For example, if SMPS is hidden, APS, UHSAS, and OPC may still be visible. This answers: "Can the model infer SMPS using all other instruments, including other sizing instruments?"

In group-out training, target modalities can belong to an exclusion group. In this run the exclusion group is:

`[size_smps, size_aps, size_uhsas, size_opc]`

When the sampled hidden target is a sizing instrument, the entire sizing group is removed from the input. For example:

| Target being scored | Inputs removed | What this tests |
|---|---|---|
| SMPS | SMPS + APS + UHSAS + OPC | Can non-sizing instruments infer SMPS-like information? |
| APS | SMPS + APS + UHSAS + OPC | Can non-sizing instruments infer coarse/aerodynamic sizing? |
| UHSAS | SMPS + APS + UHSAS + OPC | Can non-sizing instruments infer optical accumulation sizing? |
| OPC | SMPS + APS + UHSAS + OPC | Can non-sizing instruments infer OPC-like size information? |
| ACSM, CPC, CCN, neph | only the target itself | Same as normal leave-one-out |

This does not add a special crosstalk network. It changes the masking task. During group-out, the global transformer still has the same architecture; it simply receives fewer input tokens for sizing-target rows.

The reason for this diagnostic is scientific, not just machine-learning hygiene. If SMPS is predicted with APS/UHSAS/OPC visible, the model may be using overlapping sizing measurements. That is useful, because cross-instrument sizing crosstalk contains physical information. But it does not answer whether non-sizing instruments contain enough information to infer size. Group-out tests that harder question.

## Training Objective

The main reconstruction term is masked standardized MSE on observed target features. During hidden-only phases, loss is applied only to targets hidden from the input.

Additional losses are included:

| Loss term | Purpose |
|---|---|
| Per-instrument denoising pretraining | Force each modality encoder token to carry useful information before multimodal fusion training |
| Dry scattering closure | Encourage size/composition information to predict dry neph scattering proxies |
| Humidification response closure | Encourage dry/wet neph consistency |
| CCN activation-ratio closure | Encourage CCN consistency with aerosol state |
| Size log-spectrum loss | Penalize errors in log-transformed size spectra |
| Size moment loss | Penalize errors in number/surface/volume-like spectral moments |
| Size shape loss | Penalize normalized spectral-shape errors independent of total loading |
| Latent L2 | Weak regularization on the bottleneck magnitude |

The configured stage schedule was:

| Epochs | Stage | Masking / objective |
|---:|---|---|
| 1-30 | Instrument denoising token pretraining | Decode each instrument from its own noisy token |
| 31-50 | Deterministic autoencode warmup | Reconstruct visible targets |
| 51-70 | Denoising autoencode | Feature dropout and Gaussian noise |
| 71-120 | Mild random mask | Hidden-only cross-prediction with 20% input masking |
| 121-200 | Leave-one-out | Hide one target modality |
| 201-250 | Group-out | Hide all sizing instruments when scoring a sizing target |

## Training Result

![Training curve](figures/det64_192tok_pretrain_spectral_training.png)

The best validation point occurred at epoch 190, during the leave-one-out phase.

| Checkpoint | Epoch | Validation selected leave-one-out MSE | Validation strict group-out MSE |
|---|---:|---:|---:|
| Best selected checkpoint | 190 | 0.3169 | not evaluated at this epoch |
| End of leave-one-out phase | 200 | 0.3201 | 0.3359 |
| Final checkpoint | 250 | 0.3293 | 0.3392 |

The final group-out phase made the training task harder and slightly degraded normal leave-one-out validation. Therefore, the selected model for test evaluation is `checkpoint.pt` from epoch 190, not `last_checkpoint.pt` from epoch 250.

## Test Cross-Prediction

The selected epoch-190 checkpoint was evaluated on the held-out test split.

| Evaluation mode | Test standardized MSE |
|---|---:|
| Leave one modality out | 0.4043 |
| Strict sizing group out | 0.4198 |

Per-target leave-one-out test MSE:

| Target | MSE |
|---|---:|
| CCN activation | 0.0935 |
| SMPS | 0.2459 |
| APS | 0.2650 |
| CPC number | 0.3322 |
| Optical neph | 0.4334 |
| ACSM chemistry | 0.4631 |
| UHSAS | 0.6103 |
| OPC | 0.7912 |

The OPC score should be interpreted carefully because the test split only has 1,697 valid OPC rows, much less than most other targets.

## Skill Scores

Skill is defined as:

`skill = 1 - model_MSE / training_mean_baseline_MSE`

A value of `0` means the model only matches the training-mean baseline. A positive value means the model does better than predicting the training mean. All target families have positive skill in this run.

![Leave-one-out skill](figures/det64_192tok_pretrain_spectral_leave_one_out_skill.png)

| Target / case | Skill |
|---|---:|
| SMPS, leave-one-out | 0.750 |
| CCN activation | 0.742 |
| SMPS, strict sizing group-out | 0.722 |
| CPC number | 0.687 |
| ACSM chemistry | 0.633 |
| OPC, leave-one-out | 0.590 |
| Optical neph | 0.558 |
| UHSAS, strict sizing group-out | 0.529 |
| OPC, strict sizing group-out | 0.529 |
| UHSAS, leave-one-out | 0.524 |
| APS, strict sizing group-out | 0.503 |
| APS, leave-one-out | 0.473 |

![Leave-one-out MSE](figures/det64_192tok_pretrain_spectral_leave_one_out_mse.png)

![Pairwise skill](figures/det64_192tok_pretrain_spectral_pairwise_skill.png)

## Stratified Test Behavior

The same selected checkpoint was evaluated by year, season, and simple proxy regime. The values below average per-target leave-one-out MSE across target modalities, so they are diagnostic rather than final physical scores.

| Stratification | Stratum | Rows | Mean MSE | Mean skill |
|---|---|---:|---:|---:|
| Season | DJF | 5,664 | 0.469 | 0.575 |
| Season | MAM | 4,378 | 0.324 | 0.581 |
| Season | JJA | 2,920 | 0.421 | 0.643 |
| Season | SON | 4,464 | 0.241 | 0.688 |
| Regime | low_loading | 2,370 | 0.546 | 0.611 |
| Regime | mass_optics_dominated | 2,629 | 0.326 | 0.690 |
| Regime | mixed_high | 3,354 | 0.600 | 0.650 |
| Regime | moderate | 4,800 | 0.235 | 0.553 |
| Regime | number_dominated | 4,041 | 0.269 | 0.617 |
| Regime | unclassified | 232 | 2.792 | -0.286 |

The unclassified proxy-regime bin is small and performs poorly; it should not be treated as a robust aerosol regime without inspecting the underlying rows.

## Bottleneck PCA

The 64-dimensional bottleneck was encoded for all 17,426 test rows with all observed modalities visible. PCA was fit to standardized bottleneck dimensions.

![Latent PCA by proxy regime](figures/det64_192tok_pretrain_spectral_latent_pca_proxy_regime.png)

![Latent PCA by season](figures/det64_192tok_pretrain_spectral_latent_pca_season.png)

The first three PCs explain 38.6% of standardized latent variance:

| PC | Explained variance |
|---|---:|
| PC1 | 0.167 |
| PC2 | 0.126 |
| PC3 | 0.093 |

Cluster diagnostics:

| Diagnostic | Value |
|---|---:|
| k-means k=3 silhouette on PC1-PC3 | 0.361 |
| Season silhouette on PC1-PC3 | -0.069 |
| Proxy-regime silhouette on PC1-PC3 | -0.137 |

The latent space therefore looks more like continuous aerosol gradients than cleanly separated seasonal or proxy-regime clusters. That is not a failure; atmospheric aerosol state is expected to vary continuously. The positive k-means silhouette means the geometry has structure, but the simple labels used here are too crude to define clean clusters.

![Latent PCA pair grid](figures/det64_192tok_pretrain_spectral_latent_pca_pair_grid_proxy_regime.png)

## Interpretation

This run is a useful deterministic baseline for multimodal aerosol encoding:

1. The 64-dimensional bottleneck is strong enough to support positive cross-prediction skill for every instrument family.
2. Per-instrument denoising pretraining appears useful because the model reaches good validation performance after the masked phases begin.
3. The model performs especially well for CCN activation and SMPS.
4. UHSAS and OPC are harder, likely because optical sizing is instrument-specific and OPC overlap is short.
5. Group-out training did not improve the selected leave-one-out validation objective after epoch 190. It should be treated as a robustness diagnostic and possible fine-tuning objective, not automatically as the final model-selection criterion.

## Limitations

This model is still only a statistical representation of co-observed instrument behavior. It does not prove a physically unique aerosol mixing state. The following issues remain important:

1. The validation objective is standardized MSE, so scientifically important rare events may be underweighted.
2. The strict sizing group-out test is much harder and still has high errors for UHSAS and OPC.
3. OPC has limited temporal coverage, so its metrics have higher uncertainty.
4. Proxy regime labels are crude and do not produce clean latent clusters.
5. The model does not yet include a probabilistic latent distribution. A later VAE version may be useful, but this deterministic model should remain the comparison baseline.

## Artifacts

| Artifact | Path |
|---|---|
| Config | `configs/sgp_e13_no_htdma_30min_temporal_pretrain_192tok_64bottleneck_spectral_loss.yaml` |
| Best checkpoint | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/checkpoint.pt` |
| Final checkpoint | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/last_checkpoint.pt` |
| Training history | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/history.csv` |
| Test metrics | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/test_metrics.json` |
| Strict group-out test metrics | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/test_metrics_strict_group_out.json` |
| Cross-prediction skill table | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/cross_prediction_test/test_leave_one_out.csv` |
| Pairwise skill table | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/cross_prediction_test/test_pairwise.csv` |
| Stratified test table | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/stratified_test/test_stratified_leave_one_out.csv` |
| Latent PCA table | `artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/latent_pca_test/test_latent_pca.csv` |
| Network summary | `docs/det64_192tok_pretrain_spectral_network_summary.txt` |

## Reproduction

Training command:

```bash
python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_pretrain_192tok_64bottleneck_spectral_loss.yaml \
  --features artifacts/temporal_gru_30min_20161129_20230421/features/features.npz \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --output artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps \
  --device mps
```

Resume command:

```bash
python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_pretrain_192tok_64bottleneck_spectral_loss.yaml \
  --features artifacts/temporal_gru_30min_20161129_20230421/features/features.npz \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --output artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps \
  --device mps \
  --resume-checkpoint artifacts/temporal_gru_30min_20161129_20230421/run_det64_192tok_pretrain_spectral_mps/last_checkpoint.pt
```

The run stores both `checkpoint.pt` and `last_checkpoint.pt`. `checkpoint.pt` is the best validation checkpoint. `last_checkpoint.pt` is the most recent checkpoint and is used for resuming interrupted training. This follows the standard PyTorch checkpoint pattern of storing model state, optimizer state, scheduler state, epoch, and history.

## Reference

PyTorch documentation on saving and loading general training checkpoints: <https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html>
