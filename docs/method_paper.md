# Variational Multimodal Aerosol Encoder Diagnostics for ARM SGP AOS Measurements

## Abstract

This note documents the current VAE bottleneck experiments. The completed reference run is:

```text
configs/sgp_e13_no_htdma_30min_temporal_gru_32_bottleneck_vae_no_sizing_crosstalk_grokking3600.yaml
artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600
```

The goal was to learn a compact aerosol state representation from multiple ARM SGP aerosol instrument families. The model uses separate temporal encoders for chemistry, sizing, number, CCN, optical scattering, and local AOS meteorology. These modality tokens are fused by a global transformer and compressed into a 32-dimensional variational latent state. SMPS, APS, UHSAS, and OPC remain separate modalities; there is no special sizing-only crosstalk subnet in this run. Any crosstalk among sizing instruments is learned naturally inside the global transformer.

The result is scientifically useful but negative for the current architecture. The 3600-epoch run did not show useful cross-prediction grokking. The validation-selected checkpoint remained at epoch 6 with validation cross-prediction MSE `0.524`; the final epoch had worse validation cross-prediction MSE `0.647`. The final checkpoint improved some individual targets, especially CCN, UHSAS, SMPS, and CPC, but degraded ACSM chemistry and dry/wet nephelometer prediction. This says the present VAE bottleneck is not yet a good universal aerosol state representation.

## 64-D Bottleneck Diagnostic

A follow-up diagnostic tested whether the poor VAE behavior was mainly caused by the 32-D bottleneck being too small. The new diagnostic used:

```text
configs/sgp_e13_no_htdma_30min_temporal_gru_64_bottleneck_vae_no_sizing_crosstalk_earlymask.yaml
artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask
```

This model keeps the same 96-D modality-token width, but increases the aerosol bottleneck from 32 to 64 dimensions. The decoder expansion block was also deepened to let latent variables interact before target-specific decoding:

```text
z64
  -> Linear 64 -> 96
  -> GELU
  -> Linear 96 -> 96
  -> GELU
  -> Linear 96 -> 96
  -> GELU
  -> Linear 96 -> 96
  -> LayerNorm
  -> target decoders
```

The first 64-D attempt reused the long 3600-epoch curriculum. It was stopped at epoch 201 because validation cross-prediction degraded during the autoencoding warmup: training loss fell, but validation cross-prediction rose to about `0.7-0.8`. That showed that larger latent capacity alone made the autoencoding overfit faster.

The corrected 64-D diagnostic used a shorter autoencoding period, weaker KL, lower learning rate, and earlier masking. It was stopped at epoch 310 after the mild-mask stage plateaued. The best selected checkpoint was epoch 71:

| Quantity | Value |
|---|---:|
| Selected epoch | 71 |
| Stage | `mixed_mild_mask` |
| Validation selected cross-prediction MSE | 0.528 |
| Validation reconstruction MSE | 0.507 |
| Validation strict all-sizing-hidden MSE | 0.596 |
| Last observed validation selected cross-prediction MSE | 0.579 |

![64-D early-mask diagnostic training curve](../artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask/loss_curve.png)

The 64-D diagnostic improved several test target skills compared with the 32-D selected checkpoint, especially ACSM chemistry, SMPS, UHSAS, CCN, and optical neph. It still failed badly for CPC in the selected checkpoint. The interrupted last checkpoint improved CPC, but it was not selected by validation cross-prediction.

Selected 64-D checkpoint, epoch 71:

| Target | Input case | MSE | Baseline MSE | Skill |
|---|---|---:|---:|---:|
| ACSM chemistry | all other instruments | 0.667 | 1.886 | 0.646 |
| SMPS | all other instruments | 0.590 | 1.537 | 0.616 |
| SMPS | all non-sizing instruments only | 0.694 | 1.537 | 0.549 |
| APS | all other instruments | 1.493 | 1.850 | 0.193 |
| APS | all non-sizing instruments only | 1.664 | 1.850 | 0.100 |
| UHSAS | all other instruments | 0.121 | 0.478 | 0.748 |
| UHSAS | all non-sizing instruments only | 0.182 | 0.478 | 0.620 |
| OPC | all other instruments | 0.237 | 0.716 | 0.669 |
| OPC | all non-sizing instruments only | 0.400 | 0.716 | 0.442 |
| CPC | all other instruments | 1.176 | 0.702 | -0.676 |
| CCN | all other instruments | 0.303 | 1.365 | 0.778 |
| Dry/wet neph | all other instruments | 0.494 | 1.056 | 0.532 |

![64-D selected skill scores](../artifacts/temporal_gru_30min_20220607_20220620/run_vae64_no_sizing_crosstalk_earlymask/cross_prediction_test/leave_one_out_skill.png)

Conclusion from the 64-D diagnostic: the 32-D bottleneck was not the only problem. More latent capacity helps some target families, but the VAE objective and curriculum still do not produce a clearly better shared aerosol state. The next serious run should probably be deterministic 64-D first, with VAE fine-tuning added only after deterministic cross-prediction works.

## Data

The feature store covers a short E13 period:

```text
features: artifacts/temporal_gru_30min_20220607_20220620/features/features.npz
time range: 2022-06-07 00:00 to 2022-06-20 23:30
rows: 672 half-hour samples
split strategy: calendar_day_hash
train / validation / test rows: 328 / 240 / 96
```

The model does not use HTDMA. AOSMET is used as always-visible context. All other modalities are prediction targets and can be masked during training/evaluation.

| Modality | Role | Temporal representation | Encoded features |
|---|---|---:|---:|
| `met_context` | context | `30 x 12` | 360 |
| `chemistry_acsm` | target | `1 x 7` | 7 |
| `size_smps` | target | `6 x 75` | 450 |
| `size_aps` | target | `30 x 66` | 1980 |
| `size_uhsas` | target | `30 x 53` | 1590 |
| `size_opc` | target | `30 x 75` | 2250 |
| `cpc_number` | target | `30 x 8` | 240 |
| `ccn_activation` | target | `30 x 4` | 120 |
| `optical_neph` | target | `30 x 27` | 810 |

The 30-minute anchor grid was chosen to retain ACSM while preserving faster temporal structure. Within each 30-minute sample, high-frequency instruments keep sub-window time bins before being compressed to one token.

## Preprocessing

Raw variables are transformed and standardized before training. Number, mass, size-distribution, and scattering-like variables use `log1p_nonnegative` transforms. Features with less than `0.005` coverage or training standard deviation below `0.001` are removed before normalization. Missing feature values are represented by masks and are not included in reconstruction loss.

Sizing instruments are mapped onto the shared size-grid configuration:

```text
minimum diameter: 3 nm
maximum diameter: 30000 nm
diameter bins: 160
interpolation: linear in log diameter
```

SMPS, APS, UHSAS, and OPC are not merged into one size distribution. Each instrument keeps its own identity and its own modality token because the measurement physics differ.

## Model

The current model is a structured transformer VAE:

```text
instrument windows
  -> modality-specific temporal encoders
  -> one 96-D token per visible modality
  -> global transformer fusion
  -> learned latent-query readout
  -> Gaussian latent head, mu[32] and logvar[32]
  -> sampled 32-D z during training, mu during evaluation
  -> 3-layer decoder expansion, 32 -> 96
  -> target-specific decoders
```

![Architecture overview](figures/aerosol_encoder_temporal_gru_30min_vae32_no_sizing_crosstalk_grokking3600_graphviz_overview.png)

The global transformer has 3 layers, 4 attention heads, and width 96. Its input token bank contains up to 9 modality tokens plus one learned latent-query token. Hidden instruments are removed from the visible token bank through masking. The learned latent-query token is a trainable readout token: it is not a measurement, but it learns how to collect information from visible instrument tokens after transformer attention.

The VAE head maps the query output through:

```text
LayerNorm(96)
Linear 96 -> 96
GELU
Linear 96 -> 64
split into mu[32] and logvar[32]
```

The decoder expansion block maps the 32-D aerosol state back to decoder width:

```text
Linear 32 -> 96
GELU
Linear 96 -> 96
GELU
Linear 96 -> 96
LayerNorm
```

Total trainable parameters: `1,756,379`.

## Sizing Masking

![Sizing masking schematic](figures/aerosol_encoder_temporal_gru_30min_vae32_no_sizing_crosstalk_grokking3600_graphviz_sizing_masking.png)

There is no explicit sizing-crosstalk transformer in this run. The diagnostic still distinguishes two cases:

| Case | Meaning |
|---|---|
| Single-size hidden | Hide one of SMPS, APS, UHSAS, or OPC; keep the other sizing instruments visible if observed. |
| Strict all-sizing-hidden | Hide SMPS, APS, UHSAS, and OPC together; predict sizing using only non-sizing modalities and meteorology. |

This separation matters scientifically. Single-size hidden tests whether other sizing instruments help recover the hidden sizing instrument. Strict all-sizing-hidden tests how much size information is recoverable from chemistry, CPC, CCN, optical scattering, and meteorology alone.

## Training Objective

Training minimizes observed-feature MSE on the requested target set plus small regularization terms:

```text
loss = masked standardized MSE
     + KL weight * KL(q(z | visible modalities) || N(0, I))
     + small latent L2 penalty
     + closure losses when available
```

The closure losses are weak auxiliary constraints:

| Closure term | Weight |
|---|---:|
| size/composition to dry scattering | 0.03 |
| dry/wet nephelometer humidification response | 0.03 |
| CCN activation ratio | 0.02 |

The VAE KL weight was staged rather than fixed:

| Stage | Epochs | Masking mode | Loss target | KL weight |
|---|---:|---|---|---:|
| `vae_autoencode_warmup` | 540 | autoencode | all observed targets | 0.00002 |
| `vae_denoise_autoencode` | 540 | autoencode with noise/dropout | all observed targets | 0.00005 |
| `mixed_mild_mask` | 900 | random mask, probability 0.20 | hidden targets only | 0.00010 |
| `leave_one_out_hidden_only` | 900 | one target family hidden | hidden target only | 0.00015 |
| `leave_one_group_out_hidden_only` | 720 | sizing group hidden together when applicable | hidden target only | 0.00020 |

The optimizer was AdamW with learning rate `0.0006`, weight decay `0.00001`, batch size `16`, and cosine learning-rate decay to `0.00005`.

## Evaluation

Checkpoint selection uses validation leave-one-out cross-prediction, not final epoch loss. This is important because reconstruction can improve while hidden-modality prediction worsens.

Skill is reported relative to the test-set mean baseline:

```text
skill = 1 - model_MSE / mean_baseline_MSE
```

Interpretation:

| Skill value | Meaning |
|---|---|
| `1` | perfect prediction |
| `0` | same as predicting the test-set mean |
| `> 0` | better than the mean baseline |
| `< 0` | worse than the mean baseline |

All MSE values below are standardized MSE values over observed target features.

## Training Result

![Training curve](../artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/loss_curve.png)

The run completed all 3600 epochs. The selected checkpoint was still epoch 6:

| Quantity | Value |
|---|---:|
| Best validation selected cross-prediction MSE | 0.524 |
| Best validation reconstruction MSE | 0.491 |
| Best validation strict all-sizing-hidden MSE | 0.846 |
| Final validation selected cross-prediction MSE | 0.647 |
| Final validation reconstruction MSE | 0.562 |
| Final validation strict all-sizing-hidden MSE | 0.690 |

This is not a grokking result. Training loss became small, but validation cross-prediction did not make a late improvement. The final checkpoint is useful diagnostically, but the strict validation protocol selects the early checkpoint.

The run was resumed after an intentional stop. The legacy checkpoint at epoch 2160 lacked optimizer and scheduler state, so epoch 2161 was resumed from model weights and history with a fresh optimizer. The training code now writes full resume state, including optimizer, scheduler, epoch, and best-validation values.

## Test Results

Validation-selected checkpoint, epoch 6:

| Target | Input case | MSE | Baseline MSE | Skill |
|---|---|---:|---:|---:|
| ACSM chemistry | all other instruments | 0.774 | 1.886 | 0.589 |
| SMPS | all other instruments | 0.786 | 1.537 | 0.489 |
| SMPS | all non-sizing instruments only | 0.768 | 1.537 | 0.501 |
| APS | all other instruments | 1.458 | 1.850 | 0.212 |
| APS | all non-sizing instruments only | 1.512 | 1.850 | 0.183 |
| UHSAS | all other instruments | 0.167 | 0.478 | 0.652 |
| UHSAS | all non-sizing instruments only | 0.161 | 0.478 | 0.663 |
| OPC | all other instruments | 0.225 | 0.716 | 0.686 |
| OPC | all non-sizing instruments only | 0.266 | 0.716 | 0.628 |
| CPC | all other instruments | 1.167 | 0.702 | -0.664 |
| CCN | all other instruments | 0.457 | 1.365 | 0.665 |
| Dry/wet neph | all other instruments | 0.520 | 1.056 | 0.507 |

![Validation-selected checkpoint skill scores](../artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/cross_prediction_test/leave_one_out_skill.png)

Final checkpoint, epoch 3600:

| Target | Input case | MSE | Baseline MSE | Skill |
|---|---|---:|---:|---:|
| ACSM chemistry | all other instruments | 0.916 | 1.886 | 0.514 |
| SMPS | all other instruments | 0.611 | 1.537 | 0.603 |
| SMPS | all non-sizing instruments only | 0.656 | 1.537 | 0.573 |
| APS | all other instruments | 1.509 | 1.850 | 0.184 |
| APS | all non-sizing instruments only | 1.831 | 1.850 | 0.010 |
| UHSAS | all other instruments | 0.122 | 0.478 | 0.744 |
| UHSAS | all non-sizing instruments only | 0.174 | 0.478 | 0.636 |
| OPC | all other instruments | 0.243 | 0.716 | 0.661 |
| OPC | all non-sizing instruments only | 0.419 | 0.716 | 0.415 |
| CPC | all other instruments | 0.729 | 0.702 | -0.039 |
| CCN | all other instruments | 0.343 | 1.365 | 0.749 |
| Dry/wet neph | all other instruments | 0.726 | 1.056 | 0.312 |

![Final checkpoint skill scores](../artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/cross_prediction_last_test/leave_one_out_skill.png)

The final checkpoint improves CCN, UHSAS, SMPS, and CPC relative to the selected checkpoint, but it damages ACSM chemistry, APS, OPC, and optical neph. Because the selected validation metric worsens, the final checkpoint should not be treated as the main aerosol encoding without additional validation.

## Interpretation

The main lesson is that a simple global VAE bottleneck is not enough. The model can learn useful relationships for some targets, but it does not yet force the 32-D state to be a stable, transferable aerosol representation.

Three failure modes are visible:

1. Small-feature modalities can be overwhelmed. ACSM has 7 encoded features and CPC has 240, while APS and OPC have thousands of encoded values. Equal target-family averaging helps, but the large output decoders still dominate model capacity.
2. The VAE prior is weak and not well aligned with cross-prediction. The final epoch has low training loss, but validation cross-prediction remains worse than the early checkpoint.
3. The same latent must serve reconstruction, single-instrument recovery, all-sizing-hidden recovery, closure constraints, and stochastic regularization. These objectives are not yet balanced.

## Recommended Next Run

The next version should be treated as a redesign, not just a longer run:

1. Start from a deterministic multimodal transformer checkpoint, then add VAE fine-tuning after cross-prediction works.
2. Use modality-balanced or uncertainty-weighted losses so ACSM, CPC, and CCN are not dominated by high-dimensional size outputs.
3. Add CPC-specific derived targets: ultrafine/fine difference, ultrafine/fine ratio, and small-particle closure from SMPS/UHSAS.
4. Keep SMPS, APS, UHSAS, and OPC as separate modalities, but inspect attention maps and pairwise skills to verify physically meaningful crosstalk.
5. Evaluate by year, season, and aerosol regime before calling the latent space an aerosol representation.

## Artifacts

The run can be reproduced from the existing feature store with:

```bash
/Users/C832577250/miniforge3/envs/Research_DL/bin/python -m aerosol_encoding.train \
  --config configs/sgp_e13_no_htdma_30min_temporal_gru_32_bottleneck_vae_no_sizing_crosstalk_grokking3600.yaml \
  --features artifacts/temporal_gru_30min_20220607_20220620/features/features.npz \
  --output artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600
```

| Artifact | Path |
|---|---|
| Config | `configs/sgp_e13_no_htdma_30min_temporal_gru_32_bottleneck_vae_no_sizing_crosstalk_grokking3600.yaml` |
| Selected checkpoint | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/checkpoint.pt` |
| Final checkpoint | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/last_checkpoint.pt` |
| Network summary | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/network_summary.txt` |
| Loss curve | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/loss_curve.png` |
| Selected skill table | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/cross_prediction_test/test_leave_one_out.csv` |
| Final skill table | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/cross_prediction_last_test/test_leave_one_out.csv` |
| Selected latent encodings | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/encodings_selected/encodings.csv` |
| Final latent encodings | `artifacts/temporal_gru_30min_20220607_20220620/run_vae32_no_sizing_crosstalk_grokking3600/encodings_last/encodings.csv` |
