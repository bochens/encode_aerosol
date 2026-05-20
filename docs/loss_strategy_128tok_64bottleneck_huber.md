# Loss Strategy: 128-Token, 64-Bottleneck Coordinate Model With Huber Loss

This setup keeps the model deterministic and changes the objective before trying VAE regularization.

## Base Response Loss

The base reconstruction and masked cross-prediction training loss is no longer plain squared error.

```yaml
response_loss:
  kind: huber
  huber_delta: 1.0
```

The loss is applied in standardized feature space after target masking:

1. Coordinates and diagnostics are removed from the target loss.
2. Missing features are masked.
3. In masked stages, only the hidden target modality contributes.
4. A per-modality loss is computed.
5. Modality losses are averaged, so large spectra do not automatically dominate small modalities.

Validation cross-prediction is still reported as standard MSE, so the curves remain comparable to earlier experiments.

## Physical Auxiliary Losses

The same robust regression loss is also used inside the physical auxiliary objectives:

- Size log-spectrum loss.
- Size moment loss: number, surface, and volume moments.
- Size shape loss: normalized spectral shape.
- Dry scattering closure.
- Dry/wet nephelometer humidification response.
- CCN/CPC activation-ratio consistency.

The size losses are still computed in physically meaningful spaces:

- spectra in transformed log-concentration space,
- moments from inverse-transformed nonnegative concentration,
- shape from the normalized size distribution.

## Why Not Plain MSE

Plain MSE is useful as a validation metric, but it is a harsh training objective for these data because aerosol instruments have spikes, dropouts, calibration periods, and regime jumps. Squared error lets rare large residuals dominate a gradient step. Huber loss with `delta=1.0` behaves quadratically for errors below roughly one standardized feature unit and linearly for larger errors, so the model still improves accurate predictions without letting spikes control the training.

## Current Training Emphasis

This run does not use VAE and does not train straight group-out. It uses:

- per-instrument denoising token pretraining,
- deterministic autoencoding warmup,
- denoising autoencoding,
- long leave-one-modality-out masking stages.

The strict all-sizing-hidden test can still be run as a diagnostic later, but it is not the main training task in this setup.
