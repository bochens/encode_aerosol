# Google Colab training setup

This is the recommended path for long aerosol encoder runs. Build the feature
store locally from the ARM NetCDF files, then train on Colab CUDA using the
compressed feature store.

## What to upload to Google Drive

Create this folder:

```text
MyDrive/encode_aerosol/
```

Upload the project source:

```text
aerosol_encoding/
configs/
docs/
README.md
```

Upload the generated feature store:

```text
artifacts/temporal_gru_30min_20161129_20230421/features/features.npz
artifacts/temporal_gru_30min_20161129_20230421/features/metadata.json
artifacts/temporal_gru_30min_20161129_20230421/features/feature_coverage.csv
```

Do not upload raw ARM NetCDF files to Colab for this run.

## Colab runtime

In Colab, use:

```text
Runtime -> Change runtime type -> GPU
```

T4 is usable; L4/A100 is better if available.

## Notebook commands

Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Copy source and features to Colab local disk:

```python
from pathlib import Path

DRIVE_PROJECT = Path("/content/drive/MyDrive/encode_aerosol")
LOCAL_PROJECT = Path("/content/encode_aerosol")
LOCAL_FEATURE_DIR = Path("/content/aerosol_features/features")

!rm -rf "{LOCAL_PROJECT}" "{LOCAL_FEATURE_DIR.parent}"
!rsync -a --exclude artifacts "{DRIVE_PROJECT}/" "{LOCAL_PROJECT}/"
!mkdir -p "{LOCAL_FEATURE_DIR}"
!cp "{DRIVE_PROJECT}/artifacts/temporal_gru_30min_20161129_20230421/features/features.npz" "{LOCAL_FEATURE_DIR}/features.npz"
!cp "{DRIVE_PROJECT}/artifacts/temporal_gru_30min_20161129_20230421/features/metadata.json" "{LOCAL_FEATURE_DIR}/metadata.json"
!cp "{DRIVE_PROJECT}/artifacts/temporal_gru_30min_20161129_20230421/features/feature_coverage.csv" "{LOCAL_FEATURE_DIR}/feature_coverage.csv"
```

Install only non-Torch dependencies. Colab GPU runtimes normally already include
a CUDA PyTorch build.

```python
%cd /content/encode_aerosol
!python -m pip install -q pyyaml pandas numpy matplotlib scikit-learn
```

Verify CUDA:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
```

Start training:

```python
FEATURES = "/content/aerosol_features/features/features.npz"
CONFIG = "configs/sgp_e13_no_htdma_30min_temporal_gru_64_bottleneck_no_vae_long_e13_70_15_15.yaml"
DRIVE_OUTPUT = "/content/drive/MyDrive/encode_aerosol/artifacts/temporal_gru_30min_20161129_20230421/run_det64_no_vae_70_15_15_colab_cuda"

!python -m aerosol_encoding.train \
  --config "{CONFIG}" \
  --features "{FEATURES}" \
  --output "{DRIVE_OUTPUT}" \
  --device cuda
```

Plot the training curve:

```python
!python -m aerosol_encoding.plot_training \
  --history "{DRIVE_OUTPUT}/history.csv" \
  --output "{DRIVE_OUTPUT}/training_curve.png" \
  --title "64-D no-VAE long SGP aerosol encoder, Colab CUDA"
```

Run skill evaluation:

```python
!python -m aerosol_encoding.evaluate_cross_prediction \
  --features "{FEATURES}" \
  --checkpoint "{DRIVE_OUTPUT}/checkpoint.pt" \
  --output "{DRIVE_OUTPUT}/cross_prediction_test" \
  --split test \
  --batch-size 512 \
  --device cuda
```

## Current validation cadence

For this long run, the config intentionally avoids full reconstruction
validation every epoch:

```yaml
validation_interval: 100
reconstruction_validation_interval: 0
diagnostic_validation_interval: 300
```

That means training loss is printed every epoch, selected cross-prediction
validation runs at epoch 1, epoch 100, and the final epoch, and strict group-out
diagnostics run at epoch 1 and the final epoch for the current 120-epoch config.
