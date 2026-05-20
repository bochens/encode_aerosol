# Random Forest CCN Baseline

Purpose: add a non-neural, data-driven baseline for CCN concentration using the
same chemistry and size-distribution inputs as the traditional kappa closure.

## Method

Each RF sample is one observed CCN point. The target is `log1p(N_CCN)`. Inputs
are:

- CCN supersaturation percent for that observation.
- ACSM `time_bin_000` log-space organics, sulfate, ammonium, nitrate, and
  chloride. Complete ACSM is required by default so the RF and kappa baselines
  evaluate the same chemistry-available regime; use `--allow-missing-acsm` only
  for an imputed-chemistry sensitivity run.
- Merged SMPS/UHSAS/OPC/APS `log1p(dN/dlogDp)` spectrum. Each instrument is
  averaged over within-row time bins, then instruments are merged in log space
  with the same error-function transition mask used by the kappa baseline.

The model is a scikit-learn `Pipeline` with median imputation followed by
`RandomForestRegressor`. Missing spectral or ACSM features are imputed from the
training split only.

## Smoke Run

```bash
python -m kappa_ccn_baseline.random_forest_baseline \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --train-split train \
  --eval-split test \
  --max-train-samples 20000 \
  --max-eval-samples 20000 \
  --n-estimators 40 \
  --min-samples-leaf 5 \
  --output artifacts/random_forest_ccn_baseline_smoke
```

Larger baseline run:

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

Outputs:

- `{eval_split}_random_forest_ccn_predictions.csv`: row-level CCN predictions.
- `{eval_split}_random_forest_ccn_summary.json`: physical and log-space metrics.
- `random_forest_feature_importance.csv`: ranked RF feature importances.
