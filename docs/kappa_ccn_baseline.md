# Kappa-CCN Baseline

Purpose: build a traditional aerosol-chemistry baseline to compare against the
trained neural network. The baseline predicts CCN concentration from ACSM
composition and particle number size distributions.

## Method

1. ACSM chemistry is converted to a bulk hygroscopicity parameter:

   `kappa_chem = eps_org * kappa_org + eps_inorg * kappa_inorg`

   where `eps` are ACSM organic/inorganic mass fractions by default. The
   starter defaults are
   `kappa_org = 0.12` and `kappa_inorg = 0.63`, following Poehlker et al.
   (2023). ACSM organics are treated as the organic component; sulfate,
   ammonium, nitrate, and chloride are grouped as inorganic mass. Black carbon
   is not included in this first pass because the current feature set does not
   expose a BC modality. A `volume` basis is also available for sensitivity
   tests against the original component volume-fraction mixing rule.

2. SMPS, UHSAS, OPC, and APS dN/dlogDp spectra are merged onto the existing
   common diameter grid. Each instrument is first averaged over available
   within-row time bins in `log1p(dN/dlogDp)` space. Instrument overlap is then
   averaged in log space using an error-function taper in log10 diameter so
   transition regions are smooth instead of stepwise.

3. For every observed CCN supersaturation, Petters and Kreidenweis
   kappa-Koehler Eq. 10 gives the critical dry diameter. The predicted CCN
   number concentration is the integral of the merged size distribution above
   that diameter:

   `N_CCN = integral from Dcrit to infinity of dN/dlogDp dlogDp`

4. Kappa summary statistics use geometric mean and geometric standard
   deviation, matching the treatment in Chen et al. (2025).

## References Used

- The AMT kappa-closure reference discussed in this project: Sect. 2.4 derives
  kappa from measured CCN and size distributions, then summarizes kappa with a
  geometric mean.
- `kappa_ccn_baseline/reference/kappa_kohler_theory.py`: copied reference math
  from the supplied helper. The importable baseline uses a dependency-light
  subset in `kappa_ccn_baseline/kohler.py`.

## External Papers Checked

- Petters and Kreidenweis (2007), ACP, https://doi.org/10.5194/acp-7-1961-2007:
  single-parameter kappa-Koehler theory and volume-fraction mixing.
- Poehlker et al. (2023), Nature Communications,
  https://doi.org/10.1038/s41467-023-41695-8:
  `kappa_org = 0.12 +/- 0.02`, `kappa_inorg = 0.63 +/- 0.01`, and the
  ACSM organic/inorganic mass-fraction shortcut.
- Schmale et al. (2018), ACP, https://doi.org/10.5194/acp-18-2853-2018:
  long-term CCN, particle number size distribution, and chemical-composition
  closure studies using kappa-Koehler theory.

## Smoke Run

```bash
python -m kappa_ccn_baseline.run_baseline \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --split test \
  --fraction-basis mass \
  --limit-rows 5000 \
  --output artifacts/kappa_ccn_baseline_smoke
```

Full test split:

```bash
python -m kappa_ccn_baseline.run_baseline \
  --prepared-arrays artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz \
  --split test \
  --fraction-basis mass \
  --output artifacts/kappa_ccn_baseline
```

Outputs:

- `{split}_kappa_ccn_predictions.csv`: row-level predictions for each observed
  CCN time bin.
- `{split}_kappa_ccn_summary.json`: physical and log-space error metrics,
  geometric kappa statistics, and size-instrument diameter coverage.
