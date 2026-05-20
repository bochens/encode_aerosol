# ARM SGP E13 Aerosol Data Roles for Coordinate-Conditioned Training

This report audits the variables currently used from `data/DOE_SGP` and separates measured aerosol responses from coordinates, set points, operating conditions, and diagnostics.

## Main Rule

The model should predict aerosol response variables, not instrument coordinates.

For coordinate-dependent instruments, the decoder should learn a function:

| Instrument family | Learned response function |
|---|---|
| CCN | `N_CCN = f(z_aerosol, supersaturation)` |
| SMPS sizing | `dN/dlogDp = f_SMPS(z_aerosol, log10(Dp_nm))` |
| APS sizing | `dN/dlogDp = f_APS(z_aerosol, log10(Dp_nm))` |
| UHSAS sizing | `dN/dlogDp = f_UHSAS(z_aerosol, log10(Dp_nm))` |
| OPC sizing | `dN/dlogDp = f_OPC(z_aerosol, log10(Dp_nm))` |
| Humidified neph | `scattering/backscattering = f(z_aerosol, RH, wavelength channel, dry/wet state)` |

The coordinate variables can be used as decoder inputs. They are not reconstruction targets.

`z_aerosol` is the shared 64-dimensional bottleneck state for one 30-minute aerosol window. It is produced by the multimodal transformer from the visible instrument tokens, then every target decoder reads from it.

There is no explicit instrument-id coordinate for size retrieval. The selected sizing modality chooses the decoder head; diameter is the only continuous coordinate passed into that head.

## Current Prepared Feature Inventory

The prepared training array is `artifacts/temporal_gru_30min_20161129_20230421/features/prepared_arrays.npz`.

| Modality | Selected features | Target-response features used in loss | Conditioning-coordinate features | Diagnostic features |
|---|---:|---:|---:|---:|
| `chemistry_acsm` | 7 | 6 | 0 | 1 |
| `size_smps` | 450 | 402 | 0 | 48 |
| `size_aps` | 1980 | 1890 | 0 | 90 |
| `size_uhsas` | 1590 | 1500 | 0 | 90 |
| `size_opc` | 2220 | 2130 | 0 | 90 |
| `cpc_number` | 240 | 240 | 0 | 0 |
| `ccn_activation` | 180 | 30 | 120 | 30 |
| `optical_neph` | 810 | 540 | 270 | 0 |

For CCN, only mean `N_CCN` features are target responses. `supersaturation_calculated` and `supersaturation_set_point` are coordinates; `N_CCN__stat_std` is treated as a diagnostic for now.

For nephelometer data, `Bs_*` and `Bbs_*` are responses. `RH_Neph_*`, `T_Neph_*`, and `P_Neph_*` are operating conditions. The new coordinate decoder uses RH; it does not currently use T/P.

For sizing instruments, `dN_dlogDp` bins are target responses. Integrated totals and distribution summary quantities are diagnostics, because they are mostly derived from the same spectrum.

## Source NetCDF Variable Audit

### AOSMET

Stream: `sgpaosmetE13.a1`

| Variable | Units | NetCDF shape | Role |
|---|---|---:|---|
| `rh_ambient` | `%` | `time` | context |
| `temperature_ambient` | `degC` | `time` | context |
| `pressure_ambient` | `hPa` | `time` | context |
| `wind_speed` | `m/s` | `time` | context |
| `wind_direction` | `degree` | `time` | context |
| `rain_amount` | `mm/s` | `time` | context |
| `rain_intensity` | `mm/hr` | `time` | context |

These are environmental context, not target aerosol properties.

### ACSM-CDCE

Stream: `sgpacsmcdceE13.c2`

| Variable | Units | NetCDF shape | Role |
|---|---|---:|---|
| `total_organics_CDCE` | `ug/m^3` | `time` | target response |
| `sulfate_CDCE` | `ug/m^3` | `time` | target response |
| `ammonium_CDCE` | `ug/m^3` | `time` | target response |
| `nitrate_CDCE` | `ug/m^3` | `time` | target response |
| `chloride_CDCE` | `ug/m^3` | `time` | target response |
| `acsm_vol_conc` | `um^3/cm^3` | `time` | target response |
| `CDCE` | `1` | `time` | diagnostic |

ACSM is comparatively simple: the species concentrations are categorical response channels. `CDCE` is a correction factor, so the current loss mask treats it as diagnostic rather than an aerosol composition target.

### Size Spectrometers

All size instruments are kept as separate modalities because they measure different diameter concepts.

| Modality | Stream | Spectrum variable | Native coordinate | Native units | Converted coordinate used by model |
|---|---|---|---|---|---|
| `size_smps` | `sgpaossmpsE13.b1` | `dN_dlogDp` | `diameter_mobility` | `nm` | `diameter_common_nm` |
| `size_aps` | `sgpaosapsE13.b1` | `dN_dlogDp` | `diameter_aerodynamic` | `um` | `diameter_common_nm` |
| `size_uhsas` | `sgpaosuhsasE13.b1` | `dN_dlogDp` | `diameter_optical` | `nm` | `diameter_common_nm` |
| `size_opc` | `sgpaosopcE13.b1` | `dN_dlogDp` | `diameter_midpoint` | `um` | `diameter_common_nm` |

The feature builder converts native diameter units to nm and interpolates spectra onto a common logarithmic grid from 3 to 30000 nm. The decoder is now coordinate-conditioned, so the response is generated from `log10(Dp_nm)` rather than from a hard-coded output neuron per bin.

The four sizing instruments remain four separate modalities and four separate decoder heads. This keeps their different measurement physics separate. Calling `decode_size_at_diameter(z64, "size_smps", Dp_nm)` chooses the SMPS head; `"size_aps"`, `"size_uhsas"`, and `"size_opc"` choose their own heads. The modality argument is routing, not a learned continuous or categorical coordinate.

Integrated quantities such as `total_N_conc`, `total_SA_conc`, `total_V_conc`, `geometric_mean`, `median`, and `mode` are useful diagnostics, but they are not primary training targets in the new target mask.

### CPC

Streams: `sgpaoscpcfE13.b1`, `sgpaoscpcufE13.b1`

| Variable | Units | NetCDF shape | Role |
|---|---|---:|---|
| `concentration` | `1/cm^3` | `time` | target response |

CPC has no explicit continuous coordinate in the current files. The fine and ultrafine CPC streams are still separate source channels inside the `cpc_number` modality.

### CCN

Stream: `sgpaosccn2colbE13.b1`

| Variable | Units | NetCDF shape | Role |
|---|---|---:|---|
| `N_CCN` | `1/cm^3` | `time` | target response |
| `supersaturation_calculated` | `%` | `time` | conditioning coordinate |
| `supersaturation_set_point` | `%` | `time` | set point / fallback coordinate |

The physical target is not "the CCN modality vector." The physical target is `N_CCN` queried at a supersaturation.

The new coordinate decoder pairs each mean `N_CCN` time-bin target with the corresponding mean `supersaturation_calculated` value. In the prepared array there are rows where `N_CCN` is present but calculated supersaturation is missing, so the decoder uses `supersaturation_set_point` as a per-row fallback. There are zero prepared rows where mean `N_CCN` is present but both calculated and set-point supersaturation are missing. The loss does not reward reconstructing either supersaturation variable.

### Nephelometer

Streams: `sgpaosnephdry1mE13.b1`, `sgpaosnephwetE13.b1`

| Variable family | Units | NetCDF shape | Role |
|---|---|---:|---|
| `Bs_B/G/R_Dry_Neph3W` | `1/Mm` | `time` | target response |
| `Bbs_B/G/R_Dry_Neph3W` | `1/Mm` | `time` | target response |
| `Bs_B/G/R_Wet_Neph3W` | `1/Mm` | `time` | target response |
| `Bbs_B/G/R_Wet_Neph3W` | `1/Mm` | `time` | target response |
| `RH_Neph_Dry`, `RH_Neph_Wet` | `%` | `time` | conditioning coordinate |
| `T_Neph_Dry`, `T_Neph_Wet` | `degC` | `time` | operating condition |
| `P_Neph_Dry`, `P_Neph_Wet` | `hPa` | `time` | operating condition |

For humidified nephelometer data, the relevant continuous coordinate is RH, not supersaturation. The new decoder conditions scattering/backscattering on RH plus the discrete blue/green/red wavelength channel and dry/wet state. It does not claim validated interpolation to arbitrary wavelength; wavelength is encoded as the observed channel identity.

## Implemented Code Changes

The new config is:

`configs/sgp_e13_no_htdma_30min_temporal_pretrain_192tok_64bottleneck_coordinate_targets.yaml`

It enables:

```yaml
coordinate_decoders:
  ccn_activation: true
  size_spectra: true
  optical_neph: true
```

The model now includes these decoder types:

| Decoder | Query |
|---|---|
| `CoordinateCCNActivationDecoder` | `decode_ccn_at_supersaturation(z64, supersaturation_percent)` |
| `CoordinateSizeDistributionDecoder` | `decode_size_at_diameter(z64, modality, diameter_nm)`, where `modality` selects the separate SMPS/APS/UHSAS/OPC head |
| `CoordinateNephelometerDecoder` | internal observed-channel RH-conditioned decoding |

Training and cross-prediction evaluation now use feature-level target masks from `aerosol_encoding/loss_masks.py`, so coordinates and diagnostics are not counted in the target MSE or skill score.

The coordinate decoders emit the physical transformed response first, for example `log1p(N_CCN)` or `log1p(dN/dlogDp)`. During training, that transformed response is converted back into the existing standardized feature scale before MSE is computed. The public model query helpers default to returning inverse-transformed physical nonnegative concentrations.

## Remaining Decisions Before a Full New Run

1. Whether to keep `N_CCN__stat_std` as diagnostic only or add a separate uncertainty/variability head.
2. Whether `CDCE` should remain diagnostic or be a conditioning input to chemistry rather than part of the chemistry modality.
3. Whether to include nephelometer T/P as weak conditioning variables. RH is the main coordinate for humidification response; T/P are probably secondary operating conditions.

## External Checks

The role decisions are consistent with ARM's public instrument descriptions:

- ARM describes CCN as activated ambient aerosol particle number concentration as a function of supersaturation.
- ARM describes humidified nephelometer measurements as scattering/backscattering and hygroscopic growth as a function of RH.
