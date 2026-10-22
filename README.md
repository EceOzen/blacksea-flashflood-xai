# Western Black Sea Flash-Flood Precursors

**Can the synoptic setting of a catastrophic flash flood be recognised before the rain appears?**

Explainable machine-learning study of the meteorological precursors of
rapid-onset flash floods in the Western Black Sea region, anchored on the
11 August 2021 Bozkurt (Kastamonu) flood. Poster contribution to the
[ECMWF 2026 Workshop on Machine Learning for Land and Hydrology]([https://ecmwfevents.com/i/2026-workshop-on-machine-learning-for-land])
(Reading, 3–5 November 2026).

> Poster, figures and summary: [doi.org/10.5281/zenodo.23561977](https://doi.org/10.5281/zenodo.23561977)

## Key findings

- **The Bozkurt-type setting is recognisable 24 h ahead** from circulation and
  low-level flow alone: PR-AUC ≈ 0.50 at 24 h lead vs 0.30 for a seasonality
  baseline. Precipitation alone stays below seasonality.
- **The signal is simple and physically interpretable.** Logistic regression
  scores close to LightGBM; SHAP ranks MSLP, onshore flow (v850) and Z500 first
  at 0–12 h, 850–500 hPa shear at 24 h, and slow "background" variables
  (Black Sea SST, soil moisture) at 48–72 h.
- **Bozkurt, seen a day early.** A model trained without 2021 gives
  P = 0.89 for the flood time from conditions 24 h earlier (higher than 98 % of
  control days).
- **Independent flood check.** Analog days are flood-prone, not flood days:
  1.3 % of analog days vs 0.4 % of all days carry a documented core-region flood.
- **Climate context (descriptive, not attribution).** Bozkurt-type *conditions*
  doubled since 1979 (2.1 → 4.5 days/yr, p < 0.001), but the Bozkurt-type
  circulation *shape* did not (2.0 → 2.4, p = 0.93): the background warmed.

## Approach

1. **Analogs.** Every ERA5 day 1979–2024 is compared to the Bozkurt Z500
   anomaly composite (9–12 Aug 2021) with a cos(lat)-weighted anomaly
   correlation (ACC) over 30–55° N, 0–50° E. Days above the 95th percentile of
   a 1000-sample random-pair null are analogs (3-day declustering) → 158 days.
2. **Two definitions.** *Uncentred* ACC ("Bozkurt-like conditions", warm
   background included; the main definition) and *centred* ACC (domain mean
   removed, "circulation shape" only; used as a robustness check).
3. **Precursors.** Circulation (Z500, MSLP), flow (850 hPa wind, onshore v850,
   850–500 hPa shear) and thermodynamics/moisture (CAPE, CIN, TCWV, IVT, soil
   moisture, Black Sea SST anomaly), lagged 0/12/24/48/72 h. Each analog day
   gets 3 hour-matched control days. Rainfall is kept out of the main model and
   used only as a baseline.
4. **Models.** LightGBM and logistic regression with 5-fold year-blocked CV;
   PR-AUC (chance = 0.25), ROC-AUC, Brier score, bootstrap CIs, calibration.
5. **Explainability.** Out-of-fold TreeSHAP by lead time, group shares,
   permutation importance, cross-fold stability.
6. **Validation.** Flood records (DFO, Copernicus EMS, EM-DAT, HANZE) are used
   only as an independent check, never as training labels.

## Project layout

```
data/
  raw/era5/     ERA5 GRIB downloads (not tracked)
  interim/      NetCDF/Zarr, 1991–2020 day-of-year climatology, anomalies
  processed/    Analog lists and feature tables
  labels/       Curated flood-event table (derived, small)
scripts/        Numbered pipeline, run in order
figures/        Generated figures (not tracked); poster figures in figures/poster/
results/        Metrics, SHAP values, comparison tables
```

## Pipeline

| # | Script | Does |
| --- | --- | --- |
| 00 | `00_explore_bozkurt.py` | Raw Z500 / MSLP / 850 hPa wind / IVT / TCWV maps for the Bozkurt window; first sanity check. |
| 01 | `01_download_era5.py` | ERA5 pressure-level, single-level and precipitation downloads via `cdsapi`. |
| 02 | `02_build_climatology.py` | GRIB → Zarr, 1991–2020 day-of-year climatology, anomalies. |
| 03 | `03_find_analogs.py` | ACC against the Bozkurt composite, null-based threshold, declustering. |
| 03b–d | `03b_analog_diagnostics.py` … | Analog diagnostics and maps; ±5° domain-shift sensitivity; centred-ACC check. |
| 04 | `04_build_flood_labels.py` | Merges DFO / Copernicus EMS / EM-DAT / HANZE records for the region, deduplicates. |
| 05 | `05_make_features.py` | Lagged precursor table for analog and hour-matched control days. |
| 06 | `06_train_models.py` | Year-blocked CV, LightGBM + logistic regression, baselines, metrics, calibration. |
| 07 | `07_shap_analysis.py` | Out-of-fold TreeSHAP, group shares, interactions, stability. |
| 07b | `07b_climate_context.py` | Mann-Kendall / Sen's slope, 1979–2001 vs 2002–2024 change. |
| 08 | `08_make_figures.py` | A0 poster figures and the Bozkurt anticipation test (2021 withheld). |
| 09 | `09_compare_variants.py` | Uncentred vs centred analog definitions side by side. |

All shared settings (domains, dates, thresholds, paths) live in
`scripts/config.py`. Set `ANALOG_VARIANT = "uncentred"` (main results) or
`"centred"`, run 03 → 08 once per variant, then 09.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install xarray zarr cfgrib eccodes cdsapi numpy pandas scipy \
            scikit-learn lightgbm shap matplotlib cartopy pypdf
```

ERA5 access needs a CDS API key in `~/.cdsapirc` (not tracked).

## Data and licensing

- ERA5: Hersbach et al. (2020), Copernicus Climate Change Service (C3S) Climate
  Data Store. Raw ERA5 files are not redistributed here.
- Flood records: raw EM-DAT and ESWD exports are **not** committed, per their
  terms of use; only the derived event table needed for validation is.

## Limitations

Sparse flood labels; ERA5 resolution; onshore flow, IVT (truncated column) and
24 h rainfall are proxies; predicted probabilities are over-confident (ranking
metrics are unaffected); results at 72 h are close to chance and not
interpreted.

## Citation

Özen İldem, E. (2026). *Learning Meteorological Precursors of Rapid-Onset Flash
Floods from Synoptic Analog Events in the Western Black Sea Region.* Poster,
ECMWF Workshop on Machine Learning for Land and Hydrology, Reading, UK.
https://doi.org/10.5281/zenodo.23561977

## Author

Ece Özen İldem, Independent Researcher, Istanbul.
Questions and collaboration: eceozen23@gmail.com
