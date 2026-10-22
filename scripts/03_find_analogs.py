"""
Find synoptic analogs of the Bozkurt flash flood in the 1979-2024 ERA5 record.

Method (and why it's built this way):
  - Reference pattern: the Z500 anomaly, composited (averaged) over the
    Bozkurt precursor window (config.BOZKURT_WINDOW_START..END), restricted
    to the analog search domain (config.ANALOG_AREA). Compositing over the
    window instead of picking one timestamp makes the reference robust to
    exactly which 6-hour step happened to capture the deepest trough.
  - Matching field: Z500 anomaly only (not MSLP, not a blended score). This
    follows the classical Lorenz (1969) synoptic-analog method: Z500 is the
    field that best represents the large-scale flow whose "shape" we're
    trying to match, and using a single, standard field avoids introducing
    an arbitrary multi-variable weighting scheme. MSLP anomaly is still
    computed and can be cross-checked afterwards against the resulting
    analog set.
  - Similarity score: the anomaly correlation coefficient (ACC), cos(lat)
    weighted -- literally the same score ECMWF and other NWP centres use to
    verify forecasts against reanalysis, just applied here to compare two
    anomaly patterns instead of a forecast and the truth it's scored
    against. RMSE is also computed and reported, but ACC is primary: it
    captures pattern *shape* independent of amplitude, which is what
    "analog" is supposed to mean (a day with the same kind of trough, even
    if a little weaker, should score well; RMSE would penalise it for the
    amplitude difference alone).
  - Threshold: NOT a hand-picked "top N" or "top X%". Instead, a null
    distribution of ACC values is built from many random window/day pairs
    that have nothing to do with Bozkurt, and a candidate day only counts as
    a significant analog if its ACC beats the 95th percentile (by default)
    of that null distribution -- i.e. a pattern this similar would arise by
    chance less than 5% of the time. Top-N and top-X% sets are still
    computed and saved, but only as sensitivity/robustness checks reported
    alongside the significance-based set, not as the primary criterion.
  - Variant (config.ANALOG_VARIANT):
      "uncentred" (original): ACC on the raw anomaly fields. Rewards the whole
                  field, incl. its broad positive (warm) part -> "Bozkurt-like
                  conditions". Outputs in data/processed/.
      "centred":  the cos(lat)-weighted domain mean is removed from candidate
                  AND reference before the ACC -> "Bozkurt-like circulation
                  shape" only. Outputs in data/processed/centred/.
    The null distribution uses the same ACC, so each variant gets its own
    threshold.
  - Declustering: neighbouring timesteps within DECLUSTER_WINDOW_DAYS of a
    better-scoring day are dropped, so one slow-moving trough doesn't count
    as several independent analogs.

Usage:
    python scripts/03_find_analogs.py   (run from the project root)

Outputs (data/processed/):
    analogs_significant.csv       primary analog set (significance-based)
    analogs_topN.csv              sensitivity check: top ANALOG_N, declustered
    analogs_percentile.csv        sensitivity check: top ANALOG_PERCENTILE%, declustered
    analogs_null_distribution.csv the null ACC values themselves, for a histogram/figure later

This can take a while: the null-distribution step does ANALOG_N_BOOTSTRAP
(default 1000) small random reads from the Zarr store on disk, each
triggering its own little dask computation. Expect several minutes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ANOM_ZARR = config.DATA_INTERIM / "anomalies_1979_2024.zarr"
OUT_DIR = config.DATA_PROCESSED
ANALOGS_SIGNIFICANT_CSV = OUT_DIR / "analogs_significant.csv"
ANALOGS_TOPN_CSV = OUT_DIR / "analogs_topN.csv"
ANALOGS_PERCENTILE_CSV = OUT_DIR / "analogs_percentile.csv"
NULL_DIST_CSV = OUT_DIR / "analogs_null_distribution.csv"
ALL_SCORES_CSV = OUT_DIR / "analogs_all_scores.csv"  # every candidate day's ACC/RMSE, for later figures

RNG_SEED = 42  # fixed, so the null distribution (and hence the threshold) is reproducible
VARIANT = getattr(config, "ANALOG_VARIANT", "uncentred")


def check_variant_paths():
    """Refuse to run if the variant and the output folder disagree -- protects
    the uncentred results from being overwritten by a centred run (or v.v.)."""
    if VARIANT not in ("uncentred", "centred"):
        raise ValueError(f'config.ANALOG_VARIANT = "{VARIANT}" -- use exactly "uncentred" or "centred".')
    in_sub = Path(OUT_DIR).name == "centred"
    if (VARIANT == "centred") != in_sub:
        raise RuntimeError(
            f'config.ANALOG_VARIANT = "{VARIANT}" but config.DATA_PROCESSED = {OUT_DIR}.\n'
            "The variant block at the END of config.py must define ANALOG_VARIANT first and then\n"
            "DATA_PROCESSED / RESULTS_DIR / FIGURES_DIR from it, with no other ANALOG_VARIANT\n"
            "line anywhere else. Nothing was written."
        )


# ----------------------------------------------------------------------
# Domain subsetting and weighting (same pattern as 01_bozkurt_summary.py)
# ----------------------------------------------------------------------
def subset_domain(ds, box):
    sub = ds.sel(latitude=slice(box["north"], box["south"]),
                 longitude=slice(box["west"], box["east"]))
    if sub.sizes.get("latitude", 0) == 0:  # latitude was ascending instead
        sub = ds.sel(latitude=slice(box["south"], box["north"]),
                      longitude=slice(box["west"], box["east"]))
    return sub


def lat_weights(da):
    return np.cos(np.deg2rad(da.latitude))


# ----------------------------------------------------------------------
# Similarity scores
# ----------------------------------------------------------------------
def weighted_acc(candidate, reference, weights, dims):
    """Anomaly correlation coefficient, cos(lat)-weighted.

    ACC = sum(w*f*o) / sqrt(sum(w*f^2) * sum(w*o^2))

    where f is the candidate anomaly field and o is the reference anomaly
    field. Ranges from -1 (opposite pattern) to +1 (identical pattern,
    any amplitude). This is the standard NWP pattern-verification score.

    With config.ANALOG_VARIANT == "centred", the weighted domain mean is
    removed from both fields first (shape only, no uniform offset).
    """
    if VARIANT == "centred":
        candidate = candidate - candidate.weighted(weights).mean(dim=dims)
        reference = reference - reference.weighted(weights).mean(dim=dims)
    num = (candidate * reference).weighted(weights).sum(dim=dims)
    denom = np.sqrt(
        (candidate ** 2).weighted(weights).sum(dim=dims)
        * (reference ** 2).weighted(weights).sum(dim=dims)
    )
    return num / denom


def weighted_rmse(candidate, reference, weights, dims):
    sq_diff = (candidate - reference) ** 2
    return np.sqrt(sq_diff.weighted(weights).mean(dim=dims))


# ----------------------------------------------------------------------
# Declustering
# ----------------------------------------------------------------------
def decluster(df, score_col, window_days, max_keep=None):
    """Greedily keep the best-scoring row, drop anything within
    window_days of an already-kept row, repeat. max_keep stops early once
    enough rows are kept (only used for the top-N set, for speed)."""
    remaining = df.sort_values(score_col, ascending=False)
    kept_rows = []
    kept_times = []
    for _, row in remaining.iterrows():
        t = row["time"]
        if all(abs((t - kt).total_seconds()) > window_days * 86400 for kt in kept_times):
            kept_rows.append(row)
            kept_times.append(t)
            if max_keep is not None and len(kept_rows) >= max_keep:
                break
    return pd.DataFrame(kept_rows).reset_index(drop=True)


# ----------------------------------------------------------------------
def main():
    check_variant_paths()
    print(f"Analog variant: {VARIANT}  ->  outputs in {OUT_DIR}")
    print("Loading anomalies...")
    anom = xr.open_zarr(ANOM_ZARR)
    domain = subset_domain(anom, config.ANALOG_AREA)
    z500 = domain["z500_anom"]
    weights = lat_weights(z500)

    # Reference composite: mean Z500 anomaly over the Bozkurt precursor window.
    window = z500.sel(time=slice(config.BOZKURT_WINDOW_START, config.BOZKURT_WINDOW_END))
    reference = window.mean("time").compute()
    print(f"Reference composite built from {window.sizes['time']} time steps "
          f"({config.BOZKURT_WINDOW_START} to {config.BOZKURT_WINDOW_END}).")

    print("Scoring all time steps against the reference pattern (ACC + RMSE)...")
    acc = weighted_acc(z500, reference, weights, dims=("latitude", "longitude")).compute()
    rmse = weighted_rmse(z500, reference, weights, dims=("latitude", "longitude")).compute()

    scores = pd.DataFrame({
        "time": pd.to_datetime(z500.time.values),
        "acc": acc.values,
        "rmse": rmse.values,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(ALL_SCORES_CSV, index=False)
    print(f"Saved every candidate day's score: {ALL_SCORES_CSV}")

    # Mask out the Bozkurt window itself (+/- a decluster-width buffer) from
    # the candidate pool -- the event trivially matches itself, that's not
    # an analog. Kept aside as a sanity check instead (ACC should be ~1.0).
    buf = pd.Timedelta(days=config.DECLUSTER_WINDOW_DAYS)
    window_start = pd.Timestamp(config.BOZKURT_WINDOW_START) - buf
    window_end = pd.Timestamp(config.BOZKURT_WINDOW_END) + buf
    is_self = (scores["time"] >= window_start) & (scores["time"] <= window_end)
    self_scores = scores[is_self]
    candidates = scores[~is_self].reset_index(drop=True)

    print("\nSanity check -- Bozkurt window scored against its own composite "
          "(should be close to ACC=1.0):")
    print(self_scores.sort_values("acc", ascending=False).head(3).to_string(index=False))

    # ------------------------------------------------------------------
    # Null distribution: how similar do two UNRELATED windows/days look by
    # chance? Repeat many times: build a composite from a random window the
    # same length as Bozkurt's, score a random single day against it. Same
    # statistic as above, so the resulting threshold is directly comparable.
    # ------------------------------------------------------------------
    if NULL_DIST_CSV.exists():
        null_acc = pd.read_csv(NULL_DIST_CSV)["null_acc"].values
        print(f"\nFound existing {NULL_DIST_CSV} -- reusing it instead of rebuilding "
              f"(delete this file to force a fresh null distribution).")
    else:
        print(f"\nBuilding null distribution from {config.ANALOG_N_BOOTSTRAP} random "
              f"window/day pairs (reads a little data from disk each draw, "
              f"this takes a while)...")
        rng = np.random.default_rng(RNG_SEED)
        n_time = z500.sizes["time"]
        window_len = window.sizes["time"]
        null_acc = np.empty(config.ANALOG_N_BOOTSTRAP)
        for i in range(config.ANALOG_N_BOOTSTRAP):
            w_start = rng.integers(0, n_time - window_len)
            rand_ref = z500.isel(time=slice(w_start, w_start + window_len)).mean("time")
            cand_idx = rng.integers(0, n_time)
            rand_candidate = z500.isel(time=cand_idx)
            null_acc[i] = float(
                weighted_acc(rand_candidate, rand_ref, weights, dims=("latitude", "longitude")).compute()
            )
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{config.ANALOG_N_BOOTSTRAP}")
        pd.DataFrame({"null_acc": null_acc}).to_csv(NULL_DIST_CSV, index=False)

    threshold = float(np.quantile(null_acc, config.ANALOG_SIGNIFICANCE_LEVEL))
    print(f"\nNull-distribution ACC: mean={null_acc.mean():.3f}, std={null_acc.std():.3f}, "
          f"{config.ANALOG_SIGNIFICANCE_LEVEL:.0%} percentile={threshold:.3f}")
    print(f"-> a candidate day counts as a significant analog if its ACC exceeds {threshold:.3f} "
          f"(a pattern this similar would arise by chance less than "
          f"{100 * (1 - config.ANALOG_SIGNIFICANCE_LEVEL):.0f}% of the time).")

    # ------------------------------------------------------------------
    # Three analog sets: significance-based (primary) + top-N / top-X%
    # (sensitivity checks, reported for robustness, not used as the criterion).
    # ------------------------------------------------------------------
    significant = candidates[candidates["acc"] > threshold]
    significant_declustered = decluster(significant, "acc", config.DECLUSTER_WINDOW_DAYS)
    significant_declustered.to_csv(ANALOGS_SIGNIFICANT_CSV, index=False)
    print(f"\nSignificant analogs: {len(significant)} time steps -> "
          f"{len(significant_declustered)} after declustering.")
    print(f"Saved: {ANALOGS_SIGNIFICANT_CSV}")

    topn_declustered = decluster(candidates, "acc", config.DECLUSTER_WINDOW_DAYS,
                                  max_keep=config.ANALOG_N)
    topn_declustered.to_csv(ANALOGS_TOPN_CSV, index=False)
    print(f"\nTop-{config.ANALOG_N} analogs (declustered). Saved: {ANALOGS_TOPN_CSV}")

    pct_cutoff = np.quantile(candidates["acc"], 1 - config.ANALOG_PERCENTILE / 100)
    percentile_set = candidates[candidates["acc"] >= pct_cutoff]
    percentile_declustered = decluster(percentile_set, "acc", config.DECLUSTER_WINDOW_DAYS)
    percentile_declustered.to_csv(ANALOGS_PERCENTILE_CSV, index=False)
    print(f"\nTop {config.ANALOG_PERCENTILE}% analogs: {len(percentile_set)} time steps -> "
          f"{len(percentile_declustered)} after declustering.")
    print(f"Saved: {ANALOGS_PERCENTILE_CSV}")

    # Overlap between the three sets -- a robustness check for the writeup:
    # if the significance-based set mostly agrees with top-N/top-X%, the
    # choice of threshold isn't driving the result.
    sig_days = set(significant_declustered["time"].dt.date)
    topn_days = set(topn_declustered["time"].dt.date)
    pct_days = set(percentile_declustered["time"].dt.date)
    print(f"\nOverlap (by calendar day): significant & top-N = {len(sig_days & topn_days)}, "
          f"significant & top-{config.ANALOG_PERCENTILE}% = {len(sig_days & pct_days)}, "
          f"top-N & top-{config.ANALOG_PERCENTILE}% = {len(topn_days & pct_days)}")


if __name__ == "__main__":
    main()