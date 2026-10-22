"""
Is the rising frequency of Bozkurt-type analogs real, or an artefact of a
warming atmosphere?

Why it could be an artefact: Z500 rises almost everywhere as the troposphere
warms. Anomalies are relative to the 1991-2020 climatology, so recent years
carry a broad positive offset. The ACC in 03 is UNCENTRED (the domain mean is
not removed), so a uniform offset projects onto the Bozkurt composite whenever
that composite has a non-zero domain mean -- and can inflate (or deflate)
pattern similarity over time.

Check: recompute the analog set with a CENTRED ACC -- the cos(lat)-weighted
domain mean is removed from every time step AND from the reference composite,
so only the SHAPE of the pattern counts. Then repeat the trend test.
Threshold: rank-matched (same number of top time steps as the 03 baseline),
as in 03c.

Also reported: the trend of the domain-mean Z500 anomaly itself (the size of
the "warming offset"), and the domain mean of the reference composite (how
much an offset could matter).

Reuses 03_find_analogs.py (patterns, ACC, decluster) and 07b_climate_context.py
(Mann-Kendall, Sen's slope) -- both must sit in the same folder.

Usage:  python scripts/03d_trend_check.py   (run from the project root)
Outputs:
    results/analog_trend_check.csv        per-year counts, uncentred vs centred
    results/analog_trend_check_summary.csv
    figures/fig_analog_trend_check.png
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fa = _load("find_analogs", "03_find_analogs.py")
cc = _load("climate_context", "07b_climate_context.py")

RESULTS = config.PROJECT_ROOT / "results"          # base folders on purpose:
BASE_PROCESSED = config.PROJECT_ROOT / "data" / "processed"  # 03d compares against the uncentred set
FIGURES = config.PROJECT_ROOT / "figures"
YEARS = getattr(config, "FULL_YEARS", (1979, 2024))


def main():
    print("Loading anomalies...")
    anom = xr.open_zarr(fa.ANOM_ZARR)
    z500 = fa.subset_domain(anom, config.ANALOG_AREA)["z500_anom"]
    w = fa.lat_weights(z500)
    dims = ("latitude", "longitude")

    window = z500.sel(time=slice(config.BOZKURT_WINDOW_START, config.BOZKURT_WINDOW_END))
    reference = window.mean("time").compute()
    ref_mean = float(reference.weighted(w).mean(dim=dims))
    ref_rms = float(np.sqrt((reference ** 2).weighted(w).mean(dim=dims)))
    print(f"Reference composite: domain mean {ref_mean:+.1f}, RMS {ref_rms:.1f} "
          f"(mean/RMS = {ref_mean / ref_rms:+.2f}; |ratio| near 0 = an offset barely matters)")

    print("Domain-mean anomaly per time step (the 'warming offset')...")
    dmean = z500.weighted(w).mean(dim=dims).compute()
    dm = pd.Series(dmean.values, index=pd.to_datetime(z500.time.values))
    dm_year = dm.groupby(dm.index.year).mean()
    yrs = dm_year.index.to_numpy()
    s_dm, _ = cc.sens_slope(yrs, dm_year.to_numpy())
    print(f"  domain-mean Z500 anomaly trend: {s_dm * 10:+.1f} per decade")

    print("Centred ACC for every time step (several minutes)...")
    acc_c = fa.weighted_acc(z500 - dmean, reference - ref_mean, w, dims=dims).compute()
    scores = pd.DataFrame({"time": pd.to_datetime(z500.time.values), "acc": acc_c.values})

    buf = pd.Timedelta(days=config.DECLUSTER_WINDOW_DAYS)
    is_self = (scores["time"] >= pd.Timestamp(config.BOZKURT_WINDOW_START) - buf) & \
              (scores["time"] <= pd.Timestamp(config.BOZKURT_WINDOW_END) + buf)
    cand = scores[~is_self]

    base = pd.read_csv(BASE_PROCESSED / "analogs_significant.csv", parse_dates=["time"])
    all_scores = pd.read_csv(BASE_PROCESSED / "analogs_all_scores.csv", parse_dates=["time"])
    null = pd.read_csv(BASE_PROCESSED / "analogs_null_distribution.csv")["null_acc"].to_numpy()
    thr = float(np.quantile(null, config.ANALOG_SIGNIFICANCE_LEVEL))
    a_self = (all_scores["time"] >= pd.Timestamp(config.BOZKURT_WINDOW_START) - buf) & \
             (all_scores["time"] <= pd.Timestamp(config.BOZKURT_WINDOW_END) + buf)
    n_exceed = int((all_scores.loc[~a_self, "acc"] > thr).sum())
    centred = fa.decluster(cand.nlargest(n_exceed, "acc"), "acc", config.DECLUSTER_WINDOW_DAYS)
    print(f"Centred analog set: {len(centred)} days (baseline uncentred set: {len(base)} days)")

    years = np.arange(YEARS[0], YEARS[1] + 1)
    rows, summary = {"year": years}, []
    for name, times in [("uncentred (03 baseline)", base["time"]), ("centred", centred["time"])]:
        counts = pd.to_datetime(times).dt.year.value_counts().reindex(years, fill_value=0).to_numpy()
        _, z, p = cc.mann_kendall(counts)
        slope, _ = cc.sens_slope(years, counts)
        early, late = counts[years <= 2001].mean(), counts[years > 2001].mean()
        rows[name] = counts
        summary.append({"set": name, "n_days": int(counts.sum()), "mean_1979_2001": early,
                        "mean_2002_2024": late, "mk_z": z, "mk_p": p, "sen_slope_per_decade": slope * 10})
        print(f"  {name:<24} {early:.1f} -> {late:.1f} days/yr, Mann-Kendall p = {p:.3f}, "
              f"Sen {slope * 10:+.2f} days/decade")

    overlap = sum(any(abs((t - b).total_seconds()) <= config.DECLUSTER_WINDOW_DAYS * 86400
                      for b in base["time"]) for t in centred["time"])
    print(f"  {overlap}/{len(centred)} centred analog days are also in the uncentred set")
    summary.append({"set": "reference composite", "n_days": np.nan, "mean_1979_2001": np.nan,
                    "mean_2002_2024": np.nan, "mk_z": np.nan, "mk_p": np.nan,
                    "sen_slope_per_decade": np.nan, "ref_domain_mean": ref_mean, "ref_rms": ref_rms,
                    "domain_mean_trend_per_decade": s_dm * 10, "overlap_days": overlap})

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "analog_trend_check.csv", index=False)
    pd.DataFrame(summary).to_csv(RESULTS / "analog_trend_check_summary.csv", index=False)
    print(f"Saved: {RESULTS / 'analog_trend_check.csv'}")
    plot(years, rows, summary)


def plot(years, rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (name, col), s in zip([("uncentred (03 baseline)", "#2a78d6"), ("centred", "#eb6834")], summary):
        c = np.asarray(rows[name], dtype=float)
        smooth = pd.Series(c).rolling(5, center=True, min_periods=3).mean()
        ax.plot(years, smooth, color=col, lw=2, label=f"{name}: Sen {s['sen_slope_per_decade']:+.1f}/decade, "
                                                       f"p = {s['mk_p']:.3f}")
        ax.plot(years, c, color=col, lw=0, marker="o", ms=3, alpha=0.5)
    ax.set_ylabel("Analog days per year\n(dots; line = 5-yr mean)")
    ax.set_title("Analog frequency trend: robust to removing the domain-mean offset?", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig_analog_trend_check.png"
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
