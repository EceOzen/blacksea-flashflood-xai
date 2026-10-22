"""
Climate context for the Bozkurt-type analog days. Two questions:

  A. Are Bozkurt-type circulation patterns becoming more frequent?
     Annual count of significant analog days (1979-2024) -> Mann-Kendall
     trend test (with tie correction) + Sen's slope. Implemented here, no
     extra package needed.

  B. Same pattern, more fuel? Do analog days in the recent period carry more
     moisture / warmer sea / more instability than analog days in the early
     period -- BEYOND the background change seen on all days?
       - Every variable is first turned into a day-of-year anomaly (smoothed
         climatology over the full record), so a different seasonal mix of
         analog days between periods can't fake a difference.
       - delta_analog = mean anomaly on analog days, late minus early period
       - delta_all    = the same on all May-September time steps (background)
       - excess       = delta_analog - delta_all
       95% CIs from bootstrap resampling of days. "excess > 0 with a CI above
       zero" would mean analog days have moistened/warmed more than the
       season as a whole.
     This is a descriptive comparison, NOT an attribution study, and should be
     presented that way.

Inputs: data/processed/analogs_significant.csv, data/interim/features_timeseries.csv
Usage:  python scripts/07b_climate_context.py   (run from the project root)
Outputs:
    results/climate_context_trend.csv
    results/climate_context_periods.csv
    figures/fig_climate_trend.png
    figures/fig_climate_periods.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

RESULTS = getattr(config, "RESULTS_DIR", config.PROJECT_ROOT / "results")
FIGURES = getattr(config, "FIGURES_DIR", config.PROJECT_ROOT / "figures")
SEED = getattr(config, "FEATURE_RNG_SEED", 42)
N_BOOT = 2000
YEARS = getattr(config, "FULL_YEARS", (1979, 2024))
SPLIT_YEAR = 2001          # early = YEARS[0]..2001, late = 2002..YEARS[1] (23 years each for 1979-2024)

VARIABLES = {               # column in features_timeseries.csv -> label, unit
    "tcwv_mean": ("TCWV", "kg m$^{-2}$"),
    "sst_anom_mean": ("Black Sea SST anomaly", "K"),
    "ivt_mean": ("IVT", "kg m$^{-1}$ s$^{-1}$"),
    "cape_mean": ("CAPE", "J kg$^{-1}$"),
    "tp_24h_proxy_mean": ("Precip (24 h proxy)", "mm"),
}
COL_ANALOG = "#2a78d6"     # validated palette, slot 1
COL_ALL = "#8a8a8a"        # neutral for the background


# ----------------------------------------------------------------------
# A. Trend
# ----------------------------------------------------------------------
def mann_kendall(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in counts)) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return s, z, p


def sens_slope(years, x):
    slopes = [(x[j] - x[i]) / (years[j] - years[i])
              for i in range(len(x) - 1) for j in range(i + 1, len(x)) if years[j] != years[i]]
    slope = float(np.median(slopes))
    intercept = float(np.median(np.asarray(x) - slope * np.asarray(years)))
    return slope, intercept


# ----------------------------------------------------------------------
# B. Period comparison
# ----------------------------------------------------------------------
def doy_anomaly(t, v, window_days=31):
    doy = t.dt.dayofyear.clip(upper=365)
    clim = pd.Series(v.to_numpy(), index=doy.to_numpy()).groupby(level=0).mean().reindex(range(1, 366))
    padded = pd.concat([clim.iloc[-window_days:], clim, clim.iloc[:window_days]])
    smooth = padded.rolling(window_days, center=True, min_periods=5).mean().iloc[window_days:-window_days]
    return v.to_numpy() - smooth.reindex(doy.to_numpy()).to_numpy()


def boot_mean_diff(late, early, rng):
    late, early = late[~np.isnan(late)], early[~np.isnan(early)]
    d = [rng.choice(late, len(late)).mean() - rng.choice(early, len(early)).mean() for _ in range(N_BOOT)]
    return late.mean() - early.mean(), np.array(d)


def main():
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    analogs = pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv", parse_dates=["time"])
    ts = pd.read_csv(config.DATA_INTERIM / "features_timeseries.csv", parse_dates=["time"]).sort_values("time")
    ts = ts[(ts["time"].dt.year >= YEARS[0]) & (ts["time"].dt.year <= YEARS[1])].reset_index(drop=True)

    # --- A
    yrs = np.arange(YEARS[0], YEARS[1] + 1)
    counts = analogs["time"].dt.year.value_counts().reindex(yrs, fill_value=0).to_numpy()
    s, z, p = mann_kendall(counts)
    slope, intercept = sens_slope(yrs, counts)
    trend = pd.DataFrame([{"years": f"{YEARS[0]}-{YEARS[1]}", "n_analog_days": int(counts.sum()),
                           "mk_S": s, "mk_z": z, "mk_p": p,
                           "sen_slope_per_decade": slope * 10,
                           "mean_early": counts[yrs <= SPLIT_YEAR].mean(),
                           "mean_late": counts[yrs > SPLIT_YEAR].mean()}])
    trend.to_csv(RESULTS / "climate_context_trend.csv", index=False)
    print(f"A. Analog days per year: {counts[yrs <= SPLIT_YEAR].mean():.1f} ({YEARS[0]}-{SPLIT_YEAR}) -> "
          f"{counts[yrs > SPLIT_YEAR].mean():.1f} ({SPLIT_YEAR + 1}-{YEARS[1]})")
    print(f"   Mann-Kendall z = {z:.2f}, p = {p:.3f}; Sen's slope = {slope * 10:+.2f} days/decade")
    plot_trend(yrs, counts, slope, intercept, p)

    # --- B
    nearest = ts.set_index("time").index.get_indexer(analogs["time"], method="nearest")
    is_analog = np.zeros(len(ts), dtype=bool)
    is_analog[nearest] = True
    late = (ts["time"].dt.year > SPLIT_YEAR).to_numpy()

    rows = []
    print(f"\nB. Late ({SPLIT_YEAR + 1}-{YEARS[1]}) minus early ({YEARS[0]}-{SPLIT_YEAR}), day-of-year anomalies:")
    for col, (label, unit) in VARIABLES.items():
        if col not in ts.columns:
            print(f"   ({col} not in features_timeseries.csv -- skipped)")
            continue
        a = doy_anomaly(ts["time"], ts[col])
        d_an, b_an = boot_mean_diff(a[is_analog & late], a[is_analog & ~late], rng)
        d_all, b_all = boot_mean_diff(a[late], a[~late], rng)
        b_ex = b_an - b_all
        ci = lambda b: np.percentile(b, [2.5, 97.5])  # noqa: E731
        rows.append({"variable": col, "label": label, "unit": unit,
                     "n_analog_early": int((is_analog & ~late).sum()), "n_analog_late": int((is_analog & late).sum()),
                     "delta_analog": d_an, "delta_analog_lo": ci(b_an)[0], "delta_analog_hi": ci(b_an)[1],
                     "delta_all": d_all, "delta_all_lo": ci(b_all)[0], "delta_all_hi": ci(b_all)[1],
                     "excess": d_an - d_all, "excess_lo": ci(b_ex)[0], "excess_hi": ci(b_ex)[1]})
        r = rows[-1]
        flag = "  <- analog days changed more than the season" if r["excess_lo"] > 0 else ""
        print(f"   {label:<22} analog {d_an:+.2f} [{r['delta_analog_lo']:+.2f},{r['delta_analog_hi']:+.2f}]  "
              f"all days {d_all:+.2f}  excess {r['excess']:+.2f} [{r['excess_lo']:+.2f},{r['excess_hi']:+.2f}]{flag}")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "climate_context_periods.csv", index=False)
    print(f"\nSaved: {RESULTS / 'climate_context_trend.csv'}\nSaved: {RESULTS / 'climate_context_periods.csv'}")
    if not res.empty:
        plot_periods(res)


def plot_trend(yrs, counts, slope, intercept, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(yrs, counts, color=COL_ANALOG, width=0.75, edgecolor="white", linewidth=1)
    ax.plot(yrs, intercept + slope * yrs, color="#1a1a1a", lw=1.5,
            label=f"Sen's slope {slope * 10:+.1f} days/decade (Mann-Kendall p = {p:.2f})")
    if 2021 in yrs:
        ax.annotate("Bozkurt 2021", xy=(2021, counts[yrs == 2021][0]), xytext=(2014, counts.max() * 0.95),
                    fontsize=8, arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))
    ax.set_ylabel("Analog days per year")
    ax.set_title("How often does a Bozkurt-type circulation occur?", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "fig_climate_trend.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved: {out}")


def plot_periods(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n + 0.6, 3.0), squeeze=False)
    for ax, (_, r) in zip(axes[0], res.iterrows()):
        for x, key, col, lab in [(0, "delta_all", COL_ALL, "all days"), (1, "delta_analog", COL_ANALOG, "analog days")]:
            ax.errorbar(x, r[key], yerr=[[r[key] - r[f"{key}_lo"]], [r[f"{key}_hi"] - r[key]]],
                        fmt="o", color=col, ms=7, capsize=3, lw=1.5, label=lab)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_xticks([0, 1], ["all\ndays", "analog\ndays"], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{r['label']}\n({r['unit']})", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)
    axes[0][0].set_ylabel(f"Change, {SPLIT_YEAR + 1}-{YEARS[1]} minus {YEARS[0]}-{SPLIT_YEAR}", fontsize=8)
    fig.suptitle("Same pattern, more fuel? (day-of-year anomalies, 95% bootstrap CI)", fontsize=10)
    fig.tight_layout()
    out = FIGURES / "fig_climate_periods.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
