"""
Poster-ready diagnostic figures for the Bozkurt synoptic analog search.

Reads the outputs of 03_find_analogs.py (data/processed/analogs_*.csv) and
the anomaly Zarr store, and produces four figures:

  fig_analog_significance.png   Why the threshold is where it is: the null
                                 (random-pair) ACC distribution vs. the real
                                 1979-2024 candidate-day distribution, with
                                 the significance threshold and Bozkurt's own
                                 self-match marked. This is the figure that
                                 justifies the significance-based cutoff --
                                 good for a methods panel.
  fig_analog_timeline.png       When the significant analog days occur:
                                 count per year (any long-term trend?) and
                                 day-of-year / seasonal distribution.
  fig_analog_composite.png      Physical sanity check: the mean Z500 anomaly
                                 pattern of the significant analog days,
                                 side by side with the Bozkurt reference
                                 composite they were matched against. If the
                                 method is doing its job, these should look
                                 like the same kind of trough.
  fig_analog_acc_rmse.png       ACC vs RMSE scatter for every candidate day,
                                 highlighting the three analog sets -- shows
                                 that "significant" isn't just "low RMSE",
                                 i.e. pattern shape and magnitude are
                                 genuinely different things to select on.

Usage:
    python scripts/03b_analog_diagnostics.py
    (run 03_find_analogs.py first -- this script only reads its outputs)
"""

import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ANOM_ZARR = config.DATA_INTERIM / "anomalies_1979_2024.zarr"
PROC_DIR = config.DATA_PROCESSED
FIG_DIR = config.FIGURES_DIR

ALL_SCORES_CSV = PROC_DIR / "analogs_all_scores.csv"
NULL_DIST_CSV = PROC_DIR / "analogs_null_distribution.csv"
SIGNIFICANT_CSV = PROC_DIR / "analogs_significant.csv"
TOPN_CSV = PROC_DIR / "analogs_topN.csv"
PERCENTILE_CSV = PROC_DIR / "analogs_percentile.csv"


def subset_domain(ds, box):
    sub = ds.sel(latitude=slice(box["north"], box["south"]),
                 longitude=slice(box["west"], box["east"]))
    if sub.sizes.get("latitude", 0) == 0:
        sub = ds.sel(latitude=slice(box["south"], box["north"]),
                      longitude=slice(box["west"], box["east"]))
    return sub


# ----------------------------------------------------------------------
def fig_significance(all_scores, null_acc, threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_acc, bins=40, density=True, alpha=0.55, color="#888888",
            label=f"Null distribution (random window/day pairs, n={len(null_acc)})")
    ax.hist(all_scores["acc"], bins=60, density=True, alpha=0.5, color="#1f77b4",
            label=f"All 1979-2024 candidate days (n={len(all_scores)})")
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5,
               label=f"95th percentile of null (threshold = {threshold:.2f})")

    bozkurt_acc = all_scores.loc[
        all_scores["time"].between(config.BOZKURT_WINDOW_START, config.BOZKURT_WINDOW_END), "acc"
    ].max()
    ax.axvline(bozkurt_acc, color="black", linestyle=":", linewidth=1.5,
               label=f"Bozkurt window vs. its own composite ({bozkurt_acc:.2f})")

    ax.set_xlabel("Anomaly correlation coefficient (ACC)")
    ax.set_ylabel("Density")
    ax.set_title(f"Significance of the Bozkurt synoptic analog pattern match "
                 f"({getattr(config, 'ANALOG_VARIANT', 'uncentred')} ACC)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "fig_analog_significance.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


def fig_timeline(significant):
    years = significant["time"].dt.year
    doy = significant["time"].dt.dayofyear

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    year_counts = years.value_counts().reindex(
        range(config.FULL_YEARS[0], config.FULL_YEARS[1] + 1), fill_value=0
    )
    ax1.bar(year_counts.index, year_counts.values, color="#1f77b4", width=0.8)
    ax1.axvline(2021, color="crimson", linestyle="--", linewidth=1, label="2021 (Bozkurt)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Significant analog days")
    ax1.set_title("(a) Analog days per year, 1979-2024")
    ax1.legend(fontsize=8)

    ax2.hist(doy, bins=np.arange(doy.min() - 5, doy.max() + 6, 5), color="#ff7f0e", edgecolor="white")
    bozkurt_doy = pd.Timestamp(config.BOZKURT_EVENT_DATE).dayofyear
    ax2.axvline(bozkurt_doy, color="crimson", linestyle="--", linewidth=1, label="Bozkurt event day")
    ax2.set_xlabel("Day of year")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Seasonal distribution of analog days")
    ax2.legend(fontsize=8)

    fig.suptitle(f"When the {len(significant)} significant analog days occur", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_analog_timeline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


def fig_composite(domain_z500, reference, significant_times):
    analog_composite = domain_z500.sel(time=significant_times, method="nearest").mean("time").compute()

    lon = reference.longitude
    lat = reference.latitude
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), subplot_kw={"projection": proj})
    levels = np.arange(-80, 81, 10)
    cf = None

    titles = [
        f"Bozkurt reference composite\n({config.BOZKURT_WINDOW_START} to "
        f"{config.BOZKURT_WINDOW_END}, n=16 steps)",
        f"Significant analog composite\n(n={len(significant_times)} days)",
    ]
    for ax, field, title in zip(axes, [reference, analog_composite], titles):
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle=":")
        cf = ax.contourf(lon, lat, field, levels=levels, cmap="RdBu_r", extend="both", transform=proj)
        ax.plot(config.BOZKURT_LON, config.BOZKURT_LAT, marker="*", color="black", markersize=12,
                markeredgecolor="white", transform=proj, zorder=10)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title(title, fontsize=10)

    fig.colorbar(cf, ax=axes, shrink=0.8, pad=0.02, label="Z500 anomaly (gpm)")
    fig.suptitle(f"Reference vs. analog-composite Z500 anomaly pattern "
                 f"({getattr(config, 'ANALOG_VARIANT', 'uncentred')} ACC)", fontsize=13)
    out = FIG_DIR / "fig_analog_composite.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


SLOT_W_MM, SLOT_H_MM = 185, 80     # Canva template: section-1 map slot


def fig_composite_poster(domain_z500, reference, significant_times):
    """One map for the poster: shading = Bozkurt reference composite,
    contours = mean of the significant analog days. Same style as 08."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_figures", Path(__file__).resolve().parent / "08_make_figures.py")
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)
    figs.style()
    analog = domain_z500.sel(time=significant_times, method="nearest").mean("time").compute()
    lon, lat = reference.longitude, reference.latitude
    proj = ccrs.PlateCarree()
    w_in = SLOT_W_MM / 25.4 / figs.POSTER_SCALE
    h_in = SLOT_H_MM / 25.4 / figs.POSTER_SCALE
    fig = figs.plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0.0, 0.02, 0.84, 0.96], projection=proj)
    ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], crs=proj)
    cf = ax.contourf(lon, lat, reference, levels=np.arange(-80, 81, 10), cmap="RdBu_r",
                     extend="both", transform=proj)
    ax.contour(lon, lat, analog, levels=[-40, -20, 20, 40], colors=figs.C_PRECURSORS,
               linewidths=0.9, transform=proj)          # negative levels dashed by default
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor=figs.INK)
    ax.plot(config.BOZKURT_LON, config.BOZKURT_LAT, marker="*", markersize=12, markerfacecolor="white",
            markeredgecolor=figs.C_BOZKURT, markeredgewidth=1.6, transform=proj, zorder=10)
    cax = fig.add_axes([0.87, 0.12, 0.025, 0.76])
    cb = fig.colorbar(cf, cax=cax, ticks=[-80, -40, 0, 40, 80])
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.set_label("Z500 anomaly (gpm)", fontsize=7.5)
    figs.OUT = Path(config.FIGURES_DIR) / "poster"
    figs.save(fig, "fig_poster_bozkurt_map", SLOT_W_MM)


def fig_acc_rmse(all_scores, significant, topn, percentile):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_scores["rmse"], all_scores["acc"], s=6, alpha=0.15, color="gray",
               label="All candidate days")
    ax.scatter(percentile["rmse"], percentile["acc"], s=22, alpha=0.8, color="#2ca02c",
               label=f"Top {config.ANALOG_PERCENTILE}% (n={len(percentile)})")
    ax.scatter(significant["rmse"], significant["acc"], s=22, alpha=0.8, color="#1f77b4",
               label=f"Significant (n={len(significant)})")
    ax.scatter(topn["rmse"], topn["acc"], s=45, alpha=0.9, color="crimson", marker="D",
               label=f"Top-{config.ANALOG_N}")
    ax.set_xlabel("RMSE (gpm)")
    ax.set_ylabel("ACC")
    ax.set_title("Pattern shape (ACC) vs. magnitude error (RMSE)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig_analog_acc_rmse.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


# ----------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_scores = pd.read_csv(ALL_SCORES_CSV, parse_dates=["time"])
    null_acc = pd.read_csv(NULL_DIST_CSV)["null_acc"].values
    threshold = float(np.quantile(null_acc, config.ANALOG_SIGNIFICANCE_LEVEL))
    significant = pd.read_csv(SIGNIFICANT_CSV, parse_dates=["time"])
    topn = pd.read_csv(TOPN_CSV, parse_dates=["time"])
    percentile = pd.read_csv(PERCENTILE_CSV, parse_dates=["time"])

    print("Figure 1/4: significance histogram...")
    fig_significance(all_scores, null_acc, threshold)

    print("Figure 2/4: timeline...")
    fig_timeline(significant)

    print("Figure 3/4: composite maps (reopens the anomaly Zarr)...")
    anom = xr.open_zarr(ANOM_ZARR)
    domain = subset_domain(anom, config.ANALOG_AREA)
    z500 = domain["z500_anom"]
    window = z500.sel(time=slice(config.BOZKURT_WINDOW_START, config.BOZKURT_WINDOW_END))
    reference = window.mean("time").compute()
    fig_composite(z500, reference, significant["time"].values)
    fig_composite_poster(z500, reference, significant["time"].values)

    print("Figure 4/4: ACC vs RMSE scatter...")
    fig_acc_rmse(all_scores, significant, topn, percentile)

    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
