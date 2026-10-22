"""
Raw ERA5 maps for the Bozkurt flash flood (9-12 August 2021).

Draws one figure per time step (2x2 panels):
  (a) Z500 (shaded) + MSLP (contours)
  (b) 850 hPa wind speed (shaded) + wind vectors
  (c) IVT (shaded) + vapour flux vectors
  (d) TCWV (shaded)

Setup (once):
    pip install xarray cfgrib eccodes matplotlib cartopy numpy

Usage:
    1) Put your GRIB downloads in data/raw/era5/ as era5_pressure_bozkurt.grib
       and era5_single_bozkurt.grib (or edit PL_FILE / SL_FILE below).
    2) python scripts/00_explore_bozkurt.py   (run from the project root)
Figures are saved as PNG files in figures/.
"""

import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
PL_FILE = config.DATA_RAW_ERA5_BOZKURT / "era_202108_1209_baseline_pressure.grib"  # Z, U, V, Q, T (500/700/850/925 hPa)

# The single-level Bozkurt data came in two separate downloads: the original
# request (MSL, TCWV, CAPE, CIN, SST, SWVL) and a follow-up with just the two
# IVT components (viwve/viwvn) that were missing the first time. Both are
# opened and merged below -- SL_FILES can hold one file or several.
SL_FILES = [
    config.DATA_RAW_ERA5_BOZKURT / "era_202108_1209_baseline_single.grib",
    config.DATA_RAW_ERA5_BOZKURT / "era_202108_1209_baseline_single_missing.grib",
]

# Times to plot (UTC). The flood happened on 11 August, so we look at the days before it.
TIMES = [
    "2021-08-09T12:00",
    "2021-08-10T12:00",
    "2021-08-11T00:00",
    "2021-08-11T12:00",
]

OUT_DIR = config.FIGURES_DIR
BOZKURT = (config.BOZKURT_LON, config.BOZKURT_LAT)  # (lon, lat)
QUIVER_STEP = 6          # plot every Nth grid point for vectors (6 x 0.25 deg = 1.5 deg)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def open_grib(path):
    """Open a GRIB file as a single xarray Dataset.

    ERA5 single-level files can mix level types (surface, meanSea,
    depthBelowLandLayer, ...). cfgrib.open_datasets splits them into separate
    Datasets, which we then merge back together.
    """
    parts = []
    for ds in cfgrib.open_datasets(str(path)):
        if "step" in ds.dims:
            ds = ds.isel(step=0)
        # Drop scalar helper coordinates (surface, meanSea, valid_time, ...),
        # keeping only time / latitude / longitude / isobaricInhPa.
        ds = ds.reset_coords(drop=True)
        parts.append(ds)
    return xr.merge(parts, compat="override")


def open_grib_multi(paths):
    """Open and merge several GRIB files that together make up one single-level dataset."""
    datasets = [open_grib(p) for p in paths]
    return xr.merge(datasets, compat="override")


pl = open_grib(PL_FILE)
sl = open_grib_multi(SL_FILES)

print("Pressure-level variables:", list(pl.data_vars))
print("Single-level variables:", list(sl.data_vars))
print("Time range:", str(pl.time.values[0])[:16], "->", str(pl.time.values[-1])[:16])

# Variable names are cfgrib short names. If yours differ, check the printout above.
Z500 = pl["z"].sel(isobaricInhPa=500) / 9.80665      # geopotential -> gpm
U850 = pl["u"].sel(isobaricInhPa=850)
V850 = pl["v"].sel(isobaricInhPa=850)
MSLP = sl["msl"] / 100.0                              # Pa -> hPa
TCWV = sl["tcwv"]                                     # kg m-2
WS850 = np.hypot(U850, V850)

# IVT (vertical integral of water vapour transport) needs viwve/viwvn, which
# aren't always in the first download. If they're missing, skip panel (c)
# instead of crashing -- re-download the Bozkurt single-level request with
# "Vertical integral of eastward/northward water vapour flux" checked to get it.
HAS_IVT = "viwve" in sl.data_vars and "viwvn" in sl.data_vars
if HAS_IVT:
    IVT_U = sl["viwve"]                               # kg m-1 s-1 (eastward)
    IVT_V = sl["viwvn"]                               # kg m-1 s-1 (northward)
    IVT = np.hypot(IVT_U, IVT_V)
else:
    print("NOTE: viwve/viwvn not found in single-level file -- skipping the IVT panel.")
    print("      Re-download the Bozkurt single-level request with the two water")
    print("      vapour flux variables checked to get this panel.")
    
# IVT (vertical integral of water vapour transport). cfgrib usually names
# these viwve/viwvn, but sometimes falls back to their ECMWF param-table
# codes (71.162 / 72.162) instead -- check both.
IVT_U_NAME = next((n for n in ("viwve", "p71.162") if n in sl.data_vars), None)
IVT_V_NAME = next((n for n in ("viwvn", "p72.162") if n in sl.data_vars), None)
HAS_IVT = IVT_U_NAME is not None and IVT_V_NAME is not None
if HAS_IVT:
    IVT_U = sl[IVT_U_NAME]                            # kg m-1 s-1 (eastward)
    IVT_V = sl[IVT_V_NAME]                            # kg m-1 s-1 (northward)
    IVT = np.hypot(IVT_U, IVT_V)
else:
    print("NOTE: IVT components not found in single-level file -- skipping the IVT panel.")
    print("      Re-download the Bozkurt single-level request with the two water")
    print("      vapour flux variables checked to get this panel.")
 


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def base_map(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    ax.plot(*BOZKURT, marker="*", color="red", markersize=12,
            markeredgecolor="k", transform=ccrs.PlateCarree(), zorder=10)


def subsample(da, step):
    return da.isel(latitude=slice(None, None, step), longitude=slice(None, None, step))


def plot_time(t):
    tt = np.datetime64(t)
    z500 = Z500.sel(time=tt, method="nearest")
    mslp = MSLP.sel(time=tt, method="nearest")
    u850 = U850.sel(time=tt, method="nearest")
    v850 = V850.sel(time=tt, method="nearest")
    ws850 = WS850.sel(time=tt, method="nearest")
    tcwv = TCWV.sel(time=tt, method="nearest")
    if HAS_IVT:
        ivt = IVT.sel(time=tt, method="nearest")
        ivt_u = IVT_U.sel(time=tt, method="nearest")
        ivt_v = IVT_V.sel(time=tt, method="nearest")

    lon = z500.longitude
    lat = z500.latitude
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), subplot_kw={"projection": proj})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) Z500 + MSLP
    base_map(ax_a, extent)
    cf = ax_a.contourf(lon, lat, z500, levels=np.arange(5500, 5961, 20),
                       cmap="RdYlBu_r", extend="both", transform=proj)
    cs = ax_a.contour(lon, lat, mslp, levels=np.arange(990, 1041, 2),
                      colors="k", linewidths=0.7, transform=proj)
    ax_a.clabel(cs, fmt="%d", fontsize=7)
    plt.colorbar(cf, ax=ax_a, shrink=0.75, label="Z500 (gpm)")
    ax_a.set_title("(a) Z500 (shaded) and MSLP (contours, hPa)")

    # (b) 850 hPa wind
    base_map(ax_b, extent)
    cf = ax_b.contourf(lon, lat, ws850, levels=np.arange(0, 26, 2),
                       cmap="YlGnBu", extend="max", transform=proj)
    us, vs = subsample(u850, QUIVER_STEP), subsample(v850, QUIVER_STEP)
    ax_b.quiver(us.longitude, us.latitude, us.values, vs.values,
                transform=proj, scale=350, width=0.002)
    plt.colorbar(cf, ax=ax_b, shrink=0.75, label="850 hPa wind speed (m/s)")
    ax_b.set_title("(b) 850 hPa wind")

    # (c) IVT (skipped if viwve/viwvn weren't downloaded)
    if HAS_IVT:
        base_map(ax_c, extent)
        cf = ax_c.contourf(lon, lat, ivt, levels=np.arange(0, 601, 50),
                           cmap="PuRd", extend="max", transform=proj)
        us, vs = subsample(ivt_u, QUIVER_STEP), subsample(ivt_v, QUIVER_STEP)
        ax_c.quiver(us.longitude, us.latitude, us.values, vs.values,
                    transform=proj, scale=6000, width=0.002)
        plt.colorbar(cf, ax=ax_c, shrink=0.75, label="IVT (kg m-1 s-1)")
        ax_c.set_title("(c) Integrated vapour transport (IVT)")
    else:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "IVT not available\n(viwve/viwvn missing from download)",
                   ha="center", va="center", transform=ax_c.transAxes, fontsize=11)

    # (d) TCWV
    base_map(ax_d, extent)
    cf = ax_d.contourf(lon, lat, tcwv, levels=np.arange(0, 61, 5),
                       cmap="GnBu", extend="max", transform=proj)
    plt.colorbar(cf, ax=ax_d, shrink=0.75, label="TCWV (kg m-2)")
    ax_d.set_title("(d) Total column water vapour (TCWV)")

    stamp = str(np.datetime_as_string(z500.time.values, unit="h")).replace("T", " ")
    fig.suptitle(f"ERA5, {stamp} UTC (red star: Bozkurt)", fontsize=14)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUT_DIR / ("bozkurt_" + stamp.replace(" ", "_").replace(":", "") + ".png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print("Saved:", fname)


for t in TIMES:
    plot_time(t)