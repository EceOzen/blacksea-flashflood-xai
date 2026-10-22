"""
Numeric summary table for the Bozkurt flash flood (9-12 August 2021).

Turns the maps from 00_explore_bozkurt.py into one row per time step:
  - Z500 and MSLP at the Bozkurt grid point (large-scale signal)
  - Domain-wide MSLP minimum (tracks the low deepening)
  - Thermodynamic-box averages/maxima for CAPE, CIN, TCWV, IVT, 850 hPa wind
    (the small western Black Sea box from config.THERMO_AREA)

This is the reference row set that later analog days get compared against
in find_analogs.py, and it's also a sanity check in its own right: numbers
you can put in a table instead of eyeballing colors on a map.

Usage:
    python scripts/01_bozkurt_summary.py   (run from the project root)
Writes data/processed/bozkurt_summary.csv
"""

import sys
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ----------------------------------------------------------------------
# Settings (mirrors 00_explore_bozkurt.py -- same files, same times)
# ----------------------------------------------------------------------

PL_FILE = config.DATA_RAW_ERA5_BOZKURT / "era_202108_1209_baseline_pressure.grib"
SL_FILES = [
    config.DATA_RAW_ERA5_BOZKURT / "era_202108_1209_baseline_single.grib",
    config.DATA_RAW_ERA5_BOZKURT / "era_202108_1209_baseline_single_missing.grib",
]

# All available 6-hourly steps in the downloaded window (not just the 4 map times).
TIME_START = "2021-08-08T00:00"
TIME_END = "2021-08-12T18:00"

OUT_CSV = config.DATA_PROCESSED / "bozkurt_summary.csv"


# ----------------------------------------------------------------------
# Data loading (same helpers as 00_explore_bozkurt.py)
# ----------------------------------------------------------------------
def open_grib(path):
    parts = []
    for ds in cfgrib.open_datasets(str(path)):
        if "step" in ds.dims:
            ds = ds.isel(step=0)
        ds = ds.reset_coords(drop=True)
        parts.append(ds)
    return xr.merge(parts, compat="override")


def open_grib_multi(paths):
    return xr.merge([open_grib(p) for p in paths], compat="override")


pl = open_grib(PL_FILE)
sl = open_grib_multi(SL_FILES)

Z500 = pl["z"].sel(isobaricInhPa=500) / 9.80665
U850 = pl["u"].sel(isobaricInhPa=850)
V850 = pl["v"].sel(isobaricInhPa=850)
WS850 = np.hypot(U850, V850)
MSLP = sl["msl"] / 100.0
TCWV = sl["tcwv"]
CAPE = sl["cape"]
CIN = sl["cin"]

IVT_U_NAME = next((n for n in ("viwve", "p71.162") if n in sl.data_vars), None)
IVT_V_NAME = next((n for n in ("viwvn", "p72.162") if n in sl.data_vars), None)
HAS_IVT = IVT_U_NAME is not None and IVT_V_NAME is not None
if HAS_IVT:
    IVT = np.hypot(sl[IVT_U_NAME], sl[IVT_V_NAME])

# ----------------------------------------------------------------------
# Thermodynamic box (small, western Black Sea -- config.THERMO_AREA)
# ----------------------------------------------------------------------
box = config.THERMO_AREA
lat_slice = slice(box["north"], box["south"])   # ERA5 latitude is usually descending
lon_slice = slice(box["west"], box["east"])


def box_subset(da):
    sub = da.sel(latitude=lat_slice, longitude=lon_slice)
    if sub.sizes.get("latitude", 0) == 0:  # latitude was ascending instead
        sub = da.sel(latitude=slice(box["south"], box["north"]), longitude=lon_slice)
    return sub


def area_weighted_mean(da):
    """Mean weighted by cos(latitude), since grid cells shrink toward the poles."""
    weights = np.cos(np.deg2rad(da.latitude))
    return da.weighted(weights).mean(dim=("latitude", "longitude"))


# ----------------------------------------------------------------------
# Build the table
# ----------------------------------------------------------------------
times = pl.time.sel(time=slice(TIME_START, TIME_END)).values

rows = []
for t in times:
    z500_pt = float(Z500.sel(time=t, latitude=config.BOZKURT_LAT, longitude=config.BOZKURT_LON, method="nearest"))
    mslp_pt = float(MSLP.sel(time=t, latitude=config.BOZKURT_LAT, longitude=config.BOZKURT_LON, method="nearest"))
    mslp_min = float(MSLP.sel(time=t).min())

    cape_box = box_subset(CAPE.sel(time=t))
    cin_box = box_subset(CIN.sel(time=t))
    tcwv_box = box_subset(TCWV.sel(time=t))
    ws850_box = box_subset(WS850.sel(time=t))

    row = {
        "time": pd.Timestamp(t),
        "z500_at_bozkurt_gpm": round(z500_pt, 1),
        "mslp_at_bozkurt_hpa": round(mslp_pt, 1),
        "mslp_domain_min_hpa": round(mslp_min, 1),
        "cape_thermo_mean_jkg": round(float(area_weighted_mean(cape_box)), 1),
        "cape_thermo_max_jkg": round(float(cape_box.max()), 1),
        "cin_thermo_mean_jkg": round(float(area_weighted_mean(cin_box)), 1),
        "tcwv_thermo_mean_kgm2": round(float(area_weighted_mean(tcwv_box)), 1),
        "tcwv_thermo_max_kgm2": round(float(tcwv_box.max()), 1),
        "wind850_thermo_mean_ms": round(float(area_weighted_mean(ws850_box)), 1),
        "wind850_thermo_max_ms": round(float(ws850_box.max()), 1),
    }

    if HAS_IVT:
        ivt_box = box_subset(IVT.sel(time=t))
        row["ivt_thermo_mean_kgm1s1"] = round(float(area_weighted_mean(ivt_box)), 1)
        row["ivt_thermo_max_kgm1s1"] = round(float(ivt_box.max()), 1)

    rows.append(row)

df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(df.to_string(index=False))
print()
print("Saved:", OUT_CSV)
