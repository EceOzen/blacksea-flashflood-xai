"""
Build the 1991-2020 day-of-year climatology and Z500/MSLP anomalies from the
full 1979-2024 ERA5 download (the year-chunked pressure/single-level files).

Why this script is structured the way it is:
  - The full analog-search domain, 6-hourly, 46 years, is tens of GB even
    after keeping only the few variables we need. Loading it all into memory
    at once can exhaust a laptop's RAM.
  - So each downloaded file is opened, immediately trimmed down to just
    Z500, U850, V850 (pressure-level) or MSLP (single-level), and appended
    to a Zarr store on disk (data/interim/era5_subset.zarr) one file at a
    time. Peak memory is roughly the size of one input file, not the whole
    dataset.
  - The climatology and anomalies are then computed from that Zarr store
    using dask, which streams through the data in chunks instead of loading
    it all at once.

Setup (once, in addition to what 00/01 needed):
    pip install zarr dask

Usage:
    python scripts/02_build_climatology.py   (run from the project root)

Outputs:
    data/interim/era5_subset.zarr           Z500, U850, V850, MSLP, 1979-2024
    data/interim/climatology_1991_2020.nc   day-of-year x hour climatology
    data/interim/anomalies_1979_2024.zarr   Z500 and MSLP anomalies

If it crashes partway through the Zarr-building step, it's safe to just
rerun: already-appended files aren't easy to detect automatically, so the
simplest fix is to delete data/interim/era5_subset.zarr and start over.
For a partial rebuild, comment out files you've confirmed are already in it.
"""

import sys
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SUBSET_ZARR = config.DATA_INTERIM / "era5_subset.zarr"
CLIM_NC = config.DATA_INTERIM / "climatology_1991_2020.nc"
ANOM_ZARR = config.DATA_INTERIM / "anomalies_1979_2024.zarr"

SMOOTH_WINDOW_DAYS = 7  # +/- days of neighbouring calendar days averaged into
                         # each climatology value, so 30 years per exact date
                         # isn't too noisy


# ----------------------------------------------------------------------
# Step 1: trim each downloaded file down to what we need, write to Zarr
# ----------------------------------------------------------------------
def open_grib(path):
    # indexpath="" disables cfgrib's on-disk .idx cache files. Those caches
    # can go stale (especially on Windows, with large multi-GB GRIB files)
    # and cause OSError: [Errno 22] Invalid argument on read. Rebuilding the
    # index in memory each time is a little slower but far more reliable.
    #
    # errors="ignore" skips any individual GRIB message cfgrib can't decode
    # (e.g. one corrupted message from an interrupted download) instead of
    # crashing the whole file. Out of ~55,000 messages per file, losing one
    # time step for one variable is a negligible gap for a 46-year climatology.
    parts = []
    for ds in cfgrib.open_datasets(
        str(path), backend_kwargs={"indexpath": "", "errors": "ignore"}
    ):
        if "step" in ds.dims:
            ds = ds.isel(step=0)
        ds = ds.reset_coords(drop=True)
        parts.append(ds)
    return xr.merge(parts, compat="override")


def load_pressure_subset(path):
    ds = open_grib(path)
    # .drop_vars("isobaricInhPa") removes the leftover scalar level coordinate
    # each .sel() leaves behind (500 for z, 850 for u/v) -- without it,
    # combining the three into one Dataset below fails because they'd share
    # a coordinate name with conflicting values (500 vs 850).
    z500 = (ds["z"].sel(isobaricInhPa=500) / 9.80665).drop_vars("isobaricInhPa")
    u850 = ds["u"].sel(isobaricInhPa=850).drop_vars("isobaricInhPa")
    v850 = ds["v"].sel(isobaricInhPa=850).drop_vars("isobaricInhPa")
    return xr.Dataset({"z500": z500, "u850": u850, "v850": v850})


def load_single_subset(path):
    ds = open_grib(path)
    return xr.Dataset({"mslp": ds["msl"] / 100.0})


def build_subset_zarr():
    if SUBSET_ZARR.exists():
        print(f"Found existing {SUBSET_ZARR} -- skipping the build step.")
        print("(Delete this folder to force a full rebuild.)")
        return

    pl_files = sorted(config.DATA_RAW_ERA5_PRESSURE.glob("*.grib"))
    sl_files = sorted(config.DATA_RAW_ERA5_SINGLE.glob("*.grib"))
    if not pl_files:
        raise FileNotFoundError(f"No pressure-level GRIB files found in {config.DATA_RAW_ERA5_PRESSURE}")
    if not sl_files:
        raise FileNotFoundError(f"No single-level GRIB files found in {config.DATA_RAW_ERA5_SINGLE}")

    print(f"{len(pl_files)} pressure-level files, {len(sl_files)} single-level files.")

    # Pair each pressure file's time range with the matching single-level
    # file(s) isn't required here -- we build the two variable groups
    # separately and merge on time at the end of each pass, so the files
    # don't need to share the same year-chunking.
    print("\n--- Pressure-level files ---")
    for i, f in enumerate(pl_files):
        print(f"[{i + 1}/{len(pl_files)}] {f.name}")
        sub = load_pressure_subset(f).sortby("time").drop_duplicates("time")
        _append_to_zarr(sub, "pressure")

    print("\n--- Single-level files ---")
    for i, f in enumerate(sl_files):
        print(f"[{i + 1}/{len(sl_files)}] {f.name}")
        sub = load_single_subset(f).sortby("time").drop_duplicates("time")
        _append_to_zarr(sub, "single")

    print(f"\nSaved: {SUBSET_ZARR}")


def _append_to_zarr(ds, group):
    """Write (or append along time) to a named group inside the Zarr store."""
    store_exists = SUBSET_ZARR.exists() and (SUBSET_ZARR / group).exists()
    if not store_exists:
        ds.to_zarr(SUBSET_ZARR, group=group, mode="w")
    else:
        ds.to_zarr(SUBSET_ZARR, group=group, mode="a", append_dim="time")


# ----------------------------------------------------------------------
# Step 2: day-of-year x hour climatology (1991-2020), smoothed
# ----------------------------------------------------------------------
def build_climatology():
    if CLIM_NC.exists():
        print(f"Found existing {CLIM_NC} -- skipping.")
        return xr.open_dataset(CLIM_NC)

    pl = xr.open_zarr(SUBSET_ZARR, group="pressure")
    sl = xr.open_zarr(SUBSET_ZARR, group="single")
    # Belt-and-braces: if a split file's original was accidentally left in the
    # source folder alongside its _partNN pieces (or two separate downloads
    # overlapped in time), the same timestamp gets appended twice and merge()
    # fails with "duplicate values" on the time index. Drop repeats here so a
    # leftover-file mistake doesn't crash the whole run -- but this is a sign
    # to go clean up the raw folder, not a substitute for doing so.
    pl = pl.drop_duplicates("time")
    sl = sl.drop_duplicates("time")
    ds = xr.merge([pl, sl], join="inner")

    # Only the downloaded season months actually exist in the data, so bound
    # the slice to those (config.SEASON_MONTHS) rather than the full
    # calendar year -- avoids scanning months we never downloaded.
    clim_start, clim_end = config.CLIMATOLOGY_YEARS
    season_start_month, season_end_month = config.SEASON_MONTHS
    clim_period = ds.sel(time=slice(
        f"{clim_start}-{season_start_month:02d}-01",
        f"{clim_end}-{season_end_month:02d}-30",
    ))
    print(f"Climatology period {clim_start}-{clim_end}, months {season_start_month}-{season_end_month}: "
          f"{clim_period.sizes['time']} time steps")

    # Group by zero-padded (day-of-year, hour) so calendar dates line up
    # across years. Leap-year Feb 29 gets its own small group -- a minor
    # approximation, blurred out by the smoothing step below anyway.
    doy_hour = clim_period.time.dt.strftime("%j-%H")
    clim_period = clim_period.assign_coords(doy_hour=("time", doy_hour.values))
    raw_clim = clim_period.groupby("doy_hour").mean("time")
    raw_clim = raw_clim.sortby("doy_hour")
    print("Computing raw per-date climatology (this triggers the first full read)...")
    raw_clim = raw_clim.compute()

    # Smooth across neighbouring calendar days with a circular rolling mean
    # (Dec 31 wraps around to Jan 1) using the tile-and-slice trick.
    n = raw_clim.sizes["doy_hour"]
    window = SMOOTH_WINDOW_DAYS * 4 * 2 + 1  # 4 steps/day, +/- N days
    tiled = xr.concat([raw_clim, raw_clim, raw_clim], dim="doy_hour")
    smoothed_full = tiled.rolling(doy_hour=window, center=True, min_periods=1).mean()
    smoothed = smoothed_full.isel(doy_hour=slice(n, 2 * n))
    smoothed = smoothed.assign_coords(doy_hour=raw_clim.doy_hour)

    CLIM_NC.parent.mkdir(parents=True, exist_ok=True)
    smoothed.to_netcdf(CLIM_NC)
    print(f"Saved: {CLIM_NC}")
    return smoothed
# ----------------------------------------------------------------------
# Step 3: Z500 / MSLP anomalies for the full 1979-2024 period
# ----------------------------------------------------------------------
def build_anomalies(clim):
    if ANOM_ZARR.exists():
        print(f"Found existing {ANOM_ZARR} -- skipping.")
        return

    pl = xr.open_zarr(SUBSET_ZARR, group="pressure").drop_duplicates("time")
    sl = xr.open_zarr(SUBSET_ZARR, group="single").drop_duplicates("time")
    ds = xr.merge([pl, sl], join="inner")[["z500", "mslp"]]

    doy_hour = ds.time.dt.strftime("%j-%H")
    doy_hour_da = xr.DataArray(doy_hour.values, dims="time", coords={"time": ds.time})
    clim_aligned = clim[["z500", "mslp"]].sel(doy_hour=doy_hour_da)

    anomalies = ds - clim_aligned.drop_vars("doy_hour")
    anomalies = anomalies.rename({"z500": "z500_anom", "mslp": "mslp_anom"})

    print("Writing anomalies to Zarr (this triggers the full computation)...")
    anomalies.to_zarr(ANOM_ZARR, mode="w")
    print(f"Saved: {ANOM_ZARR}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    build_subset_zarr()
    climatology = build_climatology()
    build_anomalies(climatology)

    # Sanity check: print the Bozkurt-day anomaly, should be a large negative
    # Z500 anomaly (a trough/cutoff low) and negative MSLP anomaly (a low).
    anom = xr.open_zarr(ANOM_ZARR)
    check = anom.sel(
        time=config.BOZKURT_EVENT_DATE,
        latitude=config.BOZKURT_LAT,
        longitude=config.BOZKURT_LON,
        method="nearest",
    )
    print("\nSanity check -- Bozkurt day anomaly at the Bozkurt point:")
    print(check.compute())