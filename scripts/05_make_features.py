"""
Build the lagged feature table for modeling (Plan B, per the roadmap's risk
table: flood labels turned out too sparse/coarse to train on directly, so
the model's target is "is this day a significant synoptic analog of
Bozkurt", with the flood labels from 04 carried along as supporting
evidence, not the training target).

Three stages:

  1. build_extra_subset_zarr() -- 02_build_climatology.py already extracted
     Z500/U850/V850/MSLP into era5_subset.zarr. This does the same thing for
     the remaining variables needed here: U500/V500 (vertical shear), CAPE,
     CIN, TCWV, soil moisture (SWVL1-3), and IVT components -- straight from
     the same raw GRIB files. Skipped automatically if already built.
  2. build_timeseries() -- combines that, era5_subset.zarr, and the Z500/MSLP
     anomalies into ONE 1979-2024 6-hourly table of domain-aggregated
     features (area mean/max over config.THERMO_AREA, cos(lat)-weighted).
  3. build_feature_table() -- picks the sample days (158 significant analogs
     as positives, random non-analog days as negative controls, kept
     config.CONTROL_BUFFER_DAYS away from any positive), reads off each
     feature at lags 0/12/24/48/72h, and assembles the wide table
     06_train_models.py will train on. Flood labels from 04 are merged in by
     date, for reporting, not as the target.

Usage:
    python scripts/05_make_features.py   (run from the project root)

Outputs:
    data/interim/era5_features_extra.zarr
    data/interim/features_timeseries.csv
    data/processed/features_table.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EXTRA_ZARR = config.DATA_INTERIM / "era5_features_extra.zarr"   # old single store (interrupted run)
EXTRA_DIR = config.DATA_INTERIM / "era5_features_extra"         # one small store per GRIB file
SUBSET_ZARR = config.DATA_INTERIM / "era5_subset.zarr"
ANOM_ZARR = config.DATA_INTERIM / "anomalies_1979_2024.zarr"
TIMESERIES_CSV = config.DATA_INTERIM / "features_timeseries.csv"
FEATURE_TABLE_CSV = config.DATA_PROCESSED / "features_table.csv"


# ----------------------------------------------------------------------
# Stage 1: extra raw variables -> their own Zarr store (same pattern as 02)
# ----------------------------------------------------------------------
G = 9.80665  # m s-2


def accum_to_valid_time(da):
    """ERA5 accumulations (tp) come as (time=forecast base 06/18 UTC, step);
    re-index them on the actual valid time. Taking step=0 instead would drop
    half the values and stamp the rest with the wrong hour."""
    if "step" not in da.dims:
        return da
    if "time" not in da.dims:
        return da.swap_dims({"step": "valid_time"}).rename(valid_time="time")
    st = da.stack(vt=("time", "step"))
    valid = st["valid_time"].values
    st = st.drop_vars([c for c in st.coords if c not in ("latitude", "longitude")], errors="ignore")
    st = st.rename(vt="time").assign_coords(time=valid)
    return st.dropna("time", how="all").sortby("time")


def open_grib(path):
    import cfgrib  # only needed for the GRIB extraction step
    parts = []
    for ds in cfgrib.open_datasets(
        str(path), backend_kwargs={"indexpath": "", "errors": "ignore"}
    ):
        if "step" in ds.dims:
            # Forecast-stream fields (tp, but also CIN and others) come as
            # (time = forecast base 06/18 UTC, step). Taking step=0 would stamp
            # them 6 h early and leave 00/12 UTC empty -- re-index ALL of them
            # on their real valid time instead.
            ds = xr.Dataset({v: accum_to_valid_time(ds[v]) for v in ds.data_vars})
        ds = ds.reset_coords(drop=True)
        parts.append(ds)
    return xr.merge(parts, compat="override", join="outer")


def thermo_subset(ds):
    box = config.THERMO_AREA
    sub = ds.sel(latitude=slice(box["north"], box["south"]), longitude=slice(box["west"], box["east"]))
    if sub.sizes.get("latitude", 0) == 0:
        sub = ds.sel(latitude=slice(box["south"], box["north"]), longitude=slice(box["west"], box["east"]))
    return sub


def pressure_outputs(ds):
    """From one pressure-level file: U500/V500 (full domain, as before) and
    IVT computed from q, u, v -- the full-record single-level files have no
    viwve/viwvn. IVT = (1/g) * integral of q*wind dp over the available levels
    (925-500 hPa), on the thermodynamic box only. A truncated column (misses
    moisture below 925 hPa) -- documented as a limitation."""
    out = {"pressure": xr.Dataset({
        "u500": ds["u"].sel(isobaricInhPa=500).drop_vars("isobaricInhPa"),
        "v500": ds["v"].sel(isobaricInhPa=500).drop_vars("isobaricInhPa"),
    })}
    sub = thermo_subset(ds[["q", "u", "v"]]).sortby("isobaricInhPa")
    # integrate over hPa, x100 -> Pa; divide by g -> kg m-1 s-1
    out["ivt"] = xr.Dataset({
        "ivt_e": (sub["q"] * sub["u"]).integrate("isobaricInhPa") * 100.0 / G,
        "ivt_n": (sub["q"] * sub["v"]).integrate("isobaricInhPa") * 100.0 / G,
    })
    return out


def single_outputs(ds):
    data = {}
    for short in ["cape", "cin", "tcwv", "swvl1", "swvl2", "swvl3"]:
        if short in ds.data_vars:
            data[short] = ds[short]
    out = {"single": xr.Dataset(data)}
    if "tp" in ds.data_vars:
        # 1-hour accumulation ending at each valid time, metres -> mm.
        # Only 00/06/12/18 were downloaded, so this samples 4 of every 24 hours.
        out["precip"] = xr.Dataset({"tp_1h_mm": thermo_subset(ds[["tp"]])["tp"] * 1000.0})
    else:
        print("  (no tp in this file -- precipitation features will be missing)")
    if "sst" in ds.data_vars:
        # Sea points only (land is NaN in ERA5 SST), thermodynamic box.
        out["sst"] = xr.Dataset({"sst": thermo_subset(ds[["sst"]])["sst"]})
    else:
        print("  (no sst in this file -- SST features will be missing)")
    return out


GROUP_VERSION = {"single": 2}   # v2: forecast-stream fields (CIN) on their valid time


def _version_marker(group, stem):
    v = GROUP_VERSION.get(group, 1)
    return EXTRA_DIR / group / f"{stem}.v{v}" if v > 1 else None


def _write_one(ds, group, stem):
    """Each GRIB file gets its own small store, written under a temporary name
    and renamed only when complete -- so an interrupted run never leaves a
    store that looks finished but isn't, and a rerun resumes where it stopped."""
    import shutil
    folder = EXTRA_DIR / group
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"{stem}.zarr"
    tmp = folder / f"{stem}.zarr.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.to_zarr(tmp, mode="w")
    if final.exists():          # replacing an outdated store
        shutil.rmtree(final)
    tmp.rename(final)
    marker = _version_marker(group, stem)
    if marker is not None:
        marker.write_text("ok")


def _done(group, stem):
    marker = _version_marker(group, stem)
    store_ok = (EXTRA_DIR / group / f"{stem}.zarr").exists()
    return store_ok and (marker is None or marker.exists())


def build_extra_subset_zarr():
    """Resumable: files whose store already exists are skipped."""
    if EXTRA_ZARR.exists():
        raise RuntimeError(
            f"Found the old single store {EXTRA_ZARR} -- it's from the interrupted earlier run "
            f"and is probably half-written. Delete that folder and run again "
            f"(new runs write to {EXTRA_DIR.name}/ instead and can resume safely)."
        )

    pl_files = sorted(config.DATA_RAW_ERA5_PRESSURE.glob("*.grib"))
    sl_files = sorted(config.DATA_RAW_ERA5_SINGLE.glob("*.grib"))
    if not pl_files or not sl_files:
        raise FileNotFoundError(
            f"Expected pressure files in {config.DATA_RAW_ERA5_PRESSURE} and "
            f"single-level files in {config.DATA_RAW_ERA5_SINGLE} (same files 02 used)."
        )

    jobs = [(f, "pressure", ("pressure", "ivt")) for f in pl_files] + \
           [(f, "single", ("single", "precip", "sst")) for f in sl_files]
    todo = [(f, kind, [g for g in groups if not _done(g, f.stem)]) for f, kind, groups in jobs]
    todo = [t for t in todo if t[2]]
    if not todo:
        print(f"All {len(jobs)} GRIB files already extracted into {EXTRA_DIR} -- skipping the build step.")
        return
    print(f"{len(jobs) - len(todo)}/{len(jobs)} files complete, {len(todo)} still need work.")

    for i, (f, kind, missing) in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] {kind}: {f.name} -> {missing}")
        ds = open_grib(f)
        outputs = pressure_outputs(ds) if kind == "pressure" else single_outputs(ds)
        for g in missing:
            if g in outputs and outputs[g].data_vars:
                sub = outputs[g].sortby("time").drop_duplicates("time")
                _write_one(sub, g, f.stem)

    print(f"\nSaved: {EXTRA_DIR}")


def open_extra(group):
    stores = sorted((EXTRA_DIR / group).glob("*.zarr"))
    parts = [xr.open_zarr(s) for s in stores]
    return xr.concat(parts, dim="time").sortby("time").drop_duplicates("time")


def open_zarr_any(path):
    """Open a Zarr store whether its variables sit at the root or in
    sub-groups (02_build_climatology.py writes 'pressure' / 'single' groups)."""
    root = xr.open_zarr(path)
    if root.data_vars:
        return root.drop_duplicates("time") if "time" in root.dims else root
    groups = sorted(p.name for p in Path(path).iterdir()
                    if p.is_dir() and ((p / ".zgroup").exists() or (p / "zarr.json").exists()))
    if not groups:
        raise ValueError(f"No variables and no groups found in {path}")
    parts = []
    for g in groups:
        ds = xr.open_zarr(path, group=g)
        if "time" in ds.dims:
            ds = ds.drop_duplicates("time")
        parts.append(ds)
    print(f"  {Path(path).name}: groups {groups} -> variables {sorted(v for p in parts for v in p.data_vars)}")
    return xr.merge(parts, join="inner")


# ----------------------------------------------------------------------
# Stage 2: domain-aggregated feature time series, full record
# ----------------------------------------------------------------------
def subset_domain(ds, box):
    sub = ds.sel(latitude=slice(box["north"], box["south"]),
                 longitude=slice(box["west"], box["east"]))
    if sub.sizes.get("latitude", 0) == 0:
        sub = ds.sel(latitude=slice(box["south"], box["north"]),
                     longitude=slice(box["west"], box["east"]))
    return sub


EXPECTED_COLUMN_BY_GROUP = {"ivt": "ivt_mean", "precip": "tp_1h_mean", "sst": "sst_anom_mean"}
TIMESERIES_VERSION = 2   # v2: CIN on valid time (all four synoptic hours)
TIMESERIES_MARKER = TIMESERIES_CSV.with_name(TIMESERIES_CSV.stem + f".v{TIMESERIES_VERSION}")


def build_timeseries():
    if TIMESERIES_CSV.exists():
        cached = pd.read_csv(TIMESERIES_CSV, parse_dates=["time"])
        missing = [g for g, col in EXPECTED_COLUMN_BY_GROUP.items()
                   if (EXTRA_DIR / g).exists() and col not in cached.columns]
        if not missing and TIMESERIES_MARKER.exists():
            print(f"Found existing {TIMESERIES_CSV} -- skipping.")
            return cached
        if not missing:
            missing = ["CIN valid-time fix"]
        print(f"Cached {TIMESERIES_CSV.name} predates the {missing} features -- rebuilding it.")

    subset = open_zarr_any(SUBSET_ZARR)
    extra_pl = open_extra("pressure")
    extra_sl = open_extra("single")
    extra_ivt = open_extra("ivt") if (EXTRA_DIR / "ivt").exists() else None
    extra_pr = open_extra("precip") if (EXTRA_DIR / "precip").exists() else None
    extra_sst = open_extra("sst") if (EXTRA_DIR / "sst").exists() else None
    anom = open_zarr_any(ANOM_ZARR)

    needed = {"u850", "v850"} - set(subset.data_vars)
    if needed:
        raise KeyError(f"{SUBSET_ZARR.name} is missing {sorted(needed)}; it has {sorted(subset.data_vars)}. "
                       "Paste this line so the variable names can be matched.")
    if not {"z500_anom", "mslp_anom"} <= set(anom.data_vars):
        raise KeyError(f"{ANOM_ZARR.name} has {sorted(anom.data_vars)}, expected z500_anom / mslp_anom. "
                       "Paste this line so the names can be matched.")

    ds = xr.merge([subset, extra_pl, extra_sl], join="inner")
    thermo = subset_domain(ds, config.THERMO_AREA)
    weights = np.cos(np.deg2rad(thermo.latitude))
    dims = ("latitude", "longitude")

    print("Computing domain-aggregated features...")
    feats = {}

    wind850 = np.hypot(thermo["u850"], thermo["v850"])
    feats["wind850_mean"] = wind850.weighted(weights).mean(dim=dims)
    feats["wind850_max"] = wind850.max(dim=dims)

    shear = np.hypot(thermo["u500"] - thermo["u850"], thermo["v500"] - thermo["v850"])
    feats["shear_850_500_mean"] = shear.weighted(weights).mean(dim=dims)

    # Simplified orographic-forcing proxy: box-mean meridional wind at 850 hPa.
    # The Turkish coast faces NORTH, so onshore flow is NEGATIVE v850 (northerly
    # wind). Name kept for compatibility; sign convention: < 0 = onshore.
    # A coastline-normal projection would be better -- documented limitation.
    feats["onshore_flow_proxy"] = thermo["v850"].weighted(weights).mean(dim=dims)

    for var in ["cape", "cin", "tcwv"]:
        if var in thermo.data_vars:
            feats[f"{var}_mean"] = thermo[var].weighted(weights).mean(dim=dims)
            feats[f"{var}_max"] = thermo[var].max(dim=dims)

    if extra_ivt is not None:
        iv = thermo_subset(extra_ivt).reindex(time=thermo.time)
        ivt = np.hypot(iv["ivt_e"], iv["ivt_n"])
        w_iv = np.cos(np.deg2rad(iv.latitude))
        feats["ivt_mean"] = ivt.weighted(w_iv).mean(dim=dims)
        feats["ivt_max"] = ivt.max(dim=dims)
    else:
        print("  (no IVT group found -- IVT features skipped)")

    if extra_pr is not None:
        pr = thermo_subset(extra_pr).reindex(time=thermo.time)["tp_1h_mm"]
        w_pr = np.cos(np.deg2rad(pr.latitude))
        feats["tp_1h_mean"] = pr.weighted(w_pr).mean(dim=dims)
        feats["tp_1h_max"] = pr.max(dim=dims)
    else:
        print("  (no precip group found -- precipitation features skipped)")

    if extra_sst is not None:
        ss = thermo_subset(extra_sst).reindex(time=thermo.time)["sst"]
        w_ss = np.cos(np.deg2rad(ss.latitude))
        feats["sst_raw_mean"] = ss.weighted(w_ss).mean(dim=dims)   # NaN land points are skipped
    else:
        print("  (no sst group found -- SST features skipped)")

    swvl_vars = [v for v in ["swvl1", "swvl2", "swvl3"] if v in thermo.data_vars]
    if swvl_vars:
        swvl_mean = sum(thermo[v] for v in swvl_vars) / len(swvl_vars)
        feats["soil_moisture_mean"] = swvl_mean.weighted(weights).mean(dim=dims)

    # Z500 / MSLP anomaly at the Bozkurt point (trough strength over the event).
    point = anom.sel(latitude=config.BOZKURT_LAT, longitude=config.BOZKURT_LON, method="nearest")
    feats["z500_anom_bozkurt"] = point["z500_anom"]
    feats["mslp_anom_bozkurt"] = point["mslp_anom"]

    feat_ds = xr.Dataset(feats)
    print("Pulling the result into memory (reduced to a time series -- small)...")
    df = feat_ds.compute().to_dataframe().reset_index()
    df = df[[c for c in df.columns if c == "time" or c in feats]].sort_values("time")
    for c in ["tp_1h_mean", "tp_1h_max"]:
        if c in df.columns:
            # ~24 h total: 4 hourly samples (00/06/12/18) x 6. A proxy, not a true sum.
            df[c.replace("1h", "24h_proxy")] = df[c].rolling(4, min_periods=4).sum() * 6.0
    if "sst_raw_mean" in df.columns:
        df["sst_anom_mean"] = doy_anomaly(df["time"], df["sst_raw_mean"])
        df = df.drop(columns="sst_raw_mean")

    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(TIMESERIES_CSV, index=False)
    TIMESERIES_MARKER.write_text("ok")
    print(f"Saved: {TIMESERIES_CSV} ({len(df)} rows, {len(df.columns) - 1} features)")
    if "cin_mean" in df.columns:
        na_by_hour = df["cin_mean"].isna().groupby(pd.to_datetime(df["time"]).dt.hour).mean()
        print("  CIN missing fraction by hour: " + ", ".join(f"{h:02d}h {v:.0%}" for h, v in na_by_hour.items()))
    return df


def doy_anomaly(times, values, window_days=31):
    """Anomaly relative to a smoothed day-of-year climatology over
    config.CLIMATOLOGY_YEARS (same base period as the Z500/MSLP anomalies)."""
    t = pd.to_datetime(times)
    clim_years = getattr(config, "CLIMATOLOGY_YEARS", (1991, 2020))
    in_clim = (t.dt.year >= clim_years[0]) & (t.dt.year <= clim_years[1])
    doy = t.dt.dayofyear.clip(upper=365)
    clim = pd.Series(values[in_clim].to_numpy(), index=doy[in_clim].to_numpy()).groupby(level=0).mean()
    clim = clim.reindex(range(1, 366))
    # circular smoothing over +/- window/2 days (season edges have no neighbours -> min_periods)
    padded = pd.concat([clim.iloc[-window_days:], clim, clim.iloc[:window_days]])
    smooth = padded.rolling(window_days, center=True, min_periods=5).mean().iloc[window_days:-window_days]
    return values.to_numpy() - smooth.reindex(doy.to_numpy()).to_numpy()


# ----------------------------------------------------------------------
# Stage 3: samples (Plan B target), lagged features, final table
# ----------------------------------------------------------------------
def pick_samples():
    significant = pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv", parse_dates=["time"])
    all_scores = pd.read_csv(config.DATA_PROCESSED / "analogs_all_scores.csv", parse_dates=["time"])

    positives = significant[["time"]].copy()
    positives["label"] = 1

    rng = np.random.default_rng(config.FEATURE_RNG_SEED)
    positive_dates = np.array(pd.to_datetime(positives["time"]).dt.normalize(), dtype="datetime64[ns]")
    buf = np.timedelta64(config.CONTROL_BUFFER_DAYS, "D")

    pool_times = pd.to_datetime(all_scores["time"])
    pool_days = np.array(pool_times.dt.normalize(), dtype="datetime64[ns]")
    far = np.array([(np.abs(positive_dates - d) > buf).all() for d in pool_days])
    pool = all_scores.loc[far, ["time"]].copy()

    # Hour-matched controls: analog time steps are not spread evenly over
    # 00/06/12/18 UTC, so random controls would let any diurnally varying
    # feature (CAPE, wind, ...) leak "which hour is t0". Matching the hour
    # distribution removes that shortcut by design.
    pos_hours = pd.to_datetime(positives["time"]).dt.hour
    pool_hours = pd.to_datetime(pool["time"]).dt.hour
    picked = []
    for h, n_pos in pos_hours.value_counts().sort_index().items():
        cand = pool[pool_hours == h]
        n = min(config.CONTROL_PER_POSITIVE * n_pos, len(cand))
        picked.append(cand.iloc[rng.choice(len(cand), size=n, replace=False)])
    controls = pd.concat(picked)[["time"]].copy()
    controls["label"] = 0
    print("Hour mix (positives = controls by design): "
          + ", ".join(f"{h:02d}h {c / len(pos_hours):.0%}" for h, c in pos_hours.value_counts().sort_index().items()))

    samples = pd.concat([positives, controls], ignore_index=True)
    samples["sample_id"] = [f"s{i:04d}" for i in range(len(samples))]
    print(f"Samples: {len(positives)} positive (significant analog days), "
          f"{len(controls)} control (label=0), {len(samples)} total.")
    return samples


def build_feature_table(timeseries, samples):
    ts = timeseries.set_index("time").sort_index()
    feature_cols = list(ts.columns)

    rows = []
    for _, s in samples.iterrows():
        t0 = pd.Timestamp(s["time"])
        row = {"sample_id": s["sample_id"], "time": t0, "date": t0.normalize(), "label": s["label"]}
        for lag in config.FEATURE_LAG_HOURS:
            t_lag = t0 - pd.Timedelta(hours=lag)
            idx = ts.index.get_indexer([t_lag], method="nearest")[0]
            vals = ts.iloc[idx]
            for col in feature_cols:
                row[f"{col}_lag{lag}"] = vals[col]
        rows.append(row)

    table = pd.DataFrame(rows)

    labels = pd.read_csv(config.DATA_LABELS / "flood_labels.csv", parse_dates=["date"])
    table = table.merge(labels, on="date", how="left")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURE_TABLE_CSV, index=False)
    print(f"\nSaved: {FEATURE_TABLE_CSV} ({len(table)} samples, {len(table.columns)} columns)")
    for col in ["flood_core", "flood_country_core"]:
        if col in table.columns:
            print(f"  {col}: analog days {table.loc[table['label'] == 1, col].mean():.1%} "
                  f"vs control days {table.loc[table['label'] == 0, col].mean():.1%}")
    return table


if __name__ == "__main__":
    build_extra_subset_zarr()
    ts = build_timeseries()
    samples = pick_samples()
    build_feature_table(ts, samples)
