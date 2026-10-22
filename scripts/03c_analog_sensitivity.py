"""
Sensitivity of the analog set to the search-domain boundaries (+/- 5 deg).

Question: if the analog search box (config.ANALOG_AREA) were drawn a bit
differently, would we find the same Bozkurt-type days? If the list changes a
lot, the analog set is fragile; if it mostly survives, the result does not
hinge on one arbitrary box.

Method:
  - Variants of the box: each side moved 5 deg inward ("shrink N/S/E/W"),
    all four sides inward ("shrink all"), and -- only where the downloaded
    data actually extends that far -- each side moved 5 deg outward.
  - For each variant, the reference composite and the ACC of every time step
    are recomputed exactly as in 03_find_analogs.py (its functions are
    reused, not copied).
  - Threshold: RANK-MATCHED. Each variant keeps as many top-ACC time steps as
    the baseline significant set had before declustering, then declusters
    the same way. This tests whether the same DAYS come out on top, without
    re-running the 1000-draw null distribution per variant (hours). Noted
    as a simplification.
  - Agreement with the baseline set:
      recall   = share of baseline analog days found again (within +/- the
                 decluster window, config.DECLUSTER_WINDOW_DAYS)
      jaccard  = matched / (baseline + variant - matched)
      acc_corr = correlation of the full ACC time series (all candidates)

Runs independently of 04-07. Every variant re-scores the full record, so this
takes roughly (number of variants) x (the scoring step of 03). Good to start
when the computer would otherwise be idle.

Usage:
    python scripts/03c_analog_sensitivity.py   (run from the project root)

Outputs:
    results/analog_sensitivity.csv
    figures/fig_analog_sensitivity.png
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

_spec = importlib.util.spec_from_file_location("find_analogs", HERE / "03_find_analogs.py")
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)

SHIFT_DEG = 5
OUT_CSV = getattr(config, "RESULTS_DIR", config.PROJECT_ROOT / "results") / "analog_sensitivity.csv"
OUT_FIG = getattr(config, "FIGURES_DIR", config.PROJECT_ROOT / "figures") / "fig_analog_sensitivity.png"


def variants(base, extent):
    """extent = (lat_min, lat_max, lon_min, lon_max) of the available data."""
    out = {"baseline": dict(base)}
    inward = {"north": -SHIFT_DEG, "south": +SHIFT_DEG, "west": +SHIFT_DEG, "east": -SHIFT_DEG}
    for side, d in inward.items():
        b = dict(base); b[side] += d
        out[f"shrink {side}"] = b
    out["shrink all"] = {k: base[k] + inward[k] for k in base}
    limits = {"north": extent[1], "south": extent[0], "west": extent[2], "east": extent[3]}
    for side, d in inward.items():
        b = dict(base); b[side] -= d
        if (side in ("north", "east") and b[side] <= limits[side] + 1e-6) or \
           (side in ("south", "west") and b[side] >= limits[side] - 1e-6):
            out[f"expand {side}"] = b
    return out


def score_box(anom, box):
    domain = fa.subset_domain(anom, box)
    z500 = domain["z500_anom"]
    weights = fa.lat_weights(z500)
    window = z500.sel(time=slice(config.BOZKURT_WINDOW_START, config.BOZKURT_WINDOW_END))
    reference = window.mean("time").compute()
    acc = fa.weighted_acc(z500, reference, weights, dims=("latitude", "longitude")).compute()
    scores = pd.DataFrame({"time": pd.to_datetime(z500.time.values), "acc": acc.values})
    buf = pd.Timedelta(days=config.DECLUSTER_WINDOW_DAYS)
    is_self = (scores["time"] >= pd.Timestamp(config.BOZKURT_WINDOW_START) - buf) & \
              (scores["time"] <= pd.Timestamp(config.BOZKURT_WINDOW_END) + buf)
    return scores[~is_self].reset_index(drop=True)


def match(base_times, var_times, tol_days):
    base = np.array(sorted(base_times), dtype="datetime64[ns]")
    var = np.array(sorted(var_times), dtype="datetime64[ns]")
    if len(base) == 0 or len(var) == 0:
        return 0
    tol = np.timedelta64(int(tol_days * 86400), "s")
    return int(sum((np.abs(var - b) <= tol).any() for b in base))


def main():
    print("Loading anomalies...")
    anom = xr.open_zarr(fa.ANOM_ZARR)
    lat, lon = anom["latitude"].values, anom["longitude"].values
    extent = (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()))
    print(f"Data extent: lat {extent[0]}..{extent[1]}, lon {extent[2]}..{extent[3]}; "
          f"baseline box {config.ANALOG_AREA}")

    baseline_set = pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv", parse_dates=["time"])
    all_scores = pd.read_csv(config.DATA_PROCESSED / "analogs_all_scores.csv", parse_dates=["time"])
    null_acc = pd.read_csv(config.DATA_PROCESSED / "analogs_null_distribution.csv")["null_acc"].values
    threshold = float(np.quantile(null_acc, config.ANALOG_SIGNIFICANCE_LEVEL))
    # same candidate pool as 03: the Bozkurt window (+/- decluster buffer) is not a candidate
    buf = pd.Timedelta(days=config.DECLUSTER_WINDOW_DAYS)
    is_self = (all_scores["time"] >= pd.Timestamp(config.BOZKURT_WINDOW_START) - buf) & \
              (all_scores["time"] <= pd.Timestamp(config.BOZKURT_WINDOW_END) + buf)
    n_exceed = int((all_scores.loc[~is_self, "acc"] > threshold).sum())
    print(f"Baseline: {n_exceed} time steps above the significance threshold -> "
          f"{len(baseline_set)} declustered analog days. Variants keep the same number of top steps.\n")

    boxes = variants(config.ANALOG_AREA, extent)
    rows, base_scores = [], None
    for i, (name, box) in enumerate(boxes.items(), start=1):
        print(f"[{i}/{len(boxes)}] {name}: {box}")
        scores = score_box(anom, box)
        if name == "baseline":
            base_scores = scores.set_index("time")["acc"]
        top = scores.nlargest(n_exceed, "acc")
        chosen = fa.decluster(top, "acc", config.DECLUSTER_WINDOW_DAYS)
        m = match(baseline_set["time"], chosen["time"], config.DECLUSTER_WINDOW_DAYS)
        acc_corr = float(np.corrcoef(base_scores.reindex(scores["time"]).to_numpy(),
                                     scores["acc"].to_numpy())[0, 1])
        rows.append({"variant": name, **{f"box_{k}": v for k, v in box.items()},
                     "n_analogs": len(chosen), "matched": m,
                     "recall": m / len(baseline_set),
                     "jaccard": m / (len(baseline_set) + len(chosen) - m),
                     "acc_corr": acc_corr})
        if name == "baseline" and m < len(baseline_set):
            print("    (baseline should recover ~100% -- a lower value means 03's settings changed since it ran)")
        print(f"    {len(chosen)} analog days, {m}/{len(baseline_set)} baseline days recovered "
              f"({m / len(baseline_set):.0%}), ACC-series correlation {acc_corr:.3f}")

    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = res[res["variant"] != "baseline"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(sub) + 1.4))
    ax.barh(range(len(sub)), sub["recall"], color="#2a78d6", height=0.6, edgecolor="white", linewidth=2)
    for r, (rec, m, n) in enumerate(zip(sub["recall"], sub["matched"], sub["n_analogs"])):
        ax.text(min(rec + 0.01, 0.9), r, f"{rec:.0%}  ({m} days)", va="center", fontsize=8, color="#1a1a1a")
    ax.set_yticks(range(len(sub)), [f"{v} {SHIFT_DEG}°" for v in sub["variant"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of baseline analog days recovered (± decluster window)")
    ax.set_title("Analog set vs. search-domain boundaries", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
