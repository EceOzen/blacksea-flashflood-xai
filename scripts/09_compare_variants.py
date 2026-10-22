"""
Two definitions of "Bozkurt-like", side by side.

  uncentred  original 03: ACC on raw anomalies. Rewards the full anomaly field,
             including its broad positive (warm) part -> "Bozkurt-like
             CONDITIONS", warm background included.
  centred    domain mean removed from every field before the ACC -> "Bozkurt-
             like circulation SHAPE" only.

Run the chain 03 -> 08 once per variant (config.ANALOG_VARIANT), then this.
Reads both sets of outputs; recomputes nothing heavy.

Outputs (base results/ and figures/poster/ folders):
    results/variant_comparison.csv          one row per metric, one column per variant
    figures/poster/fig_poster_variants.png/.pdf   (a) skill vs lead, (b) analog days per year
    figures/poster/fig_poster_climate_shape.png/.pdf  panel (b) alone, 388 x 78 mm (poster climate slot)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

ROOT = config.PROJECT_ROOT
VARIANTS = {"uncentred": "", "centred": "centred"}
LABELS = {"uncentred": "Conditions (uncentred ACC)", "centred": "Shape only (centred ACC)"}
#COLORS = {"uncentred": "#2a78d6", "centred": "#eb6834"}   # validated palette slots 1-2
COLORS = {"uncentred": "#0E2A47", "centred": "#7A93C0"}   # conditions = gece mavisi, shape = çelik mavisi
C_PRECIP_LS = ":"
YEARS = getattr(config, "FULL_YEARS", (1979, 2024))


def paths(v):
    sub = VARIANTS[v]
    return {"proc": ROOT / "data" / "processed" / sub, "res": ROOT / "results" / sub}


def read(path, **kw):
    return pd.read_csv(path, **kw) if Path(path).exists() else None


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect(v):
    p = paths(v)
    out = {}
    sig = read(p["proc"] / "analogs_significant.csv", parse_dates=["time"])
    if sig is None:
        print(f"({v}: no analogs_significant.csv in {p['proc']} -- run 03 with ANALOG_VARIANT='{v}')")
        return None, out
    out["analog days"] = len(sig)

    trend = read(p["res"] / "climate_context_trend.csv")
    if trend is not None:
        t = trend.iloc[0]
        out["analog days/yr 1979-2001"] = round(t["mean_early"], 2)
        out["analog days/yr 2002-2024"] = round(t["mean_late"], 2)
        out["Sen slope (days/decade)"] = round(t["sen_slope_per_decade"], 2)
        out["Mann-Kendall p"] = round(t["mk_p"], 3)

    per = read(p["res"] / "climate_context_periods.csv")
    if per is not None:
        for _, r in per.iterrows():
            out[f"{r['label']}: analog-day change"] = round(r["delta_analog"], 2)
            out[f"{r['label']}: excess over all days"] = (
                f"{r['excess']:+.2f} [{r['excess_lo']:+.2f}, {r['excess_hi']:+.2f}]")

    m = read(p["res"] / "model_metrics.csv")
    if m is not None:
        for lead in [0, 24, 48]:
            for g in ["precip", "no_precip", "no_precip_logreg"]:
                s = m[(m["lead_h"] == lead) & (m["group"] == g)]
                if len(s):
                    r = s.iloc[0]
                    out[f"PR-AUC {lead} h {g}"] = f"{r['pr_auc']:.2f} [{r['pr_auc_lo']:.2f}-{r['pr_auc_hi']:.2f}]"
        seas = m[m["group"] == "seasonality"]
        if len(seas):
            out["PR-AUC seasonality"] = round(float(seas["pr_auc"].iloc[0]), 2)

    shap = read(p["res"] / "shap" / "importance_by_variable.csv")
    if shap is not None:
        s = shap.query("model == 'no_precip' and lead_h == 24").sort_values("mean_abs_shap", ascending=False)
        out["top SHAP variables (24 h)"] = ", ".join(s["variable"].head(4))
        grp = read(p["res"] / "shap" / "importance_by_group.csv")
        if grp is not None:
            g = grp.query("model == 'all' and lead_h == 24").set_index("group")["share"]
            out["SHAP share 24 h (circ/flow/thermo/precip)"] = " / ".join(
                f"{g.get(k, np.nan):.0%}" for k in ["circulation", "flow", "thermo_moist", "precip"])

    fv = read(p["res"] / "flood_validation_table.csv")
    if fv is not None:
        ind = fv[~fv["is_bozkurt_reference"]]
        for region, gg in ind.groupby("region"):
            out[f"floods above threshold: {region}"] = f"{int(gg['above_threshold'].sum())}/{len(gg)}"

    lab = read(p["proc"] / "analogs_significant_labeled.csv")
    labels = read(ROOT / "data" / "labels" / "flood_labels.csv")
    if lab is not None and labels is not None:
        for c in ["flood_core", "flood_extended"]:
            if c in lab.columns:
                out[f"{c}: analog days vs all days"] = f"{lab[c].mean():.1%} vs {labels[c].mean():.1%}"

    bz = read(p["res"] / "bozkurt_anticipation.csv", parse_dates=["target_time"])
    if bz is not None:
        r = bz.loc[bz["target_time"] == pd.Timestamp("2021-08-11 12:00")]
        if len(r):
            out["Bozkurt 11 Aug 12 UTC score (24 h ahead)"] = (
                f"{r['prob_from_24h_earlier'].iloc[0]:.2f} (> {r['percentile_among_control_days'].iloc[0]:.0%} of controls)")
    return sig, out


def overlap(a, b, tol_days):
    tol = np.timedelta64(int(tol_days * 86400), "s")
    av = np.array(sorted(a), dtype="datetime64[ns]")
    bv = np.array(sorted(b), dtype="datetime64[ns]")
    return int(sum((np.abs(bv - x) <= tol).any() for x in av))


def main():
    sets, table = {}, {}
    for v in VARIANTS:
        sig, out = collect(v)
        if sig is not None:
            sets[v] = sig
            table[v] = out
    if len(sets) < 2:
        print("Need both variants' outputs to compare -- stopping.")
        return

    k = overlap(sets["uncentred"]["time"], sets["centred"]["time"], config.DECLUSTER_WINDOW_DAYS)
    table["uncentred"]["shared analog days"] = f"{k}/{len(sets['uncentred'])}"
    table["centred"]["shared analog days"] = f"{k}/{len(sets['centred'])}"

    comp = pd.DataFrame(table)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    comp.to_csv(ROOT / "results" / "variant_comparison.csv")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(comp.to_string())
    print(f"\nSaved: {ROOT / 'results' / 'variant_comparison.csv'}")
    plot(sets)
    plot_climate_slot(sets)


def plot(sets):
    figs = _load("make_figures", "08_make_figures.py")
    figs.style()
    plt = figs.plt
    cc = _load("climate_context", "07b_climate_context.py")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(figs.FULL_W, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    for v in VARIANTS:
        m = read(paths(v)["res"] / "model_metrics.csv")
        if m is None:
            continue
        for g, ls, lw, band in [("no_precip", "-", 2.2, True), ("precip", C_PRECIP_LS, 1.6, False)]:
            s = m[m["group"] == g].sort_values("lead_h")
            ax.plot(s["lead_h"], s["pr_auc"], ls, marker="o" if g == "no_precip" else None, ms=5,
                    color=COLORS[v], lw=lw, markeredgecolor="white")
            if band:
                ax.fill_between(s["lead_h"], s["pr_auc_lo"], s["pr_auc_hi"], color=COLORS[v], alpha=0.12, lw=0)
    leads = sorted(m["lead_h"].unique())
    ax.set_xticks(leads)
    ax.set_xlabel("Lead time (hours before the analog day)")
    ax.set_ylabel("PR-AUC, out-of-fold")
    ax.set_title("a  Precursor skill (solid) vs precipitation only (dotted)", loc="left")

    years = np.arange(YEARS[0], YEARS[1] + 1)
    for v, sig in sets.items():
        counts = sig["time"].dt.year.value_counts().reindex(years, fill_value=0).to_numpy()
        slope, icpt = cc.sens_slope(years, counts)
        _, _, p = cc.mann_kendall(counts)
        smooth = pd.Series(counts).rolling(5, center=True, min_periods=3).mean()
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
        bx.plot(years, smooth, color=COLORS[v], lw=2.2,
                label=f"{LABELS[v]}: {slope * 10:+.1f}/decade, {p_txt}")
#        bx.plot(years, smooth, color=COLORS[v], lw=2.2,
 #               label=f"{LABELS[v]}: {slope * 10:+.1f}/decade, p = {p:.2f}")
        bx.plot(years, counts, lw=0, marker="o", ms=3, color=COLORS[v], alpha=0.45)
    bx.set_ylabel("Analog days per year\n(dots; line = 5-yr mean)")
    bx.set_title("b  More frequent: the conditions, not the shape?", loc="left")
    bx.legend(frameon=False, fontsize=8, loc="upper left")
    handles = [plt.Line2D([], [], color=COLORS[v], lw=2.2, label=LABELS[v]) for v in VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    figs.OUT = ROOT / "figures" / "poster"
    figs.save(fig, "fig_poster_variants", figs.FULL_MM)


SLOT_W_MM, SLOT_H_MM = 388, 78     # Canva template: climate figure slot


def plot_climate_slot(sets):
    """Poster climate panel: analog days per year, conditions vs shape, nothing else."""
    figs = _load("make_figures", "08_make_figures.py")
    figs.style()
    plt = figs.plt
    cc = _load("climate_context", "07b_climate_context.py")
    w_in = SLOT_W_MM / 25.4 / figs.POSTER_SCALE
    h_in = SLOT_H_MM / 25.4 / figs.POSTER_SCALE
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    years = np.arange(YEARS[0], YEARS[1] + 1)
    ymax = 0
    for k, (v, sig) in enumerate(sets.items()):
        counts = sig["time"].dt.year.value_counts().reindex(years, fill_value=0).to_numpy()
        slope, _ = cc.sens_slope(years, counts)
        _, _, p = cc.mann_kendall(counts)
        smooth = pd.Series(counts).rolling(5, center=True, min_periods=3).mean()
        ax.plot(years, counts, lw=0, marker="o", ms=2.4, color=COLORS[v], alpha=0.40)
        ax.plot(years, smooth, color=COLORS[v], lw=2.2)
        ymax = max(ymax, counts.max())
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
        short = "Conditions" if v == "uncentred" else "Shape only"
        ax.text(0.01, 0.97 - 0.17 * k, f"{short}: {slope * 10:+.1f} days/decade, {p_txt}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=COLORS[v],
                fontweight="bold" if v == "uncentred" else "normal")
    ax.set_ylim(0, ymax + 1)
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_xticks(range(1980, years[-1] + 1, 10))
    ax.set_ylabel("Analog days / yr", fontsize=8.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.2)
    figs.OUT = ROOT / "figures" / "poster"
    figs.save(fig, "fig_poster_climate_shape", SLOT_W_MM)


if __name__ == "__main__":
    main()
