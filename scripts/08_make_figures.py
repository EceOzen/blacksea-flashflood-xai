"""
Poster figures (final, consistent style) built from the outputs of 03-07.

Figures (PNG at 300 dpi + PDF, in figures/poster/):
  1. fig_poster_skill            Main result. (a) PR-AUC vs lead time: precursors
                                 without precipitation (boosted + logistic) vs
                                 precipitation alone vs seasonality. (b) at 24 h:
                                 each signal group alone vs all precursors
                                 together -- "no single group is enough".
  2. fig_poster_precursor_evolution
                                 How the key precursors evolve from -72 h to
                                 +24 h around analog days (median + IQR) vs
                                 control days, with Bozkurt 2021 overlaid.
                                 Variables = top-4 from 07's SHAP table
                                 (no_precip, 24 h); falls back to a default list.
  3. fig_poster_flood_validation Every documented flood event in the label
                                 regions: the best pattern match (ACC) to the
                                 Bozkurt composite on the event days, against
                                 the significance threshold. Honest view of
                                 "analog != flood".
  4. fig_poster_shap_heatmap     Top-10 variables x lead time, share of |SHAP|
                                 (07, no_precip model), group-coloured labels.
  5. fig_poster_climate          (a) analog days per year + Sen's slope (07b),
                                 (b) late-minus-early change on analog days vs
                                 all days, for the moisture/SST variables (07b).
  7. fig_poster_bozkurt_anticipation
                                 Would the precursor model have flagged Bozkurt?
                                 A 24-h-lead model is re-trained WITHOUT 2021
                                 (no leakage) and scores every 6-h time step of
                                 7-16 Aug 2021 from the conditions 24 h earlier.
                                 The real flood days were never in the training
                                 data (03 excludes them from the candidate pool).
  6. fig_poster_robustness       (a) analog-set recovery under +/-5 deg domain
                                 shifts (03c), (b) reliability diagram (06).

A0 PORTRAIT LAYOUT (841 x 1189 mm, 2 columns): every figure is drawn for a
fixed width on the poster -- FULL (one row across both columns) or HALF (one
column). Figures are designed at 1/POSTER_SCALE of their printed size with
normal font sizes, then the PDF page is scaled up by POSTER_SCALE (so the PDF
has the true printed size) and the PNG is written at 300 dpi of printed size.
Smallest text on the printed poster: 8 pt x 2.5 = 20 pt (ticks ~22 pt).
Place each PDF at 100% in the poster tool. Widths: FULL 781 mm, HALF 378 mm.

Table:
  results/flood_validation_table.csv   one row per flood event: region, dates,
                                       max ACC, analog yes/no, sources

All inputs are files written by earlier scripts; nothing is recomputed from
ERA5, so this runs in seconds and can be re-run freely while polishing.

Usage:
    python scripts/08_make_figures.py   (run from the project root)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402

POSTER_SCALE = 2.5                   # design size x 2.5 = printed size
MARGIN_MM, GUTTER_MM = 30, 25        # A0 portrait, 2 columns
FULL_MM = 841 - 2 * MARGIN_MM        # 781 mm
HALF_MM = (FULL_MM - GUTTER_MM) / 2  # 378 mm
FULL_W = FULL_MM / 25.4 / POSTER_SCALE   # design inches (~12.3)
HALF_W = HALF_MM / 25.4 / POSTER_SCALE   # design inches (~5.95)

RESULTS = getattr(config, "RESULTS_DIR", config.PROJECT_ROOT / "results")
OUT = getattr(config, "FIGURES_DIR", config.PROJECT_ROOT / "figures") / "poster"
FOCUS_LEAD = 24
BOZKURT_T0 = pd.Timestamp(getattr(config, "BOZKURT_EVENT_DATE", "2021-08-11")) + pd.Timedelta(hours=12)
OFFSETS_H = list(range(-72, 25, 6))

# Figure heights (design inches) matched to the Canva template slots:
# printed height = design height x 2.54 x POSTER_SCALE  (e.g. 2.0 in -> 12.7 cm)
H_SKILL, H_EVOLUTION, H_HEATMAP, H_FLOOD, H_ANTICIPATION = 2.0, 2.0, 2.0, 2.0, 1.35

# Validated categorical palette (CVD-safe; fixed order, same hues as 06/07).
#C_PRECURSORS = "#2b2b2b"   # "all precursors" = near-black emphasis, distinct from every group hue
#C_CIRC = "#2a78d6"         # slot 1 blue
#C_THERMO = "#eb6834"       # slot 2 orange
#C_FLOW = "#1baf7a"         # slot 3 aqua
#C_PRECIP = "#eda100"       # slot 4 yellow
#C_BOZKURT = "#e87ba4"      # slot 5 magenta
#C_NEUTRAL = "#8a8a8a"
#INK, INK_2, GRID = "#1a1a1a", "#555555", "#e6e6e6"
#GROUP_COLORS = {"circulation": C_CIRC, "thermo_moist": C_THERMO, "flow": C_FLOW, "precip": C_PRECIP}

# Poster palette (Ece) -- CVD-checked; kızıl is reserved for Bozkurt only.
C_PRECURSORS = "#0E2A47"   # all precursors / "conditions" = gece mavisi
C_CIRC = "#3E6FA8"         # circulation
C_THERMO = "#C98B3B"       # thermodynamics & moisture (ochre)
C_FLOW = "#5FA37A"         # low-level flow (sage)
C_PRECIP = "#8C96A3"       # precipitation only (duman)
C_BOZKURT = "#B3263A"      # Bozkurt 2021 (kızıl)
C_NEUTRAL = "#9AA3AE"      # control days, seasonality line
INK, INK_2, GRID = "#1B2533", "#4A5568", "#E6EAF0"
GROUP_COLORS = {"circulation": C_CIRC, "thermo_moist": C_THERMO, "flow": C_FLOW, "precip": C_PRECIP}

from matplotlib.colors import LinearSegmentedColormap
CMAP_SHAP = LinearSegmentedColormap.from_list("sis_gece", ["#EEF2F8", "#8EA6CC", "#0E2A47"])

GROUP_LABELS = {"precip": "Precipitation", "circulation": "Circulation", "flow": "Low-level flow",
                "thermo_moist": "Thermodynamics\n& moisture", "no_precip": "All precursors\n(no precip.)"}
VAR_LABELS = {
    "z500_anom_bozkurt": ("Z500 anomaly at Bozkurt", "gpm"),
    "mslp_anom_bozkurt": ("MSLP anomaly at Bozkurt", "hPa"),
    "tcwv_mean": ("Total column water vapour", "kg m$^{-2}$"),
    "tcwv_max": ("TCWV (box max)", "kg m$^{-2}$"),
    "ivt_mean": ("Integrated vapour transport", "kg m$^{-1}$ s$^{-1}$"),
    "ivt_max": ("IVT (box max)", "kg m$^{-1}$ s$^{-1}$"),
    "sst_anom_mean": ("Black Sea SST anomaly", "K"),
    "cape_mean": ("CAPE", "J kg$^{-1}$"), "cape_max": ("CAPE (box max)", "J kg$^{-1}$"),
    "cin_mean": ("CIN", "J kg$^{-1}$"), "cin_max": ("CIN (box max)", "J kg$^{-1}$"),
    "wind850_mean": ("850 hPa wind speed", "m s$^{-1}$"), "wind850_max": ("850 hPa wind (box max)", "m s$^{-1}$"),
    "shear_850_500_mean": ("Shear 850-500 hPa", "m s$^{-1}$"),
    "onshore_flow_proxy": ("Meridional wind v850 (<0 = onshore)", "m s$^{-1}$"),
    "soil_moisture_mean": ("Soil moisture", "m$^3$ m$^{-3}$"),
}
DEFAULT_VARS = ["z500_anom_bozkurt", "tcwv_mean", "ivt_mean", "sst_anom_mean"]
ALREADY_ANOMALY = {"z500_anom_bozkurt", "mslp_anom_bozkurt", "sst_anom_mean"}


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,          # editable text in Illustrator/Inkscape/PowerPoint
    })


def save(fig, name, width_mm):
    """Write PDF + PNG so the figure is exactly width_mm wide on the poster.
    The (tight-cropped) design is scaled by ~POSTER_SCALE; small deviations from
    2.5x only nudge font sizes (they stay >= ~19 pt)."""
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT / f"{name}.pdf", OUT / f"{name}.png"
    fig.savefig(pdf)
    factor = scale_pdf_to_width(pdf, width_mm)
    if factor is None:   # no pypdf: fall back to the nominal scale
        factor = POSTER_SCALE
    fig.savefig(png, dpi=300 * factor)          # = 300 dpi at printed size
    plt.close(fig)
    print(f"Saved: {OUT / name}.png/.pdf  ({width_mm:.0f} mm wide on the poster, "
          f"text scaled x{factor:.2f})")


def scale_pdf_to_width(path, width_mm):
    try:
        from pypdf import PdfReader, PdfWriter, Transformation
    except ImportError:
        print("  (pip install pypdf to get true-size PDFs; scale this one to "
              f"{width_mm:.0f} mm wide when placing it)")
        return None
    reader, writer = PdfReader(str(path)), PdfWriter()
    page = reader.pages[0]
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    factor = (width_mm / 25.4 * 72) / w
    page.add_transformation(Transformation().scale(factor, factor))
    page.mediabox.lower_left = (0, 0)
    page.mediabox.upper_right = (w * factor, h * factor)
    page.cropbox.lower_left = (0, 0)
    page.cropbox.upper_right = (w * factor, h * factor)
    writer.add_page(page)
    with open(path, "wb") as fh:
        writer.write(fh)
    return factor


def doy_anomaly(t, v, window_days=31):
    doy = pd.to_datetime(t).dt.dayofyear.clip(upper=365)
    clim = pd.Series(v.to_numpy(), index=doy.to_numpy()).groupby(level=0).mean().reindex(range(1, 366))
    padded = pd.concat([clim.iloc[-window_days:], clim, clim.iloc[:window_days]])
    smooth = padded.rolling(window_days, center=True, min_periods=5).mean().iloc[window_days:-window_days]
    return v.to_numpy() - smooth.reindex(doy.to_numpy()).to_numpy()


# ----------------------------------------------------------------------
# 1. Main skill figure
# ----------------------------------------------------------------------
def fig_skill():
    path = RESULTS / "model_metrics.csv"
    if not path.exists():
        print("(no model_metrics.csv -- run 06 first; skipping figure 1)")
        return
    m = pd.read_csv(path)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL_W, H_SKILL), gridspec_kw={"width_ratios": [1.35, 1]})

    end_labels = []

    def line(group, color, ls="-", marker="o", lw=2.2, label=None, band=True):
        s = m[m["group"] == group].sort_values("lead_h")
        if s.empty:
            return
        ax.plot(s["lead_h"], s["pr_auc"], ls, marker=marker, color=color, lw=lw, ms=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        if band:
            ax.fill_between(s["lead_h"], s["pr_auc_lo"], s["pr_auc_hi"], color=color, alpha=0.13, lw=0)
        last = s.iloc[-1]
        end_labels.append([float(last["pr_auc"]), float(last["lead_h"]), label])

    line("no_precip", C_PRECURSORS, label="Precursors (boosted trees)")
    line("no_precip_logreg", C_PRECURSORS, ls="--", marker="s", lw=1.4, band=False,
         label="Precursors (logistic reg.)")
    line("precip", C_PRECIP, label="Precipitation only")
    # direct labels at the line ends, nudged apart so they never overlap
    ylo, yhi = ax.get_ylim()
    gap = 0.13 * (yhi - ylo)
    end_labels.sort()
    for i in range(1, len(end_labels)):
        end_labels[i][0] = max(end_labels[i][0], end_labels[i - 1][0] + gap)
    for y_lab, x_end, text in end_labels:
        ax.annotate(text, (x_end, y_lab), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK, linespacing=0.95)
    seas = m[m["group"] == "seasonality"]
    if not seas.empty:
        ax.axhline(seas["pr_auc"].iloc[0], color=C_NEUTRAL, ls="--", lw=1.1, zorder=1)
        ax.text(0, seas["pr_auc"].iloc[0], "seasonality only", fontsize=8, color=INK_2, va="bottom")
    leads = sorted(m["lead_h"].unique())
    ax.set_xticks(leads, [f"{l}" for l in leads])
    ax.set_xlim(min(leads) - 4, max(leads) + 40)
    ax.set_xlabel("Lead time (hours before the analog day)")
    ax.set_ylabel("PR-AUC (95% CI)")
    ax.set_title("a  How early can a Bozkurt-type day be anticipated?", loc="left")

    # (b) groups alone at the focus lead
    order = ["precip", "circulation", "flow", "thermo_moist", "no_precip"]
    s = m[(m["lead_h"] == FOCUS_LEAD) & (m["group"].isin(order))].set_index("group").reindex(order).dropna(subset=["pr_auc"])
    y = np.arange(len(s))
    colors = [GROUP_COLORS.get(g, C_PRECURSORS) for g in s.index]
    bx.barh(y, s["pr_auc"], color=colors, height=0.62, edgecolor="white", linewidth=2, zorder=2)
    bx.errorbar(s["pr_auc"], y, xerr=[s["pr_auc"] - s["pr_auc_lo"], s["pr_auc_hi"] - s["pr_auc"]],
                fmt="none", ecolor=INK_2, elinewidth=1.1, capsize=3, zorder=3)
    for yi, v, hi in zip(y, s["pr_auc"], s["pr_auc_hi"]):
        bx.text(hi + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=9, color=INK, zorder=4)
    if not seas.empty:
        bx.axvline(seas["pr_auc"].iloc[0], color=C_NEUTRAL, ls="--", lw=1.1, zorder=1)
    short = {"thermo_moist": "Thermo & moisture", "no_precip": "All precursors"}
    bx.set_yticks(y, [short.get(g, GROUP_LABELS.get(g, g)) for g in s.index])
    bx.invert_yaxis()
    bx.set_xlim(0, max(0.75, float(s["pr_auc_hi"].max()) + 0.1))
    bx.set_xlabel(f"PR-AUC at {FOCUS_LEAD} h lead")
    bx.set_title("b  No single signal group is enough", loc="left")
    bx.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_poster_skill", FULL_MM)


# ----------------------------------------------------------------------
# 2. Precursor evolution
# ----------------------------------------------------------------------
def pick_variables(ts_cols):
    path = RESULTS / "shap" / "importance_by_variable.csv"
    if path.exists():
        v = pd.read_csv(path).query("model == 'no_precip' and lead_h == @FOCUS_LEAD")
        v = v[v["variable"].isin(ts_cols)].sort_values("mean_abs_shap", ascending=False)
        if len(v) >= 2:
            chosen = v["variable"].head(3).tolist()
            if "tcwv_mean" in ts_cols and "tcwv_mean" not in chosen:
                chosen.append("tcwv_mean")   # moisture contrast, even if not a top driver
            else:
                chosen = v["variable"].head(4).tolist()
            print(f"Precursor panels from SHAP ranking at {FOCUS_LEAD} h: {chosen}")
            return chosen
    chosen = [c for c in DEFAULT_VARS if c in ts_cols]
    print(f"(no SHAP table yet -- using default precursor panels: {chosen})")
    return chosen


def windows(ts_time, ts_vals, t0s):
    idx = pd.DatetimeIndex(ts_time)
    out = np.full((len(t0s), len(OFFSETS_H)), np.nan)
    for i, t0 in enumerate(pd.to_datetime(t0s)):
        targets = [t0 + pd.Timedelta(hours=h) for h in OFFSETS_H]
        pos = idx.get_indexer(targets)          # exact match only; gaps stay NaN
        ok = pos >= 0
        out[i, ok] = ts_vals[pos[ok]]
    return out


def fig_precursor_evolution():
    ts_path = config.DATA_INTERIM / "features_timeseries.csv"
    if not ts_path.exists():
        print("(no features_timeseries.csv -- run 05 first; skipping figure 2)")
        return
    ts = pd.read_csv(ts_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    analogs = pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv", parse_dates=["time"])["time"]
    table = pd.read_csv(config.DATA_PROCESSED / "features_table.csv", parse_dates=["time"])
    controls = table.loc[table["label"] == 0, "time"]
    variables = pick_variables(ts.columns)
    if not variables:
        print("(no usable variables -- skipping figure 2)")
        return

    fig, axes = plt.subplots(1, len(variables), figsize=(FULL_W, H_EVOLUTION), squeeze=False)
    for ax, var in zip(axes[0], variables):
        vals = ts[var] if var in ALREADY_ANOMALY else pd.Series(doy_anomaly(ts["time"], ts[var]))
        vals = vals.to_numpy(dtype=float)
        for t0s, color, lab in [(controls, C_NEUTRAL, "control days"), (analogs, C_PRECURSORS, "analog days")]:
            w = windows(ts["time"], vals, t0s)
            q25, q50, q75 = np.nanpercentile(w, [25, 50, 75], axis=0)
            ax.fill_between(OFFSETS_H, q25, q75, color=color, alpha=0.18, lw=0)
            ax.plot(OFFSETS_H, q50, color=color, lw=2.2, label=f"{lab} (median, IQR)")
        bz = windows(ts["time"], vals, [BOZKURT_T0])[0]
        if np.isfinite(bz).any():
            ax.plot(OFFSETS_H, bz, color=C_BOZKURT, lw=1.8, ls="-", marker="o", ms=3, label="Bozkurt 2021")
        ax.axvline(0, color=INK_2, lw=0.9)
        ax.axhline(0, color=GRID, lw=1.0, zorder=0)
        name, unit = VAR_LABELS.get(var, (var, ""))
        ax.set_title(f"{name} ({unit})" if unit else name, fontsize=9)
        ax.set_xticks([-72, -48, -24, 0, 24])
    fig.supxlabel("hours relative to day 0 (anomalies vs day-of-year climatology)", fontsize=9, y=0.02)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.0), fontsize=8.5)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.72, bottom=0.26, wspace=0.18)
    save(fig, "fig_poster_precursor_evolution", FULL_MM)


# ----------------------------------------------------------------------
# 3. Flood validation
# ----------------------------------------------------------------------
def flood_events(labels, col):
    """Runs of consecutive flagged days = one event. The flags were widened by
    +/- LABEL_MATCH_WINDOW_DAYS in 04, so a run spans the event +/- that window."""
    s = labels.sort_values("date").set_index("date")[col].fillna(False).astype(bool)
    events, start, prev = [], None, None
    for d, flag in s.items():
        if flag and start is None:
            start = d
        if flag:
            prev = d
        if not flag and start is not None:
            events.append((start, prev)); start = None
    if start is not None:
        events.append((start, prev))
    return events


def fig_flood_validation():
    lab_path = config.DATA_LABELS / "flood_labels.csv"
    if not lab_path.exists():
        print("(no flood_labels.csv -- run 04 first; skipping figure 3)")
        return
    labels = pd.read_csv(lab_path, parse_dates=["date"])
    scores = pd.read_csv(config.DATA_PROCESSED / "analogs_all_scores.csv", parse_dates=["time"])
    scores["date"] = scores["time"].dt.normalize()
    daily_max = scores.groupby("date")["acc"].max()
    null = pd.read_csv(config.DATA_PROCESSED / "analogs_null_distribution.csv")["null_acc"].to_numpy()
    thr = float(np.quantile(null, config.ANALOG_SIGNIFICANCE_LEVEL))
    analog_days = set(pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv",
                                  parse_dates=["time"])["time"].dt.normalize())
    bz0 = pd.Timestamp(config.BOZKURT_WINDOW_START) - pd.Timedelta(days=config.DECLUSTER_WINDOW_DAYS)
    bz1 = pd.Timestamp(config.BOZKURT_WINDOW_END) + pd.Timedelta(days=config.DECLUSTER_WINDOW_DAYS)

    rows = []
    for region, col in [("core (TR W. Black Sea)", "flood_core"), ("extended (+BG/RO coast)", "flood_extended")]:
        if col not in labels.columns:
            continue
        for a, b in flood_events(labels, col):
            if region.startswith("extended") and labels.set_index("date").loc[a:b, "flood_core"].any():
                continue  # already shown as a core event
            acc = daily_max.loc[a:b]
            is_bz = (a <= bz1) and (b >= bz0)
            rows.append({"region": region, "start": a.date(), "end": b.date(),
                         "max_acc": float(acc.max()) if len(acc) else np.nan,
                         "day_of_max": acc.idxmax().date() if len(acc) else None,
                         "is_bozkurt_reference": is_bz,
                         "analog_within_event": any(d in analog_days for d in pd.date_range(a, b)),
                         "above_threshold": bool(len(acc) and acc.max() > thr)})
    tab = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / "flood_validation_table.csv", index=False)
    print(f"Saved: {RESULTS / 'flood_validation_table.csv'}")
    if tab.empty:
        return
    for region, g in tab[~tab["is_bozkurt_reference"]].groupby("region"):
        print(f"  {region}: {int(g['above_threshold'].sum())}/{len(g)} independent flood events reach "
              f"the analog threshold (ACC > {thr:.2f}) on at least one event day")

    regions = list(dict.fromkeys(tab["region"]))
    fig, ax = plt.subplots(figsize=(HALF_W, H_FLOOD))
    rng = np.random.default_rng(0)
    for i, reg in enumerate(regions):
        g = tab[tab["region"] == reg]
        if region_is_core(reg):
            # few events: spread them evenly by ACC so labels never collide
            ind = g[~g["is_bozkurt_reference"]].sort_values("max_acc").index
            offs = np.linspace(0.16, -0.40, len(ind)) if len(ind) > 1 else [0.0]
            pos = dict(zip(ind, offs))
            jitter = [pos.get(k, 0.28) for k in g.index]
        else:
            jitter = rng.uniform(-0.18, 0.18, len(g))
        for (_, r), j in zip(g.iterrows(), jitter):
            if r["is_bozkurt_reference"]:
                ax.scatter(r["max_acc"], i + 0.28, s=110, marker="*", color=C_BOZKURT, edgecolor="white",
                           linewidth=0.8, zorder=4)
                ax.annotate("Bozkurt 2021", (r["max_acc"], i + 0.28), xytext=(-8, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=8, color=C_BOZKURT)
                continue
            above = r["above_threshold"]
            ax.scatter(r["max_acc"], i + j, s=42, color=C_CIRC if above else "white",
                       edgecolor=C_CIRC, linewidth=1.4, zorder=3)
            if region_is_core(reg):
                ax.annotate(str(r["start"])[:4], (r["max_acc"], i + j), xytext=(9, -1),
                            textcoords="offset points", ha="left", va="center", fontsize=8,
                            color=INK, zorder=5)
    ax.axvline(thr, color=INK, ls="--", lw=1.1)
    ax.text(thr, len(regions) - 0.45, f" threshold {thr:.2f}", fontsize=8, color=INK, va="center")
    ax.set_yticks(range(len(regions)), [r.replace(" (", "\n(") for r in regions])
    ax.set_ylim(-0.6, len(regions) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Best match to the Bozkurt pattern\nduring the flood (ACC)")
    ax.set_title("Documented floods vs. the pattern", loc="left")
    ax.text(1.0, 1.02, "filled = above threshold", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK_2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_poster_flood_validation", HALF_MM)


# ----------------------------------------------------------------------
# 4. SHAP heatmap (top-10 variables)
# ----------------------------------------------------------------------
def fig_shap_heatmap(top_n=8):
    path = RESULTS / "shap" / "importance_by_variable.csv"
    if not path.exists():
        print("(no SHAP table -- run 07 first; skipping figure 4)")
        return
    v = pd.read_csv(path).query("model == 'no_precip'")
    tab = v.pivot(index="variable", columns="lead_h", values="share")
    tab = tab.loc[tab.mean(axis=1).sort_values(ascending=False).index[:top_n]]
    groups = v.drop_duplicates("variable").set_index("variable")["group"]
    fig, ax = plt.subplots(figsize=(HALF_W, H_HEATMAP))
    data = tab.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=CMAP_SHAP, aspect="auto", vmin=0)
    ax.set_xticks(range(tab.shape[1]), [f"\u2212{c} h" if c else "0 h" for c in tab.columns])
    ax.set_yticks(range(len(tab)), [VAR_LABELS.get(x, (x,))[0] for x in tab.index])
    for tick, var in zip(ax.get_yticklabels(), tab.index):
        tick.set_color(GROUP_COLORS.get(groups.get(var), INK))
    vmax = np.nanmax(data)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if np.isfinite(data[r, c]):
                ax.text(c, r, f"{data[r, c]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if data[r, c] > 0.55 * vmax else INK)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("What the model relies on (share of |SHAP|, by lead time)", loc="left")
    handles = [plt.Line2D([], [], ls="", marker="s", ms=8, color=GROUP_COLORS[g], label=GROUP_LABELS[g].replace("\n", " "))
               for g in ["circulation", "flow", "thermo_moist"] if g in set(groups)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.10), ncol=len(handles),
              frameon=False, fontsize=8, handletextpad=0.3, columnspacing=0.9)
    save(fig, "fig_poster_shap_heatmap", HALF_MM)


# ----------------------------------------------------------------------
# 5. Climate context
# ----------------------------------------------------------------------
def fig_climate():
    trend_p, per_p = RESULTS / "climate_context_trend.csv", RESULTS / "climate_context_periods.csv"
    if not (trend_p.exists() and per_p.exists()):
        print("(no climate_context_*.csv -- run 07b first; skipping figure 5)")
        return
    trend = pd.read_csv(trend_p).iloc[0]
    per = pd.read_csv(per_p)
    per = per[per["variable"].isin(["tcwv_mean", "sst_anom_mean", "ivt_mean"])]
    analogs = pd.read_csv(config.DATA_PROCESSED / "analogs_significant.csv", parse_dates=["time"])
    y0, y1 = (int(x) for x in str(trend["years"]).split("-"))
    yrs = np.arange(y0, y1 + 1)
    counts = analogs["time"].dt.year.value_counts().reindex(yrs, fill_value=0).to_numpy()

    n = max(len(per), 1)
    fig = plt.figure(figsize=(FULL_W, 3.4))
    gs = fig.add_gridspec(1, 1 + n, width_ratios=[2.6] + [1] * n, wspace=0.5,
                          left=0.06, right=0.995, top=0.74, bottom=0.17)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(yrs, counts, color=C_CIRC, width=0.75, edgecolor="white", linewidth=0.8)
    slope = trend["sen_slope_per_decade"] / 10
    icpt = np.median(counts - slope * yrs)
    ax.plot(yrs, icpt + slope * yrs, color=INK, lw=1.8)
    ax.text(0.01, 0.97, f"Sen's slope {trend['sen_slope_per_decade']:+.1f} days/decade\n"
                        f"Mann-Kendall {fmt_p(trend['mk_p'])}", transform=ax.transAxes,
            va="top", fontsize=8.5, color=INK)
    ax.set_ylabel("Analog days per year")
    fig.text(ax.get_position().x0, 0.97, "a  How often do Bozkurt-type conditions occur?",
             fontsize=11, color=INK, va="top")
    ax.grid(axis="x", visible=False)

    for k, (_, r) in enumerate(per.iterrows()):
        bx = fig.add_subplot(gs[0, 1 + k])
        for x, key, col in [(0, "delta_all", C_NEUTRAL), (1, "delta_analog", C_CIRC)]:
            bx.errorbar(x, r[key], yerr=[[r[key] - r[f"{key}_lo"]], [r[f"{key}_hi"] - r[key]]],
                        fmt="o", color=col, ms=7, capsize=3, lw=1.6)
        bx.axhline(0, color=INK_2, lw=0.8)
        bx.set_xticks([0, 1], ["all\ndays", "analog\ndays"])
        bx.set_xlim(-0.6, 1.6)
        bx.set_title(f"{r['label'].replace('Black Sea ', '')}\n({r['unit']})", fontsize=9)
        bx.grid(axis="x", visible=False)
        if k == 0:
            fig.text(bx.get_position().x0 - 0.04, 0.97,
                     f"b  Change {y0}\u2013{y0 + 22} \u2192 {y0 + 23}\u2013{y1}",
                     fontsize=11, color=INK, va="top")
    save(fig, "fig_poster_climate", FULL_MM)


# ----------------------------------------------------------------------
# 6. Robustness: domain sensitivity + calibration
# ----------------------------------------------------------------------
def fig_robustness():
    sens_p, oof_p = RESULTS / "analog_sensitivity.csv", RESULTS / "oof_predictions.csv"
    if not (sens_p.exists() or oof_p.exists()):
        print("(no analog_sensitivity.csv / oof_predictions.csv -- skipping figure 6)")
        return
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(HALF_W, 6.6), gridspec_kw={"height_ratios": [1.25, 1]})
    if sens_p.exists():
        sens = pd.read_csv(sens_p)
        sens = sens[sens["variant"] != "baseline"].reset_index(drop=True)
        y = np.arange(len(sens))
        ax.barh(y, sens["recall"], color=C_CIRC, height=0.62, edgecolor="white", linewidth=2)
        for yi, rec in zip(y, sens["recall"]):
            ax.text(rec + 0.015, yi, f"{rec:.0%}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [v.replace("shrink", "\u2212").replace("expand", "+") + " 5\u00b0" for v in sens["variant"]])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Analog days recovered")
        ax.grid(axis="y", visible=False)
    ax.set_title("a  Search-domain sensitivity", loc="left")

    if oof_p.exists():
        from sklearn.calibration import calibration_curve
        oof = pd.read_csv(oof_p)
        bx.plot([0, 1], [0, 1], color=C_NEUTRAL, ls=":", lw=1.2)
        for lead, mk in [(0, "o"), (FOCUS_LEAD, "s")]:
            sub = oof.query("group == 'no_precip' and lead_h == @lead")
            if sub.empty:
                continue
            frac, mp = calibration_curve(sub["label"], sub["prob"], n_bins=8, strategy="quantile")
            bx.plot(mp, frac, "-" + mk, color=C_PRECURSORS if lead == 0 else C_CIRC, lw=1.8, ms=6,
                    markeredgecolor="white", label=f"{lead} h lead")
        bx.set_xlim(0, 1); bx.set_ylim(0, 1)
        bx.set_xlabel("Predicted probability")
        bx.set_ylabel("Observed frequency")
        bx.legend(frameon=False, loc="upper left")
    bx.set_title("b  Reliability (precursor model)", loc="left")
    fig.tight_layout(h_pad=1.6)
    save(fig, "fig_poster_robustness", HALF_MM)


# ----------------------------------------------------------------------
# 7. Would the model have anticipated Bozkurt?
# ----------------------------------------------------------------------
def fig_bozkurt_anticipation(lead=FOCUS_LEAD, group="no_precip"):
    import pickle
    try:
        import lightgbm as lgb
    except ImportError:
        print("(lightgbm not installed -- skipping figure 7)")
        return
    pkl = RESULTS / "models" / f"lead{lead}_{group}.pkl"
    if not pkl.exists():
        print(f"(no {pkl.name} -- run 06 first; skipping figure 7)")
        return
    with open(pkl, "rb") as fh:
        cols = pickle.load(fh)["features"]
    table = pd.read_csv(config.DATA_PROCESSED / "features_table.csv", parse_dates=["time"])
    train = table[table["time"].dt.year != 2021]
    model = lgb.LGBMClassifier(   # same settings as 06
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        class_weight="balanced", random_state=getattr(config, "FEATURE_RNG_SEED", 42), verbose=-1,
    ).fit(train[cols].to_numpy(dtype=float), train["label"].astype(int).to_numpy())

    ts = pd.read_csv(config.DATA_INTERIM / "features_timeseries.csv", parse_dates=["time"]).set_index("time")
    targets = pd.date_range("2021-08-07", "2021-08-16 18:00", freq="6h")
    rows = []
    for t0 in targets:
        x = []
        for c in cols:
            base, lag = c.rsplit("_lag", 1)
            t = t0 - pd.Timedelta(hours=int(lag))
            x.append(ts.at[t, base] if t in ts.index and base in ts.columns else np.nan)
        rows.append(x)
    prob = model.predict_proba(np.array(rows, dtype=float))[:, 1]

    oof = pd.read_csv(RESULTS / "oof_predictions.csv")
    ctrl = oof.query("group == @group and lead_h == @lead and label == 0")["prob"].to_numpy()
    p90 = float(np.quantile(ctrl, 0.9))
    flood_t = pd.Timestamp("2021-08-11 12:00")
    p_flood = float(prob[targets.get_loc(flood_t)])
    pct = float((ctrl < p_flood).mean())
    out = pd.DataFrame({"target_time": targets, "prob_from_24h_earlier": prob,
                        "percentile_among_control_days": [(ctrl < p).mean() for p in prob]})
    out.to_csv(RESULTS / "bozkurt_anticipation.csv", index=False)
    print(f"Bozkurt, 11 Aug 2021 12 UTC, scored from conditions 24 h earlier (model trained without 2021): "
          f"P = {p_flood:.2f}, higher than {pct:.0%} of control days")

    fig, ax = plt.subplots(figsize=(HALF_W, H_ANTICIPATION))
    ax.axvspan(pd.Timestamp("2021-08-10 12:00"), pd.Timestamp("2021-08-12 00:00"), color=C_BOZKURT,
               alpha=0.15, lw=0)
    ax.text(pd.Timestamp("2021-08-11 06:00"), 0.03, "flood", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color=C_BOZKURT, fontweight="bold")
    ax.axhline(p90, color=C_NEUTRAL, ls="--", lw=1.1)
    ax.text(targets[0], p90 - 0.04, "p90 of controls", fontsize=7.5, color=INK_2, va="top", ha="left")
    ax.plot(targets, prob, "-o", color=C_PRECURSORS, lw=2, ms=4, markeredgecolor="white")
    ax.plot(flood_t, p_flood, "o", color=C_BOZKURT, ms=8, markeredgecolor="white", zorder=5)
    ax.annotate(f"P = {p_flood:.2f}", (flood_t, p_flood),
                xytext=(pd.Timestamp("2021-08-12 12:00"), 0.22), textcoords="data",
                ha="left", va="center", fontsize=9, color=C_BOZKURT, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_BOZKURT, lw=1))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Model score\n(24 h ahead)")
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d Aug"))
    ax.set_title("Would the model have flagged Bozkurt? (2021 withheld)", loc="left")
    fig.tight_layout()
    save(fig, "fig_poster_bozkurt_anticipation", HALF_MM)


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"


def region_is_core(reg):
    return reg.startswith("core")


def main():
    style()
    fig_skill()
    fig_precursor_evolution()
    fig_flood_validation()
    fig_shap_heatmap()
    fig_climate()
    fig_robustness()
    fig_bozkurt_anticipation()


if __name__ == "__main__":
    main()
