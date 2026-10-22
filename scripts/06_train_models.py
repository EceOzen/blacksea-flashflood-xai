"""
Train and evaluate the precursor models (Plan B: target = "is this day a
significant synoptic analog of Bozkurt", label column in features_table.csv).

Why lead-time framing: the analog label is DEFINED by the Z500/MSLP pattern,
so circulation features at lag 0 nearly restate the label (leakage-like, high
skill that teaches nothing). The scientific question is about precursors, so
every model only sees features from L hours or more before the sample time,
for L in config.FEATURE_LAG_HOURS. Skill vs lead time, per feature group, is
the main result: "how early, and from which signals, can a Bozkurt-type day
be anticipated?"

Feature groups (by variable-name prefix, all lags):
  circulation   z500_anom_bozkurt, mslp_anom_bozkurt
  flow          wind850_*, shear_850_500_mean, onshore_flow_proxy
  thermo_moist  cape_*, cin_*, tcwv_*, ivt_*, soil_moisture_mean, sst_anom_mean
  precip        tp_* (precipitation itself) -- the "rain-only" baseline
  no_precip     circulation + flow + thermo_moist: do precursors carry
                information beyond the rainfall signal?
  all           everything above
  seasonality   baseline: day-of-year (sin/cos) only, no weather at all

Validation: year-blocked cross-validation (all samples of a year in the same
fold -> no leakage between neighbouring days of one event). Metrics on
out-of-fold predictions: PR-AUC (main; chance level = positive fraction),
ROC-AUC, Brier score, with 95% CIs from a year-block bootstrap.

Linear check: the no_precip set is also fit with a regularised logistic
regression ("no_precip_logreg"): does the boosted model beat a simple linear
one? If not, the relationship is mostly linear -- itself a finding.

Calibration: reliability diagram of the out-of-fold probabilities. Note that
with ~630 samples the boosted model tends to be OVERCONFIDENT (probabilities
pushed towards 0 and 1; observed frequencies are flatter). Ranking metrics
(PR-AUC, ROC-AUC) are unaffected; treat the outputs as risk rankings, not
calibrated probabilities (post-hoc isotonic/Platt calibration left for later).

Model: LightGBM if installed (needed later for SHAP TreeExplainer), otherwise
sklearn HistGradientBoosting. Small fixed hyperparameters, no tuning (time
budget; documented as such). Class imbalance handled with balanced weights.

Usage:
    python scripts/06_train_models.py   (run from the project root)

Outputs:
    results/model_metrics.csv          one row per (lead, group): metrics + CIs
    results/oof_predictions.csv        out-of-fold probabilities per sample,
                                         lead and group (input for 07 / 08)
    results/models/lead{L}_{group}.pkl  model refit on all data (for SHAP in 07)
    figures/fig_lead_time_skill.png    PR-AUC vs lead time, per group
    figures/fig_calibration.png        reliability diagram (no_precip, 0 h and 24 h)
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

FEATURE_TABLE_CSV = config.DATA_PROCESSED / "features_table.csv"
RESULTS = getattr(config, "RESULTS_DIR", config.PROJECT_ROOT / "results")
MODELS_DIR = RESULTS / "models"
FIGURES = getattr(config, "FIGURES_DIR", config.PROJECT_ROOT / "figures")

N_FOLDS = 5
N_BOOT = 500
SEED = getattr(config, "FEATURE_RNG_SEED", 42)

GROUP_PATTERNS = {
    "circulation": r"^(z500_anom|mslp_anom)",
    "flow": r"^(wind850|shear_850_500|onshore_flow)",
    "thermo_moist": r"^(cape|cin|tcwv|ivt|soil_moisture|sst)",
    "precip": r"^tp_",
}
GROUP_ORDER = ["seasonality", "precip", "circulation", "flow", "thermo_moist", "no_precip",
               "no_precip_logreg", "all"]


# ----------------------------------------------------------------------
def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            class_weight="balanced", random_state=SEED, verbose=-1,
        ), "LightGBM"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
            class_weight="balanced", random_state=SEED,
        ), "HistGradientBoosting (install lightgbm for SHAP in 07)"


def make_logreg():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


def lag_of(col):
    m = re.search(r"_lag(\d+)$", col)
    return int(m.group(1)) if m else None


def base_name(col):
    return re.sub(r"_lag\d+$", "", col)


def feature_sets(df, lead):
    """Columns usable at a given lead: weather features with lag >= lead."""
    weather = [c for c in df.columns if lag_of(c) is not None and lag_of(c) >= lead]
    sets = {"seasonality": ["doy_sin", "doy_cos"]}
    for g, pat in GROUP_PATTERNS.items():
        sets[g] = [c for c in weather if re.match(pat, base_name(c))]
    sets["all"] = [c for c in weather if any(re.match(p, base_name(c)) for p in GROUP_PATTERNS.values())]
    sets["no_precip"] = [c for c in sets["all"] if not re.match(GROUP_PATTERNS["precip"], base_name(c))]
    sets["no_precip_logreg"] = sets["no_precip"]
    return {g: cols for g, cols in sets.items() if cols}


# ----------------------------------------------------------------------
def year_block_bootstrap(y, p, years, rng):
    uniq = np.unique(years)
    idx_by_year = {yr: np.where(years == yr)[0] for yr in uniq}
    out = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_year[yr] for yr in pick])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            continue
        out["pr_auc"].append(average_precision_score(yy, pp))
        out["roc_auc"].append(roc_auc_score(yy, pp))
        out["brier"].append(brier_score_loss(yy, pp))
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def evaluate(y, p, years, rng):
    ci = year_block_bootstrap(y, p, years, rng)
    return {
        "pr_auc": average_precision_score(y, p), "pr_auc_lo": ci["pr_auc"][0], "pr_auc_hi": ci["pr_auc"][1],
        "roc_auc": roc_auc_score(y, p), "roc_auc_lo": ci["roc_auc"][0], "roc_auc_hi": ci["roc_auc"][1],
        "brier": brier_score_loss(y, p), "brier_lo": ci["brier"][0], "brier_hi": ci["brier"][1],
    }


# ----------------------------------------------------------------------
def main():
    df = pd.read_csv(FEATURE_TABLE_CSV, parse_dates=["time"])
    df = df.dropna(subset=[c for c in df.columns if lag_of(c) is not None], how="all").reset_index(drop=True)
    doy = df["time"].dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    y = df["label"].astype(int).to_numpy()
    years = df["time"].dt.year.to_numpy()
    n_years = len(np.unique(years))
    folds = list(GroupKFold(n_splits=min(N_FOLDS, n_years)).split(df, y, groups=years))
    _, model_name = make_model()
    print(f"{len(df)} samples ({y.sum()} analog days, {len(y) - y.sum()} controls), "
          f"{n_years} years, {len(folds)} year-blocked folds. Model: {model_name}")
    print(f"Chance-level PR-AUC = positive fraction = {y.mean():.3f}\n")

    rng = np.random.default_rng(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows, oof_rows = [], []

    for lead in config.FEATURE_LAG_HOURS:
        for group, cols in sorted(feature_sets(df, lead).items(), key=lambda kv: GROUP_ORDER.index(kv[0])):
            if group == "seasonality" and lead != config.FEATURE_LAG_HOURS[0]:
                continue  # weather-free baseline is the same at every lead
            X = df[cols].to_numpy(dtype=float)
            factory = make_logreg if group.endswith("_logreg") else (lambda: make_model()[0])
            oof = np.full(len(df), np.nan)
            for tr, te in folds:
                model = factory()
                model.fit(X[tr], y[tr])
                oof[te] = model.predict_proba(X[te])[:, 1]

            m = evaluate(y, oof, years, rng)
            rows.append({"lead_h": lead, "group": group, "n_features": len(cols), **m})
            oof_rows.append(pd.DataFrame({"sample_id": df["sample_id"], "lead_h": lead,
                                          "group": group, "label": y, "prob": oof}))
            print(f"lead {lead:>2}h  {group:<16} ({len(cols):>3} feats)  "
                  f"PR-AUC {m['pr_auc']:.3f} [{m['pr_auc_lo']:.3f}-{m['pr_auc_hi']:.3f}]  "
                  f"ROC-AUC {m['roc_auc']:.3f}  Brier {m['brier']:.3f}")

            final = factory()
            final.fit(X, y)
            with open(MODELS_DIR / f"lead{lead}_{group}.pkl", "wb") as fh:
                pickle.dump({"model": final, "features": cols, "lead_h": lead, "group": group}, fh)
        print()

    metrics = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    oof_all = pd.concat(oof_rows, ignore_index=True)
    oof_all.to_csv(RESULTS / "oof_predictions.csv", index=False)
    print(f"Saved: {RESULTS / 'model_metrics.csv'}")
    print(f"Saved: {RESULTS / 'oof_predictions.csv'}")
    print(f"Saved: {len(rows)} refit models in {MODELS_DIR}")

    # --- flood check (supporting evidence only, tiny n) -------------------
    flood_cols = [c for c in ["flood_core", "flood_extended"] if c in df.columns]
    if flood_cols:
        lead_ref = 24 if 24 in config.FEATURE_LAG_HOURS else config.FEATURE_LAG_HOURS[-1]
        p = oof_all.query("lead_h == @lead_ref and group == 'all'").set_index("sample_id")["prob"]
        print(f"\nFlood check (lead {lead_ref}h, 'all' model, mean out-of-fold probability):")
        for c in flood_cols:
            flag = df.set_index("sample_id")[c].fillna(False).astype(bool)
            print(f"  {c}: flood days {p[flag].mean():.2f} (n={flag.sum()}) "
                  f"vs other days {p[~flag].mean():.2f} (n={(~flag).sum()})")

    plot_lead_time(metrics, y.mean())
    plot_calibration(oof_all)


def plot_lead_time(metrics, chance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"circulation": "#2a78d6", "thermo_moist": "#eb6834", "flow": "#1baf7a",
              "precip": "#eda100", "no_precip": "#e87ba4", "all": "#222222"}  # validated CVD-safe order
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for g, c in colors.items():
        sub = metrics[metrics["group"] == g].sort_values("lead_h")
        if sub.empty:
            continue
        ax.plot(sub["lead_h"], sub["pr_auc"], "-o", color=c, label=g, lw=2 if g == "all" else 1.5)
        ax.fill_between(sub["lead_h"], sub["pr_auc_lo"], sub["pr_auc_hi"], color=c, alpha=0.12)
    lr = metrics[metrics["group"] == "no_precip_logreg"].sort_values("lead_h")
    if not lr.empty:
        ax.plot(lr["lead_h"], lr["pr_auc"], "--s", color=colors["no_precip"], ms=4, lw=1.3,
                label="no_precip (logistic reg.)")
    seas = metrics[metrics["group"] == "seasonality"]
    if not seas.empty:
        ax.axhline(seas["pr_auc"].iloc[0], color="#888888", ls="--", lw=1, label="seasonality baseline")
    ax.axhline(chance, color="#bbbbbb", ls=":", lw=1, label="chance")
    ax.set_xlabel("Lead time (hours before the analog day)")
    ax.set_ylabel("PR-AUC (out-of-fold)")
    ax.set_title("How early can a Bozkurt-type day be anticipated?")
    ax.set_xticks(sorted(metrics["lead_h"].unique()))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig_lead_time_skill.png"
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")


def plot_calibration(oof_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", ls=":", lw=1, label="perfect")
    styles = {0: ("#e87ba4", "-o"), 24: ("#2a78d6", "-s")}
    for lead, (c, st) in styles.items():
        sub = oof_all.query("group == 'no_precip' and lead_h == @lead")
        if sub.empty:
            continue
        frac, mean_p = calibration_curve(sub["label"], sub["prob"], n_bins=8, strategy="quantile")
        ax.plot(mean_p, frac, st, color=c, ms=5, lw=1.5, label=f"no_precip, {lead} h ahead")
    ax.set_xlabel("Predicted probability (out-of-fold)")
    ax.set_ylabel("Observed fraction of analog days")
    ax.set_title("Reliability", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "fig_calibration.png"
    fig.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
