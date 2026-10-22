"""
SHAP analysis of the precursor models from 06_train_models.py.

Questions it answers, per lead time (0/12/24/48/72 h before the analog day):
  1. Which variables drive the prediction?      (mean |SHAP|, by variable)
  2. How much does each variable GROUP carry?   (group SHAP = sum of members)
  3. Is the ranking stable?                     (across CV folds + bootstrap CI)
  4. Which variables act together?              (SHAP interaction values)
  5. What did the model see for Bozkurt itself and the other core flood days?
  6. Does a model-agnostic method agree?        (grouped permutation importance,
                                                 Spearman rho vs the SHAP ranking)

Design choices (and why):
  - OUT-OF-FOLD SHAP: models are retrained on the same year-blocked folds as
    06 and each sample is explained by the model that did NOT see it. SHAP of
    a model refit on all data would explain in-sample fits instead.
  - Correlated features: the same variable appears at several lags and some
    variables co-vary (CAPE/CIN/TCWV). Single-feature rankings are therefore
    unreliable, so the headline numbers are aggregated by VARIABLE (sum over
    lags) and by GROUP. SHAP values are additive, so summing is legitimate.
  - Main model: "no_precip" (circulation + flow + thermo/moisture), i.e. the
    precursors beyond rainfall. "all" is also explained, to show how much the
    precipitation features add once the precursors are known.

Requires: lightgbm, shap (pip install shap).

Usage:
    python scripts/07_shap_analysis.py   (run from the project root)

Outputs:
    results/shap/importance_by_feature.csv   lead, model, feature, variable, lag, mean_abs_shap (+CI)
    results/shap/importance_by_variable.csv  lead, model, variable, group, mean_abs_shap, share (+CI)
    results/shap/importance_by_group.csv     lead, model, group, mean_abs_shap, share
    results/shap/stability.csv               lead, model, mean Spearman rho between fold rankings
    results/shap/interactions_lead24.csv     top variable-pair interactions at 24 h
    results/shap/case_days.csv               per-sample OOF prob + top contributors for flood days
    results/shap/permutation_importance.csv  lead, model, variable, drop in PR-AUC when shuffled
    figures/fig_shap_variable_by_lead.png    heatmap: variable share of |SHAP| per lead
    figures/fig_shap_group_share.png         stacked bars: group share per lead ("all" model)
    figures/fig_shap_summary_lead24.png      beeswarm, 24 h, no_precip
    figures/fig_shap_dependence_lead24_*.png top interaction pairs, 24 h
    figures/fig_shap_case_<date>_lead24.png  waterfall for Bozkurt and other core flood days
"""

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences a harmless joblib warning on Windows
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")  # format handled in positive_class()

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

FEATURE_TABLE_CSV = config.DATA_PROCESSED / "features_table.csv"
RESULTS = getattr(config, "RESULTS_DIR", config.PROJECT_ROOT / "results")
MODELS_DIR = RESULTS / "models"
OUT = RESULTS / "shap"
FIGURES = getattr(config, "FIGURES_DIR", config.PROJECT_ROOT / "figures")

N_FOLDS = 5          # must match 06
N_BOOT = 300
N_PERM_REPEATS = 5
N_INTERACTION_PER_FOLD = 40   # interaction values are O(features^2) and slow; a subsample ranks pairs fine
SEED = getattr(config, "FEATURE_RNG_SEED", 42)
MODELS_TO_EXPLAIN = ["no_precip", "all"]
FOCUS_LEAD = 24

GROUP_PATTERNS = {   # must match 06
    "circulation": r"^(z500_anom|mslp_anom)",
    "flow": r"^(wind850|shear_850_500|onshore_flow)",
    "thermo_moist": r"^(cape|cin|tcwv|ivt|soil_moisture|sst)",
    "precip": r"^tp_",
}
# Validated categorical palette (CVD-safe, fixed order; see dataviz check).
GROUP_COLORS = {"circulation": "#2a78d6", "thermo_moist": "#eb6834",
                "flow": "#1baf7a", "precip": "#eda100"}
GROUP_LABELS = {"circulation": "Circulation", "flow": "Low-level flow",
                "thermo_moist": "Thermodynamics & moisture", "precip": "Precipitation"}
VAR_LABELS = {
    "z500_anom_bozkurt": "Z500 anomaly", "mslp_anom_bozkurt": "MSLP anomaly",
    "wind850_mean": "850 hPa wind (mean)", "wind850_max": "850 hPa wind (max)",
    "shear_850_500_mean": "Shear 850-500 hPa", "onshore_flow_proxy": "Meridional wind v850 (<0 = onshore)",
    "cape_mean": "CAPE (mean)", "cape_max": "CAPE (max)", "cin_mean": "CIN (mean)", "cin_max": "CIN (max)",
    "tcwv_mean": "TCWV (mean)", "tcwv_max": "TCWV (max)", "ivt_mean": "IVT (mean)", "ivt_max": "IVT (max)",
    "soil_moisture_mean": "Soil moisture", "sst_anom_mean": "Black Sea SST anomaly",
    "tp_1h_mean": "Precip 1 h (mean)", "tp_1h_max": "Precip 1 h (max)",
    "tp_24h_proxy_mean": "Precip 24 h (mean)", "tp_24h_proxy_max": "Precip 24 h (max)",
}


# ----------------------------------------------------------------------
def lag_of(col):
    m = re.search(r"_lag(\d+)$", col)
    return int(m.group(1)) if m else None


def base_name(col):
    return re.sub(r"_lag\d+$", "", col)


def group_of(var):
    return next((g for g, p in GROUP_PATTERNS.items() if re.match(p, var)), "other")


def pretty(col):
    lag = lag_of(col)
    name = VAR_LABELS.get(base_name(col), base_name(col))
    return f"{name} (-{lag} h)" if lag else f"{name} (0 h)"


def make_model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(   # identical to 06
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        class_weight="balanced", random_state=SEED, verbose=-1,
    )


def positive_class(sv):
    """shap returns either a list [neg, pos] or one array depending on version."""
    if isinstance(sv, list):
        return sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[-1] == 2:      # (n, features, classes)
        return sv[..., 1]
    return sv


def positive_interactions(iv):
    if isinstance(iv, list):
        return iv[1]
    iv = np.asarray(iv)
    if iv.ndim == 4:                            # (n, f, f, classes)
        return iv[..., 1]
    return iv


# ----------------------------------------------------------------------
def oof_shap(df, cols, folds, want_interactions=False):
    """Explain every sample with the fold model that didn't train on it."""
    import shap
    X = df[cols].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    sv = np.zeros_like(X)
    base = np.zeros(len(df))
    prob = np.zeros(len(df))
    fold_id = np.zeros(len(df), dtype=int)
    iv = np.full((len(df), len(cols), len(cols)), np.nan) if want_interactions else None
    rng = np.random.default_rng(SEED)
    models = []
    for k, (tr, te) in enumerate(folds):
        print(f"    fold {k + 1}/{len(folds)}", end="\r", flush=True)
        model = make_model().fit(X[tr], y[tr])
        models.append(model)
        expl = shap.TreeExplainer(model)
        sv[te] = positive_class(expl.shap_values(X[te]))
        ev = expl.expected_value
        base[te] = ev[1] if np.ndim(ev) else ev
        prob[te] = model.predict_proba(X[te])[:, 1]
        fold_id[te] = k
        if want_interactions:
            sub = np.sort(rng.choice(te, size=min(N_INTERACTION_PER_FOLD, len(te)), replace=False))
            print(f"    fold {k + 1}/{len(folds)}: SHAP interaction values for {len(sub)} samples "
                  f"({len(cols)} features -- slow, please wait)...", flush=True)
            iv[sub] = positive_interactions(expl.shap_interaction_values(X[sub]))
    return sv, base, prob, fold_id, iv, models


def grouped_permutation(df, cols, folds, models, rng):
    """Drop in out-of-fold PR-AUC when ALL lags of one variable are shuffled
    together (shuffling lags separately would let the model fall back on a
    neighbouring lag of the same variable, understating its importance)."""
    from sklearn.metrics import average_precision_score
    X = df[cols].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    variables = sorted({base_name(c) for c in cols})
    idx = {v: [j for j, c in enumerate(cols) if base_name(c) == v] for v in variables}
    drops = {v: [] for v in variables}
    for (tr, te), model in zip(folds, models):
        Xte, yte = X[te], y[te]
        if yte.min() == yte.max():
            continue
        base_score = average_precision_score(yte, model.predict_proba(Xte)[:, 1])
        for v in variables:
            for _ in range(N_PERM_REPEATS):
                Xp = Xte.copy()
                perm = rng.permutation(len(te))
                Xp[:, idx[v]] = Xte[perm][:, idx[v]]
                drops[v].append(base_score - average_precision_score(yte, model.predict_proba(Xp)[:, 1]))
    return {v: float(np.mean(d)) for v, d in drops.items()}


def year_block_boot(values, years, rng):
    """values: (n_samples, n_items) of |SHAP|; returns 2.5/97.5% of the column means."""
    uniq = np.unique(years)
    idx_by_year = {yr: np.where(years == yr)[0] for yr in uniq}
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_year[yr] for yr in pick])
        boots.append(values[idx].mean(axis=0))
    return np.percentile(np.array(boots), [2.5, 97.5], axis=0)


def fold_stability(abs_by_var, fold_id):
    from scipy.stats import spearmanr
    ranks = [abs_by_var[fold_id == k].mean(axis=0) for k in np.unique(fold_id)]
    rhos = [spearmanr(ranks[i], ranks[j])[0] for i in range(len(ranks)) for j in range(i + 1, len(ranks))]
    return float(np.nanmean(rhos)), float(np.nanmin(rhos))


# ----------------------------------------------------------------------
def main():
    import shap  # noqa: F401  (fail early with a clear message if missing)

    df = pd.read_csv(FEATURE_TABLE_CSV, parse_dates=["time"])
    df = df.dropna(subset=[c for c in df.columns if lag_of(c) is not None], how="all").reset_index(drop=True)
    years = df["time"].dt.year.to_numpy()
    folds = list(GroupKFold(n_splits=min(N_FOLDS, len(np.unique(years)))).split(df, df["label"], groups=years))
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    feat_rows, var_rows, grp_rows, stab_rows, perm_rows = [], [], [], [], []
    focus = {}

    for model_name in MODELS_TO_EXPLAIN:
        for lead in config.FEATURE_LAG_HOURS:
            pkl = MODELS_DIR / f"lead{lead}_{model_name}.pkl"
            if not pkl.exists():
                print(f"(no {pkl.name} -- run 06 first; skipping)")
                continue
            cols = pd.read_pickle(pkl)["features"]
            want_iv = (model_name == "no_precip" and lead == FOCUS_LEAD)
            sv, base, prob, fold_id, iv, models = oof_shap(df, cols, folds, want_interactions=want_iv)
            abs_sv = np.abs(sv)

            # --- per feature
            lo, hi = year_block_boot(abs_sv, years, rng)
            for j, c in enumerate(cols):
                feat_rows.append({"lead_h": lead, "model": model_name, "feature": c, "variable": base_name(c),
                                  "lag_h": lag_of(c), "group": group_of(base_name(c)),
                                  "mean_abs_shap": abs_sv[:, j].mean(), "ci_lo": lo[j], "ci_hi": hi[j]})

            # --- per variable (sum over lags, per sample, then |.|)
            variables = sorted({base_name(c) for c in cols})
            by_var = np.column_stack([sv[:, [j for j, c in enumerate(cols) if base_name(c) == v]].sum(axis=1)
                                      for v in variables])
            abs_var = np.abs(by_var)
            lo, hi = year_block_boot(abs_var, years, rng)
            tot = abs_var.mean(axis=0).sum()
            for j, v in enumerate(variables):
                var_rows.append({"lead_h": lead, "model": model_name, "variable": v, "group": group_of(v),
                                 "mean_abs_shap": abs_var[:, j].mean(), "share": abs_var[:, j].mean() / tot,
                                 "ci_lo": lo[j], "ci_hi": hi[j]})

            # --- per group
            groups = sorted({group_of(v) for v in variables})
            by_grp = np.column_stack([by_var[:, [j for j, v in enumerate(variables) if group_of(v) == g]].sum(axis=1)
                                      for g in groups])
            abs_grp = np.abs(by_grp)
            tot_g = abs_grp.mean(axis=0).sum()
            for j, g in enumerate(groups):
                grp_rows.append({"lead_h": lead, "model": model_name, "group": g,
                                 "mean_abs_shap": abs_grp[:, j].mean(), "share": abs_grp[:, j].mean() / tot_g})

            rho_mean, rho_min = fold_stability(abs_var, fold_id)
            perm = grouped_permutation(df, cols, folds, models, rng)
            for v, d in perm.items():
                perm_rows.append({"lead_h": lead, "model": model_name, "variable": v,
                                  "group": group_of(v), "pr_auc_drop": d})
            from scipy.stats import spearmanr
            rho_perm = spearmanr([perm[v] for v in variables], abs_var.mean(axis=0))[0]
            stab_rows.append({"lead_h": lead, "model": model_name, "spearman_shap_vs_perm": rho_perm,
                              "spearman_mean": rho_mean,
                              "spearman_min": rho_min, "n_variables": len(variables)})

            top = pd.Series(abs_var.mean(axis=0), index=variables).sort_values(ascending=False).head(3)
            print(f"lead {lead:>2}h  {model_name:<9}  fold rho {rho_mean:.2f} (min {rho_min:.2f}), "
                  f"SHAP~perm rho {rho_perm:.2f}  "
                  f"top: " + ", ".join(f"{VAR_LABELS.get(v, v)} {s:.2f}" for v, s in top.items()))

            if model_name == "no_precip" and lead == FOCUS_LEAD:
                focus = {"cols": cols, "sv": sv, "base": base, "prob": prob, "iv": iv}
        print()

    feat = pd.DataFrame(feat_rows); var = pd.DataFrame(var_rows)
    grp = pd.DataFrame(grp_rows); stab = pd.DataFrame(stab_rows)
    feat.to_csv(OUT / "importance_by_feature.csv", index=False)
    var.to_csv(OUT / "importance_by_variable.csv", index=False)
    grp.to_csv(OUT / "importance_by_group.csv", index=False)
    stab.to_csv(OUT / "stability.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(OUT / "permutation_importance.csv", index=False)
    print(f"Saved importance tables and stability to {OUT}")

    precip_share = grp.query("model == 'all' and group == 'precip'").set_index("lead_h")["share"]
    if len(precip_share):
        print("Share of |SHAP| carried by precipitation in the 'all' model: "
              + ", ".join(f"{l} h {s:.0%}" for l, s in precip_share.items()))

    if focus:
        interactions(df, focus)
        case_days(df, focus)
        summary_plot(df, focus)
    plot_variable_heatmap(var)
    plot_group_share(grp)


# ----------------------------------------------------------------------
def interactions(df, focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols, iv, sv = focus["cols"], focus["iv"], focus["sv"]
    variables = sorted({base_name(c) for c in cols})
    idx = {v: [j for j, c in enumerate(cols) if base_name(c) == v] for v in variables}
    rows = []
    for a in range(len(variables)):
        for b in range(a + 1, len(variables)):
            va, vb = variables[a], variables[b]
            block = iv[:, idx[va]][:, :, idx[vb]].sum(axis=(1, 2)) * 2  # off-diagonal, both halves
            rows.append({"var_a": va, "var_b": vb, "group_a": group_of(va), "group_b": group_of(vb),
                         "mean_abs_interaction": np.nanmean(np.abs(block))})
    inter = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)
    done = ~np.isnan(iv).any(axis=(1, 2))
    a = np.abs(iv[done])
    diag = np.trace(a, axis1=1, axis2=2)
    share = float((a.sum(axis=(1, 2)) - diag).sum() / a.sum())
    # Net view: for each feature, how much of its SHAP value comes from interactions
    # (phi_i - phi_ii), after positive and negative pair terms cancel. The gross
    # share above sums |.| over ~2000 tiny off-diagonal cells and is inflated by noise.
    ivd = iv[done]
    main_eff = np.diagonal(ivd, axis1=1, axis2=2)
    total = ivd.sum(axis=2)
    net = float(np.abs(total - main_eff).sum() / np.abs(total).sum())
    print(f"\nInteractions at {FOCUS_LEAD} h (n={done.sum()} samples):")
    print(f"  net share of |SHAP| from interactions: {net:.0%}  (use this one; low = mostly additive)")
    print(f"  gross share (sum of all |pair cells|): {share:.0%}  (inflated by many tiny cells -- not for the poster)")
    inter.to_csv(OUT / f"interactions_lead{FOCUS_LEAD}.csv", index=False)
    print(f"\nTop variable interactions at {FOCUS_LEAD} h:")
    for _, r in inter.head(5).iterrows():
        cross = "  (cross-group)" if r.group_a != r.group_b else ""
        print(f"  {VAR_LABELS.get(r.var_a, r.var_a)} x {VAR_LABELS.get(r.var_b, r.var_b)}: "
              f"{r.mean_abs_interaction:.3f}{cross}")

    # dependence plots for the top 2 pairs, using the strongest lag of each variable
    X = df[cols].to_numpy(dtype=float)
    mean_abs = np.abs(sv).mean(axis=0)
    for k, (_, r) in enumerate(inter.head(2).iterrows(), start=1):
        ja = max(idx[r.var_a], key=lambda j: mean_abs[j])
        jb = max(idx[r.var_b], key=lambda j: mean_abs[j])
        fig, ax = plt.subplots(figsize=(5.2, 4))
        sc = ax.scatter(X[:, ja], sv[:, ja], c=X[:, jb], cmap="viridis", s=14, alpha=0.85, linewidths=0)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(pretty(cols[jb]))
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel(pretty(cols[ja]))
        ax.set_ylabel("SHAP value (log-odds of analog day)")
        ax.set_title(f"Interaction #{k} at -{FOCUS_LEAD} h", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        out = FIGURES / f"fig_shap_dependence_lead{FOCUS_LEAD}_{k}.png"
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"Saved: {out}")


def case_days(df, focus):
    """Waterfalls for Bozkurt and the other core-region flood days."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    cols, sv, base, prob = focus["cols"], focus["sv"], focus["base"], focus["prob"]
    is_case = pd.Series(False, index=df.index)
    if "flood_core" in df.columns:
        is_case |= df["flood_core"].fillna(False).astype(bool)
    bozkurt = (df["time"] >= "2021-08-08") & (df["time"] <= "2021-08-16") & (df["label"] == 1)
    is_case |= bozkurt

    rows = []
    for i in np.where(is_case)[0]:
        contrib = pd.Series(sv[i], index=cols).sort_values(key=np.abs, ascending=False)
        rows.append({"time": df.loc[i, "time"], "label": int(df.loc[i, "label"]),
                     "flood_core": bool(df.loc[i, "flood_core"]) if "flood_core" in df.columns else None,
                     "oof_prob": prob[i],
                     "top_contributors": "; ".join(f"{pretty(c)} {v:+.2f}" for c, v in contrib.head(5).items())})
        expl = shap.Explanation(values=sv[i], base_values=base[i], data=df.loc[i, cols].to_numpy(dtype=float),
                                feature_names=[pretty(c) for c in cols])
        plt.figure()
        shap.plots.waterfall(expl, max_display=10, show=False)
        plt.title(f"{df.loc[i, 'time']:%Y-%m-%d %H UTC}  |  P(analog) = {prob[i]:.2f}  |  -{FOCUS_LEAD} h", fontsize=9)
        out = FIGURES / f"fig_shap_case_{df.loc[i, 'time']:%Y%m%d}_lead{FOCUS_LEAD}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close("all")
        print(f"Saved: {out}")
    pd.DataFrame(rows).to_csv(OUT / "case_days.csv", index=False)


def summary_plot(df, focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    cols, sv = focus["cols"], focus["sv"]
    plt.figure()
    shap.summary_plot(sv, df[cols].to_numpy(dtype=float), feature_names=[pretty(c) for c in cols],
                      max_display=15, show=False, plot_size=(7, 6))
    plt.title(f"What drives a Bozkurt-type day, {FOCUS_LEAD} h ahead (out-of-fold SHAP)", fontsize=10)
    out = FIGURES / f"fig_shap_summary_lead{FOCUS_LEAD}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close("all")
    print(f"Saved: {out}")


def plot_variable_heatmap(var):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = var[var["model"] == "no_precip"]
    if sub.empty:
        return
    tab = sub.pivot(index="variable", columns="lead_h", values="share")
    order = sorted(tab.index, key=lambda v: (list(GROUP_PATTERNS).index(group_of(v)) if group_of(v) in GROUP_PATTERNS else 9,
                                             -tab.loc[v].fillna(0).mean()))
    tab = tab.loc[order]
    fig, ax = plt.subplots(figsize=(6.2, 0.38 * len(tab) + 1.4))
    im = ax.imshow(tab.to_numpy(dtype=float), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(tab.shape[1]), [f"-{c} h" if c else "0 h" for c in tab.columns])
    ax.set_yticks(range(len(tab)), [VAR_LABELS.get(v, v) for v in tab.index])
    for tick, v in zip(ax.get_yticklabels(), tab.index):
        tick.set_color(GROUP_COLORS.get(group_of(v), "#333333"))
    vmax = np.nanmax(tab.to_numpy(dtype=float))
    for r in range(tab.shape[0]):
        for c in range(tab.shape[1]):
            val = tab.iat[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.6 * vmax else "#333333")
    ax.set_xlabel("Lead time")
    ax.set_title("Share of the model's attention, by variable and lead time", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="share of mean |SHAP|")
    fig.tight_layout()
    out = FIGURES / "fig_shap_variable_by_lead.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved: {out}")


def plot_group_share(grp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = grp[grp["model"] == "all"]
    if sub.empty:
        return
    tab = sub.pivot(index="lead_h", columns="group", values="share").fillna(0)
    groups = [g for g in GROUP_COLORS if g in tab.columns]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    left = np.zeros(len(tab))
    for g in groups:
        vals = tab[g].to_numpy()
        ax.barh(range(len(tab)), vals, left=left, color=GROUP_COLORS[g], label=GROUP_LABELS[g],
                edgecolor="white", linewidth=2, height=0.6)
        for r, (l, v) in enumerate(zip(left, vals)):
            if v >= 0.08:
                ax.text(l + v / 2, r, f"{v:.0%}", ha="center", va="center", fontsize=8, color="#1a1a1a")
        left += vals
    ax.set_yticks(range(len(tab)), [f"-{l} h" if l else "0 h" for l in tab.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of mean |SHAP| (all-variables model)")
    ax.set_title("Which signal groups the model relies on, by lead time", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "fig_shap_group_share.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
