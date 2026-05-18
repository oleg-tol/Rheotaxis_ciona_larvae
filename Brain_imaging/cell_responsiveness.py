
"""
Quantification for whole-brain calcium imaging.

  compute_dff()               — ΔF/F with baseline normalisation
  identify_responsive_cells() — peak / AUC / latency / τ per stimulus window
  assign_regions()            — cell-type → brain-region mapping
  apply_qc()                  — remove artefact ROIs
  get_per_animal_recruitment()— responsive counts per animal × strength
  save_response_table()       — cell-type × strength CSV summary

  # Stats (optional, require scipy)
  compute_stats()             — Kruskal–Wallis + pairwise Mann–Whitney U
  compute_onesample_stats()   — Wilcoxon signed-rank of adaptation index vs 0
"""

import os
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import curve_fit
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  —  edit here
# ══════════════════════════════════════════════════════════════════════════════

EXPOSURE_TIME     = 0.673474          # s per frame
BASELINE_DUR      = 20.0              # s before stim 1 used for F0
STIM1_START, STIM1_END = 60.0,  90.0
STIM2_START, STIM2_END = 150.0, 180.0
SIGMA_THRESH      = 4.0               # response threshold (× baseline σ)
TAU_CEILING       = 20.0              # max accepted decay τ (s)
STIMULI_STRENGTHS = [75, 150, 300]

CELL_TO_REGION = {
    "mn": "hindbrain", "amg": "hindbrain", "pmg": "hindbrain",
    "pnsrn": "midbrain", "prrn": "midbrain", "antrn": "midbrain", "ant": "midbrain",
    "cor": "forebrain", "pr": "forebrain",
    "palp": "pns", "rten": "pns", "aten": "pns", "dcen": "pns",
}
REGION_ORDER = ["pns", "forebrain", "midbrain", "hindbrain"]


# ══════════════════════════════════════════════════════════════════════════════
# ΔF/F
# ══════════════════════════════════════════════════════════════════════════════

def _times(n):
    return np.arange(n) * EXPOSURE_TIME

def _idx(times, t0, t1):
    return np.where((times >= t0) & (times < t1))[0]


def compute_dff(merged_df):
    """
    F0 = median fluorescence over the 20 s pre-stimulus baseline.
    ΔF/F = (F − F0) / (|F0| + ε).
    σ_ΔF/F stored per ROI for use as response threshold reference.
    Returns dff_df and the time axis (seconds).
    """
    META = ["cell", "Experiment", "Stimuli_Strength"]
    tc   = [c for c in merged_df.columns if c not in META]
    t    = _times(len(tc))
    bidx = _idx(t, STIM1_START - BASELINE_DUR, STIM1_START)

    F   = merged_df[tc].values.astype(float)
    F0  = np.median(F[:, bidx], axis=1, keepdims=True)
    dff = (F - F0) / (np.abs(F0) + 1e-8)

    out = merged_df[META].copy()
    out[tc]          = dff
    out["F0"]        = F0.ravel()
    out["sigma_F0"]  = F[:, bidx].std(axis=1)
    out["sigma_dff"] = dff[:, bidx].std(axis=1)
    return out, t


# ══════════════════════════════════════════════════════════════════════════════
# QC
# ══════════════════════════════════════════════════════════════════════════════

def apply_qc(dff_df, metrics_df, verbose=True):
    """
    Exclude ROIs where:
      - F0 ≈ 0  (near-zero baseline inflates ΔF/F)
      - max |ΔF/F| ≥ 100  (photobleach or motion artefact)
    
    """
    KEY  = ["cell", "Experiment", "Stimuli_Strength"]
    tc   = [c for c in dff_df.columns if c not in KEY + ["F0", "sigma_F0", "sigma_dff"]]

    dff_df     = dff_df.drop_duplicates(KEY).copy()
    metrics_df = metrics_df.drop_duplicates(KEY).copy()

    dff_df["max_abs_dff"] = np.nanmax(np.abs(dff_df[tc].values), axis=1)
    metrics_df = metrics_df.merge(
        dff_df[KEY + ["max_abs_dff"]].drop_duplicates(KEY), on=KEY, how="left"
    )

    metrics_df["pass_qc"] = (
        (metrics_df["F0"].abs()    > 1e-6) &
        (metrics_df["max_abs_dff"] < 100.0)
    )

    if verbose:
        n, nb = len(metrics_df), (~metrics_df["pass_qc"]).sum()
        bad   = metrics_df[~metrics_df["pass_qc"]]
        print(f"QC: removed {nb}/{n} ({100*nb/n:.1f}%)  "
              f"[F0≈0: {(bad['F0'].abs()<=1e-6).sum()}  "
              f"|ΔF/F|≥100: {(bad['max_abs_dff']>=100).sum()}]")

    metrics_df = metrics_df[metrics_df["pass_qc"]].copy()
    dff_df     = dff_df.merge(metrics_df[KEY].drop_duplicates(), on=KEY, how="inner")
    return dff_df, metrics_df


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _monoexp(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def _fit_decay(trace, tl):
    """Monoexponential decay from peak onward. Returns dict or None."""
    pi = int(np.argmax(trace))
    if pi == 0 or len(trace) - pi < 3:
        return None
    ft, ff = tl[pi:] - tl[pi], trace[pi:]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _monoexp, ft, ff,
                p0=[max(ff[0]-ff[-1], 1e-6), 1.5, ff[-1]],
                bounds=([0, 0.05, -np.inf], [np.inf, TAU_CEILING*2, np.inf]),
                method="trf", maxfev=4000,
            )
        A, tau, C = popt
        if tau <= 0 or tau > TAU_CEILING:
            return None
        fit = _monoexp(ft, *popt)
        r2  = 1 - np.sum((ff-fit)**2) / (np.sum((ff-ff.mean())**2) + 1e-12)
        return {"tau": tau, "A": A, "C": C, "r2": r2}
    except Exception:
        return None


def identify_responsive_cells(dff_df, times):
    """
    Classify each ROI as responsive to each stimulus.
    Criterion: peak ΔF/F in stimulus window > SIGMA_THRESH × baseline σ_ΔF/F.

    Per-stimulus metrics: peak, AUC, latency, τ (optional decay fit).
    Cross-stimulus:       adaptation_peak_index, adaptation_auc_index,
                          responsive_any, responsive_both, tau_similarity.
    Adaptation index = (R2 − R1) / (R2 + R1);  negative = adaptation.
    """
    META = ["cell", "Experiment", "Stimuli_Strength", "F0", "sigma_F0", "sigma_dff"]
    tc   = [c for c in dff_df.columns if c not in META]
    recs = []

    for _, row in dff_df.iterrows():
        trace = row[tc].values.astype(float)
        sigma = row["sigma_dff"]
        rec   = {k: row[k] for k in META}

        for sn, s0, s1 in [(1, STIM1_START, STIM1_END), (2, STIM2_START, STIM2_END)]:
            idx = _idx(times, s0, s1)
            if not len(idx):
                rec.update({f"stim{sn}_{k}": np.nan
                            for k in ["peak","auc","latency","tau","r2"]})
                rec.update({f"responsive_stim{sn}": False,
                            f"valid_decay_stim{sn}": False})
                continue
            st   = trace[idx]
            tl   = times[idx] - times[idx[0]]
            dec  = _fit_decay(st, tl)
            above = np.where(st > SIGMA_THRESH * sigma)[0]
            rec.update({
                f"stim{sn}_peak":        float(np.nanmax(st)),
                f"stim{sn}_auc":         float(np.trapz(np.maximum(st, 0), tl)),
                f"stim{sn}_latency":     float(tl[above[0]]) if len(above) else np.nan,
                f"stim{sn}_tau":         dec["tau"] if dec else np.nan,
                f"stim{sn}_r2":          dec["r2"]  if dec else np.nan,
                f"responsive_stim{sn}":  bool(np.nanmax(st) > SIGMA_THRESH * sigma),
                f"valid_decay_stim{sn}": bool(dec),
            })

        p1, p2 = rec.get("stim1_peak", np.nan), rec.get("stim2_peak", np.nan)
        a1, a2 = rec.get("stim1_auc",  np.nan), rec.get("stim2_auc",  np.nan)
        t1, t2 = rec.get("stim1_tau",  np.nan), rec.get("stim2_tau",  np.nan)
        r1, r2 = rec.get("responsive_stim1", False), rec.get("responsive_stim2", False)

        rec["responsive_any"]          = bool(r1 or r2)
        rec["responsive_both"]         = bool(r1 and r2)
        rec["adaptation_peak_index"]   = (p2-p1)/(p2+p1+1e-8) if r1 and p1>0 else np.nan
        rec["adaptation_auc_index"]    = (a2-a1)/(a2+a1+1e-8) if r1 and a1>0 else np.nan
        rec["tau_similarity"]          = (
            min(t1,t2)/max(t1,t2)
            if not (np.isnan(t1) or np.isnan(t2)) and max(t1,t2)>0 else np.nan
        )
        recs.append(rec)

    return pd.DataFrame(recs)


def assign_regions(metrics_df):
    d = metrics_df.copy()
    d["region"] = d["cell"].map(CELL_TO_REGION)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# RECRUITMENT & TABLES
# ══════════════════════════════════════════════════════════════════════════════

def get_per_animal_recruitment(metrics_df):
    """Responsive neuron counts and fractions per animal × stimulus strength."""
    df = metrics_df.copy()
    df["Stimuli_Strength"] = df["Stimuli_Strength"].astype(int)
    pa = (df.groupby(["Experiment", "Stimuli_Strength"])
            .agg(n_total      =("cell",             "count"),
                 n_resp_stim1 =("responsive_stim1", "sum"),
                 n_resp_stim2 =("responsive_stim2", "sum"),
                 n_resp_any   =("responsive_any",   "sum"),
                 n_resp_both  =("responsive_both",  "sum"))
            .reset_index())
    for col in ["stim1", "stim2", "any", "both"]:
        pa[f"fraction_resp_{col}"] = pa[f"n_resp_{col}"] / pa["n_total"]
    return pa


def save_response_table(metrics_df, output_dir):
    """
    Cell-type × stimulus-strength responsiveness CSV.
    Rows sorted by mean fraction responsive (desc) within each region.
    """
    df = metrics_df.dropna(subset=["region"]).copy()
    df["Stimuli_Strength"] = df["Stimuli_Strength"].astype(int)
    records = []
    for region in REGION_ORDER:
        for ct in sorted(df[df["region"] == region]["cell"].unique()):
            row, fracs = {"region": region, "cell_type": ct}, []
            for s in STIMULI_STRENGTHS:
                sub  = df[(df["cell"] == ct) & (df["Stimuli_Strength"] == s)]
                n_t, n_r = len(sub), int(sub["responsive_stim1"].sum())
                frac = round(n_r/n_t, 3) if n_t else np.nan
                row.update({f"n_total_{s}": n_t, f"n_resp_{s}": n_r, f"frac_{s}": frac})
                if not np.isnan(frac): fracs.append(frac)
            row["mean_frac"] = round(float(np.mean(fracs)), 3) if fracs else np.nan
            records.append(row)
    tbl = (pd.DataFrame(records)
           .sort_values(["region", "mean_frac"], ascending=[True, False])
           .reset_index(drop=True))
    tbl.to_csv(os.path.join(output_dir, "response_table_cellwise.csv"), index=False)
    print(f"  → response_table_cellwise.csv  ({len(tbl)} rows)")
    return tbl


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS  (optional — require scipy)
# ══════════════════════════════════════════════════════════════════════════════

def _stars(p):
    if pd.isna(p): return ""
    p = float(p)
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"


def compute_stats(df, metric, filter_col=None, label=""):
    """Kruskal–Wallis + pairwise Mann–Whitney U across stimulus strengths."""
    sub = df.dropna(subset=[metric]).copy()
    if filter_col:
        sub = sub[sub[filter_col].fillna(False)]
    sub["Stimuli_Strength"] = sub["Stimuli_Strength"].astype(int)
    groups = sorted(sub["Stimuli_Strength"].unique())
    gdata  = {g: sub[sub["Stimuli_Strength"]==g][metric].dropna().values for g in groups}
    valid  = [v for v in gdata.values() if len(v) >= 2]
    kw, kp = kruskal(*valid) if len(valid) >= 2 else (np.nan, np.nan)
    recs   = [{
        "comparison": f"{label}_KW", "group1": str(groups), "group2": "",
        "n1": sum(len(v) for v in valid), "n2": "",
        "stat": round(kw,3) if not np.isnan(kw) else np.nan,
        "p":    round(kp,4) if not np.isnan(kp) else np.nan,
        "significance": _stars(kp), "test": "Kruskal-Wallis",
    }]
    for g1, g2 in combinations(groups, 2):
        v1, v2 = gdata[g1], gdata[g2]
        if len(v1) < 2 or len(v2) < 2:
            recs.append({"comparison": f"{label}_{g1}v{g2}",
                         "group1": g1, "group2": g2,
                         "n1": len(v1), "n2": len(v2),
                         "stat": np.nan, "p": np.nan,
                         "significance": "insufficient n", "test": "Mann-Whitney"})
            continue
        u, p = mannwhitneyu(v1, v2, alternative="two-sided")
        recs.append({"comparison": f"{label}_{g1}v{g2}",
                     "group1": g1, "group2": g2,
                     "n1": len(v1), "n2": len(v2),
                     "stat": round(u,1), "p": round(p,4),
                     "significance": _stars(p), "test": "Mann-Whitney"})
    return pd.DataFrame(recs)


def compute_onesample_stats(metrics_df, output_dir):
    """Wilcoxon signed-rank test of adaptation_auc_index vs 0, per region × strength."""
    df = metrics_df[metrics_df["responsive_stim1"]].dropna(
        subset=["adaptation_auc_index"]).copy()
    df["Stimuli_Strength"] = df["Stimuli_Strength"].astype(int)
    records = []
    for scope, sub in [("wholebrain", df)] + [(r, df[df["region"]==r]) for r in REGION_ORDER]:
        for s in STIMULI_STRENGTHS:
            vals = sub[sub["Stimuli_Strength"]==s]["adaptation_auc_index"].dropna().values
            n = len(vals)
            if n < 4:
                records.append({"scope": scope, "strength": s, "n": n,
                                 "median": np.nan, "W": np.nan, "p": np.nan,
                                 "significance": "insufficient n", "direction": ""})
                continue
            try:    W, p = wilcoxon(vals, alternative="two-sided")
            except: W, p = np.nan, np.nan
            records.append({"scope": scope, "strength": s, "n": n,
                             "median":       round(float(np.median(vals)), 3),
                             "W":            round(W, 1) if not np.isnan(W) else np.nan,
                             "p":            round(p, 4) if not np.isnan(p) else np.nan,
                             "significance": _stars(p) if not np.isnan(p) else "",
                             "direction":    "adaptation" if np.median(vals)<0 else "sensitization"})
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(output_dir, "stats_onesample_adaptation.csv"), index=False)
    return out


def run_all_stats(metrics_df, per_animal, output_dir):
    """Run all group-comparison stats and save to stats_results.csv."""
    all_stats = []
    for metric, col in [("stim1_peak","peak"), ("adaptation_auc_index","adapt")]:
        for scope, sub in [("wholebrain", metrics_df)] + \
                           [(r, metrics_df[metrics_df["region"]==r]) for r in REGION_ORDER]:
            st = compute_stats(sub, metric, filter_col="responsive_stim1",
                               label=f"{col}_{scope}")
            all_stats.append(st.assign(metric=metric, scope=scope))
    st = compute_stats(per_animal, "fraction_resp_stim1", label="recruitment_per_animal")
    all_stats.append(st.assign(metric="fraction_resp_stim1", scope="wholebrain"))
    combined = pd.concat(all_stats, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "stats_results.csv"), index=False)
    sig = combined[combined["significance"].isin(["*","**","***"])]
    print(f"\nSignificant comparisons: {len(sig)}")
    if len(sig):
        print(sig[["metric","scope","comparison","n1","n2","p","significance"]]
              .to_string(index=False))
    return combined