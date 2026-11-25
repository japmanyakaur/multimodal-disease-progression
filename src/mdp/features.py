"""Turns cohort windows into model inputs.

Neural models get, for every visit in the history:
  x   - test scores, standardised, last value carried forward (LOCF)
  m   - 1 if the score was really measured at this visit, 0 if carried forward/absent
  dt  - log(1 + years since the score was last measured)
  gap - years since the previous visit
  tti - years before the index visit
plus static covariates, MRI measures (roi), availability flags and reliability features.

Classical models get one row of summary features per window (last value and slope).
All scaling statistics are computed on the TRAINING split only.
"""
import numpy as np
import pandas as pd

from .cohort import MRI_COLS, SCORES

SCORE_NAMES = ["MMSE", "CDR-SB", "ADAS-Cog13", "FAQ"]
STATIC_NAMES = ["Age", "Female", "Education", "APOE4", "APOE4 missing"]
MRI_NAMES = ["Hippocampus", "Entorhinal", "Ventricles", "Whole brain", "ICV"]
REL_NAMES = ["Clinical available", "MRI available", "MRI quality good", "MRI quality acceptable",
             "MRI quality poor", "Scan age (years)", "History length", "Years since previous visit"]
DT_CAP = 5.0


def _mri_measures(v):
    """Head-size-normalised MRI measures (ratios to ICV) plus ICV itself."""
    icv = v["ICV"]
    return np.column_stack([v["Hippocampus"] / icv * 1e3, v["Entorhinal"] / icv * 1e3,
                            v["Ventricles"] / icv * 1e2, v["WholeBrain"] / icv, icv / 1e6])


def extract_raw(visits, cohort, max_len):
    """Unscaled arrays for every window (NaN where unobserved)."""
    n = len(cohort)
    vals = np.full((n, max_len, len(SCORES)), np.nan)
    times = np.zeros((n, max_len))
    lengths = np.zeros(n, dtype=int)
    first_exam = visits.groupby("RID").EXAMDATE.min()
    static = np.full((n, 4), np.nan)
    mri = np.full((n, len(MRI_COLS)), np.nan)
    qc = np.zeros(n, dtype=int)
    scan_age = np.zeros(n)

    for i, r in enumerate(cohort.itertuples()):
        hist = [int(x) for x in str(r.hist_rows).split(";")]
        v = visits.loc[hist]
        k = len(hist)
        lengths[i] = k
        vals[i, :k] = v[SCORES].to_numpy(float)
        times[i, :k] = (v.EXAMDATE - r.index_date).dt.days.to_numpy() / 365.25
        idx_visit = visits.loc[r.index_row]
        age_at_index = idx_visit.AGE + (r.index_date - first_exam[r.RID]).days / 365.25
        static[i] = [age_at_index, float(idx_visit.PTGENDER == "Female"), idx_visit.PTEDUCAT, idx_visit.APOE4]
        if r.mri_row >= 0:
            mv = visits.loc[[r.mri_row]]
            mri[i] = _mri_measures(mv)[0]
            qc[i] = int(mv.MRI_QC.iloc[0])
            scan_age[i] = (r.index_date - mv.MRI_SCANDATE.iloc[0]).days / 365.25
    return dict(vals=vals, times=times, lengths=lengths, static=static, mri=mri, qc=qc, scan_age=scan_age)


def fit_scalers(raw, train_mask):
    tr = np.where(train_mask)[0]
    v = raw["vals"][tr]
    lens = raw["lengths"][tr]
    valid = np.arange(v.shape[1])[None, :] < lens[:, None]
    score_vals = v[valid]
    mri_tr = raw["mri"][tr]
    mri_tr = mri_tr[~np.isnan(mri_tr).any(axis=1)]
    st = raw["static"][tr]
    return dict(
        score_mean=np.nanmean(score_vals, axis=0), score_std=np.nanstd(score_vals, axis=0) + 1e-6,
        age_mean=np.nanmean(st[:, 0]), age_std=np.nanstd(st[:, 0]) + 1e-6,
        educ_mean=np.nanmean(st[:, 2]), educ_std=np.nanstd(st[:, 2]) + 1e-6,
        mri_mean=mri_tr.mean(axis=0), mri_std=mri_tr.std(axis=0) + 1e-6,
    )


def transform(raw, sc):
    n, L, F = raw["vals"].shape
    z = (raw["vals"] - sc["score_mean"]) / sc["score_std"]
    observed = ~np.isnan(z)
    x = np.zeros((n, L, F))
    dt = np.full((n, L, F), DT_CAP)
    gap = np.zeros((n, L, 1))
    tti = np.zeros((n, L, 1))
    for i in range(n):
        k = raw["lengths"][i]
        t = raw["times"][i, :k]
        last_val = np.zeros(F)
        last_time = np.full(F, np.nan)
        for step in range(k):
            obs = observed[i, step]
            last_val[obs] = z[i, step, obs]
            last_time[obs] = t[step]
            x[i, step] = last_val
            since = t[step] - last_time
            dt[i, step] = np.where(np.isnan(since), DT_CAP, np.minimum(since, DT_CAP))
            gap[i, step, 0] = 0.0 if step == 0 else t[step] - t[step - 1]
            tti[i, step, 0] = -t[step]
    m = observed.astype(float)
    pad = np.arange(L)[None, :] >= raw["lengths"][:, None]
    m[pad] = 0.0
    dt[pad] = 0.0

    st = raw["static"]
    apoe_missing = np.isnan(st[:, 3])
    static = np.column_stack([
        (st[:, 0] - sc["age_mean"]) / sc["age_std"], st[:, 1],
        (st[:, 2] - sc["educ_mean"]) / sc["educ_std"],
        np.nan_to_num(st[:, 3]) / 2.0, apoe_missing.astype(float),
    ])

    a_mri = (~np.isnan(raw["mri"]).any(axis=1)).astype(float)
    roi = np.nan_to_num((raw["mri"] - sc["mri_mean"]) / sc["mri_std"])
    qc = raw["qc"]
    rel = np.column_stack([
        np.ones(n), a_mri, (qc == 1).astype(float), (qc == 2).astype(float), (qc == 3).astype(float),
        raw["scan_age"], raw["lengths"] / L, gap[np.arange(n), raw["lengths"] - 1, 0],
    ])
    arrays = dict(x=x, m=m, dt=np.log1p(dt), gap=gap, tti=tti, lengths=raw["lengths"],
                  static=static, roi=roi, a_clin=np.ones(n), a_mri=a_mri, rel=rel, qc=qc)
    return {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in arrays.items()}


def summary_features(raw):
    """One row per window for the classical baselines (E1)."""
    n = raw["vals"].shape[0]
    out = {}
    for f, name in enumerate(SCORE_NAMES):
        last, slope = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            k = raw["lengths"][i]
            y, t = raw["vals"][i, :k, f], raw["times"][i, :k]
            ok = ~np.isnan(y)
            if ok.any():
                last[i] = y[ok][-1]
            if ok.sum() >= 2 and np.ptp(t[ok]) > 0.1:
                slope[i] = np.polyfit(t[ok], y[ok], 1)[0]
        out[f"{name} (last)"] = last
        out[f"{name} (slope/yr)"] = slope
    out["Visits in history"] = raw["lengths"].astype(float)
    out["Years of history"] = -raw["times"][:, 0]
    for j, name in enumerate(["Age", "Female", "Education", "APOE4"]):
        out[name] = raw["static"][:, j]
    clinical_cols = list(out)
    for j, name in enumerate(MRI_NAMES):
        out[name] = raw["mri"][:, j]
    return pd.DataFrame(out), clinical_cols
