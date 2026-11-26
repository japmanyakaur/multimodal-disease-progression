"""Cohort construction, 24-month target, MRI linkage, participant splits and leakage checks.

Unit of analysis: one participant at one MCI "index visit" (time t0).
Inputs may only use data recorded up to the predictor cut-off (t0 + 30 days);
the label is read from visits after t0.
"""
import hashlib
import warnings

import numpy as np
import pandas as pd

SCORES = ["MMSE", "CDRSB", "ADAS13", "FAQ"]
MRI_COLS = ["Hippocampus", "Entorhinal", "Ventricles", "WholeBrain", "ICV"]
REQUIRED = ["RID", "VISCODE", "EXAMDATE", "DX", "DX_bl", "AGE", "PTGENDER",
            "PTEDUCAT", "APOE4"] + SCORES + MRI_COLS
DAYS_PER_MONTH = 30.44


class LeakageError(AssertionError):
    pass


# --------------------------------------------------------------------------- loading
def load_visits(path):
    """Read an ADNIMERGE-format CSV, validate it and clean it. Returns (visits, flow)."""
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required ADNIMERGE columns: {missing}")
    flow = [("Visits in data release", len(df), df.RID.nunique())]

    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    for col in SCORES + MRI_COLS + ["AGE", "PTEDUCAT", "APOE4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["DX"] = df["DX"].replace({"AD": "Dementia", "NL": "CN"})

    if "MRI_QC" not in df.columns:
        warnings.warn("No MRI_QC column found: every available scan is treated as good quality.")
        df["MRI_QC"] = np.where(df[MRI_COLS].notna().all(axis=1), 1, np.nan)
    if "MRI_SCANDATE" not in df.columns:
        df["MRI_SCANDATE"] = df["EXAMDATE"]
    df["MRI_SCANDATE"] = pd.to_datetime(df["MRI_SCANDATE"], errors="coerce")
    df["MRI_QC"] = pd.to_numeric(df["MRI_QC"], errors="coerce")

    df = df[df.RID.notna() & df.EXAMDATE.notna()]
    flow.append(("After removing records without participant ID or date", len(df), df.RID.nunique()))
    df = df.drop_duplicates(["RID", "EXAMDATE"]).drop_duplicates(["RID", "VISCODE"])
    flow.append(("After removing duplicate visits", len(df), df.RID.nunique()))

    df["mri_ok"] = df[MRI_COLS].notna().all(axis=1) & df.MRI_QC.isin([1, 2, 3]) & df.MRI_SCANDATE.notna()
    df = df.sort_values(["RID", "EXAMDATE"]).reset_index(drop=True)
    return df, flow


# --------------------------------------------------------------------------- cohort
def build_cohort(df, cfg):
    """Create one row per eligible index visit. Returns (cohort, flow_rows, linkage_counts)."""
    c = cfg["cohort"]
    h, tol = c["horizon_months"], c["tolerance_months"]
    cutoff_days = c["cutoff_days_after_index"]
    before, after = pd.Timedelta(days=c["mri_days_before_index"]), pd.Timedelta(days=c["mri_days_after_index"])

    counts = dict(mci_visits=0, prior_dementia=0, no_scores=0, insufficient_followup=0, overlap=0)
    linkage = dict(exact=0, window=0, none=0)
    rows = []
    for rid, g in df.groupby("RID", sort=True):
        idx = g.index.to_numpy()
        dates = g.EXAMDATE.to_numpy()
        dx = g.DX.to_numpy()
        for j in range(len(g)):
            if dx[j] != c["index_diagnosis"]:
                continue
            counts["mci_visits"] += 1
            if np.any(dx[:j] == "Dementia"):
                counts["prior_dementia"] += 1
                continue
            if g.iloc[j][SCORES].isna().all():
                counts["no_scores"] += 1
                continue

            months = (dates[j + 1:] - dates[j]) / np.timedelta64(1, "D") / DAYS_PER_MONTH
            fut_dx = dx[j + 1:]
            known = pd.notna(fut_dx)
            in_window = months <= h + tol
            if np.any(in_window & (fut_dx == "Dementia")):
                label = 1
                months_to_event = float(months[in_window & (fut_dx == "Dementia")][0])
            elif np.any(known & (months >= h - tol) & (fut_dx != "Dementia")):
                label, months_to_event = 0, np.nan
            else:
                counts["insufficient_followup"] += 1
                continue

            # time to first dementia diagnosis (any time after t0), censored at the last known visit
            dem = np.where(fut_dx == "Dementia")[0]
            tte, tte_event = (float(months[dem[0]]), 1) if len(dem) else (float(months[known].max()), 0)
            # secondary regression target: CDR-SB at the visit closest to t0 + h within the tolerance
            cdr = g.CDRSB.to_numpy()[j + 1:]
            ok_cdr = (np.abs(months - h) <= tol) & ~np.isnan(cdr)
            cdrsb_h = float(cdr[ok_cdr][np.argmin(np.abs(months[ok_cdr] - h))]) if ok_cdr.any() else np.nan

            index_date = pd.Timestamp(dates[j])
            cutoff = index_date + pd.Timedelta(days=cutoff_days)
            if len(months) and pd.Timestamp(dates[j + 1]) <= cutoff:
                counts["overlap"] += 1
                continue

            # MRI linkage: same visit first, otherwise nearest earlier visit in the window
            mri_row, match = -1, "none"
            g_past = g.iloc[: j + 1]
            ok = g_past.mri_ok & (g_past.MRI_SCANDATE >= index_date - before) & \
                (g_past.MRI_SCANDATE <= index_date + after) & (g_past.MRI_SCANDATE <= cutoff)
            if ok.iloc[-1]:
                mri_row, match = int(idx[j]), "exact"
            elif ok.any():
                mri_row, match = int(g_past.index[ok.to_numpy()][-1]), "window"
            linkage[match] += 1

            hist = idx[max(0, j + 1 - c["max_history_visits"]): j + 1]
            rows.append(dict(
                RID=int(rid), index_row=int(idx[j]), index_viscode=g.VISCODE.iloc[j],
                index_date=index_date, cutoff_date=cutoff, label=label,
                months_to_event=months_to_event, tte_months=tte, tte_event=tte_event, cdrsb_target=cdrsb_h,
                index_cdrsb=float(g.CDRSB.iloc[j]), mri_row=mri_row, mri_match=match,
                hist_rows=";".join(map(str, hist)),
                first_outcome_date=pd.Timestamp(dates[j + 1]),
            ))
    cohort = pd.DataFrame(rows)
    cohort.insert(0, "window_id", np.arange(len(cohort)))

    flow = [
        ("MCI visits (candidate index visits)", counts["mci_visits"], None),
        ("Excluded: dementia diagnosed before the index visit", -counts["prior_dementia"], None),
        ("Excluded: no cognitive score at the index visit", -counts["no_scores"], None),
        ("Excluded: outcome unknown at 24 months (insufficient follow-up)", -counts["insufficient_followup"], None),
        ("Excluded: next visit inside the predictor cut-off", -counts["overlap"], None),
        ("Analysis cohort (index windows)", len(cohort), cohort.RID.nunique()),
    ]
    return cohort, flow, linkage


# --------------------------------------------------------------------------- feasibility audit (Section 4.2)
def feasibility_audit(df, cfg):
    """Counts, missingness and outcome availability recorded before modelling."""
    per = df.groupby("RID")
    follow = (per.EXAMDATE.max() - per.EXAMDATE.min()).dt.days / 365.25
    nvis = per.size()
    scans = df[df.MRI_QC.notna()]

    def med(x):
        return f"{np.median(x):.1f} [{np.percentile(x, 25):.1f}, {np.percentile(x, 75):.1f}]"

    rows = [("Participants", f"{df.RID.nunique():,}"), ("Visits", f"{len(df):,}"),
            ("Visits per participant, median [IQR]", med(nvis)),
            ("Follow-up years, median [IQR]", med(follow))]
    for grp, n in per.DX_bl.first().value_counts().items():
        rows.append((f"Baseline group {grp}", f"{n:,}"))
    for col in SCORES:
        rows.append((f"{col} missing (% of visits)", f"{df[col].isna().mean():.1%}"))
    rows.append(("APOE4 missing (% of participants)", f"{per.APOE4.first().isna().mean():.1%}"))
    rows.append(("Visits with a usable MRI", f"{df.mri_ok.mean():.1%}"))
    for q, name in ((1, "good"), (2, "acceptable"), (3, "poor"), (0, "failed QC")):
        rows.append((f"MRI scans rated {name}", f"{(scans.MRI_QC == q).mean():.1%}"))
    return pd.DataFrame(rows, columns=["Audit item", "Value"])


def horizon_audit(df, cfg, horizons=(12, 24, 36)):
    """Outcome availability at candidate forecast horizons."""
    rows = []
    for h in horizons:
        c2 = {**cfg, "cohort": {**cfg["cohort"], "horizon_months": h}}
        coh, _, _ = build_cohort(df, c2)
        rows.append({"Horizon (months)": h, "Index windows": len(coh), "Participants": coh.RID.nunique(),
                     "Converters": int(coh.label.sum()), "Conversion rate": f"{coh.label.mean():.1%}",
                     "Windows with MRI": f"{(coh.mri_row >= 0).mean():.1%}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- splits
def split_participants(cohort, cfg, seed):
    """Participant-level split, stratified by whether the participant ever converts."""
    s = cfg["splits"]
    rng = np.random.default_rng(seed)
    part_label = cohort.groupby("RID").label.max()
    assign = {}
    for lab in (0, 1):
        ids = rng.permutation(part_label.index[part_label == lab].to_numpy())
        n = len(ids)
        n_test, n_cal = round(n * s["test"]), round(n * s["calibration"])
        n_val = round((n - n_test - n_cal) * s["validation_of_development"])
        for k, rid in enumerate(ids):
            if k < n_test:
                assign[rid] = "test"
            elif k < n_test + n_cal:
                assign[rid] = "calibration"
            elif k < n_test + n_cal + n_val:
                assign[rid] = "validation"
            else:
                assign[rid] = "train"
    manifest = pd.DataFrame({"RID": list(assign), "split": list(assign.values())}).sort_values("RID")
    return manifest.reset_index(drop=True)


def manifest_hash(manifest):
    data = manifest.sort_values("RID").to_csv(index=False).encode()
    return hashlib.sha256(data).hexdigest()[:16]


# --------------------------------------------------------------------------- leakage checks
def assert_no_participant_overlap(cohort):
    n_splits = cohort.groupby("RID").split.nunique()
    bad = n_splits[n_splits > 1]
    if len(bad):
        raise LeakageError(f"{len(bad)} participant(s) appear in more than one split, e.g. RID {bad.index[0]}")


def assert_predictors_before_cutoff(cohort, visits):
    """Every history visit and the linked MRI must precede the cut-off; outcomes must follow it."""
    for r in cohort.itertuples():
        hist = [int(x) for x in str(r.hist_rows).split(";")]
        if visits.loc[hist, "EXAMDATE"].max() > r.index_date:
            raise LeakageError(f"Window {r.window_id}: a history visit is after the index date")
        if r.mri_row >= 0 and visits.loc[r.mri_row, "MRI_SCANDATE"] > r.cutoff_date:
            raise LeakageError(f"Window {r.window_id}: MRI acquired after the predictor cut-off")
        if r.first_outcome_date <= r.cutoff_date:
            raise LeakageError(f"Window {r.window_id}: outcome visit inside the predictor window")
