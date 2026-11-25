"""Simulated cohort written in the ADNIMERGE format.

Each participant has a hidden disease stage s(t) that grows over time:

    ds/dt = r * exp(kappa * (s - 3))

* r (personal speed) is larger for APOE-e4 carriers; the exponential term
  makes the disease accelerate as it advances.
* MRI atrophy follows a sigmoid of s that starts EARLY (centre 2.6).
* Cognition and daily function follow sigmoids of s_cog = s - reserve that
  start LATER (centres 4.2 and 4.8). "reserve" grows with education, so
  well-educated people show symptoms later than their atrophy suggests.
* Diagnosis is a noisy clinical judgement of s_cog:
  CN < 1.8 <= MCI < 4.6 <= Dementia.

The latent variables are saved separately (simulation_truth.csv) for sanity
checks only; the pipeline never reads them.
"""
import numpy as np
import pandas as pd

GROUPS = ["CN", "SMC", "EMCI", "LMCI", "AD"]
GROUP_PROB = [0.24, 0.06, 0.26, 0.29, 0.15]
SCOG_RANGE = {"CN": (-1.5, 1.6), "SMC": (0.3, 1.7), "EMCI": (1.9, 3.0),
              "LMCI": (2.9, 4.3), "AD": (4.7, 6.3)}
DX_AT_ENTRY = {"CN": "CN", "SMC": "CN", "EMCI": "MCI", "LMCI": "MCI", "AD": "Dementia"}
SCHEDULE = {
    "CN": [0, 6, 12, 24, 36, 48, 60, 72, 84, 96],
    "SMC": [0, 6, 12, 24, 36, 48, 60, 72],
    "EMCI": [0, 6, 12, 24, 36, 48, 60, 72, 84, 96],
    "LMCI": [0, 6, 12, 18, 24, 36, 48, 60, 72, 84],
    "AD": [0, 6, 12, 24, 36],
}
KAPPA = 0.45
MCI_THRESHOLD, DEMENTIA_THRESHOLD = 1.8, 4.6
QC_NOISE = {1: 1.0, 2: 2.0, 3: 4.0}   # measurement-noise multiplier by MRI quality


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stage_at(s0, rate, years):
    """Closed-form solution of ds/dt = r * exp(kappa * (s - 3)), capped at 10."""
    inner = np.exp(-KAPPA * (s0 - 3.0)) - KAPPA * rate * years
    return np.where(inner > 1e-6, 3.0 - np.log(np.maximum(inner, 1e-6)) / KAPPA, 10.0).clip(max=10.0)


def _viscode(month):
    return "bl" if month == 0 else f"m{month:02d}"


def simulate(n_participants=1800, seed=2026):
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for i in range(n_participants):
        rid = i + 1
        group = rng.choice(GROUPS, p=GROUP_PROB)
        female = rng.random() < (0.52 if group in ("CN", "SMC") else 0.42)
        age0 = float(np.clip(rng.normal({"CN": 74, "SMC": 72, "AD": 75}.get(group, 72.5), 7), 55, 90))
        educ = int(np.clip(round(rng.normal(16, 2.8)), 6, 20))
        apoe_p = {"CN": [.72, .25, .03], "SMC": [.68, .28, .04], "AD": [.33, .47, .20]}.get(group, [.48, .40, .12])
        apoe4 = int(rng.choice([0, 1, 2], p=apoe_p))
        apoe_missing = rng.random() < 0.04

        reserve = 0.35 * (educ - 16) / 2.8 + rng.normal(0, 0.4)
        s_cog0 = rng.uniform(*SCOG_RANGE[group])
        s0 = s_cog0 + reserve
        rate = float(np.exp(rng.normal(np.log(0.10), 0.5)) * (1 + 0.35 * apoe4))

        icv = rng.normal(1.35e6 if female else 1.50e6, 1.2e5)
        icv_scale = icv / 1.45e6
        hipp_var = rng.normal(1.0, 0.08)
        vent_var = float(np.exp(rng.normal(0.0, 0.25)))

        start = pd.Timestamp("2005-09-01") + pd.Timedelta(days=int(rng.integers(0, 9 * 365)))
        max_follow = rng.uniform(24, 120)
        truth.append(dict(RID=rid, group=group, s0=s0, rate=rate, reserve=reserve, s_cog0=s_cog0))

        dx_prev = None
        for month in SCHEDULE[group]:
            if month > max_follow:
                break
            if month > 0:
                if rng.random() < 0.05 + (0.05 if dx_prev == "Dementia" else 0.0):
                    break                                   # drop-out
                if rng.random() < 0.06:
                    continue                                # skipped visit
            jitter = 0 if month == 0 else int(rng.normal(0, 15))
            exam_date = start + pd.Timedelta(days=int(month * 30.44) + jitter)
            years = (exam_date - start).days / 365.25
            age = age0 + years
            age_eff = (age - 73) / 10
            s = float(stage_at(s0, rate, years))
            s_cog = s - reserve

            if month == 0:
                dx = DX_AT_ENTRY[group]
            else:
                lat = s_cog + rng.normal(0, 0.25)
                if dx_prev == "Dementia" and lat > DEMENTIA_THRESHOLD - 0.6:
                    dx = "Dementia"
                else:
                    dx = "CN" if lat < MCI_THRESHOLD else ("MCI" if lat < DEMENTIA_THRESHOLD else "Dementia")
            dx_prev = dx

            cog = _sigmoid((s_cog - 4.2) / 1.0)
            fun = _sigmoid((s_cog - 4.8) / 0.8)
            row = dict(
                RID=rid, PTID=f"SIM_S_{rid:04d}", VISCODE=_viscode(month), M=month,
                EXAMDATE=exam_date.date().isoformat(), DX_bl=group, DX=dx,
                AGE=round(age0, 1), PTGENDER="Female" if female else "Male",
                PTEDUCAT=educ, APOE4=apoe4,
                MMSE=float(np.clip(round(29.6 - 9 * cog - 3 * fun - 0.4 * age_eff
                                         + 0.1 * (educ - 16) + rng.normal(0, 1.1)), 0, 30)),
                ADAS13=float(np.clip(round(6 + 36 * cog + 6 * fun + 1.5 * age_eff
                                           - 0.4 * (educ - 16) + rng.normal(0, 2.8), 2), 0, 85)),
                CDRSB=float(np.clip(round(2 * (0.1 + 1.5 * cog + 9.5 * fun + rng.normal(0, 0.5))) / 2, 0, 18)),
                FAQ=float(np.clip(round(0.2 + 2 * cog + 22 * fun + rng.normal(0, 1.6)), 0, 30)),
            )
            for col, p_missing in (("MMSE", .03), ("ADAS13", .06), ("CDRSB", .02), ("FAQ", .05)):
                if month > 0 and rng.random() < p_missing:
                    row[col] = np.nan
            if apoe_missing:
                row["APOE4"] = np.nan

            p_scan = 0.88 if month == 0 else (0.78 if month <= 48 else 0.60)
            p_scan -= 0.10 if dx == "Dementia" else 0.0
            row.update(Hippocampus=np.nan, Entorhinal=np.nan, Ventricles=np.nan,
                       WholeBrain=np.nan, ICV=np.nan, MRI_QC=np.nan, MRI_SCANDATE=None)
            if rng.random() < p_scan:
                qc = int(rng.choice([1, 2, 3, 0], p=[0.70, 0.20, 0.08, 0.02]))   # 0 = failed QC
                scan_date = exam_date + pd.Timedelta(days=int(np.clip(rng.normal(-7, 12), -45, 25)))
                row.update(MRI_QC=qc, MRI_SCANDATE=scan_date.date().isoformat())
                if qc > 0:
                    atrophy = _sigmoid((s - 2.6) / 1.1)
                    nf = QC_NOISE[qc]
                    row.update(
                        Hippocampus=round(icv_scale * 7600 * hipp_var * (1 - 0.32 * atrophy)
                                          * (1 - 0.04 * age_eff) + rng.normal(0, 180 * nf)),
                        Entorhinal=round(icv_scale * 3900 * np.sqrt(hipp_var) * (1 - 0.30 * atrophy)
                                         * (1 - 0.03 * age_eff) + rng.normal(0, 200 * nf)),
                        Ventricles=round(icv_scale * 22000 * vent_var * (1 + 1.6 * atrophy)
                                         * (1 + 0.35 * age_eff) + rng.normal(0, 2000 * nf)),
                        WholeBrain=round(icv_scale * 1.06e6 * (1 - 0.09 * atrophy)
                                         * (1 - 0.035 * age_eff) + rng.normal(0, 1.2e4 * nf)),
                        ICV=round(icv + rng.normal(0, 1e4 * nf)),
                    )
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(truth)
