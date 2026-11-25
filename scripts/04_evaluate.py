"""Step 4: evaluate every model from the saved predictions (E1-E7, hypotheses, ablations, subgroups).

1. Seed ensembles: the 5 seeds of each neural model are averaged (mean logit).
2. Temperature T and decision threshold are fitted on the CALIBRATION split, then fixed.
3. The locked TEST split is used once, here, for every reported number.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402

from mdp import metrics as M  # noqa: E402
from mdp.config import MODELS, PREDICTIONS, PROCESSED, RESULTS, RUNS, TABLES, load_config  # noqa: E402
from mdp.explain import FEATURE_NAMES, feature_vector, integrated_gradients, stability, temporal_profile  # noqa: E402
from mdp.models import FusionModel  # noqa: E402
from mdp.train import to_tensors  # noqa: E402

MAIN = {
    "e1_lr_clin": "E1 Logistic regression (clinical)",
    "e1_gbm_clin": "E1 Gradient boosting (clinical)",
    "e1_lr_clinmri": "E1 Logistic regression (clinical + MRI)",
    "e1_gbm_clinmri": "E1 Gradient boosting (clinical + MRI)",
    "e2_clinical": "E2 GRU, clinical history only",
    "e3_mri": "E3 MRI only",
    "e4_early": "E4 Early fusion",
    "e4_late": "E4 Late fusion",
    "e4_xattn": "E4 Cross-attention fusion",
    "e5_proposed": "E5 Proposed reliability-aware fusion",
}
ABLATIONS = {
    "e5_proposed": "Full proposed model",
    "abl_no_gate": "- reliability gate",
    "abl_no_xattn": "- cross-attention",
    "abl_no_time": "- explicit time gaps",
    "abl_no_mask": "- missingness masks",
    "abl_no_moddrop": "- modality dropout",
}
SENSITIVITY = {"sens_pdrop_015": 0.15, "e5_proposed": 0.30, "sens_pdrop_050": 0.50}
CONVENTIONAL = ["e4_early", "e4_late", "e4_xattn"]
SINGLE = ["e1_lr_clin", "e1_gbm_clin", "e2_clinical"]
FUSION = ["e1_lr_clinmri", "e1_gbm_clinmri", "e4_early", "e4_late", "e4_xattn", "e5_proposed"]
H2_COMPARATOR = "e4_xattn"          # same architecture without the gate and without modality dropout
STRESS_MODELS = ["e2_clinical", "e1_gbm_clinmri", "e4_early", "e4_late", "e4_xattn", "e5_proposed"]


# --------------------------------------------------------------------------- loading
def late_fusion(a_mri):
    clin = pd.read_csv(PREDICTIONS / "e2_clinical.csv.gz")
    mri = pd.read_csv(PREDICTIONS / "e3_mri.csv.gz")
    m = clin.merge(mri, on=["window_id", "seed", "condition"], suffixes=("_c", "_m"))
    pc, pm = M.sigmoid(m.logit_c), M.sigmoid(m.logit_m)
    has_mri = (a_mri[m.window_id] == 1) & (m.condition != "mri_removed")
    p = np.where(m.condition == "clinical_removed", pm, np.where(has_mri, (pc + pm) / 2, pc))
    out = m[["window_id", "seed", "condition"]].copy()
    out["logit"], out["alpha_mri"] = M.logit(p), np.nan
    return out


class Model:
    def __init__(self, mid, df, cal_idx, y):
        self.id, self.seeds = mid, df
        g = df.groupby(["condition", "window_id"])
        self.logit = g.logit.mean().unstack(0)          # rows: window_id, columns: condition
        self.alpha = g.alpha_mri.mean().unstack(0)
        z = self.logit["natural"].reindex(cal_idx)
        ok = z.notna().to_numpy()
        self.T = M.fit_temperature(z.to_numpy()[ok], y[cal_idx][ok])
        self.thr = M.youden_threshold(y[cal_idx][ok], self.prob("natural", cal_idx[ok]))

    def prob(self, condition, idx, calibrated=True):
        if condition not in self.logit.columns:              # condition not applicable to this model
            return np.full(len(idx), np.nan)
        z = self.logit[condition].reindex(idx).to_numpy()
        return M.sigmoid(z / (self.T if calibrated else 1.0))

    def seed_aurocs(self, y, idx, condition="natural"):
        out = []
        for _, g in self.seeds[self.seeds.condition == condition].groupby("seed"):
            z = g.set_index("window_id").logit.reindex(idx).to_numpy()
            out.append(M.auroc(y[idx], z))
        return np.array(out)


def metric_row(model, y, idx, boot, bins, condition="natural"):
    p = model.prob(condition, idx)
    yv = y[idx]
    seeds = model.seed_aurocs(y, idx, condition)
    row = {"AUROC [95% CI]": M.fmt_ci(*boot.ci(M.auroc, yv, p)),
           "AUPRC [95% CI]": M.fmt_ci(*boot.ci(M.auprc, yv, p)),
           "Brier [95% CI]": M.fmt_ci(*boot.ci(M.brier, yv, p)),
           "ECE [95% CI]": M.fmt_ci(*boot.ci(lambda a, b: M.ece(a, b, bins), yv, p))}
    t = M.threshold_metrics(yv, p, model.thr)
    row.update({"Sensitivity": f"{t['sensitivity']:.3f}", "Specificity": f"{t['specificity']:.3f}",
                "F1": f"{t['f1']:.3f}",
                "AUROC per seed, mean (SD)": f"{seeds.mean():.3f} ({seeds.std():.3f})" if len(seeds) > 1 else "-"})
    return row, M.auroc(yv, p)


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    ev = cfg["evaluation"]
    bins, n_boot = ev["ece_bins"], ev["bootstrap_replicates"]
    cohort = pd.read_csv(PROCESSED / "cohort.csv")
    arrays = dict(np.load(PROCESSED / "features.npz"))
    assert (cohort.window_id.to_numpy() == np.arange(len(cohort))).all()
    y, rid, split = cohort.label.to_numpy(), cohort.RID.to_numpy(), cohort.split.to_numpy()
    a_mri = arrays["a_mri"]

    cal = np.where(split == "calibration")[0]
    val = np.where(split == "validation")[0]
    test = np.where(split == "test")[0]
    test_complete = test[a_mri[test] == 1]
    test_nomri = test[a_mri[test] == 0]
    boot_all = M.ClusterBootstrap(rid[test], n_boot, seed=1)
    boot_cc = M.ClusterBootstrap(rid[test_complete], n_boot, seed=2)
    summary = {"n_test_windows": len(test), "n_test_participants": int(len(np.unique(rid[test]))),
               "n_test_complete": len(test_complete), "n_test_no_mri": len(test_nomri),
               "test_events": int(y[test].sum())}

    ids = [m for m in MAIN if m != "e4_late"] + [m for m in list(ABLATIONS) + list(SENSITIVITY) if m not in MAIN]
    models = {}
    for mid in ids:
        df = pd.read_csv(PREDICTIONS / f"{mid}.csv.gz")
        c = cal[a_mri[cal] == 1] if mid == "e3_mri" else cal
        models[mid] = Model(mid, df, c, y)
    models["e4_late"] = Model("e4_late", late_fusion(a_mri), cal, y)

    # ---- validation-based selections (made before looking at test results)
    val_auc = {m: M.auroc(y[val], models[m].prob("natural", val)) for m in models if m != "e3_mri"}
    best_conv = max(CONVENTIONAL, key=val_auc.get)
    best_single = max(SINGLE, key=val_auc.get)
    best_fusion = max(FUSION, key=val_auc.get)
    summary.update(validation_auroc={k: round(v, 4) for k, v in val_auc.items()},
                   strongest_conventional_fusion=best_conv, strongest_single_modality=best_single,
                   strongest_fusion=best_fusion)

    # ---- Table: main results on the full locked test set (natural availability)
    rows, cc_rows = [], []
    test_auc = {}
    for mid, label in MAIN.items():
        m = models[mid]
        idx, boot = (test_complete, boot_cc) if mid == "e3_mri" else (test, boot_all)
        r, test_auc[mid] = metric_row(m, y, idx, boot, bins)
        rows.append({"Model": label + (" *" if mid == "e3_mri" else ""), "Windows": len(idx), **r})
        r2, _ = metric_row(m, y, test_complete, boot_cc, bins)
        cc_rows.append({"Model": label, **r2})
    pd.DataFrame(rows).to_csv(TABLES / "table_main_results.csv", index=False)
    pd.DataFrame(cc_rows).to_csv(TABLES / "table_complete_case_results.csv", index=False)
    summary["test_auroc"] = {k: round(v, 4) for k, v in test_auc.items()}

    # ---- Hypotheses (paired participant-level bootstrap)
    def paired(a, b, idx, boot, fn, condition="natural"):
        return boot.paired_diff(fn, (y[idx], models[a].prob(condition, idx)), (y[idx], models[b].prob(condition, idx)))

    hyp = []
    d_auc = paired("e5_proposed", best_conv, test, boot_all, M.auroc)
    d_bri = paired("e5_proposed", best_conv, test, boot_all, M.brier)
    h1 = d_auc[1] > 0 and d_auc[0] >= ev["min_meaningful_difference"] and d_bri[0] <= ev["brier_noninferiority_margin"]
    hyp.append({"Question": "H1 / RQ1b", "Comparison": f"Proposed - {MAIN[best_conv]}",
                "Population": "full test set", "Delta AUROC [95% CI]": M.fmt_ci(*d_auc),
                "Delta Brier [95% CI]": M.fmt_ci(*d_bri), "Decision": "supported" if h1 else "not supported"})
    d_rq1 = paired(best_fusion, best_single, test, boot_all, M.auroc)
    rq1 = d_rq1[1] > 0 and d_rq1[0] >= ev["min_meaningful_difference"]
    hyp.append({"Question": "RQ1", "Comparison": f"{MAIN[best_fusion]} - {MAIN[best_single]}",
                "Population": "full test set", "Delta AUROC [95% CI]": M.fmt_ci(*d_rq1),
                "Delta Brier [95% CI]": M.fmt_ci(*paired(best_fusion, best_single, test, boot_all, M.brier)),
                "Decision": "fusion better" if rq1 else "no meaningful improvement"})
    for other in ("e1_gbm_clinmri", "e2_clinical"):
        d = paired("e5_proposed", other, test, boot_all, M.auroc)
        hyp.append({"Question": "Supplementary", "Comparison": f"Proposed - {MAIN[other]}",
                    "Population": "full test set", "Delta AUROC [95% CI]": M.fmt_ci(*d),
                    "Delta Brier [95% CI]": M.fmt_ci(*paired("e5_proposed", other, test, boot_all, M.brier)),
                    "Decision": "-"})

    def degradation(yv, p_nat, p_rem):
        return M.auroc(yv, p_nat) - M.auroc(yv, p_rem)

    h2_rows = {}
    for comp in CONVENTIONAL:
        yc = y[test_complete]
        args_c = (yc, models[comp].prob("natural", test_complete), models[comp].prob("mri_removed", test_complete))
        args_p = (yc, models["e5_proposed"].prob("natural", test_complete),
                  models["e5_proposed"].prob("mri_removed", test_complete))
        h2_rows[comp] = boot_cc.paired_diff(degradation, args_c, args_p)    # degradation(comp) - degradation(proposed)
    d_h2 = h2_rows[H2_COMPARATOR]
    hyp.append({"Question": "H2", "Comparison": f"AUROC loss when MRI removed: {MAIN[H2_COMPARATOR]} minus Proposed",
                "Population": "test windows with MRI", "Delta AUROC [95% CI]": M.fmt_ci(*d_h2),
                "Delta Brier [95% CI]": "-", "Decision": "supported" if d_h2[1] > 0 else "not supported"})
    for comp in ("e4_early", "e4_late"):
        hyp.append({"Question": "H2 (supplementary)",
                    "Comparison": f"AUROC loss when MRI removed: {MAIN[comp]} minus Proposed",
                    "Population": "test windows with MRI", "Delta AUROC [95% CI]": M.fmt_ci(*h2_rows[comp]),
                    "Delta Brier [95% CI]": "-", "Decision": "-"})
    pd.DataFrame(hyp).to_csv(TABLES / "table_hypotheses.csv", index=False)
    summary["hypotheses"] = hyp

    # ---- E6 missing-modality stress tests (common subset = test windows with MRI)
    rng = np.random.default_rng(7)
    draws = {f: [rng.random(len(test_complete)) < f for _ in range(50)] for f in ev["missing_fractions"]}
    stress, curve = [], []
    for mid in STRESS_MODELS:
        m = models[mid]
        yc = y[test_complete]
        ref = M.auroc(yc, m.prob("natural", test_complete))
        row = {"Model": MAIN[mid], "All modalities present": ref}
        for cond, name in (("mri_removed", "Clinical only (MRI masked)"), ("clinical_removed", "MRI only (history masked)"),
                           ("mri_noisy_flagged", "Degraded MRI, flagged by QC"),
                           ("mri_noisy_silent", "Degraded MRI, not flagged")):
            p = m.prob(cond, test_complete)
            row[name] = M.auroc(yc, p) if not np.isnan(p).any() else np.nan
        p_nat, p_rem = m.prob("natural", test_complete), m.prob("mri_removed", test_complete)
        for f in ev["missing_fractions"]:
            vals = [M.auroc(yc, np.where(d, p_rem, p_nat)) for d in draws[f]]
            curve.append({"model": mid, "fraction_mri_removed": f, "auroc": float(np.mean(vals)),
                          "sd_over_draws": float(np.std(vals))})
            if f in (0.25, 0.50):
                row[f"Random MRI removal {int(f * 100)}%"] = float(np.mean(vals))
        row["Naturally missing MRI (separate windows)"] = M.auroc(y[test_nomri], m.prob("natural", test_nomri))
        stress.append(row)
    stress_df = pd.DataFrame(stress)
    stress_df.to_csv(TABLES / "table_missing_modality.csv", index=False, float_format="%.3f")
    pd.DataFrame(curve).to_csv(TABLES / "data_degradation_curve.csv", index=False)
    summary["stress_populations"] = {"common_subset_windows": len(test_complete),
                                     "naturally_missing_windows": len(test_nomri),
                                     "naturally_missing_events": int(y[test_nomri].sum())}

    # ---- Ablations and modality-dropout sensitivity
    abl = []
    full = models["e5_proposed"]
    for mid, label in ABLATIONS.items():
        m = models[mid]
        auc = boot_all.ci(M.auroc, y[test], m.prob("natural", test))
        row = {"Variant": label, "AUROC [95% CI]": M.fmt_ci(*auc),
               "Brier": f"{M.brier(y[test], m.prob('natural', test)):.3f}",
               "AUROC, MRI removed": f"{M.auroc(y[test_complete], m.prob('mri_removed', test_complete)):.3f}"}
        if mid != "e5_proposed":
            row["Delta AUROC vs full [95% CI]"] = M.fmt_ci(*paired(mid, "e5_proposed", test, boot_all, M.auroc))
        else:
            row["Delta AUROC vs full [95% CI]"] = "-"
        abl.append(row)
    pd.DataFrame(abl).to_csv(TABLES / "table_ablation.csv", index=False)
    sens = [{"MRI drop probability p_MRI": p, "AUROC (full test)": round(M.auroc(y[test], models[m].prob("natural", test)), 3),
             "AUROC (MRI removed)": round(M.auroc(y[test_complete], models[m].prob("mri_removed", test_complete)), 3),
             "Validation AUROC": round(val_auc[m], 3)} for m, p in SENSITIVITY.items()]
    pd.DataFrame(sens).to_csv(TABLES / "table_sensitivity_moddrop.csv", index=False)

    # ---- E7a calibration
    calib, rel_bins = [], []
    for mid in ["e1_lr_clinmri", "e1_gbm_clinmri", "e2_clinical", "e4_early", "e4_late", "e4_xattn", "e5_proposed"]:
        m = models[mid]
        p_raw, p_cal = m.prob("natural", test, calibrated=False), m.prob("natural", test)
        calib.append({"Model": MAIN[mid], "Temperature T": round(m.T, 3),
                      "ECE before": round(M.ece(y[test], p_raw, bins), 3),
                      "ECE after [95% CI]": M.fmt_ci(*boot_all.ci(lambda a, b: M.ece(a, b, bins), y[test], p_cal)),
                      "Brier before": round(M.brier(y[test], p_raw), 3),
                      "Brier after [95% CI]": M.fmt_ci(*boot_all.ci(M.brier, y[test], p_cal)),
                      "Threshold (from calibration split)": round(m.thr, 3)})
        for stage, p in (("before", p_raw), ("after", p_cal)):
            for b, n, conf, freq in M.reliability_bins(y[test], p, bins):
                rel_bins.append({"model": mid, "stage": stage, "bin": b, "n": n, "mean_predicted": conf,
                                 "observed_rate": freq})
    pd.DataFrame(calib).to_csv(TABLES / "table_calibration.csv", index=False)
    pd.DataFrame(rel_bins).to_csv(TABLES / "data_reliability_bins.csv", index=False)

    # ---- E7b conformal prediction sets
    conf = []
    alpha = ev["conformal_alpha"]
    for mid in ["e1_gbm_clinmri", best_conv, "e5_proposed"]:
        m = models[mid]
        has0, has1, q = M.conformal_sets(m.prob("natural", cal), y[cal], m.prob("natural", test), alpha)
        covered = np.where(y[test] == 1, has1, has0)
        size = has0.astype(int) + has1.astype(int)
        with_mri = a_mri[test] == 1
        conf.append({"Model": MAIN[mid], "Target coverage": 1 - alpha, "q_hat": round(q, 3),
                     "Coverage (all)": round(covered.mean(), 3),
                     "Coverage (with MRI)": round(covered[with_mri].mean(), 3),
                     "Coverage (MRI missing)": round(covered[~with_mri].mean(), 3),
                     "Coverage (converters)": round(covered[y[test] == 1].mean(), 3),
                     "Mean set size": round(size.mean(), 3), "Singleton sets": f"{(size == 1).mean():.1%}",
                     "Both-class sets (uncertain)": f"{(size == 2).mean():.1%}",
                     "Empty sets": f"{(size == 0).mean():.1%}"})
    pd.DataFrame(conf).to_csv(TABLES / "table_conformal.csv", index=False)

    # ---- Gate weights of the proposed model by MRI quality
    alpha_nat = full.alpha["natural"].reindex(test).to_numpy()
    alpha_noisy = full.alpha["mri_noisy_flagged"].reindex(test).to_numpy()
    qc = arrays["qc"][test]
    gate = []
    for q_val, q_name in ((1, "good"), (2, "acceptable"), (3, "poor")):
        sel = qc == q_val
        gate.append({"MRI quality": q_name, "Windows": int(sel.sum()),
                     "Mean MRI weight alpha_MRI": round(float(np.nanmean(alpha_nat[sel])), 3),
                     "SD": round(float(np.nanstd(alpha_nat[sel])), 3)})
    gate.append({"MRI quality": "degraded + flagged poor (stress test)", "Windows": int((qc > 0).sum()),
                 "Mean MRI weight alpha_MRI": round(float(np.nanmean(alpha_noisy[qc > 0])), 3),
                 "SD": round(float(np.nanstd(alpha_noisy[qc > 0])), 3)})
    gate.append({"MRI quality": "MRI missing", "Windows": int((qc == 0).sum()),
                 "Mean MRI weight alpha_MRI": round(float(np.nanmean(alpha_nat[qc == 0])), 3), "SD": 0.0})
    pd.DataFrame(gate).to_csv(TABLES / "table_gate_weights.csv", index=False)
    pd.DataFrame({"window_id": test, "qc": qc, "alpha_mri": alpha_nat}).to_csv(
        TABLES / "data_gate_weights.csv", index=False)

    # ---- Subgroups (RQ5)
    age = arrays["static"][:, 0]
    meta = json.load(open(PROCESSED / "feature_meta.json"))
    age_years = age * meta["scalers"]["age_std"] + meta["scalers"]["age_mean"]
    tab = pd.read_csv(PROCESSED / "tabular.csv")
    dx_bl = cohort.merge(_dx_bl(cfg), on="RID", how="left").DX_bl.to_numpy()
    groups = {
        "Sex": np.where(tab["Female"].to_numpy() == 1, "Female", "Male"),
        "Age band": np.select([age_years < 70, age_years < 80], ["<70", "70-79"], ">=80"),
        "Baseline diagnosis": np.where(np.isin(dx_bl, ["CN", "SMC"]), "CN/SMC at entry", dx_bl),
    }
    sub = []
    for gname, gval in groups.items():
        for level in sorted(np.unique(gval[test])):
            idx = test[gval[test] == level]
            n_ev = int(y[idx].sum())
            row = {"Subgroup": gname, "Level": level, "Participants": int(len(np.unique(rid[idx]))),
                   "Windows": len(idx), "Converters": n_ev}
            for mid in ("e5_proposed", "e1_gbm_clinmri"):
                if n_ev < 5 or n_ev > len(idx) - 5:
                    row[f"AUROC {MAIN[mid]}"] = "suppressed (<5 in a class)"
                else:
                    b = M.ClusterBootstrap(rid[idx], n_boot, seed=3)
                    row[f"AUROC {MAIN[mid]}"] = M.fmt_ci(*b.ci(M.auroc, y[idx], models[mid].prob("natural", idx)))
            sub.append(row)
    pd.DataFrame(sub).to_csv(TABLES / "table_subgroups.csv", index=False)

    # ---- Error analysis of the proposed model at its fixed threshold
    p = full.prob("natural", test)
    pred = p >= full.thr
    lengths = arrays["lengths"][test]
    err = []
    for kind, sel in (("False positive", pred & (y[test] == 0)), ("False negative", ~pred & (y[test] == 1))):
        cats = np.select([lengths[sel] <= 2, a_mri[test][sel] == 0, qc[sel] == 3, np.abs(p[sel] - full.thr) < 0.1],
                         ["Sparse history (<= 2 visits)", "MRI missing", "Poor-quality MRI", "Close to threshold (+/-0.10)"],
                         "Other (unusual trajectory)")
        for c in ["Sparse history (<= 2 visits)", "MRI missing", "Poor-quality MRI", "Close to threshold (+/-0.10)",
                  "Other (unusual trajectory)"]:
            err.append({"Error type": kind, "Primary category": c, "Count": int((cats == c).sum())})
    pd.DataFrame(err).to_csv(TABLES / "table_error_analysis.csv", index=False)

    # ---- E7c explanations and their stability
    summary["stability"] = explanation_analysis(cfg, arrays, cohort, test, models)

    # ---- Computation
    recs = [json.loads(line) for line in open(RUNS / "run_records.jsonl")]
    comp = pd.DataFrame(recs).groupby("model").agg(parameters=("parameters", "mean"),
                                                    seconds_per_run=("train_seconds", "mean"),
                                                    runs=("run_id", "size")).reset_index()
    comp.to_csv(TABLES / "table_compute.csv", index=False, float_format="%.1f")

    # ---- Saved calibrated test predictions (used by figures and dashboard)
    out = []
    for mid in MAIN:
        idx = test_complete if mid == "e3_mri" else test
        for cond in ("natural", "mri_removed"):
            if cond == "mri_removed" and mid in ("e3_mri", "e1_lr_clin", "e1_gbm_clin"):
                continue
            out.append(pd.DataFrame({"window_id": idx, "model": mid, "condition": cond, "label": y[idx],
                                     "p_raw": models[mid].prob(cond, idx, calibrated=False),
                                     "p_calibrated": models[mid].prob(cond, idx)}))
    pd.concat(out).to_csv(PREDICTIONS / "test_calibrated.csv.gz", index=False)
    summary["temperatures"] = {k: round(v.T, 3) for k, v in models.items()}
    summary["thresholds"] = {k: round(v.thr, 3) for k, v in models.items()}
    with open(RESULTS / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("strongest_conventional_fusion", "strongest_single_modality",
                                                "strongest_fusion", "test_auroc")}, indent=1))
    for h in hyp:
        print(f"{h['Question']:<20} {h['Comparison'][:70]:<70} {h['Delta AUROC [95% CI]']}  {h['Decision']}")


def _dx_bl(cfg):
    from mdp.config import data_csv
    v = pd.read_csv(data_csv(cfg), usecols=["RID", "DX_bl"])
    return v.groupby("RID").DX_bl.first().reset_index()


def explanation_analysis(cfg, arrays, cohort, test, models):
    ev = cfg["evaluation"]
    k = ev["stability_top_k"]
    data = to_tensors(arrays)
    kw = json.load(open(MODELS / "e5_proposed" / "model_kwargs.json"))
    seed_vecs, temporal = [], []
    for s in cfg["training"]["seeds"]:
        model = FusionModel(**kw)
        model.load_state_dict(torch.load(MODELS / "e5_proposed" / f"seed{s}.pt"))
        attr = integrated_gradients(model, data, test)
        seed_vecs.append(feature_vector(attr))
        temporal += [(s, *row) for row in temporal_profile(attr, arrays["tti"][test], arrays["lengths"][test])]
        if s == cfg["training"]["seeds"][0]:
            base_model, base_vec = model, seed_vecs[-1]

    rho_seed, jac_seed = [], []
    n_seeds = len(seed_vecs)
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            r, jk = stability(seed_vecs[i], seed_vecs[j], k)
            rho_seed.append(r)
            jac_seed.append(jk)
    rho_seed, jac_seed = np.concatenate(rho_seed), np.concatenate(jac_seed)

    g = torch.Generator().manual_seed(99)
    rho_pert, jac_pert = [], []
    for _ in range(5):
        pert = {kk: v.clone() for kk, v in data.items()}
        valid = (torch.arange(pert["x"].shape[1])[None, :] < pert["lengths"][:, None]).float()[..., None]
        pert["x"] += ev["perturbation_sd"] * torch.randn(pert["x"].shape, generator=g) * valid
        pert["static"][:, [0, 2]] += ev["perturbation_sd"] * torch.randn((len(pert["static"]), 2), generator=g)
        pert["roi"] += ev["perturbation_sd"] * torch.randn(pert["roi"].shape, generator=g) * pert["a_mri"][:, None]
        r, jk = stability(base_vec, feature_vector(integrated_gradients(base_model, pert, test)), k)
        rho_pert.append(r)
        jac_pert.append(jk)
    rho_pert, jac_pert = np.concatenate(rho_pert), np.concatenate(jac_pert)

    # SHAP for gradient boosting (clinical + MRI)
    with open(MODELS / "e1_gbm_clinmri.pkl", "rb") as f:
        gb = pickle.load(f)
    tab = pd.read_csv(PROCESSED / "tabular.csv")
    X = tab[gb["columns"]].to_numpy(float)
    explainer = shap.TreeExplainer(gb["model"])
    sv = np.asarray(explainer.shap_values(X[test]))
    sd = np.nanstd(X[cohort.split.to_numpy() == "train"], axis=0)
    binary = np.array([c == "Female" for c in gb["columns"]])
    rng = np.random.default_rng(99)
    rho_shap, jac_shap = [], []
    for _ in range(5):
        Xp = X[test] + ev["perturbation_sd"] * rng.standard_normal(X[test].shape) * sd * ~binary
        r, jk = stability(sv, np.asarray(explainer.shap_values(Xp)), k)
        rho_shap.append(r)
        jac_shap.append(jk)
    rho_shap, jac_shap = np.concatenate(rho_shap), np.concatenate(jac_shap)

    def q(a):
        return f"{np.median(a):.3f} [{np.percentile(a, 25):.3f}, {np.percentile(a, 75):.3f}]"

    rows = [
        {"Model / explainer": "Proposed / Integrated Gradients", "Perturbation": "5 training seeds (10 pairs)",
         "Spearman rho, median [IQR]": q(rho_seed), f"Top-{k} overlap J_k, median [IQR]": q(jac_seed),
         "Comparisons": len(rho_seed)},
        {"Model / explainer": "Proposed / Integrated Gradients", "Perturbation": f"input noise {ev['perturbation_sd']} SD (x5)",
         "Spearman rho, median [IQR]": q(rho_pert), f"Top-{k} overlap J_k, median [IQR]": q(jac_pert),
         "Comparisons": len(rho_pert)},
        {"Model / explainer": "Gradient boosting (clinical + MRI) / SHAP",
         "Perturbation": f"input noise {ev['perturbation_sd']} SD (x5)",
         "Spearman rho, median [IQR]": q(rho_shap), f"Top-{k} overlap J_k, median [IQR]": q(jac_shap),
         "Comparisons": len(rho_shap)},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "table_stability.csv", index=False)
    np.savez(TABLES / "data_stability.npz", rho_seed=rho_seed, jac_seed=jac_seed, rho_pert=rho_pert,
             jac_pert=jac_pert, rho_shap=rho_shap, jac_shap=jac_shap)

    mean_abs = np.mean([np.abs(v).mean(axis=0) for v in seed_vecs], axis=0)
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "proposed_mean_abs_IG": mean_abs})
    imp.to_csv(TABLES / "data_importance_proposed.csv", index=False)
    pd.DataFrame({"feature": gb["columns"], "gbm_mean_abs_SHAP": np.abs(sv).mean(axis=0)}).to_csv(
        TABLES / "data_importance_gbm.csv", index=False)
    pd.DataFrame(temporal, columns=["seed", "period", "steps", "mean_abs_attribution"]).to_csv(
        TABLES / "data_temporal_attribution.csv", index=False)
    np.save(MODELS / "e5_proposed" / "test_attributions_seed0.npy", base_vec)
    stable = bool(np.median(rho_seed) >= ev["stability_min_spearman"])
    with open(MODELS / "e5_proposed" / "stability.json", "w") as f:
        json.dump({"median_seed_spearman": float(np.median(rho_seed)), "threshold": ev["stability_min_spearman"],
                   "stable": stable}, f)
    return {"seed_spearman_median": float(np.median(rho_seed)), "seed_jaccard_median": float(np.median(jac_seed)),
            "perturbation_spearman_median": float(np.median(rho_pert)), "shap_spearman_median": float(np.median(rho_shap)),
            "meets_threshold": stable}


if __name__ == "__main__":
    main()
