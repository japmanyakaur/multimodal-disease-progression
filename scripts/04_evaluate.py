"""Step 4: evaluate all models from saved predictions, following configs/preregistration_stage2.yaml.

Evaluation sets
  dev_test   - development locked test split (opened in Stage 1, so Stage 2 results there are exploratory)
  external_A - new independent cohort, same population              (confirmatory, primary)
  external_B - new independent cohort, degraded-imaging site        (confirmatory, secondary)
Temperature, decision threshold and conformal quantiles always come from the development calibration split.
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
from mdp.config import MODELS, PREDICTIONS, PROCESSED, RESULTS, RUNS, TABLES, data_csv, load_config  # noqa: E402
from mdp.explain import FEATURE_NAMES, feature_vector, integrated_gradients, stability, temporal_profile  # noqa: E402
from mdp.models import FusionModel  # noqa: E402
from mdp.train import to_tensors  # noqa: E402

LABEL = {
    "e1_lr_clin": "E1 Logistic regression (clinical)", "e1_gbm_clin": "E1 Gradient boosting (clinical)",
    "e1_lr_clinmri": "E1 Logistic regression (clinical + MRI)", "e1_gbm_clinmri": "E1 Gradient boosting (clinical + MRI)",
    "e2_clinical": "E2 GRU, clinical history only", "e3_mri": "E3 MRI measures only", "e3_cnn": "E3 3D CNN, MRI only",
    "e4_early": "E4 Early fusion", "e4_late": "E4 Late fusion", "e4_xattn": "E4 Cross-attention fusion",
    "e4_early_cnn": "E4 Early fusion (3D CNN)", "e5_proposed": "E5 Proposed reliability-aware fusion",
    "e5_proposed_cnn": "E5 Proposed (3D CNN)",
}
ABLATIONS = {"e5_proposed": "Full proposed model", "abl_no_gate": "- reliability gate",
             "abl_no_xattn": "- cross-attention", "abl_no_time": "- explicit time gaps",
             "abl_no_mask": "- missingness masks", "abl_no_consistency": "- consistency loss",
             "abl_no_moddrop": "- modality dropout (and consistency loss)"}
SENSITIVITY = {"sens_pdrop_015": 0.15, "e5_proposed": 0.30, "sens_pdrop_050": 0.50}
REG = {"persistence": "Persistence (last CDR-SB carried forward)", "e1_ridge_reg": "Ridge regression",
       "e1_gbm_reg": "Gradient boosting", "e2_clinical_reg": "E2 GRU, clinical history only",
       "e5_proposed_reg": "E5 Proposed reliability-aware fusion"}
CONVENTIONAL = ["e4_early", "e4_late", "e4_xattn"]
SINGLE = ["e1_lr_clin", "e1_gbm_clin", "e2_clinical"]
FUSION = ["e1_lr_clinmri", "e1_gbm_clinmri", "e4_early", "e4_late", "e4_xattn", "e5_proposed"]
MRI_ONLY = ("e3_mri", "e3_cnn")
H2_COMPARATOR = "e4_xattn"
STRESS = ["e2_clinical", "e1_gbm_clinmri", "e4_early", "e4_late", "e4_xattn", "e5_proposed", "e4_early_cnn",
          "e5_proposed_cnn"]
SETS = {"dev_test": "Development test (exploratory)", "external_A": "External A", "external_B": "External B"}


class Model:
    """Seed-ensemble (mean logit) with temperature and threshold fitted on the development calibration split."""

    def __init__(self, mid, df, cal_idx, y_cal):
        self.id = mid
        self.seeds = df
        g = df.groupby(["cohort", "condition", "window_id"])
        self.logit = g.logit.mean()
        self.alpha = g.alpha_mri.mean()
        z = self.raw("dev", "natural", cal_idx)
        ok = ~np.isnan(z)
        self.T = M.fit_temperature(z[ok], y_cal[ok])
        self.thr = M.youden_threshold(y_cal[ok], M.sigmoid(z[ok] / self.T))

    def raw(self, cohort, condition, idx):
        try:
            s = self.logit.loc[(cohort, condition)]
        except KeyError:
            return np.full(len(idx), np.nan)
        return s.reindex(idx).to_numpy()

    def prob(self, cohort, condition, idx, calibrated=True):
        return M.sigmoid(self.raw(cohort, condition, idx) / (self.T if calibrated else 1.0))

    def alpha_mri(self, cohort, condition, idx):
        return self.alpha.loc[(cohort, condition)].reindex(idx).to_numpy()

    def seed_aurocs(self, cohort, y, idx):
        d = self.seeds[(self.seeds.cohort == cohort) & (self.seeds.condition == "natural")]
        return np.array([M.auroc(y, g.set_index("window_id").logit.reindex(idx).to_numpy())
                         for _, g in d.groupby("seed")])


def late_fusion():
    clin = pd.read_csv(PREDICTIONS / "e2_clinical.csv.gz")
    mri = pd.read_csv(PREDICTIONS / "e3_mri.csv.gz")
    m = clin.merge(mri, on=["cohort", "window_id", "seed", "condition"], suffixes=("_c", "_m"))
    pc, pm = M.sigmoid(m.logit_c), M.sigmoid(m.logit_m)
    has_mri = m.logit_m.notna() & (m.condition != "mri_removed")
    p = np.where(m.condition == "clinical_removed", pm, np.where(has_mri, (pc + pm) / 2, pc))
    out = m[["cohort", "window_id", "seed", "condition"]].copy()
    out["logit"], out["alpha_mri"] = M.logit(p), np.nan
    return out


class EvalSet:
    def __init__(self, key, cohort_name, cohort, arrays, idx, n_boot):
        self.key, self.cohort, self.idx = key, cohort_name, idx
        self.y = cohort.label.to_numpy()
        self.rid = cohort.RID.to_numpy()
        a = arrays["a_mri"]
        self.idx_mri, self.idx_nomri = idx[a[idx] == 1], idx[a[idx] == 0]
        self.boot = M.ClusterBootstrap(self.rid[idx], n_boot, seed=1)
        self.boot_mri = M.ClusterBootstrap(self.rid[self.idx_mri], n_boot, seed=2)
        self.qc = arrays["qc"]
        self.lengths = arrays["lengths"]
        self.a_mri = a
        self.frame = cohort

    def pick(self, mid):
        return (self.idx_mri, self.boot_mri) if mid in MRI_ONLY else (self.idx, self.boot)


def metric_cells(model, es, bins, full=True):
    idx, boot = es.pick(model.id)
    p, y = model.prob(es.cohort, "natural", idx), es.y[idx]
    cells = {"Windows": len(idx), "AUROC [95% CI]": M.fmt_ci(*boot.ci(M.auroc, y, p))}
    if full:
        t = M.threshold_metrics(y, p, model.thr)
        seeds = model.seed_aurocs(es.cohort, y, idx)
        cells.update({"AUPRC [95% CI]": M.fmt_ci(*boot.ci(M.auprc, y, p)),
                      "Brier [95% CI]": M.fmt_ci(*boot.ci(M.brier, y, p)),
                      "ECE [95% CI]": M.fmt_ci(*boot.ci(lambda a, b: M.ece(a, b, bins), y, p)),
                      "Sensitivity": f"{t['sensitivity']:.3f}", "Specificity": f"{t['specificity']:.3f}",
                      "F1": f"{t['f1']:.3f}",
                      "AUROC per seed, mean (SD)": f"{seeds.mean():.3f} ({seeds.std():.3f})" if len(seeds) > 1 else "-"})
    return cells


def main():
    cfg = load_config()
    ev = cfg["evaluation"]
    bins, n_boot, alpha = ev["ece_bins"], ev["bootstrap_replicates"], ev["conformal_alpha"]
    cohorts, arrays = {"dev": pd.read_csv(PROCESSED / "cohort.csv")}, {"dev": dict(np.load(PROCESSED / "features.npz"))}
    for name in ("external_A", "external_B"):
        cohorts[name] = pd.read_csv(PROCESSED / f"cohort_{name}.csv")
        arrays[name] = dict(np.load(PROCESSED / f"features_{name}.npz"))
    dev = cohorts["dev"]
    assert all((c.window_id.to_numpy() == np.arange(len(c))).all() for c in cohorts.values())
    split = dev.split.to_numpy()
    cal, val, test = (np.where(split == s)[0] for s in ("calibration", "validation", "test"))
    y_dev = dev.label.to_numpy()
    a_dev = arrays["dev"]["a_mri"]

    sets = {"dev_test": EvalSet("dev_test", "dev", dev, arrays["dev"], test, n_boot)}
    for name in ("external_A", "external_B"):
        sets[name] = EvalSet(name, name, cohorts[name], arrays[name], np.arange(len(cohorts[name])), n_boot)

    ids = [m for m in LABEL if m != "e4_late"] + [m for m in list(ABLATIONS) + list(SENSITIVITY) if m not in LABEL]
    models = {}
    for mid in ids:
        df = pd.read_csv(PREDICTIONS / f"{mid}.csv.gz")
        c = cal[a_dev[cal] == 1] if mid in MRI_ONLY else cal
        models[mid] = Model(mid, df, c, y_dev[c])
    models["e4_late"] = Model("e4_late", late_fusion(), cal, y_dev[cal])

    val_auc = {m: M.auroc(y_dev[val], models[m].prob("dev", "natural", val)) for m in models if m not in MRI_ONLY}
    best_conv = max(CONVENTIONAL, key=val_auc.get)
    best_single = max(SINGLE, key=val_auc.get)
    best_fusion = max(FUSION, key=val_auc.get)
    summary = {"strongest_conventional_fusion": best_conv, "strongest_single_modality": best_single,
               "strongest_fusion": best_fusion, "validation_auroc": {k: round(v, 4) for k, v in val_auc.items()},
               "temperatures": {k: round(v.T, 3) for k, v in models.items()},
               "thresholds": {k: round(v.thr, 3) for k, v in models.items()},
               "sets": {k: {"windows": len(s.idx), "participants": int(len(np.unique(s.rid[s.idx]))),
                            "events": int(s.y[s.idx].sum()), "with_mri": len(s.idx_mri)} for k, s in sets.items()}}

    # ---- discrimination of every model on every evaluation set
    auroc_rows, test_auc = [], {}
    for mid, label in LABEL.items():
        row = {"Model": label + (" *" if mid in MRI_ONLY else "")}
        for key, es in sets.items():
            idx, boot = es.pick(mid)
            p = models[mid].prob(es.cohort, "natural", idx)
            row[SETS[key]] = M.fmt_ci(*boot.ci(M.auroc, es.y[idx], p))
            test_auc.setdefault(key, {})[mid] = round(M.auroc(es.y[idx], p), 4)
        auroc_rows.append(row)
    pd.DataFrame(auroc_rows).to_csv(TABLES / "table_auroc_all_sets.csv", index=False)
    summary["auroc"] = test_auc
    for key in ("external_A", "external_B", "dev_test"):
        rows = [{"Model": LABEL[m] + (" *" if m in MRI_ONLY else ""), **metric_cells(models[m], sets[key], bins)}
                for m in LABEL]
        pd.DataFrame(rows).to_csv(TABLES / f"table_metrics_{key}.csv", index=False)

    # ---- pre-registered hypothesis tests
    def paired(a, b, es, fn, idx=None, boot=None, condition="natural"):
        idx = es.idx if idx is None else idx
        boot = es.boot if boot is None else boot
        return boot.paired_diff(fn, (es.y[idx], models[a].prob(es.cohort, condition, idx)),
                                (es.y[idx], models[b].prob(es.cohort, condition, idx)))

    def degradation(yv, p_nat, p_rem):
        return M.auroc(yv, p_nat) - M.auroc(yv, p_rem)

    def deg_contrast(comp, es):
        args = [(es.y[es.idx_mri], models[m].prob(es.cohort, "natural", es.idx_mri),
                 models[m].prob(es.cohort, "mri_removed", es.idx_mri)) for m in (comp, "e5_proposed")]
        return es.boot_mri.paired_diff(degradation, *args)

    hyp = []
    for key in ("external_A", "external_B", "dev_test"):
        es = sets[key]
        role = {"external_A": "confirmatory (primary)", "external_B": "confirmatory (secondary)",
                "dev_test": "exploratory"}[key]
        d_auc, d_bri = paired("e5_proposed", best_conv, es, M.auroc), paired("e5_proposed", best_conv, es, M.brier)
        ok = d_auc[1] > 0 and d_auc[0] >= ev["min_meaningful_difference"] and d_bri[0] <= ev["brier_noninferiority_margin"]
        hyp.append({"Set": SETS[key], "Role": role, "Question": "H1" if key == "external_A" else "H1 (replication)",
                    "Comparison": f"Proposed - {LABEL[best_conv]}", "Delta [95% CI]": M.fmt_ci(*d_auc),
                    "Delta Brier [95% CI]": M.fmt_ci(*d_bri), "Decision": "supported" if ok else "not supported"})
        d = paired(best_fusion, best_single, es, M.auroc)
        hyp.append({"Set": SETS[key], "Role": role, "Question": "RQ1",
                    "Comparison": f"{LABEL[best_fusion]} - {LABEL[best_single]}", "Delta [95% CI]": M.fmt_ci(*d),
                    "Delta Brier [95% CI]": M.fmt_ci(*paired(best_fusion, best_single, es, M.brier)),
                    "Decision": "fusion better" if d[1] > 0 and d[0] >= ev["min_meaningful_difference"]
                    else "no meaningful improvement"})
        for comp in [H2_COMPARATOR, "e4_early", "e4_late"]:
            d = deg_contrast(comp, es)
            main = comp == H2_COMPARATOR
            hyp.append({"Set": SETS[key], "Role": role if main else "supplementary",
                        "Question": "H2" if main else "H2 (other comparator)",
                        "Comparison": f"AUROC loss when MRI removed: {LABEL[comp]} minus Proposed",
                        "Delta [95% CI]": M.fmt_ci(*d), "Delta Brier [95% CI]": "-",
                        "Decision": ("supported" if d[1] > 0 else "not supported") if main else
                        ("proposed degrades less" if d[1] > 0 else ("proposed degrades more" if d[2] < 0 else "no difference"))})
    for comp, key in [(c, k) for k in ("external_A", "external_B") for c in ("e1_gbm_clinmri", "e2_clinical")]:
        es = sets[key]
        hyp.append({"Set": SETS[key], "Role": "supplementary (not pre-registered)", "Question": "Supplementary",
                    "Comparison": f"Proposed - {LABEL[comp]}",
                    "Delta [95% CI]": M.fmt_ci(*paired("e5_proposed", comp, es, M.auroc)),
                    "Delta Brier [95% CI]": M.fmt_ci(*paired("e5_proposed", comp, es, M.brier)), "Decision": "-"})
    for a, b in (("e5_proposed_cnn", "e4_early_cnn"), ("e5_proposed_cnn", "e5_proposed")):
        for key in ("external_A", "external_B"):
            es = sets[key]
            hyp.append({"Set": SETS[key], "Role": "secondary (3D CNN)", "Question": "3D CNN",
                        "Comparison": f"{LABEL[a]} - {LABEL[b]}",
                        "Delta [95% CI]": M.fmt_ci(*paired(a, b, es, M.auroc)),
                        "Delta Brier [95% CI]": M.fmt_ci(*paired(a, b, es, M.brier)), "Decision": "-"})
    pd.DataFrame(hyp).to_csv(TABLES / "table_hypotheses.csv", index=False)
    summary["hypotheses"] = hyp

    # ---- E6 missing-modality stress tests on each set (common subset = windows with MRI)
    curve = []
    for key, es in sets.items():
        rng = np.random.default_rng(7)
        draws = {f: [rng.random(len(es.idx_mri)) < f for _ in range(50)] for f in ev["missing_fractions"]}
        rows = []
        for mid in STRESS:
            m, yc = models[mid], es.y[es.idx_mri]
            row = {"Model": LABEL[mid], "All modalities present": M.auroc(yc, m.prob(es.cohort, "natural", es.idx_mri))}
            for cond, name in (("mri_removed", "Clinical only (MRI masked)"),
                               ("clinical_removed", "MRI only (history masked)"),
                               ("mri_noisy_flagged", "Degraded MRI, flagged by QC"),
                               ("mri_noisy_silent", "Degraded MRI, not flagged")):
                p = m.prob(es.cohort, cond, es.idx_mri)
                row[name] = M.auroc(yc, p) if not np.isnan(p).any() else np.nan
            p_nat, p_rem = m.prob(es.cohort, "natural", es.idx_mri), m.prob(es.cohort, "mri_removed", es.idx_mri)
            for f in ev["missing_fractions"]:
                vals = [M.auroc(yc, np.where(d, p_rem, p_nat)) for d in draws[f]]
                curve.append({"set": key, "model": mid, "fraction_mri_removed": f, "auroc": float(np.mean(vals))})
                if f in (0.25, 0.50):
                    row[f"Random MRI removal {int(f * 100)}%"] = float(np.mean(vals))
            row["Naturally missing MRI (separate windows)"] = M.auroc(es.y[es.idx_nomri],
                                                                      m.prob(es.cohort, "natural", es.idx_nomri))
            rows.append(row)
        pd.DataFrame(rows).to_csv(TABLES / f"table_missing_modality_{key}.csv", index=False, float_format="%.3f")
    pd.DataFrame(curve).to_csv(TABLES / "data_degradation_curve.csv", index=False)

    # ---- ablations and sensitivity (external A, with external B for context)
    es_a, es_b = sets["external_A"], sets["external_B"]
    abl = []
    for mid, label in ABLATIONS.items():
        if mid not in models:
            continue
        m = models[mid]
        row = {"Variant": label,
               "AUROC, external A [95% CI]": M.fmt_ci(*es_a.boot.ci(M.auroc, es_a.y, m.prob("external_A", "natural", es_a.idx))),
               "Delta vs full, external A [95% CI]": "-" if mid == "e5_proposed" else
               M.fmt_ci(*paired(mid, "e5_proposed", es_a, M.auroc), 4),
               "AUROC MRI removed, external A": f"{M.auroc(es_a.y[es_a.idx_mri], m.prob('external_A', 'mri_removed', es_a.idx_mri)):.3f}",
               "AUROC, external B": f"{M.auroc(es_b.y, m.prob('external_B', 'natural', es_b.idx)):.3f}",
               "Delta vs full, external B [95% CI]": "-" if mid == "e5_proposed" else
               M.fmt_ci(*paired(mid, "e5_proposed", es_b, M.auroc), 4)}
        abl.append(row)
    pd.DataFrame(abl).to_csv(TABLES / "table_ablation.csv", index=False)
    sens = [{"MRI drop probability p_MRI": p,
             "AUROC external A": round(M.auroc(es_a.y, models[m].prob("external_A", "natural", es_a.idx)), 3),
             "AUROC external A, MRI removed": round(M.auroc(es_a.y[es_a.idx_mri],
                                                            models[m].prob("external_A", "mri_removed", es_a.idx_mri)), 3),
             "AUROC external B": round(M.auroc(es_b.y, models[m].prob("external_B", "natural", es_b.idx)), 3),
             "Validation AUROC": round(val_auc[m], 3)} for m, p in SENSITIVITY.items()]
    pd.DataFrame(sens).to_csv(TABLES / "table_sensitivity_moddrop.csv", index=False)

    # ---- E7a calibration (temperature fitted on development calibration split, applied unchanged)
    calib, rel_bins = [], []
    for mid in ["e1_lr_clinmri", "e1_gbm_clinmri", "e2_clinical", "e4_early", "e4_late", "e4_xattn", "e5_proposed",
                "e5_proposed_cnn"]:
        m = models[mid]
        row = {"Model": LABEL[mid], "Temperature T": round(m.T, 3)}
        for key in ("external_A", "external_B"):
            es = sets[key]
            p_raw, p_cal = m.prob(es.cohort, "natural", es.idx, False), m.prob(es.cohort, "natural", es.idx)
            tag = "A" if key == "external_A" else "B"
            row[f"ECE before ({tag})"] = round(M.ece(es.y[es.idx], p_raw, bins), 3)
            row[f"ECE after ({tag}) [95% CI]"] = M.fmt_ci(*es.boot.ci(lambda a, b: M.ece(a, b, bins), es.y[es.idx], p_cal))
            row[f"Brier after ({tag})"] = round(M.brier(es.y[es.idx], p_cal), 3)
            for stage, p in (("before", p_raw), ("after", p_cal)):
                for b, n, conf, freq in M.reliability_bins(es.y[es.idx], p, bins):
                    rel_bins.append({"set": key, "model": mid, "stage": stage, "bin": b, "n": n,
                                     "mean_predicted": conf, "observed_rate": freq})
        calib.append(row)
    pd.DataFrame(calib).to_csv(TABLES / "table_calibration.csv", index=False)
    pd.DataFrame(rel_bins).to_csv(TABLES / "data_reliability_bins.csv", index=False)

    # ---- E7b conformal prediction sets: standard vs class-conditional (Mondrian)
    conf, qhat = [], {}
    for mid in ["e1_gbm_clinmri", best_conv, "e5_proposed"]:
        m = models[mid]
        p_cal = m.prob("dev", "natural", cal)
        for key, es in sets.items():
            p = m.prob(es.cohort, "natural", es.idx)
            y = es.y[es.idx]
            for method in ("Standard", "Mondrian"):
                if method == "Standard":
                    has0, has1, q = M.conformal_sets(p_cal, y_dev[cal], p, alpha)
                    qhat.setdefault(mid, {})["standard"] = q
                else:
                    has0, has1, qs = M.mondrian_sets(p_cal, y_dev[cal], p, alpha)
                    qhat.setdefault(mid, {})["mondrian"] = qs
                covered = np.where(y == 1, has1, has0)
                size = has0.astype(int) + has1.astype(int)
                with_mri = es.a_mri[es.idx] == 1
                conf.append({"Model": LABEL[mid], "Set": SETS[key], "Method": method,
                             "Coverage (all)": round(covered.mean(), 3),
                             "Coverage (converters)": round(covered[y == 1].mean(), 3),
                             "Coverage (non-converters)": round(covered[y == 0].mean(), 3),
                             "Coverage (MRI missing)": round(covered[~with_mri].mean(), 3),
                             "Mean set size": round(size.mean(), 3),
                             "Both-class sets": f"{(size == 2).mean():.1%}", "Empty sets": f"{(size == 0).mean():.1%}"})
    pd.DataFrame(conf).to_csv(TABLES / "table_conformal.csv", index=False)
    summary["conformal_qhat"] = qhat

    # ---- secondary regression: CDR-SB at 24 months, with split-conformal intervals
    reg_rows = []
    reg_preds = {}
    for mid in REG:
        if mid == "persistence":                          # last recorded CDR-SB up to the index visit
            reg_preds[mid] = {c: pd.Series(pd.read_csv(PROCESSED / ("tabular.csv" if c == "dev" else f"tabular_{c}.csv"))
                                           ["CDR-SB (last)"].to_numpy(), index=cohorts[c].window_id) for c in cohorts}
        else:
            df = pd.read_csv(PREDICTIONS / f"{mid}.csv.gz")
            reg_preds[mid] = {c: g.groupby("window_id").pred.mean() for c, g in df.groupby("cohort")}
    def usable(cohort, idx, target):                    # windows with a target and a prediction from every model
        ok = ~np.isnan(target[idx])
        for m in REG:
            ok &= ~np.isnan(reg_preds[m][cohort].reindex(idx).to_numpy())
        return idx[ok]

    t_dev = dev.cdrsb_target.to_numpy()
    cal_r = usable("dev", cal, t_dev)
    for mid, label in REG.items():
        q = M.regression_interval(t_dev[cal_r] - reg_preds[mid]["dev"].reindex(cal_r).to_numpy(), alpha)
        row = {"Model": label, "Interval half-width q (CDR-SB points)": round(q, 2)}
        for key, es in sets.items():
            tgt = es.frame.cdrsb_target.to_numpy()
            idx = usable(es.cohort, es.idx, tgt)
            pred = reg_preds[mid][es.cohort].reindex(idx).to_numpy()
            b = M.ClusterBootstrap(es.rid[idx], n_boot, seed=4)
            tag = {"dev_test": "dev", "external_A": "A", "external_B": "B"}[key]
            row[f"MAE {tag} [95% CI]"] = M.fmt_ci(*b.ci(M.mae, tgt[idx], pred), 2)
            row[f"RMSE {tag}"] = round(M.rmse(tgt[idx], pred), 2)
            row[f"R2 {tag}"] = round(M.r2(tgt[idx], pred), 3)
            row[f"90% interval coverage {tag}"] = round(float(np.mean(np.abs(tgt[idx] - pred) <= q)), 3)
        reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(TABLES / "table_regression.csv", index=False)
    for key, es in sets.items():
        tgt = es.frame.cdrsb_target.to_numpy()
        idx = usable(es.cohort, es.idx, tgt)
        summary.setdefault("regression_n", {})[key] = len(idx)
        if key == "external_A":
            pd.DataFrame({"observed": tgt[idx], **{m: reg_preds[m][es.cohort].reindex(idx).to_numpy() for m in REG}}).to_csv(
                TABLES / "data_regression_external_A.csv", index=False)

    # ---- supplementary time-to-event concordance (Harrell C-index, 200 participant bootstrap replicates)
    cidx = []
    for mid in ["e1_gbm_clinmri", "e2_clinical", "e4_early", "e5_proposed"]:
        row = {"Model": LABEL[mid]}
        for key, es in sets.items():
            t, e = es.frame.tte_months.to_numpy()[es.idx], es.frame.tte_event.to_numpy()[es.idx]
            r = models[mid].raw(es.cohort, "natural", es.idx)
            b = M.ClusterBootstrap(es.rid[es.idx], 200, seed=5)
            row[SETS[key]] = M.fmt_ci(*b.ci(M.c_index, t, e, r))
        cidx.append(row)
    pd.DataFrame(cidx).to_csv(TABLES / "table_cindex.csv", index=False)

    # ---- reliability gate: weight given to MRI by scan quality
    gate, gate_data = [], []
    for key, es in sets.items():
        a = models["e5_proposed"].alpha_mri(es.cohort, "natural", es.idx)
        qc = es.qc[es.idx]
        for q_val, q_name in ((1, "good"), (2, "acceptable"), (3, "poor")):
            sel = qc == q_val
            gate.append({"Set": SETS[key], "MRI quality": q_name, "Windows": int(sel.sum()),
                         "Mean alpha_MRI": round(float(np.nanmean(a[sel])), 3), "SD": round(float(np.nanstd(a[sel])), 3)})
        gate_data.append(pd.DataFrame({"set": key, "qc": qc, "alpha_mri": a}))
        flagged = models["e5_proposed"].alpha_mri(es.cohort, "mri_noisy_flagged", es.idx_mri)
        gate.append({"Set": SETS[key], "MRI quality": "degraded + flagged poor (stress test)", "Windows": len(es.idx_mri),
                     "Mean alpha_MRI": round(float(np.nanmean(flagged)), 3), "SD": round(float(np.nanstd(flagged)), 3)})
    pd.DataFrame(gate).to_csv(TABLES / "table_gate_weights.csv", index=False)
    pd.concat(gate_data).to_csv(TABLES / "data_gate_weights.csv", index=False)

    # ---- subgroups (RQ5) on external A
    es = sets["external_A"]
    meta = json.load(open(PROCESSED / "feature_meta.json"))
    age = arrays["external_A"]["static"][:, 0] * meta["scalers"]["age_std"] + meta["scalers"]["age_mean"]
    tab = pd.read_csv(PROCESSED / "tabular_external_A.csv")
    dx_bl = es.frame.merge(_dx_bl(cfg["data"]["external"]["external_A"]["csv"]), on="RID", how="left").DX_bl.to_numpy()
    groups = {"Sex": np.where(tab["Female"].to_numpy() == 1, "Female", "Male"),
              "Age band": np.select([age < 70, age < 80], ["<70", "70-79"], ">=80"),
              "Baseline diagnosis": np.where(np.isin(dx_bl, ["CN", "SMC"]), "CN/SMC at entry", dx_bl),
              "MRI quality at index": np.array(["no MRI", "good", "acceptable", "poor"])[es.qc]}
    sub = []
    for gname, gval in groups.items():
        for level in sorted(np.unique(gval)):
            idx = es.idx[gval == level]
            n_ev = int(es.y[idx].sum())
            row = {"Subgroup": gname, "Level": level, "Participants": int(len(np.unique(es.rid[idx]))),
                   "Windows": len(idx), "Converters": n_ev}
            for mid in ("e5_proposed", "e4_early", "e1_gbm_clinmri"):
                if n_ev < 5 or n_ev > len(idx) - 5:
                    row[f"AUROC {LABEL[mid]}"] = "suppressed (<5 in a class)"
                else:
                    b = M.ClusterBootstrap(es.rid[idx], n_boot, seed=3)
                    row[f"AUROC {LABEL[mid]}"] = M.fmt_ci(*b.ci(M.auroc, es.y[idx], models[mid].prob(es.cohort, "natural", idx)))
            sub.append(row)
    pd.DataFrame(sub).to_csv(TABLES / "table_subgroups.csv", index=False)

    # ---- error analysis of the proposed model on external A
    full = models["e5_proposed"]
    p = full.prob("external_A", "natural", es.idx)
    pred = p >= full.thr
    qc, lengths, has = es.qc[es.idx], es.lengths[es.idx], es.a_mri[es.idx]
    err = []
    cats = ["Sparse history (<= 2 visits)", "MRI missing", "Poor-quality MRI", "Close to threshold (+/-0.10)",
            "Other (unusual trajectory)"]
    for kind, sel in (("False positive", pred & (es.y == 0)), ("False negative", ~pred & (es.y == 1))):
        c = np.select([lengths[sel] <= 2, has[sel] == 0, qc[sel] == 3, np.abs(p[sel] - full.thr) < 0.1], cats[:4], cats[4])
        err += [{"Error type": kind, "Primary category": k, "Count": int((c == k).sum())} for k in cats]
    pd.DataFrame(err).to_csv(TABLES / "table_error_analysis.csv", index=False)

    # ---- explanations and their stability (external A)
    summary["stability"] = explanation_analysis(cfg, arrays["external_A"], es.idx)

    # ---- computation
    recs = pd.DataFrame([json.loads(line) for line in open(RUNS / "run_records.jsonl")])
    comp = recs.groupby("model").agg(parameters=("parameters", "mean"), seconds_per_run=("train_seconds", "mean"),
                                     runs=("run_id", "size")).reset_index()
    comp.to_csv(TABLES / "table_compute.csv", index=False, float_format="%.1f")

    # ---- calibrated predictions for figures and dashboard
    out = []
    for mid in LABEL:
        for key, es in sets.items():
            idx = es.pick(mid)[0]
            for cond in ("natural", "mri_removed"):
                p = models[mid].prob(es.cohort, cond, idx)
                if np.isnan(p).all():
                    continue
                out.append(pd.DataFrame({"set": key, "window_id": idx, "model": mid, "condition": cond,
                                         "label": es.y[idx], "p_raw": models[mid].prob(es.cohort, cond, idx, False),
                                         "p_calibrated": p}))
    pd.concat(out).to_csv(PREDICTIONS / "calibrated_predictions.csv.gz", index=False)
    summary.update(test_auroc=test_auc["dev_test"], n_test_windows=summary["sets"]["dev_test"]["windows"],
                   n_test_participants=summary["sets"]["dev_test"]["participants"])
    with open(RESULTS / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for h in hyp:
        print(f"{h['Set'][:18]:<18} {h['Question']:<22} {h['Comparison'][:62]:<62} {h['Delta [95% CI]']:<24} {h['Decision']}")


def _dx_bl(csv):
    from mdp.config import ROOT
    v = pd.read_csv(ROOT / csv, usecols=["RID", "DX_bl"])
    return v.groupby("RID").DX_bl.first().reset_index()


def explanation_analysis(cfg, arrays, idx):
    ev = cfg["evaluation"]
    k = ev["stability_top_k"]
    data = to_tensors(arrays)
    kw = json.load(open(MODELS / "e5_proposed" / "model_kwargs.json"))
    seed_vecs, temporal = [], []
    for s in cfg["training"]["seeds"]:
        model = FusionModel(**kw)
        model.load_state_dict(torch.load(MODELS / "e5_proposed" / f"seed{s}.pt"))
        attr = integrated_gradients(model, data, idx)
        seed_vecs.append(feature_vector(attr))
        temporal += [(s, *row) for row in temporal_profile(attr, arrays["tti"][idx], arrays["lengths"][idx])]
        if s == cfg["training"]["seeds"][0]:
            base_model, base_vec = model, seed_vecs[-1]
    rho_seed, jac_seed = [], []
    for i in range(len(seed_vecs)):
        for j in range(i + 1, len(seed_vecs)):
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
        r, jk = stability(base_vec, feature_vector(integrated_gradients(base_model, pert, idx)), k)
        rho_pert.append(r)
        jac_pert.append(jk)
    rho_pert, jac_pert = np.concatenate(rho_pert), np.concatenate(jac_pert)

    with open(MODELS / "e1_gbm_clinmri.pkl", "rb") as f:
        gb = pickle.load(f)
    tab = pd.read_csv(PROCESSED / "tabular_external_A.csv")
    X = tab[gb["columns"]].to_numpy(float)
    dev_tab = pd.read_csv(PROCESSED / "tabular.csv")
    sd = np.nanstd(dev_tab[gb["columns"]].to_numpy(float), axis=0)
    explainer = shap.TreeExplainer(gb["model"])
    sv = np.asarray(explainer.shap_values(X[idx]))
    binary = np.array([c == "Female" for c in gb["columns"]])
    rng = np.random.default_rng(99)
    rho_shap, jac_shap = [], []
    for _ in range(5):
        Xp = X[idx] + ev["perturbation_sd"] * rng.standard_normal(X[idx].shape) * sd * ~binary
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
    pd.DataFrame({"feature": FEATURE_NAMES, "proposed_mean_abs_IG": mean_abs}).to_csv(
        TABLES / "data_importance_proposed.csv", index=False)
    pd.DataFrame({"feature": gb["columns"], "gbm_mean_abs_SHAP": np.abs(sv).mean(axis=0)}).to_csv(
        TABLES / "data_importance_gbm.csv", index=False)
    pd.DataFrame(temporal, columns=["seed", "period", "steps", "mean_abs_attribution"]).to_csv(
        TABLES / "data_temporal_attribution.csv", index=False)
    stable = bool(np.median(rho_seed) >= ev["stability_min_spearman"])
    with open(MODELS / "e5_proposed" / "stability.json", "w") as f:
        json.dump({"median_seed_spearman": float(np.median(rho_seed)), "threshold": ev["stability_min_spearman"],
                   "stable": stable}, f)
    return {"seed_spearman_median": float(np.median(rho_seed)), "seed_jaccard_median": float(np.median(jac_seed)),
            "perturbation_spearman_median": float(np.median(rho_pert)), "shap_spearman_median": float(np.median(rho_shap)),
            "meets_threshold": stable}


if __name__ == "__main__":
    main()
