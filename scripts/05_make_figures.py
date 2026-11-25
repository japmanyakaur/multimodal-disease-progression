"""Step 5: every figure of the Results chapter, generated from the saved prediction and table files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from mdp.config import FIGURES, PREDICTIONS, RESULTS, TABLES  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"e5_proposed": "#2a78d6", "e4_early": "#eb6834", "e4_xattn": "#1baf7a", "e4_late": "#eda100",
         "e1_gbm_clinmri": "#e87ba4", "e2_clinical": "#008300", "e3_mri": "#6250d6", "e1_lr_clinmri": "#e34948"}
MARKER = {"e5_proposed": "o", "e4_early": "s", "e4_xattn": "^", "e4_late": "D", "e1_gbm_clinmri": "v",
          "e2_clinical": "P", "e3_mri": "X", "e1_lr_clinmri": "*"}
SHORT = {"e1_lr_clin": "LR (clinical)", "e1_gbm_clin": "GBM (clinical)", "e1_lr_clinmri": "LR (clinical+MRI)",
         "e1_gbm_clinmri": "GBM (clinical+MRI)", "e2_clinical": "GRU (clinical only)", "e3_mri": "MRI only",
         "e4_early": "Early fusion", "e4_late": "Late fusion", "e4_xattn": "Cross-attention fusion",
         "e5_proposed": "Proposed (reliability-aware)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print("  wrote", name)


def parse_ci(s):
    point, rest = s.split(" [")
    lo, hi = rest.rstrip("]").split(", ")
    return float(point), float(lo), float(hi)


def fig_cohort_flow():
    flow = pd.read_csv(TABLES / "table_cohort_flow.csv")
    split = pd.read_csv(TABLES / "table_split_counts.csv").set_index("split")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    first = flow.iloc[0]
    mci = flow[flow.Stage.str.startswith("MCI visits")].iloc[0]
    final = flow.iloc[-1]
    main = [f"Data release\n{int(first.Count):,} visits, {int(first.Participants):,} participants",
            f"Candidate index visits (diagnosis = MCI)\n{int(mci.Count):,} visits",
            f"Analysis cohort\n{int(final.Count):,} index windows, {int(final.Participants):,} participants"]
    ys = [9.0, 6.6, 3.5]
    for text, yy in zip(main, ys):
        ax.text(3.6, yy, text, ha="center", va="center", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f0efec", ec=AXIS))
    for a, b in zip(ys[:-1], ys[1:]):
        ax.annotate("", xy=(3.6, b + 0.55), xytext=(3.6, a - 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    excl = flow[flow.Stage.str.startswith("Excluded")]
    short = {"dementia": "Dementia before index", "no cognitive": "No score at index",
             "outcome unknown": "Outcome unknown at 24 mo", "next visit": "Next visit inside cut-off"}
    lines = "\n".join(f"{next(v for k, v in short.items() if k in st)}: {abs(int(c)):,}"
                      for st, c in zip(excl.Stage, excl.Count))
    ax.text(6.3, 5.05, "Excluded\n" + lines, ha="left", va="center", fontsize=8, color=INK2,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=AXIS))
    ax.annotate("", xy=(6.25, 5.05), xytext=(3.6, 5.05), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    names = [("train", "Training"), ("validation", "Validation"), ("calibration", "Calibration"),
             ("test", "Locked test")]
    for k, (key, name) in enumerate(names):
        x = 1.1 + k * 2.45
        r = split.loc[key]
        ax.text(x, 1.0, f"{name}\n{int(r.participants)} participants\n{int(r.windows)} windows\n"
                        f"{int(r.converters)} converters", ha="center", va="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=COLOR["e5_proposed"] if key == "test" else AXIS))
        ax.annotate("", xy=(x, 1.75), xytext=(3.6, 2.9), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.text(5.2, 2.6, "participant-level split", ha="left", fontsize=7.5, color=MUTED, style="italic")
    save(fig, "fig_cohort_flow")


def fig_main_auroc():
    t = pd.read_csv(TABLES / "table_main_results.csv")
    keys = list(SHORT)
    vals = [parse_ci(s) for s in t["AUROC [95% CI]"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, (k, (p, lo, hi)) in enumerate(zip(keys, vals)):
        c = COLOR["e5_proposed"] if k == "e5_proposed" else INK2
        ax.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(p, i, "o", color=c, ms=7, mec="white", mew=1.5)
        ax.text(hi + 0.006, i, f"{p:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(keys)), [SHORT[k] + (" *" if k == "e3_mri" else "") for k in keys])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC on locked test set (95% participant-bootstrap CI)")
    ax.set_title("Discrimination of all models")
    ax.text(0, -0.2, "* MRI-only model evaluated on the test windows that have an MRI.", transform=ax.transAxes,
            fontsize=7.5, color=MUTED)
    save(fig, "fig_main_auroc")


def fig_roc_pr():
    pr = pd.read_csv(PREDICTIONS / "test_calibrated.csv.gz")
    pr = pr[pr.condition == "natural"]
    sel = ["e5_proposed", "e4_early", "e1_gbm_clinmri", "e2_clinical"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for k in sel:
        d = pr[pr.model == k]
        fpr, tpr, _ = roc_curve(d.label, d.p_calibrated)
        prec, rec, _ = precision_recall_curve(d.label, d.p_calibrated)
        axes[0].plot(fpr, tpr, color=COLOR[k], lw=1.6, label=SHORT[k])
        axes[1].plot(rec, prec, color=COLOR[k], lw=1.6, label=SHORT[k])
    axes[0].plot([0, 1], [0, 1], ls="--", color=AXIS, lw=1)
    axes[1].axhline(pr[pr.model == sel[0]].label.mean(), ls="--", color=AXIS, lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves")
    axes[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision-recall curves")
    axes[1].legend(loc="lower left", fontsize=8)
    save(fig, "fig_roc_pr")


def fig_reliability():
    b = pd.read_csv(TABLES / "data_reliability_bins.csv")
    cal = pd.read_csv(TABLES / "table_calibration.csv")
    pairs = [("e5_proposed", "E5 Proposed reliability-aware fusion"),
             ("e1_gbm_clinmri", "E1 Gradient boosting (clinical + MRI)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), sharey=True)
    for ax, (k, full) in zip(axes, pairs):
        row = cal[cal.Model == full].iloc[0]
        for stage, color, mk in (("before", MUTED, "s"), ("after", COLOR["e5_proposed"], "o")):
            d = b[(b.model == k) & (b.stage == stage)]
            lab = (f"Before scaling (ECE {row['ECE before']:.3f})" if stage == "before" else
                   f"After temperature scaling (ECE {parse_ci(row['ECE after [95% CI]'])[0]:.3f})")
            ax.plot(d.mean_predicted, d.observed_rate, marker=mk, color=color, lw=1.5, ms=5, label=lab)
        ax.plot([0, 1], [0, 1], ls="--", color=AXIS, lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
               title=f"{SHORT[k]}  (T = {row['Temperature T']:.2f})")
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("Observed conversion rate")
    save(fig, "fig_reliability")


def fig_missing_modality():
    c = pd.read_csv(TABLES / "data_degradation_curve.csv")
    s = pd.read_csv(TABLES / "table_missing_modality.csv")
    order = ["e5_proposed", "e4_early", "e4_xattn", "e4_late", "e1_gbm_clinmri", "e2_clinical"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for k in order:
        d = c[c.model == k]
        ax.plot(d.fraction_mri_removed * 100, d.auroc, color=COLOR[k], marker=MARKER[k], lw=1.6, ms=6,
                mec="white", mew=1, label=SHORT[k])
    ax.set(xlabel="Test windows with MRI removed (%)", ylabel="AUROC", title="Random MRI removal")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(0.85, 0.90)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax2 = axes[1]
    labels = {"E5 Proposed reliability-aware fusion": "e5_proposed", "E4 Early fusion": "e4_early",
              "E4 Cross-attention fusion": "e4_xattn", "E4 Late fusion": "e4_late"}
    rows = s[s.Model.isin(labels)]
    y = np.arange(len(rows))
    for i, (_, r) in enumerate(rows.iterrows()):
        k = labels[r.Model]
        ax2.barh(i, r["MRI only (history masked)"], color=COLOR[k], height=0.6)
        ax2.text(r["MRI only (history masked)"] + 0.005, i, f"{r['MRI only (history masked)']:.3f}", va="center",
                 fontsize=8, color=INK2)
    ax2.set_yticks(y, [SHORT[labels[m]] for m in rows.Model])
    ax2.invert_yaxis()
    ax2.set_xlim(0.5, 0.82)
    ax2.grid(axis="y", visible=False)
    ax2.set(xlabel="AUROC", title="Clinical history removed (MRI only)")
    fig.subplots_adjust(wspace=0.75)
    save(fig, "fig_missing_modality")


def fig_ablation():
    a = pd.read_csv(TABLES / "table_ablation.csv")
    a = a[a.Variant != "Full proposed model"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, (_, r) in enumerate(a.iterrows()):
        p, lo, hi = parse_ci(r["Delta AUROC vs full [95% CI]"])
        c = COLOR["e5_proposed"] if hi < 0 or lo > 0 else INK2
        ax.plot([lo, hi], [i, i], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(p, i, "o", color=c, ms=7, mec="white", mew=1.5)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(a)), [v.replace("- ", "without ") for v in a.Variant])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Change in test AUROC vs full proposed model (95% CI)", title="Ablation study")
    ax.text(0, -0.32, "Blue: confidence interval excludes zero. Negative = removing the component hurts.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    save(fig, "fig_ablation")


def fig_importance():
    ig = pd.read_csv(TABLES / "data_importance_proposed.csv").sort_values("proposed_mean_abs_IG")
    sh = pd.read_csv(TABLES / "data_importance_gbm.csv").sort_values("gbm_mean_abs_SHAP").tail(13)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].barh(ig.feature, ig.proposed_mean_abs_IG, color=COLOR["e5_proposed"], height=0.65)
    axes[0].set(xlabel="Mean |Integrated Gradients| (logit units)", title="Proposed model")
    axes[1].barh(sh.feature, sh.gbm_mean_abs_SHAP, color=COLOR["e5_proposed"], height=0.65)
    axes[1].set(xlabel="Mean |SHAP value| (log-odds)", title="Gradient boosting (clinical + MRI)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.75)
    save(fig, "fig_importance")


def fig_temporal():
    t = pd.read_csv(TABLES / "data_temporal_attribution.csv")
    g = t.groupby("period", sort=False).mean_abs_attribution
    m, lo, hi = g.mean(), g.min(), g.max()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(m))
    ax.bar(x, m.to_numpy(), color=COLOR["e5_proposed"], width=0.6)
    ax.errorbar(x, m.to_numpy(), yerr=[m - lo, hi - m], fmt="none", ecolor=INK2, lw=1, capsize=3)
    ax.set_xticks(x, ["Index visit", "6 mo", "12 mo", "18 mo", "24 mo", "> 2.5 y"], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Mean |attribution| per visit", xlabel="Time before the index visit",
           title="Which visits drive the forecast (proposed model)")
    ax.text(0, -0.32, "Bars: mean over 5 seeds; whiskers: range across seeds.", transform=ax.transAxes,
            fontsize=7.5, color=MUTED)
    save(fig, "fig_temporal_attribution")


def fig_gate():
    g = pd.read_csv(TABLES / "data_gate_weights.csv")
    g = g[g.qc > 0]
    names = {1: "Good", 2: "Acceptable", 3: "Poor"}
    shades = {1: "#1c5cab", 2: "#3987e5", 3: "#86b6ef"}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    rng = np.random.default_rng(0)
    for i, q in enumerate((1, 2, 3)):
        v = g[g.qc == q].alpha_mri.to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=shades[q], alpha=0.6, lw=0)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, lw=2)
        ax.text(i + 0.32, v.mean(), f"{v.mean():.3f}", va="center", fontsize=8, color=INK2)
    ax.axhline(0.5, ls="--", color=AXIS, lw=1)
    ax.set_xticks([0, 1, 2], [f"{names[q]}\n(n={int((g.qc == q).sum())})" for q in (1, 2, 3)])
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Weight given to MRI (alpha_MRI)", xlabel="MRI quality rating",
           title="Reliability gate: MRI weight by scan quality")
    save(fig, "fig_gate_weights")


def fig_stability():
    d = np.load(TABLES / "data_stability.npz")
    panels = [("rho_seed", "Proposed: across 5 seeds"), ("rho_pert", "Proposed: small input noise"),
              ("rho_shap", "GBM SHAP: small input noise")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True)
    for ax, (k, title) in zip(axes, panels):
        v = d[k]
        ax.hist(v, bins=np.linspace(0.0, 1, 41), color=COLOR["e5_proposed"], edgecolor="white", linewidth=1)
        ax.axvline(np.median(v), color=INK, lw=1.5)
        ax.axvline(0.7, color=MUTED, lw=1, ls="--")
        ax.text(0.03, ax.get_ylim()[1] * 0.92, f"median {np.median(v):.2f}", ha="left", fontsize=8, color=INK)
        ax.set(title=title, xlabel="Spearman rank correlation")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Pairs of explanations")
    fig.text(0.01, -0.1, "Dashed line: pre-specified stability threshold (0.70). Solid line: median.", fontsize=7.5,
             color=MUTED)
    save(fig, "fig_stability")


def fig_subgroups():
    s = pd.read_csv(TABLES / "table_subgroups.csv")
    col = [c for c in s.columns if c.startswith("AUROC E5")][0]
    s = s[~s[col].str.startswith("suppressed")]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, (_, r) in enumerate(s.iterrows()):
        p, lo, hi = parse_ci(r[col])
        ax.plot([lo, hi], [i, i], color=COLOR["e5_proposed"], lw=1.6)
        ax.plot(p, i, "o", color=COLOR["e5_proposed"], ms=6, mec="white", mew=1.2)
    overall = json.load(open(RESULTS / "metrics_summary.json"))["test_auroc"]["e5_proposed"]
    ax.axvline(overall, ls="--", color=AXIS, lw=1)
    ax.set_yticks(range(len(s)), [f"{a}: {b} (n={w}, events={e})" for a, b, w, e in
                                  zip(s.Subgroup, s.Level, s.Windows, s.Converters)], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="AUROC (95% CI)", title="Subgroup performance of the proposed model")
    save(fig, "fig_subgroups")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for f in (fig_cohort_flow, fig_main_auroc, fig_roc_pr, fig_reliability, fig_missing_modality, fig_ablation,
              fig_importance, fig_temporal, fig_gate, fig_stability, fig_subgroups):
        f()


if __name__ == "__main__":
    main()
