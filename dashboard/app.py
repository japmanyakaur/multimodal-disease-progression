"""Research dashboard (thesis Chapter 9).   Run:  streamlit run dashboard/app.py

Shows simulated test cases only. Research prototype - not for diagnosis or treatment decisions.
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from mdp import metrics as M  # noqa: E402
from mdp.config import MODELS, PROCESSED, RESULTS, RUNS, data_csv, load_config  # noqa: E402
from mdp.explain import FEATURE_NAMES, feature_vector, integrated_gradients  # noqa: E402
from mdp.features import SCORE_NAMES  # noqa: E402
from mdp.models import FusionModel  # noqa: E402
from mdp.train import apply_condition, subset, to_tensors  # noqa: E402

st.set_page_config(page_title="Disease Progression Forecast (research)", layout="wide")
NOTICE = ("**Research prototype - not a diagnostic tool.** Forecasts are produced on a simulated cohort and must "
          "not be used for decisions about patient care.")
QC_NAME = {0: "no MRI", 1: "good", 2: "acceptable", 3: "poor"}


@st.cache_resource
def load():
    needed = [PROCESSED / "cohort.csv", PROCESSED / "features.npz", MODELS / "e5_proposed" / "model_kwargs.json",
              RESULTS / "metrics_summary.json"]
    if not all(p.exists() for p in needed):
        return None
    cfg = load_config()
    cohort = pd.read_csv(PROCESSED / "cohort.csv", parse_dates=["index_date"])
    arrays = dict(np.load(PROCESSED / "features.npz"))
    visits = pd.read_csv(data_csv(cfg), parse_dates=["EXAMDATE"])
    kw = json.load(open(MODELS / "e5_proposed" / "model_kwargs.json"))
    models, digest = [], hashlib.sha256()
    for s in cfg["training"]["seeds"]:
        path = MODELS / "e5_proposed" / f"seed{s}.pt"
        digest.update(path.read_bytes())
        m = FusionModel(**kw)
        m.load_state_dict(torch.load(path))
        models.append(m.eval())
    summary = json.load(open(RESULTS / "metrics_summary.json"))
    q_mondrian = {int(k): float(v) for k, v in summary["conformal_qhat"]["e5_proposed"]["mondrian"].items()}
    stability = json.load(open(MODELS / "e5_proposed" / "stability.json"))
    dev = cohort.split != "test"
    return dict(cfg=cfg, cohort=cohort, arrays=arrays, data=to_tensors(arrays), visits=visits, models=models,
                version=digest.hexdigest()[:12], T=summary["temperatures"]["e5_proposed"],
                thr=summary["thresholds"]["e5_proposed"], q=q_mondrian, stability=stability, summary=summary,
                base_rate=float(cohort[dev].label.mean()))


def forecast(S, wid, hide_mri):
    b = subset(S["data"], np.array([wid]))
    if hide_mri:
        b = apply_condition(b, "mri_removed")
    with torch.no_grad():
        outs = [m(b) for m in S["models"]]
    logit = float(np.mean([o[0].item() for o in outs]))
    alpha = float(np.mean([o[1]["alpha"][0, 1].item() for o in outs]))
    p = float(M.sigmoid(logit / S["T"]))
    attr = np.mean([feature_vector(integrated_gradients(m, b, np.array([0]))) for m in S["models"]], axis=0)[0]
    per_visit = np.mean([np.abs(integrated_gradients(m, b, np.array([0]))["x"][0]).sum(1) for m in S["models"]], axis=0)
    return p, alpha, attr, per_visit, b


def audit(S, wid, hide_mri):
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "dashboard_audit.jsonl", "a") as f:
        f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "model_version": S["version"],
                            "case": int(wid), "mri_hidden": hide_mri, "temperature": S["T"]}) + "\n")


S = load()
st.title("Forecasting conversion from MCI to dementia")
st.warning(NOTICE)
if S is None:
    st.error("No trained model found. Run `python run_all.py` first, then reload this page.")
    st.stop()

cohort = S["cohort"]
test = cohort[cohort.split == "test"].reset_index(drop=True)
with st.sidebar:
    st.header("Case selection")
    st.caption("Simulated test-set cases only (never seen during training).")
    first_with_mri = int(np.argmax(S["arrays"]["a_mri"][test.window_id.to_numpy()] == 1))
    k = st.selectbox("Case", range(len(test)), index=first_with_mri,
                     format_func=lambda i: f"Case {i + 1:03d} (simulated)")
    hide_mri = st.toggle("Hide MRI (simulate a missing scan)", value=False)
    st.divider()
    st.caption(f"Model version `{S['version']}` - data source: `{S['cfg']['data']['source']}`")

row = test.iloc[k]
wid = int(row.window_id)
p, alpha_mri, attr, per_visit, b = forecast(S, wid, hide_mri)
audit(S, wid, hide_mri)
has_mri = bool(S["arrays"]["a_mri"][wid]) and not hide_mri
qc = int(S["arrays"]["qc"][wid]) if has_mri else 0
n_visits = int(S["arrays"]["lengths"][wid])

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Timeline & inputs", "Forecast & uncertainty", "Explanation"])

with tab1:
    st.subheader("What this system does")
    st.markdown(
        f"- **Target:** conversion from mild cognitive impairment (MCI) to dementia within "
        f"**{S['cfg']['cohort']['horizon_months']} months** of the index visit.\n"
        "- **Inputs:** up to 8 previous visits (MMSE, CDR-SB, ADAS-Cog13, FAQ, with time gaps and missingness) "
        "and MRI-derived brain volumes from the index window.\n"
        "- **Model:** reliability-aware fusion (GRU + MRI encoder + cross-attention + reliability gate), "
        "5-seed ensemble, temperature-scaled.\n"
        "- **Intended use:** academic research and demonstration. **Excluded uses:** diagnosis, screening, "
        "treatment selection, triage.")
    s = S["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test AUROC (proposed)", f"{s['test_auroc']['e5_proposed']:.3f}")
    c2.metric("Test windows / participants", f"{s['n_test_windows']} / {s['n_test_participants']}")
    c3.metric("Conversion rate in development data", f"{S['base_rate']:.1%}")
    st.caption("All numbers come from the saved evaluation files in results/. Cohort: simulated ADNIMERGE-format data.")

with tab2:
    st.subheader("Visit history up to the index visit")
    hist = [int(x) for x in str(row.hist_rows).split(";")]
    v = S["visits"].loc[hist, ["EXAMDATE", "DX", "MMSE", "CDRSB", "ADAS13", "FAQ", "Hippocampus"]].copy()
    v.insert(0, "Months relative to index", ((v.EXAMDATE - row.index_date).dt.days / 30.44).round(1))
    v["MRI at visit"] = np.where(v.Hippocampus.notna(), "yes", "no")
    v = v.drop(columns=["EXAMDATE", "Hippocampus"]).rename(columns={"DX": "Diagnosis", "CDRSB": "CDR-SB",
                                                                     "ADAS13": "ADAS-Cog13"})
    shown = v.copy()
    for col in ("MMSE", "CDR-SB", "ADAS-Cog13", "FAQ"):
        shown[col] = [f"{x:g}" if pd.notna(x) else "not measured" for x in v[col]]
    st.dataframe(shown, hide_index=True, use_container_width=True)
    st.caption("'not measured' values are never silently filled: the model carries the last value forward "
               "together with a missingness flag and the time since it was measured.")
    chart = v.set_index("Months relative to index")[["MMSE", "CDR-SB", "ADAS-Cog13", "FAQ"]]
    st.line_chart(chart, height=260)
    c1, c2 = st.columns(2)
    c1.metric("Visits used", n_visits)
    c2.metric("MRI used for this forecast", f"yes ({QC_NAME[qc]} quality)" if has_mri else "no")
    if hide_mri:
        st.info("MRI hidden by the user: the forecast uses clinical history only.")

with tab3:
    st.subheader(f"Forecast: probability of dementia within {S['cfg']['cohort']['horizon_months']} months")
    has0, has1 = p <= S["q"][0], (1 - p) <= S["q"][1]          # class-conditional (Mondrian) conformal set
    pred_set = [n for n, inc in (("stable MCI", has0), ("converter", has1)) if inc]
    c1, c2, c3 = st.columns(3)
    c1.metric("Calibrated probability", f"{p:.1%}")
    c2.metric("90% conformal prediction set (per class)",
              "both (uncertain)" if len(pred_set) == 2 else (pred_set[0] if pred_set else "empty"))
    c3.metric("Weight on MRI (reliability gate)", f"{alpha_mri:.2f}")
    st.progress(min(max(p, 0.0), 1.0))
    st.markdown(
        f"- Decision threshold fixed on the calibration split: **{S['thr']:.2f}** -> "
        f"{'above' if p >= S['thr'] else 'below'} threshold.\n"
        f"- Reference: {S['base_rate']:.1%} of development-set windows converted within 24 months.\n"
        f"- Probabilities are temperature-scaled (T = {S['T']:.2f}).\n"
        "- The threshold is tuned to catch converters (high sensitivity), so it flags more cases; the conformal "
        "set lists every outcome compatible with 90% coverage, separately for converters and non-converters. "
        "The two can differ and answer different questions.")
    if len(pred_set) == 2:
        st.warning("The prediction set contains both outcomes: the model is uncertain for this case.")
    if not has_mri:
        st.warning("No MRI used: the forecast relies on clinical history only.")
    elif qc == 3:
        st.warning("Poor-quality MRI: the reliability gate gives it less weight; interpret with caution.")
    if n_visits <= 2:
        st.warning("Sparse history (<= 2 visits): forecasts with short histories are less reliable.")
    st.caption(f"Model version {S['version']}. Research estimate, not a statement that the person will or will "
               "not develop dementia. No treatment recommendation is generated.")

with tab4:
    st.subheader("What influenced this forecast")
    if not S["stability"]["stable"]:
        st.error("Explanation stability is below the pre-specified threshold; explanations are withheld.")
    else:
        st.caption(f"Integrated Gradients averaged over 5 seeds. Explanation stability across seeds: median "
                   f"Spearman {S['stability']['median_seed_spearman']:.2f} (threshold "
                   f"{S['stability']['threshold']:.2f}). Positive = pushes towards conversion.")
        e = pd.DataFrame({"feature": FEATURE_NAMES, "attribution": attr})
        e = e.reindex(e.attribution.abs().sort_values(ascending=False).index).head(8)
        st.altair_chart(alt.Chart(e).mark_bar().encode(
            x=alt.X("attribution:Q", title="Attribution (logit units)"),
            y=alt.Y("feature:N", sort=list(e.feature), title=None),
            tooltip=["feature", alt.Tooltip("attribution:Q", format=".3f")]).properties(height=280),
            use_container_width=True)
        months = np.round(np.asarray(b["tti"][0, :n_visits, 0]) * 12).astype(int)
        labels = [f"{k + 1}. {mo} mo before" if mo > 0 else f"{k + 1}. index visit" for k, mo in enumerate(months)]
        tv = pd.DataFrame({"visit": labels, "influence": per_visit[:n_visits]})
        st.markdown("**Influence of each visit** (sum of absolute attributions of its test scores)")
        st.altair_chart(alt.Chart(tv).mark_bar().encode(
            x=alt.X("visit:N", sort=labels, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("influence:Q", title="Influence"),
            tooltip=["visit", alt.Tooltip("influence:Q", format=".3f")]).properties(height=220),
            use_container_width=True)
        st.info("Attributions describe the model's behaviour, not causes of disease. "
                f"Scores shown: {', '.join(SCORE_NAMES)}; MRI measures are normalised by head size.")
