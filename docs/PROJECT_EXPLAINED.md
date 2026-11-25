# Project Explained (plain English)

This guide explains what the project does, how each part works, and how to
answer the questions a judge is likely to ask. Read it top to bottom once; it
takes about 15 minutes.

---

## 1. The one-line summary

> For a patient with **Mild Cognitive Impairment (MCI)**, we look at their past
> clinic visits and their **MRI brain measurements** and predict whether they
> will develop **Alzheimer's dementia within the next 24 months**. Besides the
> prediction, the system reports **how confident** it is, **why** it made the
> prediction, and **how it behaves when the MRI is missing**.

"Trustworthy" in the title means we do not only measure accuracy; we also
measure calibration (are the probabilities honest?), uncertainty (does the
model say when it is unsure?), robustness (does it still work with missing
data?) and explanation stability (do the explanations stay the same if we
retrain?).

---

## 2. The data

The thesis is designed for **ADNI** (Alzheimer's Disease Neuroimaging
Initiative). ADNI data are free but need an approved application (a data use
agreement), so the code was developed and run on a **simulated cohort in
exactly the ADNI format** (`ADNIMERGE.csv` column names: `RID`, `VISCODE`,
`EXAMDATE`, `DX`, `MMSE`, `CDRSB`, `ADAS13`, `FAQ`, `Hippocampus`, ...).

* When ADNI access is granted, put `ADNIMERGE.csv` in `data/adni/`, set
  `data.source: adni` in `configs/config.yaml`, and re-run. No code changes.
* **Integrity rule:** results from the simulated cohort must be described as
  *simulated* in the thesis. They show that the pipeline works and how the
  models compare on data with known properties; they are **not** clinical
  findings about real patients.

### How the simulator works (so you can explain it)

Each simulated person has a hidden **disease stage** that increases over time.

1. **Speed of progression** differs per person; carrying the **APOE-e4 gene**
   makes it faster, and the disease accelerates as it advances.
2. **Brain shrinkage (MRI) appears before memory problems** (the well-known
   "biomarker cascade" of Jack et al., 2010). So the MRI carries information
   that test scores do not yet show.
3. **Cognitive reserve:** people with more education show symptoms later than
   their brain damage would suggest. This is another reason MRI adds
   information.
4. Visits happen roughly every 6–12 months with random delays, skipped visits
   and drop-outs. Test scores are noisy. About 20–30% of visits have no MRI and
   some MRIs are low quality (noisier measurements).

| Variable | Meaning | Higher means |
|---|---|---|
| MMSE (0–30) | Mini-Mental State Exam, memory/thinking test | better |
| ADAS-Cog13 (0–85) | Alzheimer's cognitive test | worse |
| CDR-SB (0–18) | Clinical Dementia Rating, sum of boxes | worse |
| FAQ (0–30) | Daily-living function questionnaire | worse |
| Hippocampus, Entorhinal | Volume of memory regions (MRI) | healthier |
| Ventricles | Fluid spaces in the brain (MRI) | more shrinkage |
| WholeBrain, ICV | Total brain volume, skull size (MRI) | - |

---

## 3. The pipeline, step by step

Run everything with one command: `python run_all.py`.

### Step 1 – `scripts/01_simulate_data.py`
Creates `data/synthetic/ADNIMERGE_synthetic.csv` (one row per visit).

### Step 2 – `scripts/02_build_cohort.py` (cohort, label, split)
* **Index visit:** any visit where the patient is diagnosed MCI. This is the
  moment the forecast is made (time *t0*).
* **Inputs:** only data recorded **on or before** t0 (up to the last 8 visits)
  plus the MRI from the index window (up to ~6 months before t0).
* **Label:** 1 if the patient is diagnosed with dementia within 24 months
  (+3 months tolerance) after t0, otherwise 0. Visits without enough follow-up
  are excluded (we cannot know their label).
* **Split by patient:** 70% train (of which 15% validation), 10% calibration,
  20% locked test. All visits of one person stay in one split; otherwise the
  model would "recognise" people and look better than it is.
* **Leakage checks:** automatic assertions that (a) no patient is in two
  splits and (b) every input date is before the cut-off. A unit test plants
  leakage on purpose and confirms the checks catch it.

### Step 3 – `scripts/03_train_models.py` (experiments E1–E5)
| Exp. | Model | Uses |
|---|---|---|
| E1 | Logistic regression, gradient-boosted trees | summary features (last value, slope) |
| E2 | GRU (recurrent network) | visit history only |
| E3 | MLP | MRI only |
| E4 | Early fusion / late fusion / cross-attention fusion | both |
| E5 | **Proposed reliability-aware fusion** | both |

**The proposed model in one paragraph.** A GRU reads the visit history
(scores, which values were missing, and the time gaps between visits). A small
network reads the MRI measurements. **Cross-attention** lets each branch look
at the other (e.g. the MRI summary looks at which visits matter). A
**reliability estimator** looks at *whether* each input is present and *how
good* it is (MRI quality, number of visits, how old the scan is) and produces
trust weights that sum to 1. A missing input gets weight 0. The final
prediction uses the trust-weighted combination. During training we randomly
hide the MRI 30% of the time (**modality dropout**) so the model learns to work
without it.

Each neural model is trained with **5 random seeds**; the 5 copies are averaged
(a small ensemble) and the spread across seeds is reported too.

### Step 4 – `scripts/04_evaluate.py` (E6, E7, statistics)
* **Main metric:** AUROC (0.5 = coin flip, 1.0 = perfect ranking). Also AUPRC,
  Brier score, ECE, sensitivity, specificity, F1.
* **95% confidence intervals:** we resample *patients* 2,000 times
  (bootstrap) and recompute the metric each time.
* **E6 missing-modality stress test:** remove the MRI (for everyone, 25%, 50%),
  remove the clinical history, add noise to the MRI, and look at patients
  whose MRI is genuinely missing.
* **E7 calibration:** temperature scaling divides the model's raw score by one
  number *T* learned on the calibration split so probabilities become honest.
* **E7 conformal prediction:** gives a *set* of possible answers ({converter},
  {stable} or {both}) that contains the truth for at least 90% of patients.
  "Both" means "the model is unsure".
* **E7 explanation stability:** we compute which features drove each
  prediction (Integrated Gradients for neural nets, SHAP for trees), then
  check whether the top-3 features stay the same across the 5 seeds and under
  small noise (Spearman correlation and top-k overlap).
* **Ablations:** remove one part of the proposed model at a time (gate,
  cross-attention, time gaps, missingness masks, modality dropout).
* **Subgroups:** sex, age band, baseline diagnosis group.

### Step 5 – `scripts/05_make_figures.py`
Every table and figure in the Results chapter is generated from the saved
prediction files in `results/` – nothing is typed by hand.

### Dashboard – `streamlit run dashboard/app.py`
Pick a (simulated) test patient, see the visit timeline, the forecast with its
uncertainty, what happens if the MRI is hidden, and the explanation. Always
labelled "research prototype – not for diagnosis".

---

## 4. Key terms in one line each

* **AUROC** – probability that a random converter gets a higher risk score
  than a random non-converter.
* **AUPRC** – like AUROC but focuses on the positive (converter) class; useful
  when positives are fewer.
* **Brier score** – average squared error of the probabilities (lower is
  better).
* **ECE** – average gap between predicted probability and actual frequency
  (lower is better).
* **Bootstrap** – resampling patients with replacement to estimate how much a
  number would vary with a different sample.
* **Temperature scaling** – one-parameter fix for over/under-confident
  probabilities; does not change AUROC.
* **Conformal prediction** – turns any model into one with a guaranteed
  coverage rate on exchangeable data.
* **Modality dropout** – hiding a whole input type during training.
* **Leakage** – when information from the future or from test patients sneaks
  into training; makes results look better than they are.

---

## 5. Likely judge questions and short answers

**Q: Why not use real patient data?**
ADNI needs an approved data use agreement. The pipeline reads the official
ADNIMERGE format, so it will run on real data unchanged once access is
granted. The simulated cohort lets us test every component, including
leakage checks, before touching restricted data.

**Q: Why MRI measurements instead of raw images?**
The thesis allows "validated image-derived measures" (Table 5.1). Volumes such
as the hippocampus are the standard, clinically meaningful MRI markers of
Alzheimer's, they are already in ADNIMERGE, and they run on a laptop. A 3D CNN
on raw scans is listed as future work.

**Q: How do you prevent leakage?**
Patient-level splits, a strict date cut-off for inputs, preprocessing fitted on
the training split only, and automated tests that detect deliberately planted
leakage.

**Q: What is new compared with semester 3?**
A fixed 24-month forecasting task, reliability-aware fusion, missing-modality
stress tests, calibration, conformal uncertainty and explanation-stability
measurements, all compared against simpler baselines with confidence
intervals.

**Q: What if the proposed model does not win?**
The thesis pre-specifies that a negative result is valid. We report all
baselines, including any that beat the proposed model.
