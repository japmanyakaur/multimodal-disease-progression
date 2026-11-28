# Trustworthy Multimodal Learning for Longitudinal Disease Progression Prediction

Forecasts whether a patient with Mild Cognitive Impairment converts to Alzheimer's dementia within **24 months**,
using longitudinal clinical/cognitive history and MRI-derived brain measures,
and evaluates calibration, uncertainty, missing-modality robustness and
explanation stability.


> Research prototype. Not for diagnosis or treatment decisions.
> Results in `results/` were produced on a **simulated ADNI-format cohort**.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py            # data -> cohort -> training -> evaluation -> figures
pytest -q                    # unit tests (leakage checks, shapes, probabilities)
streamlit run dashboard/app.py
```

## Layout

| Path | What it contains |
|---|---|
| `configs/config.yaml` | every setting (horizon, splits, seeds, metrics) |
| `configs/preregistration_stage2.yaml` | Stage 2 hypotheses and decision rules, committed before the external cohorts existed |
| `src/mdp/simulate.py` | simulated cohort in ADNIMERGE format (development + external cohorts A and B) |
| `src/mdp/imaging.py` | rendered 3D MRI volumes, 3D CNN, Grad-CAM |
| `src/mdp/cohort.py` | index visits, 24-month label, MRI linkage, patient splits, leakage checks |
| `src/mdp/features.py` | visit sequences, masks, time gaps, summary features |
| `src/mdp/models.py` | GRU, MRI encoder, early / late / cross-attention / proposed fusion |
| `src/mdp/train.py` | training loop, classical baselines |
| `src/mdp/metrics.py` | AUROC, calibration, bootstrap, temperature scaling, conformal |
| `src/mdp/explain.py` | Integrated Gradients, SHAP, stability measures |
| `scripts/01..06_*.py` | pipeline steps (data, cohort, training, 3D CNN, evaluation, figures, cards) |
| `results/` | Stage 2 predictions, tables, figures, run records (generated) |
| `results/stage1/` | Stage 1 locked-evaluation results, kept unchanged |
| `docs/DATASET_CARD.md`, `docs/MODEL_CARD.md` | dataset and model documentation (generated) |
| `dashboard/app.py` | Streamlit research dashboard |

## Running on real ADNI data

1. Obtain access through the LONI Image and Data Archive and download `ADNIMERGE.csv`.
2. Place it at `data/adni/ADNIMERGE.csv` (this folder is git-ignored).
3. Set `data.source: adni` in `configs/config.yaml` and run `python run_all.py`.

If your ADNIMERGE has no MRI quality column, all scans are treated as
"good" quality and a warning is printed.
