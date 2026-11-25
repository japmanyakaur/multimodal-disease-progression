# Trustworthy Multimodal Learning for Longitudinal Disease Progression Prediction

M.Tech final-semester project (IIT Patna). Forecasts whether a patient with
Mild Cognitive Impairment converts to Alzheimer's dementia within **24 months**,
using longitudinal clinical/cognitive history and MRI-derived brain measures,
and evaluates calibration, uncertainty, missing-modality robustness and
explanation stability.

**New to the project? Read [`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md) first.**

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
| `src/mdp/simulate.py` | simulated cohort in ADNIMERGE format |
| `src/mdp/cohort.py` | index visits, 24-month label, MRI linkage, patient splits, leakage checks |
| `src/mdp/features.py` | visit sequences, masks, time gaps, summary features |
| `src/mdp/models.py` | GRU, MRI encoder, early / late / cross-attention / proposed fusion |
| `src/mdp/train.py` | training loop, classical baselines |
| `src/mdp/metrics.py` | AUROC, calibration, bootstrap, temperature scaling, conformal |
| `src/mdp/explain.py` | Integrated Gradients, SHAP, stability measures |
| `scripts/01..05_*.py` | the five pipeline steps |
| `results/` | predictions, tables, figures, run records (generated) |
| `dashboard/app.py` | Streamlit research dashboard |

## Running on real ADNI data

1. Obtain access through the LONI Image and Data Archive and download `ADNIMERGE.csv`.
2. Place it at `data/adni/ADNIMERGE.csv` (this folder is git-ignored).
3. Set `data.source: adni` in `configs/config.yaml` and run `python run_all.py`.

If your ADNIMERGE has no MRI quality column, all scans are treated as
"good" quality and a warning is printed.
