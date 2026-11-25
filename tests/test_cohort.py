"""Tests for target construction, splitting and the leakage validators.

The leakage tests plant violations on purpose and check that each is detected.
They test the safeguard, not model performance.
"""
import numpy as np
import pandas as pd
import pytest

from mdp import cohort as C
from mdp import features as F
from mdp.config import load_config


def _visit(rid, month, dx, mri=True, start="2010-01-01"):
    date = pd.Timestamp(start) + pd.Timedelta(days=int(month * 30.44))
    return dict(RID=rid, VISCODE="bl" if month == 0 else f"m{month:02d}", EXAMDATE=date.date().isoformat(),
                DX=dx, DX_bl="LMCI", AGE=72.0, PTGENDER="Female", PTEDUCAT=16, APOE4=1,
                MMSE=27, CDRSB=1.5, ADAS13=15.0, FAQ=3,
                Hippocampus=6500 if mri else np.nan, Entorhinal=3500 if mri else np.nan,
                Ventricles=40000 if mri else np.nan, WholeBrain=1.0e6 if mri else np.nan,
                ICV=1.45e6 if mri else np.nan, MRI_QC=1 if mri else np.nan,
                MRI_SCANDATE=date.date().isoformat() if mri else None)


@pytest.fixture
def toy(tmp_path):
    rows = (
        [_visit(1, m, d) for m, d in [(0, "MCI"), (6, "MCI"), (12, "Dementia"), (24, "Dementia")]]   # converter
        + [_visit(2, m, "MCI") for m in (0, 6, 12, 24, 36)]                                          # stable
        + [_visit(3, m, "MCI") for m in (0, 6)]                                                      # lost to follow-up
        + [_visit(4, 0, "MCI", mri=False), _visit(4, 24, "MCI", mri=False)]                          # no MRI
    )
    path = tmp_path / "toy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    visits, _ = C.load_visits(path)
    cfg = load_config()
    cohort, _, linkage = C.build_cohort(visits, cfg)
    return visits, cohort, linkage, cfg


def test_labels(toy):
    _, cohort, _, _ = toy
    first = cohort.groupby("RID").first()
    assert first.loc[1, "label"] == 1           # dementia 12 months after baseline
    assert first.loc[2, "label"] == 0           # still MCI at month 24
    assert 3 not in first.index                 # cannot know the 24-month outcome
    assert (cohort[cohort.RID == 1].index_viscode == ["bl", "m06"]).all()


def test_mri_linkage(toy):
    _, cohort, linkage, _ = toy
    assert (cohort[cohort.RID == 4].mri_row == -1).all()
    assert linkage["none"] >= 1 and linkage["exact"] >= 1


def test_clean_cohort_passes_leakage_checks(toy):
    visits, cohort, _, cfg = toy
    cohort = cohort.merge(C.split_participants(cohort, cfg, seed=0), on="RID")
    C.assert_no_participant_overlap(cohort)
    C.assert_predictors_before_cutoff(cohort, visits)


def test_detects_participant_in_two_splits(toy):
    _, cohort, _, _ = toy
    cohort = cohort.assign(split="train")
    cohort.loc[cohort.index[0], "split"] = "test"
    cohort.loc[cohort.index[1], "split"] = "train"     # same participant (RID 1) in two splits
    with pytest.raises(C.LeakageError):
        C.assert_no_participant_overlap(cohort)


def test_detects_future_visit_in_history(toy):
    visits, cohort, _, _ = toy
    bad = cohort.copy()
    future_row = visits.index[(visits.RID == 2) & (visits.VISCODE == "m36")][0]
    i = bad.index[bad.RID == 2][0]
    bad.loc[i, "hist_rows"] = f"{bad.loc[i, 'hist_rows']};{future_row}"
    with pytest.raises(C.LeakageError):
        C.assert_predictors_before_cutoff(bad, visits)


def test_detects_mri_after_cutoff(toy):
    visits, cohort, _, _ = toy
    bad_visits = visits.copy()
    r = cohort[cohort.mri_row >= 0].iloc[0]
    bad_visits.loc[r.mri_row, "MRI_SCANDATE"] = r.index_date + pd.Timedelta(days=90)
    with pytest.raises(C.LeakageError):
        C.assert_predictors_before_cutoff(cohort, bad_visits)


def test_features_masks_and_train_only_scaling(toy):
    visits, cohort, _, cfg = toy
    raw = F.extract_raw(visits, cohort, cfg["cohort"]["max_history_visits"])
    train = np.zeros(len(cohort), bool)
    train[:2] = True
    sc = F.fit_scalers(raw, train)
    raw2 = {k: v.copy() for k, v in raw.items()}
    raw2["vals"][2:] += 100.0                                        # change non-training windows only
    sc2 = F.fit_scalers(raw2, train)
    assert np.allclose(sc["score_mean"], sc2["score_mean"])          # scaler ignores non-training data
    arr = F.transform(raw, sc)
    pad = np.arange(arr["x"].shape[1])[None, :] >= arr["lengths"][:, None]
    assert (arr["m"][pad] == 0).all() and (arr["x"][pad] == 0).all()
    assert set(np.unique(arr["a_mri"])) <= {0.0, 1.0}
