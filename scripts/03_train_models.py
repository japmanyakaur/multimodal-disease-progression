"""Step 3: train E1-E5, ablations and sensitivity runs; save predictions for every input condition.

Validation split  -> early stopping
Calibration split -> used later for temperature scaling, threshold and conformal prediction
Test split        -> predictions saved; metrics computed only in step 4
"""
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from mdp.config import MODELS, PREDICTIONS, PROCESSED, RUNS, ensure_dirs, load_config  # noqa: E402
from mdp.models import FusionModel, count_parameters  # noqa: E402
from mdp.records import write_record  # noqa: E402
from mdp.train import CONDITIONS, apply_condition, fit_classical, predict, to_tensors, train_neural  # noqa: E402

NEURAL = {
    "e2_clinical": dict(mode="clinical"),
    "e3_mri": dict(mode="mri"),
    "e4_early": dict(mode="early"),
    "e4_xattn": dict(mode="xattn"),
    "e5_proposed": dict(mode="proposed", moddrop=True),
    "abl_no_gate": dict(mode="proposed", moddrop=True, use_gate=False),
    "abl_no_xattn": dict(mode="proposed", moddrop=True, use_xattn=False),
    "abl_no_time": dict(mode="proposed", moddrop=True, use_time=False),
    "abl_no_mask": dict(mode="proposed", moddrop=True, use_mask=False),
    "abl_no_moddrop": dict(mode="proposed", moddrop=False),
    "sens_pdrop_015": dict(mode="proposed", moddrop=True, p_drop_mri=0.15),
    "sens_pdrop_050": dict(mode="proposed", moddrop=True, p_drop_mri=0.50),
}
CLASSICAL = {"e1_lr_clin": ("lr", False), "e1_gbm_clin": ("gbm", False),
             "e1_lr_clinmri": ("lr", True), "e1_gbm_clinmri": ("gbm", True)}

_cache = {}


def load_inputs():
    if not _cache:
        cohort = pd.read_csv(PROCESSED / "cohort.csv")
        arrays = dict(np.load(PROCESSED / "features.npz"))
        _cache.update(cohort=cohort, arrays=arrays, data=to_tensors(arrays))
    return _cache["cohort"], _cache["arrays"], _cache["data"]


def model_kwargs(spec, arrays, tcfg):
    spec = dict(spec)
    moddrop = spec.pop("moddrop", False)
    kw = dict(n_feat=arrays["x"].shape[2], n_static=arrays["static"].shape[1], n_roi=arrays["roi"].shape[1],
              n_rel=arrays["rel"].shape[1], hidden=tcfg["hidden_size"], dropout=tcfg["dropout"],
              heads=tcfg["attention_heads"])
    if moddrop:
        kw.update(p_drop_mri=tcfg["p_drop_mri"], p_drop_clinical=tcfg["p_drop_clinical"])
    kw.update(spec)
    return kw


def run_neural(job):
    model_id, seed, cfg = job
    torch.set_num_threads(1)
    cohort, arrays, data = load_inputs()
    tcfg = cfg["training"]
    y = cohort.label.to_numpy()
    split = cohort.split.to_numpy()
    kw = model_kwargs(NEURAL[model_id], arrays, tcfg)
    torch.manual_seed(seed)
    model = FusionModel(**kw)

    idx_tr, idx_va = np.where(split == "train")[0], np.where(split == "validation")[0]
    if kw["mode"] == "mri":                         # MRI-only model needs an MRI
        idx_tr = idx_tr[arrays["a_mri"][idx_tr] == 1]
        idx_va = idx_va[arrays["a_mri"][idx_va] == 1]
    model, info = train_neural(model, data, y, idx_tr, idx_va, tcfg, seed)

    test_idx = np.where(split == "test")[0]
    frames = []
    for cond in CONDITIONS:
        d = apply_condition(data, cond, noise_sd=cfg["evaluation"]["mri_noise_sd"])
        idx = np.arange(len(y)) if cond == "natural" else test_idx
        logit, alpha = predict(model, d, idx)
        if kw["mode"] == "mri" and cond in ("mri_removed",):
            logit = np.full_like(logit, np.nan)
        if kw["mode"] == "clinical" and cond == "clinical_removed":
            logit = np.full_like(logit, np.nan)
        frames.append(pd.DataFrame({"window_id": cohort.window_id.to_numpy()[idx], "seed": seed,
                                    "condition": cond, "logit": logit,
                                    "alpha_mri": alpha if alpha is not None else np.nan}))
    out_dir = MODELS / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"seed{seed}.pt")
    with open(out_dir / "model_kwargs.json", "w") as f:
        json.dump(kw, f, indent=1)
    info.update(parameters=count_parameters(model))
    return model_id, seed, pd.concat(frames), info, kw


def run_classical(cfg, cohort, arrays):
    tab = pd.read_csv(PROCESSED / "tabular.csv")
    meta = json.load(open(PROCESSED / "feature_meta.json"))
    mri_cols = [c for c in tab.columns if c not in meta["clinical_cols"]]
    mri_std = np.asarray(meta["scalers"]["mri_std"])
    split = cohort.split.to_numpy()
    y = cohort.label.to_numpy()
    tr = split == "train"
    test_idx = np.where(split == "test")[0]
    g = torch.Generator().manual_seed(12345)                    # same noise draw as the neural models
    noise = (cfg["evaluation"]["mri_noise_sd"] * torch.randn((len(tab), len(mri_cols)), generator=g)).numpy()

    for model_id, (kind, use_mri) in CLASSICAL.items():
        cols = meta["clinical_cols"] + (mri_cols if use_mri else [])
        model, info = fit_classical(kind, tab[cols].to_numpy(float)[tr], y[tr], cohort.RID.to_numpy()[tr],
                                    seed=cfg["seed"])
        variants = {"natural": tab.copy()}
        removed = tab.copy()
        removed[mri_cols] = np.nan
        noisy = tab.copy()
        noisy[mri_cols] = tab[mri_cols].to_numpy() + noise * mri_std
        variants.update(mri_removed=removed, mri_noisy_flagged=noisy, mri_noisy_silent=noisy)
        frames = []
        for cond, X in variants.items():
            idx = np.arange(len(y)) if cond == "natural" else test_idx
            logit = model.decision_function(X[cols].to_numpy(float)[idx])
            frames.append(pd.DataFrame({"window_id": cohort.window_id.to_numpy()[idx], "seed": 0,
                                        "condition": cond, "logit": logit, "alpha_mri": np.nan}))
        pd.concat(frames).to_csv(PREDICTIONS / f"{model_id}.csv.gz", index=False)
        with open(MODELS / f"{model_id}.pkl", "wb") as f:
            pickle.dump({"model": model, "columns": cols}, f)
        write_record(RUNS / "run_records.jsonl", model=model_id, seed=cfg["seed"], target="MCI->dementia 24m",
                     selection_criterion="grouped 5-fold CV AUROC on training split", **info)
        print(f"  {model_id:<16} CV AUROC {info['cv_auroc']:.3f}  params {info['best_params']}")


def main():
    cfg = load_config()
    ensure_dirs()
    (RUNS / "run_records.jsonl").unlink(missing_ok=True)
    cohort, arrays, _ = load_inputs()
    split_info = json.load(open(RUNS / "split_info.json"))

    print("E1 classical baselines")
    run_classical(cfg, cohort, arrays)

    jobs = [(m, s, cfg) for m in NEURAL for s in cfg["training"]["seeds"]]
    print(f"Training {len(jobs)} neural runs ({len(NEURAL)} models x {len(cfg['training']['seeds'])} seeds)")
    start = time.time()
    results = {}
    with mp.get_context("spawn").Pool(processes=max(1, min(4, mp.cpu_count()))) as pool:
        for model_id, seed, preds, info, kw in pool.imap_unordered(run_neural, jobs):
            results.setdefault(model_id, []).append(preds)
            write_record(RUNS / "run_records.jsonl", model=model_id, seed=seed, target="MCI->dementia 24m",
                         split_manifest=split_info["split_manifest_sha256_16"], hyperparameters=kw,
                         selection_criterion="validation AUROC (early stopping)", **info)
            print(f"  {model_id:<16} seed {seed}  val AUROC {info['best_val_auroc']:.3f}  "
                  f"epoch {info['best_epoch']:>2}  {info['train_seconds']:>5.1f}s")
    for model_id, frames in results.items():
        pd.concat(frames).sort_values(["seed", "condition", "window_id"]).to_csv(
            PREDICTIONS / f"{model_id}.csv.gz", index=False)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
