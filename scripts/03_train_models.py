"""Step 3: equal-budget hyper-parameter search, then train every model with 5 seeds and save predictions.

Search : 8 configurations per neural model, scored by mean validation AUROC over the search seeds.
Final  : selected configuration x seeds 0-4. Ablations and sensitivity runs reuse the proposed model's
         selected configuration with one component changed.
Predictions are saved for the development cohort (all windows; stress-test conditions on the test split)
and for every window of external cohorts A and B. Nothing here looks at test or external labels.
"""
import itertools
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

from mdp.config import MODELS, PREDICTIONS, PROCESSED, RUNS, TABLES, ensure_dirs, load_config  # noqa: E402
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
    "abl_no_consistency": dict(mode="proposed", moddrop=True, no_consistency=True),
    "sens_pdrop_015": dict(mode="proposed", moddrop=True, p_drop_mri=0.15),
    "sens_pdrop_050": dict(mode="proposed", moddrop=True, p_drop_mri=0.50),
}
SEARCHED = {"e2_clinical": "grid_baselines", "e3_mri": "grid_baselines", "e4_early": "grid_baselines",
            "e4_xattn": "grid_baselines", "e5_proposed": "grid_proposed"}
REGRESSION = {"e2_clinical_reg": "e2_clinical", "e5_proposed_reg": "e5_proposed"}
CLASSICAL = {"e1_lr_clin": ("lr", False), "e1_gbm_clin": ("gbm", False),
             "e1_lr_clinmri": ("lr", True), "e1_gbm_clinmri": ("gbm", True)}
CLASSICAL_REG = {"e1_ridge_reg": "lr", "e1_gbm_reg": "gbm"}

_cache = {}


def load_inputs(cfg):
    if not _cache:
        cohorts = {"dev": pd.read_csv(PROCESSED / "cohort.csv")}
        arrays = {"dev": dict(np.load(PROCESSED / "features.npz"))}
        for name in cfg["data"].get("external", {}):
            if (PROCESSED / f"features_{name}.npz").exists():
                cohorts[name] = pd.read_csv(PROCESSED / f"cohort_{name}.csv")
                arrays[name] = dict(np.load(PROCESSED / f"features_{name}.npz"))
        _cache.update(cohorts=cohorts, arrays=arrays, data={k: to_tensors(v) for k, v in arrays.items()})
    return _cache["cohorts"], _cache["arrays"], _cache["data"]


def grid(cfg, name):
    g = cfg["training"][name]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def build(model_id, overrides, arrays, cfg):
    """Model constructor arguments, training settings and consistency weight for one run."""
    tcfg = cfg["training"]
    spec = dict(NEURAL[model_id])
    moddrop = spec.pop("moddrop", False)
    no_cons = spec.pop("no_consistency", False)
    p_mri = spec.pop("p_drop_mri", overrides.get("p_drop_mri", tcfg["p_drop_mri"]))
    kw = dict(n_feat=arrays["x"].shape[2], n_static=arrays["static"].shape[1], n_roi=arrays["roi"].shape[1],
              n_rel=arrays["rel"].shape[1], hidden=overrides.get("hidden_size", tcfg["hidden_size"]),
              dropout=overrides.get("dropout", tcfg["dropout"]), heads=tcfg["attention_heads"], **spec)
    if moddrop:
        kw.update(p_drop_mri=p_mri, p_drop_clinical=overrides.get("p_drop_clinical", tcfg["p_drop_clinical"]))
    train_cfg = {**tcfg, "learning_rate": overrides.get("learning_rate", tcfg["learning_rate"])}
    consistency = 0.0 if (not moddrop or no_cons) else overrides.get("consistency_weight", tcfg["consistency_weight"])
    return kw, train_cfg, consistency


def split_idx(cohort, arrays, mode):
    split = cohort.split.to_numpy()
    tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
    if mode == "mri":                                   # the MRI-only model needs an MRI
        tr, va = tr[arrays["a_mri"][tr] == 1], va[arrays["a_mri"][va] == 1]
    return tr, va


def run_search(job):
    model_id, overrides, seed, cfg = job
    torch.set_num_threads(1)
    cohorts, arrays, data = load_inputs(cfg)
    kw, tcfg, cons = build(model_id, overrides, arrays["dev"], cfg)
    tr, va = split_idx(cohorts["dev"], arrays["dev"], kw["mode"])
    torch.manual_seed(seed)
    _, info = train_neural(FusionModel(**kw), data["dev"], cohorts["dev"].label.to_numpy(), tr, va, tcfg, seed,
                           consistency=cons)
    return model_id, json.dumps(overrides, sort_keys=True), seed, info["best_val_score"]


def run_final(job):
    model_id, seed, cfg, overrides, task = job
    torch.set_num_threads(1)
    cohorts, arrays, data = load_inputs(cfg)
    base = REGRESSION.get(model_id, model_id)
    kw, tcfg, cons = build(base, overrides, arrays["dev"], cfg)
    dev = cohorts["dev"]
    tr, va = split_idx(dev, arrays["dev"], kw["mode"])
    if task == "regression":
        target = dev.cdrsb_target.to_numpy()
        tr, va = tr[~np.isnan(target[tr])], va[~np.isnan(target[va])]
        mu, sd = target[tr].mean(), target[tr].std()
        y = np.nan_to_num((target - mu) / sd)
    else:
        y = dev.label.to_numpy()
    torch.manual_seed(seed)
    model, info = train_neural(FusionModel(**kw), data["dev"], y, tr, va, tcfg, seed, task=task, consistency=cons)

    frames = []
    for name, d_all in data.items():
        n = len(cohorts[name])
        all_idx = np.arange(n)
        eval_idx = np.where(dev.split.to_numpy() == "test")[0] if name == "dev" else all_idx
        conds = ["natural"] if task == "regression" else CONDITIONS
        for cond in conds:
            d = apply_condition(d_all, cond, noise_sd=cfg["evaluation"]["mri_noise_sd"])
            idx = all_idx if cond == "natural" else eval_idx
            out, alpha = predict(model, d, idx)
            if (kw["mode"] == "mri" and cond == "mri_removed") or (kw["mode"] == "clinical" and cond == "clinical_removed"):
                out = np.full_like(out, np.nan)
            frame = pd.DataFrame({"cohort": name, "window_id": cohorts[name].window_id.to_numpy()[idx], "seed": seed,
                                  "condition": cond})
            if task == "regression":
                frame["pred"] = out * sd + mu
            else:
                frame["logit"], frame["alpha_mri"] = out, (alpha if alpha is not None else np.nan)
            frames.append(frame)
    out_dir = MODELS / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"seed{seed}.pt")
    with open(out_dir / "model_kwargs.json", "w") as f:
        json.dump(kw, f, indent=1)
    info.update(parameters=count_parameters(model), consistency_weight=cons, training=overrides)
    return model_id, seed, pd.concat(frames), info, kw


def run_classical(cfg, cohorts):
    meta = json.load(open(PROCESSED / "feature_meta.json"))
    tabs = {"dev": pd.read_csv(PROCESSED / "tabular.csv")}
    tabs.update({n: pd.read_csv(PROCESSED / f"tabular_{n}.csv") for n in cohorts if n != "dev"})
    clinical_cols = meta["clinical_cols"]
    mri_cols = [c for c in tabs["dev"].columns if c not in clinical_cols]
    mri_std = np.asarray(meta["scalers"]["mri_std"])
    dev = cohorts["dev"]
    split = dev.split.to_numpy()
    tr = split == "train"
    groups = dev.RID.to_numpy()

    def variants(name):
        tab = tabs[name]
        g = torch.Generator().manual_seed(12345)                # same noise draw as the neural models
        noise = (cfg["evaluation"]["mri_noise_sd"] * torch.randn((len(tab), len(mri_cols)), generator=g)).numpy()
        removed, noisy = tab.copy(), tab.copy()
        removed[mri_cols] = np.nan
        noisy[mri_cols] = tab[mri_cols].to_numpy() + noise * mri_std
        return {"natural": tab, "mri_removed": removed, "mri_noisy_flagged": noisy, "mri_noisy_silent": noisy}

    for model_id, (kind, use_mri) in CLASSICAL.items():
        cols = clinical_cols + (mri_cols if use_mri else [])
        y = dev.label.to_numpy()
        model, info = fit_classical(kind, tabs["dev"][cols].to_numpy(float)[tr], y[tr], groups[tr], seed=cfg["seed"])
        frames = []
        for name in cohorts:
            eval_idx = np.where(split == "test")[0] if name == "dev" else np.arange(len(cohorts[name]))
            for cond, X in variants(name).items():
                idx = np.arange(len(cohorts[name])) if cond == "natural" else eval_idx
                frames.append(pd.DataFrame({"cohort": name, "window_id": cohorts[name].window_id.to_numpy()[idx],
                                            "seed": 0, "condition": cond,
                                            "logit": model.decision_function(X[cols].to_numpy(float)[idx]),
                                            "alpha_mri": np.nan}))
        pd.concat(frames).to_csv(PREDICTIONS / f"{model_id}.csv.gz", index=False)
        with open(MODELS / f"{model_id}.pkl", "wb") as f:
            pickle.dump({"model": model, "columns": cols}, f)
        write_record(RUNS / "run_records.jsonl", model=model_id, seed=cfg["seed"], target="MCI->dementia 24m",
                     selection_criterion="grouped 5-fold CV AUROC on training split", **info)
        print(f"  {model_id:<16} CV AUROC {info['cv_score']:.3f}  params {info['best_params']}")

    target = dev.cdrsb_target.to_numpy()
    ok = tr & ~np.isnan(target)
    cols = clinical_cols + mri_cols
    for model_id, kind in CLASSICAL_REG.items():
        model, info = fit_classical(kind, tabs["dev"][cols].to_numpy(float)[ok], target[ok], groups[ok],
                                    seed=cfg["seed"], task="regression")
        frames = [pd.DataFrame({"cohort": name, "window_id": cohorts[name].window_id.to_numpy(), "seed": 0,
                                "condition": "natural", "pred": model.predict(tabs[name][cols].to_numpy(float))})
                  for name in cohorts]
        pd.concat(frames).to_csv(PREDICTIONS / f"{model_id}.csv.gz", index=False)
        write_record(RUNS / "run_records.jsonl", model=model_id, seed=cfg["seed"], target="CDR-SB at 24m",
                     selection_criterion="grouped 5-fold CV MAE on training split", **info)
        print(f"  {model_id:<16} CV MAE {-info['cv_score']:.3f}  params {info['best_params']}")


def run_pool(fn, jobs):
    with mp.get_context("spawn").Pool(processes=max(1, min(4, mp.cpu_count()))) as pool:
        yield from pool.imap_unordered(fn, jobs)


def main():
    cfg = load_config()
    ensure_dirs()
    (RUNS / "run_records.jsonl").unlink(missing_ok=True)
    cohorts, _, _ = load_inputs(cfg)
    split_info = json.load(open(RUNS / "split_info.json"))
    start = time.time()

    print("E1 classical baselines (classification and CDR-SB regression)")
    run_classical(cfg, cohorts)

    jobs = [(m, cfgd, s, cfg) for m, g in SEARCHED.items() for cfgd in grid(cfg, g)
            for s in cfg["training"]["search_seeds"]]
    print(f"Hyper-parameter search: {len(jobs)} runs ({len(SEARCHED)} models x 8 configurations x "
          f"{len(cfg['training']['search_seeds'])} seeds)")
    rows = list(run_pool(run_search, jobs))
    search = pd.DataFrame(rows, columns=["model", "configuration", "seed", "val_auroc"])
    summ = search.groupby(["model", "configuration"]).val_auroc.mean().rename("mean_val_auroc").reset_index()
    summ["selected"] = summ.groupby("model").mean_val_auroc.transform("max") == summ.mean_val_auroc
    summ = summ.sort_values(["model", "mean_val_auroc"], ascending=[True, False])
    summ = summ[~summ.duplicated("model") | ~summ.selected]          # keep one winner if tied
    summ.to_csv(TABLES / "table_hparam_search.csv", index=False)
    selected = {m: json.loads(c) for m, c in summ[summ.selected][["model", "configuration"]].values}
    print(summ[summ.selected].to_string(index=False))
    with open(RUNS / "selected_configurations.json", "w") as f:
        json.dump(selected, f, indent=1)

    final = []
    for m in NEURAL:
        if m == "abl_no_consistency" and selected["e5_proposed"].get("consistency_weight", 0) == 0:
            continue                                                    # nothing to remove
        overrides = selected.get(m, selected["e5_proposed"])
        final += [(m, s, cfg, overrides, "classification") for s in cfg["training"]["seeds"]]
    for m, base in REGRESSION.items():
        final += [(m, s, cfg, selected[base], "regression") for s in cfg["training"]["seeds"]]
    print(f"Final training: {len(final)} runs")
    results = {}
    for model_id, seed, preds, info, kw in run_pool(run_final, final):
        results.setdefault(model_id, []).append(preds)
        write_record(RUNS / "run_records.jsonl", model=model_id, seed=seed,
                     target="CDR-SB at 24m" if model_id.endswith("_reg") else "MCI->dementia 24m",
                     split_manifest=split_info["split_manifest_sha256_16"], hyperparameters=kw,
                     selection_criterion="validation score (early stopping)", **info)
        print(f"  {model_id:<20} seed {seed}  val score {info['best_val_score']:.3f}  epoch {info['best_epoch']:>2}"
              f"  {info['train_seconds']:>5.1f}s")
    for model_id, frames in results.items():
        pd.concat(frames).sort_values(["cohort", "seed", "condition", "window_id"]).to_csv(
            PREDICTIONS / f"{model_id}.csv.gz", index=False)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
