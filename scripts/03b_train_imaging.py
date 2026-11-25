"""Step 3b (secondary configuration): 3D CNN on MRI volumes, CNN-based fusion models and Grad-CAM.

1. Train the 3D CNN (MRI only, E3-CNN) on development training scans, one model per seed.
2. Freeze it and compute an embedding (64-d) and a 4x4x4 feature map (64 tokens) for every scan,
   for the original and for a degraded (noisier) version of each volume.
3. Train early fusion and the proposed model on top of the frozen CNN features (same seed).
4. Grad-CAM: where in the volume the CNN looks, with an anatomical-overlap check and a
   randomised-weights sanity check (Adebayo et al.).
"""
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from mdp.config import MODELS, PREDICTIONS, PROCESSED, RUNS, TABLES, load_config  # noqa: E402
from mdp.imaging import MRICNN, augment, grad_cam, normalise, roi_masks  # noqa: E402
from mdp.models import FusionModel, count_parameters  # noqa: E402
from mdp.records import write_record  # noqa: E402
from mdp.train import CONDITIONS, apply_condition, predict, to_tensors, train_neural  # noqa: E402

COHORTS = ["dev", "external_A", "external_B"]


def load_cohort(name):
    suffix = "" if name == "dev" else f"_{name}"
    cohort = pd.read_csv(PROCESSED / f"cohort{suffix}.csv")
    arrays = dict(np.load(PROCESSED / f"features{suffix}.npz"))
    return cohort, arrays


def load_volumes(name, idx=None):
    vols = np.load(PROCESSED / f"volumes_{name}.npy", mmap_mode="r")
    idx = np.arange(len(vols)) if idx is None else idx
    return torch.tensor(np.stack([normalise(vols[i]) for i in idx]))[:, None]


@torch.no_grad()
def cnn_forward(model, x, batch=128):
    model.eval()
    logits, embs, toks = [], [], []
    for s in range(0, len(x), batch):
        logit, emb, f2 = model(x[s:s + batch])
        logits.append(logit.numpy())
        embs.append(emb.numpy())
        toks.append(f2.flatten(2).transpose(1, 2).numpy())           # (B, 64 positions, 64 channels)
    return np.concatenate(logits), np.concatenate(embs), np.concatenate(toks)


def train_cnn(job):
    seed, cfg = job
    torch.set_num_threads(1)
    icfg = cfg["imaging_3d"]
    cohort, arrays = load_cohort("dev")
    split, has = cohort.split.to_numpy(), arrays["a_mri"] == 1
    tr, va = np.where((split == "train") & has)[0], np.where((split == "validation") & has)[0]
    x_tr, x_va = load_volumes("dev", tr), load_volumes("dev", va)
    y = torch.tensor(cohort.label.to_numpy(), dtype=torch.float32)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MRICNN()
    opt = torch.optim.AdamW(model.parameters(), lr=icfg["learning_rate"], weight_decay=icfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, wait, start = -np.inf, None, 0, time.time()
    for epoch in range(icfg["epochs"]):
        model.train()
        perm = torch.randperm(len(tr), generator=gen)
        for s in range(0, len(tr), icfg["batch_size"]):
            bi = perm[s:s + icfg["batch_size"]]
            logit, _, _ = model(augment(x_tr[bi], gen))
            loss = loss_fn(logit, y[tr][bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = roc_auc_score(cohort.label.to_numpy()[va], cnn_forward(model, x_va)[0])
        if auc > best + 1e-4:
            best, best_state, best_epoch, wait = auc, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            wait += 1
            if wait >= icfg["patience"]:
                break
    model.load_state_dict(best_state)
    (MODELS / "e3_cnn").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "e3_cnn" / f"seed{seed}.pt")
    train_auc = roc_auc_score(cohort.label.to_numpy()[tr], cnn_forward(model, x_tr)[0])
    return seed, dict(best_val_score=float(best), train_auroc=float(train_auc), best_epoch=best_epoch,
                      epochs_run=epoch + 1, train_seconds=round(time.time() - start, 1),
                      parameters=count_parameters(model))


def embed_all(seed, cfg):
    """CNN outputs for every window with an MRI, original and degraded volumes, all cohorts."""
    model = MRICNN()
    model.load_state_dict(torch.load(MODELS / "e3_cnn" / f"seed{seed}.pt"))
    out = {}
    for name in COHORTS:
        cohort, arrays = load_cohort(name)
        idx = np.where(arrays["a_mri"] == 1)[0]
        x = load_volumes(name, idx)
        g = torch.Generator().manual_seed(777)
        x_noisy = x + cfg["imaging_3d"]["degraded_noise_sd"] * torch.randn(x.shape, generator=g) * (x != 0)
        n = len(cohort)
        for tag, xx in (("", x), ("_noisy", x_noisy)):
            logit, emb, tok = cnn_forward(model, xx)
            L, E, T = np.full(n, np.nan, np.float32), np.zeros((n, 64), np.float32), np.zeros((n, 64, 64), np.float32)
            L[idx], E[idx], T[idx] = logit, emb, tok
            out[f"{name}_logit{tag}"], out[f"{name}_emb{tag}"], out[f"{name}_tok{tag}"] = L, E, T
    np.savez(PROCESSED / f"cnn_features_seed{seed}.npz", **out)
    return out


def e3_cnn_predictions(seed, feats, cohorts):
    frames = []
    for name in COHORTS:
        cohort = cohorts[name]
        test = np.where(cohort.split.to_numpy() == "test")[0] if name == "dev" else np.arange(len(cohort))
        for cond in CONDITIONS:
            idx = np.arange(len(cohort)) if cond == "natural" else test
            src = f"{name}_logit_noisy" if cond.startswith("mri_noisy") else f"{name}_logit"
            logit = feats[src][idx].copy()
            if cond == "mri_removed":
                logit[:] = np.nan
            frames.append(pd.DataFrame({"cohort": name, "window_id": cohort.window_id.to_numpy()[idx], "seed": seed,
                                        "condition": cond, "logit": logit, "alpha_mri": np.nan}))
    return pd.concat(frames)


def train_fusion(job):
    model_id, base, seed, cfg, overrides = job
    torch.set_num_threads(1)
    feats = dict(np.load(PROCESSED / f"cnn_features_seed{seed}.npz"))
    data, cohorts = {}, {}
    for name in COHORTS:
        cohorts[name], arrays = load_cohort(name)
        arrays.update(img_emb=feats[f"{name}_emb"], img_tokens=feats[f"{name}_tok"],
                      img_emb_noisy=feats[f"{name}_emb_noisy"], img_tokens_noisy=feats[f"{name}_tok_noisy"])
        data[name] = to_tensors(arrays)
        if name == "dev":
            dev_arrays = arrays
    tcfg = cfg["training"]
    mode = "proposed" if base == "e5_proposed" else "early"
    kw = dict(mode=mode, n_feat=dev_arrays["x"].shape[2], n_static=dev_arrays["static"].shape[1],
              n_roi=dev_arrays["roi"].shape[1], n_rel=dev_arrays["rel"].shape[1],
              hidden=overrides.get("hidden_size", tcfg["hidden_size"]), dropout=overrides.get("dropout", tcfg["dropout"]),
              heads=tcfg["attention_heads"], mri_kind="cnn")
    cons = 0.0
    if mode == "proposed":
        kw.update(p_drop_mri=tcfg["p_drop_mri"], p_drop_clinical=overrides.get("p_drop_clinical", tcfg["p_drop_clinical"]))
        cons = overrides.get("consistency_weight", 0.0)
    train_cfg = {**tcfg, "learning_rate": overrides.get("learning_rate", tcfg["learning_rate"])}
    dev = cohorts["dev"]
    split = dev.split.to_numpy()
    tr, va = np.where(split == "train")[0], np.where(split == "validation")[0]
    torch.manual_seed(seed)
    model, info = train_neural(FusionModel(**kw), data["dev"], dev.label.to_numpy(), tr, va, train_cfg, seed,
                               consistency=cons)
    frames = []
    for name in COHORTS:
        n = len(cohorts[name])
        test = np.where(split == "test")[0] if name == "dev" else np.arange(n)
        for cond in CONDITIONS:
            d = apply_condition(data[name], cond, noise_sd=cfg["evaluation"]["mri_noise_sd"])
            idx = np.arange(n) if cond == "natural" else test
            logit, alpha = predict(model, d, idx)
            frames.append(pd.DataFrame({"cohort": name, "window_id": cohorts[name].window_id.to_numpy()[idx],
                                        "seed": seed, "condition": cond, "logit": logit,
                                        "alpha_mri": alpha if alpha is not None else np.nan}))
    (MODELS / model_id).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / model_id / f"seed{seed}.pt")
    info.update(parameters=count_parameters(model), consistency_weight=cons, training=overrides)
    return model_id, seed, pd.concat(frames), info


def gradcam_analysis(cfg):
    """Anatomical overlap of Grad-CAM maps and a randomised-weights sanity check (seed 0 CNN)."""
    cohort, arrays = load_cohort("dev")
    idx = np.where((cohort.split.to_numpy() == "test") & (arrays["a_mri"] == 1))[0]
    x = load_volumes("dev", idx)
    trained = MRICNN()
    trained.load_state_dict(torch.load(MODELS / "e3_cnn" / "seed0.pt"))
    torch.manual_seed(123)
    random_model = MRICNN()
    hippo, vent, head = roi_masks()
    rows, cams_t, cams_r = [], [], []
    for s in range(0, len(x), 64):
        cams_t.append(grad_cam(trained, x[s:s + 64]))
        cams_r.append(grad_cam(random_model, x[s:s + 64]))
    cams_t, cams_r = np.concatenate(cams_t), np.concatenate(cams_r)
    for name, cams in (("Trained 3D CNN", cams_t), ("Randomised weights (sanity check)", cams_r)):
        mass = cams * head
        total = mass.sum(axis=(1, 2, 3)) + 1e-8
        rows.append({"Model": name, "Scans": len(idx),
                     "CAM mass in hippocampal/temporal region": f"{np.mean((mass * hippo).sum(axis=(1, 2, 3)) / total):.3f}",
                     "CAM mass in ventricular region": f"{np.mean((mass * vent).sum(axis=(1, 2, 3)) / total):.3f}",
                     "Share of head volume (hippocampal + ventricular)": f"{(hippo | vent)[head].mean():.3f}",
                     "CAM mass in hippocampal + ventricular regions": f"{np.mean((mass * (hippo | vent)).sum(axis=(1, 2, 3)) / total):.3f}"})
    rho = [spearmanr(a.ravel(), b.ravel()).statistic for a, b in zip(cams_t[:200], cams_r[:200])]
    rows.append({"Model": "Trained vs randomised maps, Spearman rho (median)", "Scans": min(200, len(idx)),
                 "CAM mass in hippocampal/temporal region": f"{np.nanmedian(rho):.3f}"})
    pd.DataFrame(rows).to_csv(TABLES / "table_gradcam.csv", index=False)
    frac = ((cams_t * head) * (hippo | vent)).sum(axis=(1, 2, 3)) / ((cams_t * head).sum(axis=(1, 2, 3)) + 1e-8)
    pick = [int(np.argmax(frac)), int(np.argmin(frac))]                # most and least plausible map
    vols = np.load(PROCESSED / "volumes_dev.npy", mmap_mode="r")
    np.savez_compressed(TABLES / "data_gradcam_examples.npz", volumes=np.stack([vols[idx[i]] for i in pick]),
                        cams=cams_t[pick].astype(np.float16), random_cams=cams_r[pick].astype(np.float16),
                        mean_cam=cams_t.mean(0).astype(np.float16), hippo=hippo, vent=vent,
                        fraction=frac[pick], labels=cohort.label.to_numpy()[idx[pick]])
    return rows


def main():
    cfg = load_config()
    seeds = cfg["training"]["seeds"]
    selected = json.load(open(RUNS / "selected_configurations.json"))
    start = time.time()
    print(f"Training the 3D CNN ({len(seeds)} seeds)")
    with mp.get_context("spawn").Pool(processes=max(1, min(4, mp.cpu_count()))) as pool:
        for seed, info in pool.imap_unordered(train_cnn, [(s, cfg) for s in seeds]):
            write_record(RUNS / "run_records.jsonl", model="e3_cnn", seed=seed, target="MCI->dementia 24m",
                         selection_criterion="validation AUROC (early stopping)", **info)
            print(f"  e3_cnn seed {seed}: val AUROC {info['best_val_score']:.3f}, train AUROC {info['train_auroc']:.3f},"
                  f" epoch {info['best_epoch']}, {info['train_seconds']}s")

    torch.set_num_threads(4)
    cohorts = {n: load_cohort(n)[0] for n in COHORTS}
    frames = []
    for s in seeds:
        frames.append(e3_cnn_predictions(s, embed_all(s, cfg), cohorts))
    pd.concat(frames).to_csv(PREDICTIONS / "e3_cnn.csv.gz", index=False)
    print(f"Embedded all scans ({time.time() - start:.0f}s)")

    jobs = [(m, base, s, cfg, selected[base]) for m, base in (("e4_early_cnn", "e4_early"),
                                                              ("e5_proposed_cnn", "e5_proposed")) for s in seeds]
    results = {}
    with mp.get_context("spawn").Pool(processes=max(1, min(4, mp.cpu_count()))) as pool:
        for model_id, seed, preds, info in pool.imap_unordered(train_fusion, jobs):
            results.setdefault(model_id, []).append(preds)
            write_record(RUNS / "run_records.jsonl", model=model_id, seed=seed, target="MCI->dementia 24m",
                         selection_criterion="validation AUROC (early stopping)", **info)
            print(f"  {model_id:<16} seed {seed}  val AUROC {info['best_val_score']:.3f}")
    for model_id, fr in results.items():
        pd.concat(fr).to_csv(PREDICTIONS / f"{model_id}.csv.gz", index=False)

    torch.set_num_threads(4)
    for r in gradcam_analysis(cfg):
        print("  Grad-CAM:", r)
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
