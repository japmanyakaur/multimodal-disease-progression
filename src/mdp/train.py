"""Training loops, missing-modality input conditions and classical baselines."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import DT_CAP

KEYS = ["x", "m", "dt", "gap", "tti", "lengths", "static", "roi", "a_clin", "a_mri", "rel"]
CNN_KEYS = ["img_emb", "img_tokens", "img_emb_noisy", "img_tokens_noisy"]
CONDITIONS = ["natural", "mri_removed", "clinical_removed", "mri_noisy_flagged", "mri_noisy_silent"]


def to_tensors(arrays):
    out = {k: torch.as_tensor(np.asarray(arrays[k]), dtype=torch.long if k == "lengths" else torch.float32)
           for k in KEYS}
    for k in CNN_KEYS:
        if k in arrays:
            out[k] = torch.as_tensor(np.asarray(arrays[k]), dtype=torch.float32)
    return out


def subset(data, idx):
    return {k: v[idx] for k, v in data.items()}


def apply_condition(data, condition, noise_sd=1.0, seed=12345):
    """Return a copy of the inputs with a modality removed or degraded (stress tests, E6)."""
    d = {k: v.clone() for k, v in data.items()}
    if condition == "natural":
        return d
    if condition == "mri_removed":
        d["roi"].zero_()
        d["a_mri"].zero_()
        d["rel"][:, 1:6] = 0.0
        for k in ("img_emb", "img_tokens"):
            if k in d:
                d[k].zero_()
    elif condition == "clinical_removed":
        for k in ("x", "m", "gap", "tti", "static"):
            d[k].zero_()
        d["dt"].zero_()
        d["dt"][:, 0] = float(np.log1p(DT_CAP))
        d["lengths"].fill_(1)
        d["a_clin"].zero_()
        d["rel"][:, [0, 6, 7]] = 0.0
    elif condition in ("mri_noisy_flagged", "mri_noisy_silent"):
        g = torch.Generator().manual_seed(seed)
        has = d["a_mri"][:, None]
        d["roi"] += noise_sd * torch.randn(d["roi"].shape, generator=g) * has
        for k in ("img_emb", "img_tokens"):                # CNN models: embeddings of the noisy volumes
            if k in d and f"{k}_noisy" in d:
                d[k] = d[f"{k}_noisy"].clone()
        if condition == "mri_noisy_flagged":               # quality control would mark the scan as poor
            d["rel"][:, 2:5] = torch.tensor([0.0, 0.0, 1.0]) * has
    else:
        raise ValueError(condition)
    return d


@torch.no_grad()
def predict(model, data, idx=None, batch_size=1024):
    model.eval()
    n = len(data["a_mri"]) if idx is None else len(idx)
    idx = np.arange(n) if idx is None else idx
    logits, alphas = [], []
    for s in range(0, n, batch_size):
        logit, extras = model(subset(data, idx[s:s + batch_size]))
        logits.append(logit.numpy())
        if "alpha" in extras:
            alphas.append(extras["alpha"][:, 1].numpy())
    return np.concatenate(logits), (np.concatenate(alphas) if alphas else None)


def train_neural(model, data, y, idx_train, idx_val, tcfg, seed, task="classification", consistency=0.0):
    """AdamW with early stopping on validation AUROC (classification) or MAE (regression).

    consistency > 0 adds lambda * L_cons (Eq. 6.6): the prediction made with modality dropout is pulled
    towards the prediction made from the full input of the same participant.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    classification = task == "classification"
    loss_fn = nn.BCEWithLogitsLoss() if classification else nn.HuberLoss(delta=1.0)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    best, best_state, best_epoch, wait = -np.inf, None, 0, 0
    start = time.time()
    for epoch in range(tcfg["max_epochs"]):
        model.train()
        perm = rng.permutation(idx_train)
        for s in range(0, len(perm), tcfg["batch_size"]):
            b = subset(data, perm[s:s + tcfg["batch_size"]])
            target = yt[perm[s:s + tcfg["batch_size"]]]
            out, _ = model(b)
            loss = loss_fn(out, target)
            if consistency > 0:
                full, _ = model(b, moddrop=False)
                cons = (Fnn.binary_cross_entropy_with_logits(out, torch.sigmoid(full).detach()) if classification
                        else Fnn.mse_loss(out, full.detach()))
                loss = loss + consistency * cons
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_pred = predict(model, data, idx_val)[0]
        score = roc_auc_score(y[idx_val], val_pred) if classification else -mean_absolute_error(y[idx_val], val_pred)
        if score > best + 1e-4:
            best, best_state, best_epoch, wait = score, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, dict(best_val_score=float(best), best_epoch=best_epoch, epochs_run=epoch + 1,
                       train_seconds=round(time.time() - start, 1))


# --------------------------------------------------------------------------- classical baselines (E1)
def _make_classical(kind, params, seed, task):
    if kind == "lr":
        est = (LogisticRegression(C=params["C"], max_iter=5000) if task == "classification"
               else Ridge(alpha=params["alpha"]))
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)
    cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
    return cls(max_iter=200, early_stopping=False, min_samples_leaf=20, l2_regularization=1.0,
               random_state=seed, **params)


def _score(model, X, y, task):
    if task == "classification":
        return roc_auc_score(y, model.decision_function(X))
    return -mean_absolute_error(y, model.predict(X))


def fit_classical(kind, X, y, groups, seed=0, task="classification"):
    """Pick hyper-parameters by grouped 5-fold CV on the training split, then refit."""
    if kind == "lr":
        grid = ([{"C": c} for c in (0.01, 0.1, 1.0, 10.0)] if task == "classification"
                else [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)])
    else:
        grid = [{"learning_rate": lr, "max_leaf_nodes": n} for lr in (0.05, 0.1) for n in (8, 31)]
    cv = GroupKFold(n_splits=5)
    scores = []
    for params in grid:
        fold = [_score(_make_classical(kind, params, seed, task).fit(X[tr], y[tr]), X[te], y[te], task)
                for tr, te in cv.split(X, y, groups)]
        scores.append(np.mean(fold))
    best = grid[int(np.argmax(scores))]
    start = time.time()
    model = _make_classical(kind, best, seed, task).fit(X, y)
    return model, dict(best_params=best, cv_score=float(max(scores)), train_seconds=round(time.time() - start, 2))
