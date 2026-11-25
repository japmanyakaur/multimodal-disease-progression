"""Explanations and their stability.

Neural models : Integrated Gradients on the clinical values, static covariates and MRI measures
                (baseline = the training mean, i.e. 0 after standardisation).
Tree model    : SHAP TreeExplainer.
Stability     : Spearman rank correlation and top-k overlap J_k (Eq. 7.5) between two attribution vectors.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from .features import MRI_NAMES, SCORE_NAMES, STATIC_NAMES
from .train import subset

FEATURE_NAMES = SCORE_NAMES + STATIC_NAMES[:4] + MRI_NAMES     # APOE4-missing flag folded into APOE4


def integrated_gradients(model, data, idx, steps=32, batch_size=256):
    """Returns per-window attributions: x (n, L, F), static (n, S), roi (n, R)."""
    model.eval()
    keys = ("x", "static", "roi")
    out = {k: [] for k in keys}
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps        # midpoint Riemann sum
    for s in range(0, len(idx), batch_size):
        b = subset(data, idx[s:s + batch_size])
        grads = {k: torch.zeros_like(b[k]) for k in keys}
        for a in alphas:
            bb = dict(b)
            for k in keys:
                bb[k] = (a * b[k]).detach().requires_grad_(True)
            logit, _ = model(bb)
            g = torch.autograd.grad(logit.sum(), [bb[k] for k in keys])
            for k, gk in zip(keys, g):
                grads[k] += gk / steps
        for k in keys:
            out[k].append((grads[k] * b[k]).detach().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def feature_vector(attr):
    """Collapse IG attributions to one value per named feature (scores summed over visits)."""
    scores = attr["x"].sum(axis=1)                                         # (n, 4)
    static = attr["static"][:, :4].copy()
    static[:, 3] += attr["static"][:, 4]                                   # APOE4 + its missing flag
    return np.concatenate([scores, static, attr["roi"]], axis=1)


def temporal_profile(attr, tti, lengths, bins=(0.25, 0.75, 1.25, 1.75, 2.5, 100)):
    """Mean |attribution| of the clinical history by years before the index visit."""
    per_step = np.abs(attr["x"]).sum(axis=2)                               # (n, L)
    valid = np.arange(per_step.shape[1])[None, :] < lengths[:, None]
    years = tti[..., 0]
    edges = (-1e-9,) + tuple(bins)
    labels = ["index visit", "~6 mo before", "~12 mo before", "~18 mo before", "~24 mo before", ">2.5 y before"]
    rows = []
    for k in range(len(edges) - 1):
        sel = valid & (years > edges[k]) & (years <= edges[k + 1])
        rows.append((labels[k], int(sel.sum()), float(per_step[sel].mean()) if sel.any() else np.nan))
    return rows


def topk_overlap(e1, e2, k):
    a = set(np.argsort(-np.abs(e1))[:k])
    b = set(np.argsort(-np.abs(e2))[:k])
    return len(a & b) / len(a | b)


def stability(e1, e2, k):
    """Per-window Spearman correlation of |attributions| and top-k Jaccard overlap."""
    rho, jac = [], []
    for u, v in zip(e1, e2):
        if np.allclose(u, 0) or np.allclose(v, 0) or np.std(np.abs(u)) == 0 or np.std(np.abs(v)) == 0:
            continue
        rho.append(spearmanr(np.abs(u), np.abs(v)).statistic)
        jac.append(topk_overlap(u, v, k))
    return np.array(rho), np.array(jac)
