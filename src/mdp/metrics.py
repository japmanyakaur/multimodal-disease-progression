"""Metrics, participant-level bootstrap, temperature scaling and conformal prediction sets."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def auroc(y, s):
    """Mann-Whitney form of the area under the ROC curve (handles ties)."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auprc(y, s):
    return average_precision_score(y, s) if 0 < np.sum(y) < len(y) else np.nan


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))        # Eq. 7.2


def ece(y, p, bins=10):
    """Expected calibration error with equal-width bins on the predicted probability (Eq. 7.3)."""
    y, p = np.asarray(y), np.asarray(p)
    which = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(total)


def reliability_bins(y, p, bins=10):
    which = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        sel = which == b
        if sel.any():
            rows.append((b, sel.sum(), float(np.mean(p[sel])), float(np.mean(y[sel]))))
    return rows


def threshold_metrics(y, p, thr):
    y, pred = np.asarray(y), np.asarray(p) >= thr
    tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
    fn, tn = np.sum(~pred & (y == 1)), np.sum(~pred & (y == 0))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def fit_temperature(logits, y):
    """Temperature scaling (Eq. 7.1): T > 0 minimising negative log-likelihood."""
    z, y = np.asarray(logits, float), np.asarray(y, float)

    def nll(log_t):
        p = np.clip(sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def conformal_sets(p_cal, y_cal, p_test, alpha):
    """Split conformal prediction sets (LAC score = 1 - probability of the true class)."""
    scores = 1 - np.where(np.asarray(y_cal) == 1, p_cal, 1 - np.asarray(p_cal))
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = float(np.quantile(scores, level, method="higher"))
    has_1 = (1 - np.asarray(p_test)) <= q
    has_0 = np.asarray(p_test) <= q
    return has_0, has_1, q


class ClusterBootstrap:
    """Resample participants (not windows) with replacement; all windows of a participant move together."""

    def __init__(self, groups, n_boot=2000, seed=0):
        uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.where(inv == g)[0] for g in range(len(uniq))]
        rng = np.random.default_rng(seed)
        self.samples = [np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
                        for _ in range(n_boot)]

    def ci(self, fn, *arrays):
        point = fn(*arrays)
        reps = np.array([fn(*(a[s] for a in arrays)) for s in self.samples])
        reps = reps[~np.isnan(reps)]
        return float(point), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))

    def paired_diff(self, fn, arrays_a, arrays_b):
        """CI for fn(model A) - fn(model B) on the same resampled participants."""
        point = fn(*arrays_a) - fn(*arrays_b)
        reps = np.array([fn(*(a[s] for a in arrays_a)) - fn(*(b[s] for b in arrays_b)) for s in self.samples])
        reps = reps[~np.isnan(reps)]
        return float(point), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def fmt_ci(point, lo, hi, digits=3):
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
