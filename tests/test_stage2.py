"""Tests for Stage 2 components: Mondrian conformal sets, C-index, regression intervals, 3D CNN and Grad-CAM,
and training with the consistency loss."""
import numpy as np
import torch

from mdp import metrics as M
from mdp.imaging import MRICNN, grad_cam, normalise, render
from mdp.models import FusionModel
from mdp.train import train_neural


def test_mondrian_gives_per_class_coverage():
    rng = np.random.default_rng(0)
    y = (rng.random(6000) < 0.2).astype(int)                       # imbalanced, like the cohort
    p = np.clip(0.2 + 0.35 * (y - 0.2) + rng.normal(0, 0.18, 6000), 0.01, 0.99)
    has0, has1, _ = M.mondrian_sets(p[:3000], y[:3000], p[3000:], alpha=0.1)
    yt = y[3000:]
    covered = np.where(yt == 1, has1, has0)
    assert covered[yt == 1].mean() >= 0.87 and covered[yt == 0].mean() >= 0.87


def test_c_index_perfect_and_reversed():
    t = np.array([1, 2, 3, 4, 5.0])
    e = np.ones(5, int)
    assert M.c_index(t, e, -t) == 1.0
    assert M.c_index(t, e, t) == 0.0
    assert M.c_index(t, e, np.zeros(5)) == 0.5


def test_regression_interval_coverage():
    rng = np.random.default_rng(1)
    resid = rng.normal(0, 1, 4000)
    q = M.regression_interval(resid[:2000], alpha=0.1)
    assert 0.88 <= np.mean(np.abs(resid[2000:]) <= q) <= 0.93
    y = rng.normal(size=500)
    assert M.mae(y, y) == 0 and M.rmse(y, y) == 0 and np.isclose(M.r2(y, y), 1.0)


def test_render_cnn_and_gradcam_shapes():
    vol = render(7000, 3800, 30000, 1.05e6, 1.45e6, qc=2, seed=0)
    assert vol.shape == (32, 32, 32) and np.isfinite(vol.astype(float)).all()
    x = torch.tensor(np.stack([normalise(vol)] * 2))[:, None]
    model = MRICNN().eval()
    logit, emb, fmap = model(x)
    assert logit.shape == (2,) and emb.shape == (2, 64) and fmap.shape == (2, 64, 4, 4, 4)
    cam = grad_cam(model, x)
    assert cam.shape == (2, 32, 32, 32) and cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_training_with_consistency_loss_runs():
    g = torch.Generator().manual_seed(0)
    n, L = 64, 4
    data = {"x": torch.randn(n, L, 4, generator=g), "m": torch.ones(n, L, 4), "dt": torch.zeros(n, L, 4),
            "gap": torch.zeros(n, L, 1), "tti": torch.zeros(n, L, 1), "lengths": torch.full((n,), L),
            "static": torch.randn(n, 5, generator=g), "roi": torch.randn(n, 5, generator=g),
            "a_clin": torch.ones(n), "a_mri": torch.ones(n), "rel": torch.rand(n, 8, generator=g)}
    y = (data["x"][:, -1, 0] > 0).float().numpy()
    model = FusionModel("proposed", 4, 5, 5, 8, hidden=8, p_drop_mri=0.3, p_drop_clinical=0.3)
    cfg = {"learning_rate": 0.01, "weight_decay": 0.0, "max_epochs": 2, "patience": 2, "batch_size": 16}
    model, info = train_neural(model, data, y, np.arange(48), np.arange(48, 64), cfg, seed=0, consistency=0.5)
    assert np.isfinite(info["best_val_score"])
