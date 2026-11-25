"""Model and metric tests: output shapes, valid probabilities, availability masking, determinism."""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from mdp import metrics as M
from mdp.models import FusionModel
from mdp.train import apply_condition

N, L, F, S, R, REL = 16, 5, 4, 5, 5, 8


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, L + 1, (N,), generator=g)
    a_mri = (torch.rand(N, generator=g) > 0.3).float()
    rel = torch.rand(N, REL, generator=g)
    rel[:, 1] = a_mri
    return {"x": torch.randn(N, L, F, generator=g), "m": torch.ones(N, L, F), "dt": torch.rand(N, L, F),
            "gap": torch.rand(N, L, 1), "tti": torch.rand(N, L, 1), "lengths": lengths,
            "static": torch.randn(N, S, generator=g), "roi": torch.randn(N, R, generator=g) * a_mri[:, None],
            "a_clin": torch.ones(N), "a_mri": a_mri, "rel": rel}


def make(mode, **kw):
    torch.manual_seed(0)
    return FusionModel(mode, F, S, R, REL, hidden=16, **kw)


@pytest.mark.parametrize("mode", ["clinical", "mri", "early", "xattn", "proposed"])
def test_output_shape_and_probability_range(mode, batch):
    model = make(mode).eval()
    logit, _ = model(batch)
    assert logit.shape == (N,)
    p = torch.sigmoid(logit)
    assert torch.all((p >= 0) & (p <= 1)) and torch.isfinite(logit).all()


def test_gate_gives_zero_weight_to_missing_mri(batch):
    model = make("proposed").eval()
    _, extras = model(batch)
    alpha = extras["alpha"]
    assert torch.allclose(alpha.sum(1), torch.ones(N), atol=1e-6)
    assert torch.all(alpha[batch["a_mri"] == 0, 1] == 0)


def test_removed_mri_has_no_effect_on_proposed(batch):
    """With the MRI masked, changing the MRI values must not change the forecast."""
    model = make("proposed").eval()
    d = apply_condition(batch, "mri_removed")
    d2 = {k: v.clone() for k, v in d.items()}
    d2["roi"] = torch.randn_like(d2["roi"]) * 10
    assert torch.allclose(model(d)[0], model(d2)[0], atol=1e-6)


def test_deterministic_inference(batch):
    model = make("proposed", p_drop_mri=0.3, p_drop_clinical=0.1).eval()
    assert torch.equal(model(batch)[0], model(batch)[0])


def test_modality_dropout_keeps_at_least_one_modality(batch):
    model = make("proposed", p_drop_mri=0.9, p_drop_clinical=0.9).train()
    for _ in range(20):
        a_c, a_m = model.availability(batch)
        assert torch.all(a_c + a_m >= 1)


def test_auroc_matches_sklearn_and_temperature_keeps_ranking():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    z = y + rng.normal(0, 1, 300)
    assert np.isclose(M.auroc(y, z), roc_auc_score(y, z))
    T = M.fit_temperature(z, y)
    assert T > 0 and np.isclose(M.auroc(y, M.sigmoid(z / T)), M.auroc(y, z))


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    p = np.clip(0.5 + (y - 0.5) * 0.4 + rng.normal(0, 0.2, 4000), 0.01, 0.99)
    has0, has1, _ = M.conformal_sets(p[:2000], y[:2000], p[2000:], alpha=0.1)
    covered = np.where(y[2000:] == 1, has1, has0)
    assert covered.mean() >= 0.88


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
    rng = np.random.default_rng(2)
    y = (rng.random(len(p)) < p).astype(int)
    assert M.ece(y, p) < 0.02
