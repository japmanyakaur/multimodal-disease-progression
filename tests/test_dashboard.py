"""Dashboard smoke test: the app renders, and still renders when the MRI is hidden."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
READY = (ROOT / "data" / "processed" / "features.npz").exists() and (ROOT / "results" / "metrics_summary.json").exists()


@pytest.mark.skipif(not READY, reason="run `python run_all.py` first")
def test_dashboard_renders_with_and_without_mri():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert any("Calibrated probability" in m.label for m in at.metric)
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value(5).run()
    assert not at.exception
