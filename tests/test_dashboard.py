"""Smoke test: the dashboard renders for the group and every outlet without errors.

Needs the demo outputs (`make sample forecast plan`); skipped otherwise, e.g. in CI.
"""

import pytest

from kaizen.config import load_client

OUTPUTS = load_client("demo").outputs_dir
pytestmark = pytest.mark.skipif(not (OUTPUTS / "staffing_plan.csv").exists(), reason="run `make sample forecast plan` first")


def test_dashboard_renders_every_outlet():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../dashboard/app.py", default_timeout=60).run()
    assert not at.exception
    for outlet in at.selectbox[0].options[1:]:
        at.selectbox[0].select(outlet).run()
        assert not at.exception, outlet
