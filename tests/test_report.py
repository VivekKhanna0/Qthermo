"""HTML report."""

import pytest

import qthermo as qt


def test_report_contains_audit_currents_and_data(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "valve.html"
    page = qt.report(qt.models.two_qubit_heat_valve(master_equation="local"), path)
    assert path.read_text() == page
    for fragment in ("<h2>Audit</h2>", "local vs global", "<h2>Heat currents</h2>",
                     "data:image/png;base64", "boundary work", "&quot;currents&quot;"):
        assert fragment in page


def test_report_of_a_clean_model_says_so():
    page = qt.report(qt.models.absorption_refrigerator(), fluctuations=True)
    assert "no problems found" in page
    assert "TUR ratio" in page
