"""The one-call model audit must catch each known trap and pass clean models."""

import numpy as np

import qthermo as qt
from qthermo import models


def checks(report, severity):
    return {f.check for f in report.by_severity(severity)}


def test_clean_model_passes():
    report = qt.audit(models.absorption_refrigerator())
    assert report.ok
    assert "second law" in checks(report, "ok")
    assert "local vs global" in checks(report, "ok")


def test_levy_kosloff_regime_is_an_error():
    report = qt.audit(models.two_qubit_heat_valve(master_equation="local"))
    assert not report.ok
    assert "local vs global" in checks(report, "error")


def test_second_law_violation_is_an_error_when_heat_uses_full_H():
    m = models.two_qubit_heat_valve(master_equation="local")
    report = qt.audit(m.H, m.baths)            # bare call: heat measured with H
    assert "second law" in checks(report, "error")
    assert "heat direction" in checks(report, "error")


def test_mislabelled_bath_temperature_is_caught():
    H = qt.qubit_hamiltonian(1.0)
    hot = qt.Bath("hot", qt.thermal_bath(0.3, 1.0, 3.0), 2.0)   # rates say T=3
    cold = qt.Bath("cold", qt.thermal_bath(0.3, 1.0, 0.5), 0.5)
    report = qt.audit(H, [hot, cold])
    assert "bath temperature" in checks(report, "error")
    assert "T=3" in report.report()


def test_non_unique_steady_state_is_an_error():
    dims = [2, 2]
    H = qt.embed(qt.qubit_hamiltonian(1.0), 0, dims) + qt.embed(
        qt.qubit_hamiltonian(1.0), 1, dims)
    collective = qt.embed(qt.sigma_x, 0, dims) + qt.embed(qt.sigma_x, 1, dims)
    report = qt.audit(H, [qt.davies_bath(H, collective, 0.5)])
    assert "steady state" in checks(report, "error")


def test_global_chain_flags_unresolvable_bond_currents():
    report = qt.audit(models.spin_chain(3, master_equation="global"))
    assert "internal currents" in checks(report, "warning")


def test_maser_tur_violation_is_reported():
    m = models.three_level_maser(1.0, 3.0, 1.0, 200.0, 0.05, 0.05, 0.001)
    report = qt.audit(m)
    assert any("beats the classical TUR" in f.summary for f in report.findings)
