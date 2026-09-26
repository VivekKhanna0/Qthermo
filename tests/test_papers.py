import numpy as np
import pytest

import qthermo as qt
from qthermo.validation import QThermoError


@pytest.mark.parametrize("detuning", [0.0, 0.1])
def test_redfield_matches_exact_current(detuning):
    r = qt.papers.local_vs_global(couplings=[0.003, 0.03], detuning=detuning)
    assert r.errors()["redfield"] < 5e-3


def test_each_approximation_fails_where_expected():
    resonant = qt.papers.local_vs_global(couplings=[0.002], detuning=0.0)
    assert resonant.errors()["global"] > 1.0          # finite current as g -> 0
    assert resonant.errors()["local"] < 0.01
    detuned = qt.papers.local_vs_global(couplings=[0.02], detuning=0.1)
    assert detuned.errors()["local"] > 0.05
    assert 0.02 not in detuned.valid_range("local")
    assert 0.02 in detuned.valid_range("redfield")


def test_exact_current_limits():
    # no coupling, no current; equal temperatures, no current
    assert qt.papers.exact_oscillator_current(1, 1, 0.0, 0.02, 0.5, 0.25) == 0.0
    assert abs(qt.papers.exact_oscillator_current(1, 1.1, 0.05, 0.02, 0.4, 0.4)) < 1e-15


def test_redfield_single_qubit_is_thermal():
    H = qt.qubit_hamiltonian(1.0)
    bath = qt.redfield_bath(H, qt.sigma_x, 0.7, gamma=0.1)
    rho = qt.steady_state(H, [bath])
    assert np.allclose(rho, qt.thermal_state(H, 0.7), atol=1e-10)


def test_redfield_conserves_energy_and_refuses_jumps():
    a = np.diag(np.sqrt(np.arange(1, 4)), 1)
    a1, a2 = np.kron(a, np.eye(4)), np.kron(np.eye(4), a)
    H = a1.T @ a1 + 1.2 * a2.T @ a2 + 0.05 * (a1.T @ a2 + a2.T @ a1)
    baths = [qt.redfield_bath(H, a1 + a1.T, 0.4, gamma=0.02, name="h"),
             qt.redfield_bath(H, a2 + a2.T, 0.2, gamma=0.02, name="c")]
    s = qt.analyze(H, baths)
    assert s.currents["h"] > 0
    assert abs(s.total_current) < 1e-12 * abs(s.currents["h"])
    assert "Redfield" in repr(baths[0])
    with pytest.raises(QThermoError, match="Redfield"):
        baths[0].c_ops
    L = qt.liouvillian(H, baths)
    assert np.allclose(L @ s.rho.reshape(-1, order="F"), 0, atol=1e-12)


def test_too_few_levels_is_refused():
    with pytest.raises(QThermoError, match="levels"):
        qt.papers.local_vs_global(T_hot=3.0, levels=4)


def test_build_model_redfield_between_local_and_global_limits():
    model = qt.models.two_qubit_heat_valve()
    red = model.rebuild("redfield")
    assert red.master_equation == "redfield"
    s = red.analyze()
    assert abs(s.total_current) < 1e-12
    assert np.isfinite(qt.relaxation_time(red.H, red.baths, populations_only=True))
    assert qt.audit(red).ok
