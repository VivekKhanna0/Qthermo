"""Site-resolved flows, virtual temperatures, correlation measures, plots."""

import numpy as np
import pytest

import qthermo as qt
from qthermo import models


def bell():
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())


def test_virtual_temperature_of_a_gibbs_qubit_is_its_temperature():
    H = qt.qubit_hamiltonian(1.7)
    for T in (0.2, 1.0, 7.5):
        assert np.isclose(qt.virtual_temperature(qt.thermal_state(H, T), H), T)


def test_virtual_temperature_limits():
    H = qt.qubit_hamiltonian(1.0)
    assert qt.virtual_temperature(np.diag([1.0, 0.0]), H) == 0.0
    assert np.isinf(qt.virtual_temperature(np.eye(2) / 2, H))
    assert qt.virtual_temperature(np.diag([0.3, 0.7]), H) < 0   # inversion


def test_concurrence_and_negativity_of_bell_and_product_states():
    assert np.isclose(qt.concurrence(bell()), 1.0)
    assert np.isclose(qt.negativity(bell(), 0, 1, [2, 2]), 0.5)
    product = np.kron(np.diag([0.7, 0.3]), np.diag([0.4, 0.6])).astype(complex)
    assert qt.concurrence(product) < 1e-12
    assert abs(qt.negativity(product, 0, 1, [2, 2])) < 1e-12


def test_werner_state_entanglement_threshold():
    # Werner state p|Bell><Bell| + (1-p) I/4 is entangled iff p > 1/3,
    # with concurrence max(0, (3p - 1)/2).
    for p in (0.2, 1 / 3, 0.5, 0.9):
        rho = p * bell() + (1 - p) * np.eye(4) / 4
        assert np.isclose(qt.concurrence(rho), max(0.0, (3 * p - 1) / 2), atol=1e-10)


@pytest.mark.parametrize("builder", [
    lambda: models.absorption_refrigerator(),
    lambda: models.spin_chain(4, J=0.3, master_equation="local"),
    lambda: models.spin_chain(3, master_equation="global"),
    lambda: models.two_qubit_heat_valve(master_equation="local", g=0.2),
])
def test_site_energy_balance_closes_in_steady_state(builder):
    fm = qt.heat_flow_map(builder().analyze())
    assert np.max(np.abs(fm.site_balance())) < 1e-12


def test_bath_currents_decompose_exactly():
    m = models.spin_chain(3, J=0.4, master_equation="local")
    r = m.analyze(energy=None)
    fm = qt.heat_flow_map(r)
    for name, J in r.currents.items():
        assert np.isclose(fm.bath_current(name), J, atol=1e-14)


def test_local_chain_current_is_uniform_along_the_chain():
    # Continuity: in a steady state the same current crosses every bond,
    # and it equals the heat delivered to site 0 by the left bath.
    m = models.spin_chain(5, J=0.3, T_left=3.0, T_right=0.5, master_equation="local")
    fm = qt.heat_flow_map(m.analyze())
    currents = [fm.bond_current((i, i + 1)) for i in range(4)]
    assert np.allclose(currents, currents[0], rtol=1e-9)
    assert currents[0] > 0
    assert np.isclose(currents[0], fm.bath_to_site[("left", 0)], rtol=1e-9)


def test_virtual_temperature_profile_is_monotone_hot_to_cold():
    m = models.spin_chain(5, J=0.3, T_left=3.0, T_right=0.5, master_equation="local")
    T = qt.heat_flow_map(m.analyze()).virtual_temperature
    assert np.all(np.diff(T) < 0)
    assert 0.5 < T[-1] < T[0] < 3.0


def test_refrigerator_cold_qubit_is_colder_than_every_bath():
    m = models.absorption_refrigerator(T_c=1.0, T_h=6.0, T_r=1.5)
    fm = qt.heat_flow_map(m.analyze())
    assert fm.coldest_site() == 0
    assert fm.virtual_temperature[0] < 1.0


def test_global_steady_state_flags_vanishing_bond_currents():
    fm = qt.heat_flow_map(models.spin_chain(3, master_equation="global").analyze())
    assert fm.extras.get("secular_blind")
    assert "vanishes identically" in fm.report()


def test_plots_render():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for m in (models.absorption_refrigerator(),
              models.spin_chain(4, master_equation="local"),
              models.two_qubit_heat_valve()):
        fig = qt.plot_machine(m.analyze())
        plt.close(fig)
