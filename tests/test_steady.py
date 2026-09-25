"""Global baths, steady states, and continuous machines."""

import warnings

import numpy as np
import pytest

import qthermo as qt
from qthermo import models
from qthermo.baths import bohr_decomposition, davies_bath, local_bath, ohmic_spectrum


def random_hermitian(d, rng):
    m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return (m + m.conj().T) / 2


# --- Davies construction ----------------------------------------------------

def test_davies_reduces_to_thermal_bath_for_a_qubit():
    H = qt.qubit_hamiltonian(1.3)
    bath = davies_bath(H, qt.sigma_x, 0.7, gamma=0.4)
    reference = qt.thermal_bath(0.4, 1.3, 0.7)
    # ordering: absorption (w < 0) first, then emission
    assert np.allclose(bath.c_ops[0], reference[1])
    assert np.allclose(bath.c_ops[1], reference[0])


@pytest.mark.parametrize("dim", [3, 4, 8])
def test_davies_fixed_point_is_gibbs_for_random_hamiltonians(dim):
    rng = np.random.default_rng(dim)
    H, A = random_hermitian(dim, rng), random_hermitian(dim, rng)
    bath = davies_bath(H, A, 0.9, spectrum=ohmic_spectrum(0.3, cutoff=5.0))
    rho = qt.steady_state(H, [bath])
    assert np.max(np.abs(rho - qt.thermal_state(H, 0.9))) < 1e-10


def test_bohr_components_sum_to_the_coupling():
    rng = np.random.default_rng(3)
    H, A = random_hermitian(6, rng), random_hermitian(6, rng)
    parts = bohr_decomposition(H, A)
    assert np.allclose(sum(parts.values()), A, atol=1e-12)
    # each component lowers the energy by its frequency: [H, A(w)] = -w A(w)
    for omega, A_w in parts.items():
        assert np.allclose(H @ A_w - A_w @ H, -omega * A_w, atol=1e-9)


def test_degenerate_spectrum_is_grouped():
    # Two identical uncoupled qubits: the Bohr frequency w appears on two
    # different level pairs and must become ONE jump operator per direction.
    dims = [2, 2]
    H = qt.embed(qt.qubit_hamiltonian(1.0), 0, dims) + qt.embed(
        qt.qubit_hamiltonian(1.0), 1, dims)
    coupling = qt.embed(qt.sigma_x, 0, dims) + qt.embed(qt.sigma_x, 1, dims)
    parts = bohr_decomposition(H, coupling)
    assert sorted(round(k, 9) for k in parts) == [-1.0, 1.0]


def test_local_bath_matches_embedded_thermal_bath():
    dims = [2, 2, 2]
    bath = local_bath(qt.qubit_hamiltonian(1.2), qt.sigma_x, 0.8, 1, dims, gamma=0.3)
    expected = [qt.embed(L, 1, dims) for L in qt.thermal_bath(0.3, 1.2, 0.8)]
    assert np.allclose(bath.c_ops[1], expected[0])
    assert np.allclose(bath.c_ops[0], expected[1])
    assert bath.kind == "local"


def test_bath_without_transitions_is_rejected():
    H = qt.qubit_hamiltonian(1.0)
    with pytest.raises(qt.QThermoError):
        davies_bath(H, qt.sigma_z, 1.0)   # commutes with H, and w=0 rate is 0


# --- steady states ----------------------------------------------------------

def test_steady_state_matches_long_time_evolution():
    m = models.spin_chain(3, master_equation="global")
    rho_ss = qt.steady_state(m.H, m.baths)
    c_ops = [L for b in m.baths for L in b.c_ops]
    out = qt.evolve(np.eye(8) / 8, m.H, c_ops, duration=800.0, steps=50)
    assert np.max(np.abs(out["states"][-1] - rho_ss)) < 1e-6


def test_liouvillian_sparse_and_dense_agree():
    m = models.spin_chain(3, master_equation="local")
    dense = qt.liouvillian(m.H, m.baths, sparse=False)
    sparse = qt.liouvillian(m.H, m.baths, sparse=True)
    assert np.allclose(dense, sparse.toarray())


def test_non_unique_steady_state_is_detected():
    # Identical qubits on a common bath: the singlet is dark.
    dims = [2, 2]
    H = qt.embed(qt.qubit_hamiltonian(1.0), 0, dims) + qt.embed(
        qt.qubit_hamiltonian(1.0), 1, dims)
    collective = qt.embed(qt.sigma_x, 0, dims) + qt.embed(qt.sigma_x, 1, dims)
    bath = davies_bath(H, collective, 0.5)
    with pytest.raises(qt.QThermoError, match="not unique"):
        qt.steady_state(H, [bath])

    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(singlet, singlet.conj())
    rho = qt.steady_state(H, [bath], initial_state=rho0)
    assert np.isclose(np.real(singlet.conj() @ rho @ singlet), 1.0, atol=1e-8)


# --- continuous machines ----------------------------------------------------

def test_two_bath_currents_conserve_energy_and_flow_hot_to_cold():
    m = models.spin_chain(3, T_left=3.0, T_right=0.5, master_equation="global")
    r = m.analyze()
    assert abs(r.total_current) < 1e-12
    assert r.current("left") > 0 > r.current("right")
    assert r.entropy_production_rate > 0


def test_equal_temperatures_give_zero_current():
    m = models.spin_chain(3, T_left=1.3, T_right=1.3, master_equation="global")
    r = m.analyze()
    assert abs(r.current("left")) < 1e-12
    assert np.max(np.abs(r.rho - qt.thermal_state(m.H, 1.3))) < 1e-10


def test_absorption_fridge_tight_coupling_local():
    m = models.absorption_refrigerator(omega_c=1.0, omega_h=3.0)
    r = m.analyze()
    assert r.is_cooling("cold")
    assert np.isclose(r.cop("cold", "hot"), 1.0 / 3.0, atol=1e-10)
    assert np.isclose(r.current("room") / r.current("hot"), -4.0 / 3.0, atol=1e-10)
    assert r.cop("cold", "hot") <= r.absorption_carnot_cop("cold", "hot", "room")


def test_absorption_fridge_global_obeys_second_law_and_carnot():
    for T_h in (2.0, 4.0, 8.0):
        m = models.absorption_refrigerator(T_h=T_h, master_equation="global")
        r = m.analyze()
        assert r.entropy_production_rate > -1e-14
        if r.is_cooling("cold"):
            assert r.cop("cold", "hot") <= r.absorption_carnot_cop("cold", "hot", "room")


def test_maser_efficiency_is_scovil_schulz_dubois():
    for omega_c, omega_h in ((1.0, 3.0), (0.5, 4.0)):
        m = models.three_level_maser(omega_c=omega_c, omega_h=omega_h, T_h=20.0)
        r = m.analyze()
        assert r.power_output > 0
        assert np.isclose(r.efficiency("hot"), 1 - omega_c / omega_h, atol=1e-10)
        assert r.efficiency("hot") <= 1 - 1.0 / 20.0


def test_local_master_equation_violates_second_law_levy_kosloff():
    local = models.two_qubit_heat_valve(master_equation="local")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = local.analyze(energy=None)   # measure heat with the full H
    assert r.current("hot") < 0             # heat flows cold -> hot
    assert r.entropy_production_rate < 0
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_boundary_work_restores_local_consistency():
    # Measuring heat with the bare Hamiltonian makes the local model
    # consistent, with the coupling's boundary work as the balancing term.
    local = models.two_qubit_heat_valve(master_equation="local")
    r = local.analyze()    # auto: H0 for local baths
    assert r.entropy_production_rate >= 0
    assert r.work_rate > 0


def test_global_master_equation_never_violates_second_law():
    rng = np.random.default_rng(11)
    for _ in range(40):
        w1, w2 = rng.uniform(0.3, 3.0, 2)
        g = rng.uniform(0.0, 1.5)
        T_c, T_h = np.sort(rng.uniform(0.2, 5.0, 2))
        m = models.two_qubit_heat_valve(w1, w2, g, "xx", T_h, T_c,
                                        gamma=10 ** rng.uniform(-2, 0),
                                        master_equation="global")
        r = m.analyze()
        assert r.entropy_production_rate > -1e-12
        assert r.current("hot") > -1e-12


def test_comparison_flags_opposite_flows():
    text = models.two_qubit_heat_valve().compare().report()
    assert "OPPOSITE" in text and "second law" in text


def test_analyze_rejects_mismatched_bath():
    m = models.spin_chain(3)
    bad = qt.Bath("bad", qt.thermal_bath(1.0, 1.0, 1.0), 1.0)
    with pytest.raises(qt.QThermoError, match="dimension"):
        qt.analyze(m.H, [m.baths[0], bad])
