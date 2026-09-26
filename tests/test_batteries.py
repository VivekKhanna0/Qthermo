"""Quantum-battery ergotropy decompositions and collective charging."""

import numpy as np

import qthermo as qt
from qthermo.batteries import (
    asymptotic_ergotropy,
    charge,
    collective_advantage,
    dicke_battery,
    ergotropy_split,
    locked_ergotropy,
    multi_copy_ergotropy,
)

H = qt.qubit_hamiltonian(1.0)


def test_superposition_stores_ergotropy_only_in_coherence():
    plus = np.full((2, 2), 0.5, dtype=complex)
    split = ergotropy_split(plus, H)
    assert np.isclose(split["total"], 0.5) and np.isclose(split["coherent"], 0.5)
    assert abs(split["incoherent"]) < 1e-12


def test_diagonal_states_have_no_coherent_ergotropy():
    rng = np.random.default_rng(2)
    Hq = np.diag([0.0, 0.7, 1.9]).astype(complex)
    for _ in range(10):
        p = rng.dirichlet(np.ones(3))
        split = ergotropy_split(np.diag(p).astype(complex), Hq)
        assert abs(split["coherent"]) < 1e-12
        assert np.isclose(split["total"], split["incoherent"])


def test_coherent_part_is_never_negative():
    rng = np.random.default_rng(5)
    Hq = np.diag([0.0, 1.0, 1.5, 3.0]).astype(complex)
    for _ in range(20):
        a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        rho = a @ a.conj().T
        rho /= np.trace(rho)
        assert ergotropy_split(rho, Hq)["coherent"] > -1e-12


def test_bell_pair_ergotropy_is_fully_locked():
    bell = np.zeros((4, 4), dtype=complex)
    bell[0, 0] = bell[0, 3] = bell[3, 0] = bell[3, 3] = 0.5
    out = locked_ergotropy(bell, [2, 2], [H, H])
    assert np.isclose(out["global"], 1.0) and np.isclose(out["locked"], 1.0)
    assert abs(out["local"]) < 1e-12


def test_product_states_lock_nothing():
    a = np.diag([0.2, 0.8]).astype(complex)
    b = np.full((2, 2), 0.5, dtype=complex)
    assert abs(locked_ergotropy(np.kron(a, b), [2, 2], [H, H])["locked"]) < 1e-12


def test_activation_of_a_passive_non_gibbs_state():
    """Passive but not completely passive: copies unlock work (Alicki-Fannes)."""
    Hq = np.diag([0.0, 1.0, 2.0]).astype(complex)
    rho = np.diag([0.5, 0.4, 0.1]).astype(complex)
    limit = asymptotic_ergotropy(rho, Hq)
    assert abs(qt.ergotropy(rho, Hq)) < 1e-12
    per_copy = [multi_copy_ergotropy(rho, Hq, k) for k in (1, 3, 6)]
    assert per_copy[0] < 1e-12 < per_copy[1]
    assert per_copy[2] >= per_copy[1] - 1e-12        # superadditivity: W(6) >= 2 W(3)
    assert all(w <= limit + 1e-12 for w in per_copy)


def test_gibbs_states_are_completely_passive():
    Hq = np.diag([0.0, 1.0, 2.5]).astype(complex)
    gibbs = qt.thermal_state(Hq, 0.8)
    assert abs(asymptotic_ergotropy(gibbs, Hq)) < 1e-9
    assert abs(multi_copy_ergotropy(gibbs, Hq, 3)) < 1e-10


def test_single_cell_charges_fully_with_one_photon():
    b = dicke_battery(1, g=0.05, rotating_wave=True)
    run = charge(b["H"], b["H_battery"], b["state"], np.linspace(0, 40, 801))
    # Jaynes-Cummings vacuum Rabi oscillation: full charge at t = pi / (2 g)
    assert np.isclose(run.energy.max(), 1.0, atol=1e-6)
    assert np.isclose(run.charging_time(), np.pi / (2 * 0.05), rtol=0.01)


def test_dicke_collective_advantage_scales_as_sqrt_N():
    """Ferraro et al., PRL 120, 117702 (2018)."""
    out = collective_advantage([1, 2, 4, 6, 8, 10])
    assert np.all(np.diff(out["advantage"]) > 0)
    assert abs(out["large_N_exponent"] - 0.5) < 0.05
