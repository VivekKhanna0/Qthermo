"""Full counting statistics and thermodynamic / kinetic uncertainty relations."""

import numpy as np
import pytest

import qthermo as qt
from qthermo.baths import davies_bath
from qthermo.fluctuations import current_statistics, jump_energy, scaled_cgf

MASER_VIOLATION = dict(omega_c=1.0, omega_h=3.0, T_c=1.0, T_h=200.0,
                       drive=0.05, gamma_c=0.05, gamma_h=0.001)


def qubit_two_baths(T_h=3.0, T_c=0.5, g_h=0.3, g_c=0.7):
    H = qt.qubit_hamiltonian(1.0)
    baths = [davies_bath(H, qt.sigma_x, T_h, gamma=g_h, name="hot"),
             davies_bath(H, qt.sigma_x, T_c, gamma=g_c, name="cold")]
    return H, baths


def test_mean_and_noise_match_an_independent_classical_rate_model():
    """A qubit between two baths is a two-state Markov jump process."""
    H, baths = qubit_two_baths()
    stats = current_statistics(H, {"hot": "quanta"}, baths=baths)

    n = lambda T: 1 / np.expm1(1 / T)
    up_h, down_h = 0.3 * n(3.0), 0.3 * (1 + n(3.0))
    up_c, down_c = 0.7 * n(0.5), 0.7 * (1 + n(0.5))

    def theta(s):
        M = np.array([[-(up_h + up_c), down_h * np.exp(-s) + down_c],
                      [up_h * np.exp(s) + up_c, -(down_h + down_c)]])
        return np.max(np.linalg.eigvals(M).real)

    h = 1e-4
    J = (theta(h) - theta(-h)) / (2 * h)
    D = (theta(h) - 2 * theta(0) + theta(-h)) / h ** 2
    assert np.isclose(stats.mean, J, rtol=1e-8)
    assert np.isclose(stats.noise, D, rtol=1e-5)


def test_exact_noise_matches_tilted_liouvillian():
    m = qt.models.three_level_maser()
    count = {"hot": "energy", "cold": "energy"}
    stats = current_statistics(m, count)
    h = 1e-3
    theta = scaled_cgf(m, count, [-h, 0.0, h])
    assert abs(theta[1]) < 1e-12              # normalisation: theta(0) = 0
    assert np.isclose(stats.mean, (theta[2] - theta[0]) / (2 * h), rtol=1e-6)
    assert np.isclose(stats.noise, (theta[2] - 2 * theta[1] + theta[0]) / h ** 2,
                      rtol=1e-5)


def test_mean_matches_steady_state_heat_current():
    m = qt.models.absorption_refrigerator()
    r = m.analyze()
    for name in ("cold", "hot", "room"):
        assert np.isclose(current_statistics(m, name).mean, r.current(name),
                          rtol=1e-10)


def test_poisson_limit_has_unit_fano_factor():
    # At infinite temperature up and down flips have equal rates, so the
    # flips of a qubit form a Poisson process: Var[N_t] = <N_t>.
    H = qt.qubit_hamiltonian(1.0)
    bath = davies_bath(H, qt.sigma_x, 1e6, gamma=0.4, name="b")
    stats = current_statistics(H, {"b": 1.0}, baths=[bath])
    assert np.isclose(stats.fano_factor, 1.0, rtol=1e-4)


def test_tur_and_kur_hold_for_classical_davies_machines():
    """Random non-degenerate Davies machines are classical jump processes."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        d = int(rng.integers(2, 5))
        m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        H = (m + m.conj().T) / 2
        baths = []
        for k in range(int(rng.integers(2, 4))):
            a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            baths.append(davies_bath(H, (a + a.conj().T) / 2, rng.uniform(0.2, 5.0),
                                     gamma=10 ** rng.uniform(-2, 0), name=f"b{k}"))
        stats = current_statistics(H, {"b0": "energy"}, baths=baths)
        if abs(stats.mean) > 1e-9:
            assert stats.tur_ratio >= 2.0 - 1e-6
            assert stats.kur_ratio >= 1.0 - 1e-6


def test_three_level_maser_violates_the_tur():
    """Coherent driving beats the classical precision-dissipation bound.

    Kalaee, Wacker & Potts, PRE 104, L012103 (2021).
    """
    stats = current_statistics(qt.models.three_level_maser(**MASER_VIOLATION),
                               {"hot": "energy", "cold": "energy"})
    assert stats.mean > 0
    assert stats.violates_tur
    assert 1.9 < stats.tur_ratio < 2.0
    assert "VIOLATED" in stats.report()


def test_jump_energy_refuses_non_eigenoperators():
    m = qt.models.two_qubit_heat_valve(master_equation="local", g=0.4)
    local_jump = m.baths[0].c_ops[-1]
    assert np.isclose(abs(jump_energy(local_jump, m.H0)), 1.0)
    with pytest.raises(qt.QThermoError, match="definite amount"):
        jump_energy(local_jump, m.H)


def test_counting_unknown_bath_raises():
    H, baths = qubit_two_baths()
    with pytest.raises(qt.QThermoError, match="unknown bath"):
        current_statistics(H, {"nope": "energy"}, baths=baths)
