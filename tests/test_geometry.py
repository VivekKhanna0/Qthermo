"""Thermodynamic geometry: friction metric and minimum-dissipation schedules."""

import numpy as np
import pytest

import qthermo as qt
from qthermo.baths import ohmic_spectrum
from qthermo.geometry import excess_work, friction, optimal_schedule
from qthermo.information import erasure_friction

T = 0.7


def rotating_drive(th):
    return qt.qubit_hamiltonian(1.0 + th) + 0.8 * th * qt.sigma_x


def baths_of(H):
    return [qt.davies_bath(H, qt.sigma_x + qt.sigma_z, T,
                           spectrum=ohmic_spectrum(0.5, reference=1.0))]


def test_general_friction_reproduces_the_erasure_closed_form():
    bath = lambda H: [qt.davies_bath(H, qt.sigma_x, 1.0,
                                     spectrum=ohmic_spectrum(1.0, reference=1.0))]
    for w in (0.3, 1.0, 3.0, 8.0):
        assert np.isclose(friction(qt.qubit_hamiltonian, bath, 1.0, w),
                          erasure_friction(w, 1.0, 1.0), rtol=1e-6)


def test_friction_is_positive_along_a_non_commuting_drive():
    assert all(friction(rotating_drive, baths_of, T, th) > 0
               for th in np.linspace(0.0, 2.0, 7))


def test_friction_rejects_non_thermalising_dissipation():
    wrong_T = lambda H: [qt.davies_bath(H, qt.sigma_x, 2.0 * T)]
    with pytest.raises(qt.QThermoError, match="fixed point"):
        friction(rotating_drive, wrong_T, T, 0.5)


def test_optimal_schedule_attains_L2_over_tau_and_beats_linear():
    """Quantum (non-commuting) drive: prediction vs full simulation."""
    schedule = optimal_schedule(rotating_drive, baths_of, T, (0.0, 2.0), points=61)
    tau = 120.0
    optimal = excess_work(schedule.hamiltonian(rotating_drive, tau), baths_of, T, tau,
                          steps=1200)["dissipation"]
    linear = excess_work(lambda t: rotating_drive(2.0 * t / tau), baths_of, T, tau,
                         steps=1200)["dissipation"]
    assert np.isclose(optimal, schedule.minimum_excess(tau), rtol=0.02)
    assert np.isclose(linear, schedule.predicted_excess(lambda x: x, tau), rtol=0.02)
    assert optimal < 0.8 * linear


def test_constant_speed_schedule_minimises_the_predicted_excess():
    schedule = optimal_schedule(rotating_drive, baths_of, T, (0.0, 2.0), points=81)
    best = schedule.predicted_excess(schedule.s_of, 1.0)
    for other in (lambda x: x, lambda x: x ** 2, lambda x: np.sqrt(x)):
        assert schedule.predicted_excess(other, 1.0) >= best - 1e-6
    assert np.isclose(best, schedule.length ** 2, rtol=1e-3)
