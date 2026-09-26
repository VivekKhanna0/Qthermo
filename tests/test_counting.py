"""Exact counting statistics of stroke cycles."""

import numpy as np
import pytest
from scipy.linalg import expm

import qthermo as qt
from qthermo.counting import cycle_counting


def otto(tau_iso=12.0):
    ramp = lambda a, b, tau: (lambda t: qt.qubit_hamiltonian(a + (t / tau) * (b - a)))
    return qt.Cycle([
        qt.Stroke("cold_iso", qt.qubit_hamiltonian(1.0), tau_iso,
                  qt.thermal_bath(1.0, 1.0, 1.0), temperature=1.0),
        qt.Stroke("compress", ramp(1.0, 1.5, 0.02), 0.02),
        qt.Stroke("hot_iso", qt.qubit_hamiltonian(1.5), tau_iso,
                  qt.thermal_bath(1.0, 1.5, 1.3), temperature=1.3),
        qt.Stroke("expand", ramp(1.5, 1.0, 0.02), 0.02),
    ])


def test_spontaneous_decay_counts_are_exact():
    # Excited qubit, zero-temperature decay for tau: one emission with
    # probability 1 - exp(-gamma tau), none otherwise.
    gamma, tau = 0.7, 1.3
    cycle = qt.Cycle([qt.Stroke("decay", qt.qubit_hamiltonian(1.0), tau,
                                qt.amplitude_damping(gamma))])
    excited = np.diag([0.0, 1.0]).astype(complex)
    dist = cycle_counting(cycle, {"decay": "quanta"}, rho_start=excited).distribution(4)
    assert np.isclose(dist[-1], 1 - np.exp(-gamma * tau), atol=1e-12)
    assert np.isclose(dist[0], np.exp(-gamma * tau), atol=1e-12)
    assert all(abs(p) < 1e-12 for n, p in dist.items() if n not in (-1, 0))


def test_thermal_qubit_matches_independent_classical_telegraph():
    """Net quanta from a bath over time tau: two-state jump process."""
    gamma, omega, T, tau = 0.8, 1.0, 0.7, 2.5
    n_bar = 1 / np.expm1(omega / T)
    up, down = gamma * n_bar, gamma * (1 + n_bar)
    cycle = qt.Cycle([qt.Stroke("contact", qt.qubit_hamiltonian(omega), tau,
                                qt.thermal_bath(gamma, omega, T), temperature=T)])
    rho0 = qt.thermal_state(qt.qubit_hamiltonian(omega), T)
    dist = cycle_counting(cycle, {"contact": "quanta"}, rho_start=rho0).distribution(6)

    def G(chi):
        # populations (ground, excited); absorption +1, emission -1
        M = np.array([[-up, down * np.exp(-1j * chi)],
                      [up * np.exp(1j * chi), -down]])
        p0 = np.array([np.real(rho0[0, 0]), np.real(rho0[1, 1])])
        return np.sum(expm(M * tau) @ p0)
    points = 24
    chis = 2 * np.pi * np.arange(points) / points
    reference = np.fft.fft([G(c) for c in chis]) / points
    for n in range(-6, 7):
        assert np.isclose(dist[n], np.real(reference[n % points]), atol=1e-12)
    # net exchange in a stationary two-state process is -1, 0 or +1
    assert np.isclose(dist[-1] + dist[0] + dist[1], 1.0, atol=1e-12)


def test_cycle_mean_matches_the_master_equation_heat():
    cycle = otto()
    stats = cycle_counting(cycle, {"cold_iso": "quanta"})
    result = cycle.run(stats.rho_start)
    heat_quanta = result.heat_from("cold_iso") / 1.0
    assert np.isclose(stats.single_cycle_mean(), heat_quanta, atol=1e-9)
    assert np.isclose(stats.long_run()["mean"], heat_quanta, rtol=1e-4)
    assert stats.truncation_error() < 1e-12


def test_distribution_is_normalised_and_non_negative():
    dist = cycle_counting(otto(), {"cold_iso": "quanta", "hot_iso": "quanta"}).distribution(8)
    assert np.isclose(sum(dist.values()), 1.0, atol=1e-12)
    assert min(dist.values()) > -1e-12


def test_unknown_stroke_rejected():
    with pytest.raises(qt.QThermoError):
        cycle_counting(otto(), {"nope": "quanta"})


def test_sampled_trajectories_agree_with_exact_counting():
    """Regression: zero-heat trajectories must count as q <= 0.

    Before round-off snapping, ~35% of trajectories carried heat ~ +1e-16 and
    P(q <= 0) came out ~0.45-0.53 instead of the exact 0.7955.
    """
    cycle = otto()
    exact = cycle_counting(cycle, {"cold_iso": "quanta"})
    p_exact = exact.probability(lambda n: n <= 0)
    ensemble = qt.unravel(cycle, exact.rho_start, trajectories=1500, steps=1600, seed=4)
    p_sampled = ensemble.probability_of("cold_iso", lambda q: q <= 0)
    sigma = np.sqrt(p_exact * (1 - p_exact) / 1500)
    assert abs(p_sampled - p_exact) < 4 * sigma
