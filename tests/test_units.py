"""Laboratory unit conversion."""

import numpy as np

import qthermo as qt


def test_five_ghz_qubit_at_twenty_millikelvin():
    lab = qt.LabUnits(5.0)
    # h f / k_B T = 6.626e-34 * 5e9 / (1.381e-23 * 0.02) = 12.0
    assert np.isclose(1 / lab.temperature(20.0), 11.998, rtol=1e-4)
    assert lab.frequency(5.0) == 1.0


def test_round_trips():
    lab = qt.LabUnits(6.2)
    assert np.isclose(lab.to_mK(lab.temperature(37.0)), 37.0)
    assert np.isclose(lab.to_seconds(lab.time(2.5e-6)), 2.5e-6)
    assert np.isclose(lab.to_per_second(lab.rate(1 / 80e-6)), 1 / 80e-6)
    assert np.isclose(lab.to_GHz(lab.frequency(4.3)), 4.3)


def test_rate_and_time_are_consistent():
    # a rate times a time is dimensionless in both unit systems
    lab = qt.LabUnits(5.0)
    gamma, t = 1 / 50e-6, 20e-6
    assert np.isclose(lab.rate(gamma) * lab.time(t), gamma * t)


def test_thermal_occupation_matches_lab_formula():
    lab = qt.LabUnits(5.0)
    n_lab = 1 / np.expm1(qt.PLANCK * 5e9 / (qt.BOLTZMANN * 0.05))
    assert np.isclose(qt.mean_occupation(lab.frequency(5.0), lab.temperature(50.0)), n_lab)
