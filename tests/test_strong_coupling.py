"""Reaction-coordinate mapping: strong coupling limits and consistency."""

import numpy as np
import pytest

import qthermo as qt
from qthermo.strong_coupling import (
    mean_force_state,
    rc_convergence,
    reaction_coordinate_model,
    ultrastrong_limit_state,
)

H_S = qt.qubit_hamiltonian(1.0)
S = qt.sigma_x


def two_bath(lam, n_levels=14):
    return reaction_coordinate_model(
        H_S, S, 2.0, lam=lam, Omega=2.0, kappa=0.1, n_levels=n_levels,
        name="hot", weak_baths=[dict(coupling=S, T=0.5, gamma=0.05, name="cold")])


def test_single_bath_steady_state_is_gibbs_of_the_enlarged_system():
    m = reaction_coordinate_model(H_S, S, 0.5, lam=0.5, Omega=2.0, n_levels=10)
    r = m.analyze()
    assert np.max(np.abs(r.rho - qt.thermal_state(m.H, 0.5))) < 1e-10


def test_weak_coupling_recovers_gibbs_with_lambda_squared_correction():
    T = 0.5
    gibbs = qt.thermal_state(H_S, T)
    d1 = qt.trace_distance(mean_force_state(H_S, S, T, 0.01, 2.0, 40), gibbs)
    d2 = qt.trace_distance(mean_force_state(H_S, S, T, 0.02, 2.0, 40), gibbs)
    assert d1 < 1e-4
    assert np.isclose(d2 / d1, 4.0, rtol=0.02)


def test_ultrastrong_coupling_approaches_cresser_anders_limit():
    T = 0.5
    limit = ultrastrong_limit_state(H_S, S, T)
    distances = [qt.trace_distance(mean_force_state(H_S, S, T, lam, 2.0, 60), limit)
                 for lam in (0.3, 1.0, 2.0, 4.0)]
    assert all(b < a for a, b in zip(distances, distances[1:]))
    assert distances[-1] < 0.02


def test_ultrastrong_limit_for_sigma_x_coupling_is_maximally_mixed():
    # P_n H_S P_n = 0 in the sigma_x basis for H_S ~ sigma_z.
    assert np.allclose(ultrastrong_limit_state(H_S, S, 0.3), np.eye(2) / 2)


def test_heat_current_turns_over_with_coupling_strength():
    """Weak coupling predicts J ~ lam^2 forever; the RC shows the turnover."""
    lams = [0.05, 0.1, 0.4, 0.8, 1.6, 2.4]
    currents = [two_bath(lam).analyze().current("hot") for lam in lams]
    assert np.isclose(currents[1] / currents[0], 4.0, rtol=0.1)  # weak: ~lam^2
    peak = int(np.argmax(currents))
    assert 0 < peak < len(lams) - 1
    assert currents[-1] < 0.5 * currents[peak]


def test_two_bath_rc_model_obeys_second_law_and_conservation():
    for lam in (0.1, 0.8, 2.0):
        r = two_bath(lam).analyze()
        assert r.entropy_production_rate > 0
        assert abs(r.total_current) < 1e-12
        assert r.current("hot") > 0 > r.current("cold")


def test_truncation_convergence_check():
    report = rc_convergence(lambda n: two_bath(1.2, n))
    assert report["converged"]
    assert report["relative_change"] < 1e-3


def test_rc_model_rejects_bad_parameters():
    with pytest.raises(qt.QThermoError):
        reaction_coordinate_model(H_S, S, 1.0, 0.5, Omega=-1.0)
    with pytest.raises(qt.QThermoError):
        reaction_coordinate_model(H_S, np.eye(3), 1.0, 0.5, 1.0)
