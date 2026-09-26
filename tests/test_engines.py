"""Otto engines with interacting multi-qubit working media."""

import numpy as np
import pytest

import qthermo as qt
from qthermo.engines import adiabatic_pairing, ideal_otto, otto_cycle

D = [2, 2]
E, X, Y, Z = qt.embed, qt.sigma_x, qt.sigma_y, qt.sigma_z


def heisenberg(B, J):
    return (-0.5 * B * (E(Z, 0, D) + E(Z, 1, D))
            + J * (E(X, 0, D) @ E(X, 1, D) + E(Y, 0, D) @ E(Y, 1, D)
                   + E(Z, 0, D) @ E(Z, 1, D)))


def ising(h, J=1.0):
    return -J * E(Z, 0, D) @ E(Z, 1, D) - h * (E(X, 0, D) + E(X, 1, D))


def test_single_qubit_limit_is_the_textbook_otto_efficiency():
    lim = ideal_otto(qt.qubit_hamiltonian(1.0), qt.qubit_hamiltonian(2.5), 0.5, 4.0)
    assert np.isclose(lim.efficiency, 1 - 1.0 / 2.5, atol=1e-14)


def test_uniformly_scaled_spectrum_gives_one_minus_ratio():
    # H_hot = r H_cold: every gap scales by r, so eta = 1 - 1/r regardless
    # of the interactions.
    H = heisenberg(1.0, 0.3)
    lim = ideal_otto(H, 2.0 * H, 0.3, 5.0)
    assert np.isclose(lim.efficiency, 0.5, atol=1e-12)


def test_heisenberg_coupling_enhances_otto_efficiency():
    """Thomas & Johal, PRE 83, 031135 (2011): eta > 1 - B_c/B_h for J > 0."""
    eta = [ideal_otto(heisenberg(2, J), heisenberg(4, J), 0.5, 4.0).efficiency
           for J in (0.0, 0.2)]
    assert np.isclose(eta[0], 0.5)
    assert eta[1] > eta[0]
    assert eta[1] < 1 - 0.5 / 4.0


def test_simulated_coupled_otto_matches_quasi_static_limit():
    couplings = [E(X, 0, D), E(X, 1, D)]
    cycle = otto_cycle(heisenberg(2, 0.2), heisenberg(4, 0.2), 0.5, 4.0,
                       couplings, tau_iso=30, tau_ramp=1.0)
    result, _, converged = cycle.limit_cycle(np.eye(4) / 4)
    lim = ideal_otto(heisenberg(2, 0.2), heisenberg(4, 0.2), 0.5, 4.0)
    assert converged
    assert np.isclose(result.efficiency(("hot_iso",)), lim.efficiency, atol=1e-6)
    assert result.max_first_law_residual < 1e-12
    for stroke in result.strokes:
        if stroke.entropy_production is not None:
            assert stroke.entropy_production > -1e-10


def test_symmetry_blocked_thermalisation_is_refused():
    # sigma_x couplings commute with the Ising parity X1 X2: the isochores
    # could never reach the Gibbs state.
    with pytest.raises(qt.QThermoError, match="symmetry"):
        otto_cycle(ising(0.5), ising(2.0), 0.3, 3.0, [E(X, 0, D), E(X, 1, D)])


def test_non_commuting_ising_otto_approaches_the_adiabatic_limit():
    couplings = [E(Z, 0, D), E(Z, 1, D)]
    lim = ideal_otto(ising(0.5), ising(2.0), 0.3, 3.0)
    works = []
    for tau_ramp in (2.0, 20.0):
        cycle = otto_cycle(ising(0.5), ising(2.0), 0.3, 3.0, couplings,
                           tau_iso=40, tau_ramp=tau_ramp)
        result, _, _ = cycle.limit_cycle(np.eye(4) / 4)
        works.append(result.net_work)
    # slower ramps extract more work (less quantum friction) ...
    assert works[1] < works[0] < 0
    # ... and approach the quasi-static value from above
    assert abs(works[1] - lim.work) < 0.02 * abs(lim.work)
    assert works[1] > lim.work


def test_level_crossing_changes_the_quasi_static_cycle():
    # Two uncoupled qubits whose fields swap order: |01> and |10> cross
    # exactly, and a quasi-static ramp keeps each on its own branch.
    def H(b1, b2):
        return -0.5 * b1 * E(Z, 0, D) - 0.5 * b2 * E(Z, 1, D)
    H_c, H_h = H(1.0, 2.0), H(4.0, 3.0)
    perm = adiabatic_pairing(H_h, H_c)
    assert np.any(perm != np.arange(4))
    followed = ideal_otto(H_c, H_h, 0.4, 3.0)
    naive = ideal_otto(H_c, H_h, 0.4, 3.0, follow_crossings=False)
    assert followed.crossings
    assert not np.isclose(followed.work, naive.work)

    # the finite-time cycle (commuting drive: populations exactly conserved)
    # agrees with the followed pairing, not the naive one
    cycle = otto_cycle(H_c, H_h, 0.4, 3.0, [E(X, 0, D), E(X, 1, D)],
                       tau_iso=60, tau_ramp=1.0, gamma=0.5)
    result, _, _ = cycle.limit_cycle(np.eye(4) / 4)
    assert np.isclose(result.net_work, followed.work, rtol=1e-4)


def test_degenerate_levels_with_flat_spectrum_warn_and_slow_isochore_is_flagged():
    # B_c = 4J makes the singlet and |up,up> exactly degenerate; a flat
    # spectrum has no finite zero-frequency rate, so the direct channel
    # between them is off and the cold isochore thermalises very slowly.
    with pytest.warns(RuntimeWarning) as caught:
        otto_cycle(heisenberg(2, 0.5), heisenberg(4, 0.5), 0.5, 4.0,
                   [E(X, 0, D), E(X, 1, D)], tau_iso=30, tau_ramp=1.0)
    messages = " ".join(str(w.message) for w in caught)
    assert "degenerate" in messages
    assert "will not reach its Gibbs state" in messages


def test_ohmic_spectrum_removes_the_degeneracy_problem():
    import warnings
    from qthermo.baths import ohmic_spectrum
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no warning of either kind
        cycle = otto_cycle(heisenberg(2, 0.5), heisenberg(4, 0.5), 0.5, 4.0,
                           [E(X, 0, D), E(X, 1, D)], tau_iso=60, tau_ramp=1.0,
                           spectrum=ohmic_spectrum(0.5))
    result, _, _ = cycle.limit_cycle(np.eye(4) / 4)
    ideal = ideal_otto(heisenberg(2, 0.5), heisenberg(4, 0.5), 0.5, 4.0)
    assert np.isclose(result.net_work, ideal.work, rtol=1e-6)


def test_relaxation_time_of_a_qubit_is_the_inverse_T1_rate():
    # Populations of a qubit under a thermal bath relax at gamma (1 + 2 n):
    # emission gamma (1 + n) plus absorption gamma n. Coherences decay at half
    # that, so they set the slowest time, 2 / (gamma (1 + 2 n)).
    H = qt.qubit_hamiltonian(1.0)
    gamma, T = 0.3, 0.8
    n = 1 / np.expm1(1 / T)
    bath = qt.davies_bath(H, qt.sigma_x, T, gamma=gamma)
    assert np.isclose(qt.relaxation_time(H, [bath]), 2 / (gamma * (1 + 2 * n)))
    assert np.isclose(qt.relaxation_time(H, [bath], populations_only=True),
                      1 / (gamma * (1 + 2 * n)))
