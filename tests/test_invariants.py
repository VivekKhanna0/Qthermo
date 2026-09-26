"""Randomised property tests: invariants that must hold for ANY machine.

Each case draws a random system (qubits, qutrits, mixed dimensions, sometimes
a degenerate spectrum), random couplings, baths and temperatures, and checks
identities that thermodynamics fixes independently of the model. A failure
here is a bug, whatever the parameters.
"""

import warnings

import numpy as np
import pytest

import qthermo as qt

SEEDS = range(24)


def random_hermitian(d, rng, scale=1.0):
    m = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return scale * (m + m.conj().T) / 2


def random_machine(seed):
    rng = np.random.default_rng(seed)
    dims = [[2, 2], [3], [2, 3], [2, 2, 2], [4]][seed % 5]
    d = int(np.prod(dims))
    local = [np.diag(np.sort(rng.uniform(0, 2, n))).astype(complex) for n in dims]
    if seed % 7 == 3:                       # degenerate local spectrum
        local[0] = np.diag(np.r_[0.0, np.ones(dims[0] - 1)]).astype(complex)
    H0 = sum(qt.embed(h, i, dims) for i, h in enumerate(local))
    H = H0 + random_hermitian(d, rng, scale=rng.uniform(0.05, 0.6))
    n_baths = int(rng.integers(1, 4))
    baths = []
    for k in range(n_baths):
        A = random_hermitian(d, rng)
        spectrum = qt.ohmic_spectrum(10 ** rng.uniform(-2, 0)) if k % 2 else None
        baths.append(qt.davies_bath(H, A, rng.uniform(0.3, 4.0),
                                    gamma=10 ** rng.uniform(-2, 0), spectrum=spectrum,
                                    name=f"b{k}"))
    return H, baths, dims, local


@pytest.mark.parametrize("seed", SEEDS)
def test_global_machine_invariants(seed):
    H, baths, dims, local = random_machine(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = qt.analyze(H, baths)
    rho = r.rho
    scale = max(max(abs(J) for J in r.currents.values()), 1e-300)
    # a valid state
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-10)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() > -1e-10
    # stationary
    L = qt.liouvillian(H, baths, sparse=False)
    assert np.max(np.abs(L @ rho.reshape(-1, order="F"))) < 1e-9
    # first law: currents measured with H sum to zero
    assert abs(r.total_current) < 1e-9 * max(scale, 1.0)
    # second law for Davies baths
    assert r.entropy_production_rate > -1e-10 * max(scale, 1.0)
    # single bath -> Gibbs state
    if len(baths) == 1:
        assert np.max(np.abs(rho - qt.thermal_state(H, baths[0].temperature))) < 1e-8
    # counting statistics mean equals the steady-state current; noise >= 0
    for b in baths:
        stats = qt.current_statistics(H, b.name, baths=baths)
        assert np.isclose(stats.mean, r.current(b.name), atol=1e-10 * max(scale, 1.0))
        assert stats.noise > -1e-12
    # site-resolved flows close exactly
    fm = qt.heat_flow_map(r, dims=dims, local_H=local,
                          interaction_terms={tuple(range(len(dims))): H - sum(
                              qt.embed(h, i, dims) for i, h in enumerate(local))})
    assert np.max(np.abs(fm.site_balance())) < 1e-9 * max(scale, 1.0)
    for b in baths:
        assert np.isclose(fm.bath_current(b.name), r.current(b.name), atol=1e-12)


@pytest.mark.parametrize("seed", SEEDS[:12])
def test_dynamics_invariants(seed):
    H, baths, dims, _ = random_machine(seed)
    d = H.shape[0]
    rng = np.random.default_rng(100 + seed)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho0 = a @ a.conj().T
    rho0 /= np.trace(rho0)
    out = qt.evolve(rho0, H, [L for b in baths for L in b.c_ops], duration=3.0, steps=60)
    for rho in out["states"]:
        assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)
        assert np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() > -1e-8
    # first law along the trajectory, exact by construction
    dU = np.real(np.trace(H @ out["states"][-1]) - np.trace(H @ rho0))
    assert np.isclose(out["heat"] + out["work"], dU, atol=1e-10)
    assert abs(out["work"]) < 1e-12                      # static H: no work


@pytest.mark.parametrize("seed", SEEDS[:12])
def test_ergotropy_and_passivity_invariants(seed):
    rng = np.random.default_rng(200 + seed)
    d = int(rng.integers(2, 6))
    H = random_hermitian(d, rng)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    W = qt.ergotropy(rho, H)
    split = qt.ergotropy_split(rho, H)
    assert W > -1e-12
    assert split["coherent"] > -1e-12 and split["incoherent"] > -1e-12
    assert abs(qt.ergotropy(qt.passive_state(rho, H), H)) < 1e-12
    assert abs(qt.ergotropy(qt.thermal_state(H, rng.uniform(0.2, 3)), H)) < 1e-12
    assert qt.batteries.asymptotic_ergotropy(rho, H) >= W - 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_local_baths_obey_the_second_law_with_bare_hamiltonian_heat(seed):
    """De Chiara et al., NJP 20, 113024 (2018), for arbitrary couplings.

    Each local dissipator obeys detailed balance with respect to H0, so
    Spohn's inequality gives -sum_k J_k^(H0) / T_k >= 0 in any steady state,
    however strong the (unphysical-for-local-ME) coupling.
    """
    rng = np.random.default_rng(300 + seed)
    dims = [[2, 2], [2, 3], [2, 2, 2]][seed % 3]
    local = [np.diag(np.sort(rng.uniform(0.3, 2, n))).astype(complex) for n in dims]
    H0 = sum(qt.embed(h, i, dims) for i, h in enumerate(local))
    H = H0 + random_hermitian(int(np.prod(dims)), rng, scale=rng.uniform(0.1, 1.5))
    baths = []
    for k, site in enumerate(rng.permutation(len(dims))[: int(rng.integers(1, len(dims) + 1))]):
        A = random_hermitian(dims[site], rng)
        baths.append(qt.local_bath(local[site], A, rng.uniform(0.3, 4.0), int(site), dims,
                                   gamma=10 ** rng.uniform(-2, 0), name=f"b{k}"))
    try:
        r = qt.analyze(H, baths, energy=H0)
    except qt.QThermoError:
        pytest.skip("random coupling produced a non-unique steady state")
    scale = max(abs(J) for J in r.currents.values())
    assert r.entropy_production_rate > -1e-10 * max(scale, 1.0)
