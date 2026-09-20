"""Physics tests for the spatially resolved (Phase 3) machinery."""

import numpy as np
import pytest

from qthermo import (
    Cycle,
    QThermoError,
    Stroke,
    embed,
    mutual_information,
    partial_trace,
    qubit_hamiltonian,
    resolve_cycle,
    resolve_stroke,
    sigma_x,
    sigma_z,
    thermal_bath,
    thermal_state,
    total_correlation,
    von_neumann_entropy,
)

DIMS2 = [2, 2]
DIMS3 = [2, 2, 2]


def product_state(*singles):
    rho = np.array([[1.0 + 0.0j]])
    for s in singles:
        rho = np.kron(rho, s)
    return rho


def bell_state():
    psi = np.zeros(4, dtype=complex)
    psi[0] = psi[3] = 1 / np.sqrt(2)
    return np.outer(psi, psi.conj())


# --- partial trace ----------------------------------------------------------

def test_partial_trace_of_product_returns_factors():
    a = thermal_state(qubit_hamiltonian(1.0), 0.5)
    b = thermal_state(qubit_hamiltonian(2.0), 1.5)
    rho = product_state(a, b)
    assert np.allclose(partial_trace(rho, 0, DIMS2), a, atol=1e-12)
    assert np.allclose(partial_trace(rho, 1, DIMS2), b, atol=1e-12)


def test_partial_trace_preserves_trace_and_hermiticity():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = m @ m.conj().T
    rho /= np.trace(rho).real
    for keep in (0, 1, 2, [0, 2], [1, 2]):
        reduced = partial_trace(rho, keep, DIMS3)
        assert np.isclose(np.trace(reduced).real, 1.0, atol=1e-12)
        assert np.allclose(reduced, reduced.conj().T, atol=1e-12)
        assert np.linalg.eigvalsh(reduced).min() > -1e-12


def test_partial_trace_middle_site_of_three():
    a = thermal_state(qubit_hamiltonian(1.0), 0.4)
    b = thermal_state(qubit_hamiltonian(3.0), 0.9)
    c = thermal_state(qubit_hamiltonian(2.0), 2.0)
    rho = product_state(a, b, c)
    assert np.allclose(partial_trace(rho, 1, DIMS3), b, atol=1e-12)
    assert np.allclose(partial_trace(rho, [0, 2], DIMS3), np.kron(a, c), atol=1e-12)


def test_partial_trace_rejects_bad_dims():
    rho = np.eye(4, dtype=complex) / 4
    with pytest.raises(QThermoError):
        partial_trace(rho, 0, [2, 2, 2])
    with pytest.raises(QThermoError):
        partial_trace(rho, 5, DIMS2)


# --- correlations -----------------------------------------------------------

def test_product_state_has_zero_correlation():
    a = thermal_state(qubit_hamiltonian(1.0), 0.5)
    b = thermal_state(qubit_hamiltonian(2.0), 1.5)
    assert abs(total_correlation(product_state(a, b), DIMS2)) < 1e-12


def test_bell_state_saturates_mutual_information():
    # Maximally entangled: each marginal is maximally mixed (ln 2), the joint
    # state is pure (0), so I = 2 ln 2 -- the algebraic maximum for two qubits.
    rho = bell_state()
    assert np.isclose(mutual_information(rho, 0, 1, DIMS2), 2 * np.log(2), atol=1e-10)
    assert np.isclose(total_correlation(rho, DIMS2), 2 * np.log(2), atol=1e-10)


def test_total_correlation_is_non_negative():
    rng = np.random.default_rng(7)
    for _ in range(5):
        m = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        rho = m @ m.conj().T
        rho /= np.trace(rho).real
        assert total_correlation(rho, DIMS3) > -1e-12


# --- embedding --------------------------------------------------------------

def test_embed_places_operator_on_the_right_site():
    assert np.allclose(embed(sigma_z, 0, DIMS2), np.kron(sigma_z, np.eye(2)))
    assert np.allclose(embed(sigma_z, 1, DIMS2), np.kron(np.eye(2), sigma_z))
    assert np.allclose(
        embed(sigma_x, 1, DIMS3),
        np.kron(np.kron(np.eye(2), sigma_x), np.eye(2)),
    )


def test_embed_rejects_wrong_site():
    with pytest.raises(QThermoError):
        embed(sigma_z, 3, DIMS2)


# --- thermodynamic resolution ----------------------------------------------

def two_qubit_cycle(coupling=0.0, t_hot=2.0, t_cold=0.3):
    """Two qubits, each with its own bath, optionally coupled."""
    omega_a, omega_b = 1.0, 1.4
    H_a, H_b = qubit_hamiltonian(omega_a), qubit_hamiltonian(omega_b)
    H = embed(H_a, 0, DIMS2) + embed(H_b, 1, DIMS2)
    if coupling:
        H = H + coupling * (embed(sigma_x, 0, DIMS2) @ embed(sigma_x, 1, DIMS2))

    c_ops = [
        embed(op, 0, DIMS2) for op in thermal_bath(0.4, omega_a, t_hot)
    ] + [embed(op, 1, DIMS2) for op in thermal_bath(0.4, omega_b, t_hot)]

    cycle = Cycle([Stroke("contact", H, duration=1.5, c_ops=c_ops, temperature=t_hot)])
    return cycle, [H_a, H_b]


def test_uncoupled_sites_have_no_interaction_energy():
    cycle, local_H = two_qubit_cycle(coupling=0.0)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    result = cycle.run(rho0)
    breakdown = resolve_stroke(result.strokes[0], DIMS2, local_H)
    # H = H_a (x) I + I (x) H_b exactly, so local energies must add up.
    assert abs(breakdown.interaction_energy_change) < 1e-9


def test_coupled_sites_have_nonzero_interaction_energy():
    cycle, local_H = two_qubit_cycle(coupling=0.35)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    result = cycle.run(rho0)
    breakdown = resolve_stroke(result.strokes[0], DIMS2, local_H)
    # The XX term stores energy in the coupling: the per-site picture cannot
    # account for the whole dU, and the module must say so rather than hide it.
    assert abs(breakdown.interaction_energy_change) > 1e-6


def test_entropy_balance_identity_holds():
    """dS_total = sum_i dS_i - dI_corr, exactly, coupled or not."""
    for coupling in (0.0, 0.35):
        cycle, local_H = two_qubit_cycle(coupling=coupling)
        rho0 = product_state(
            thermal_state(qubit_hamiltonian(1.0), 0.3),
            thermal_state(qubit_hamiltonian(1.4), 0.3),
        )
        result = cycle.run(rho0)
        breakdown = resolve_stroke(result.strokes[0], DIMS2, local_H)
        assert breakdown.entropy_balance_residual < 1e-9


def test_local_first_law_holds_per_site():
    cycle, local_H = two_qubit_cycle(coupling=0.0)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    result = cycle.run(rho0)
    breakdown = resolve_stroke(result.strokes[0], DIMS2, local_H)
    for site in breakdown.sites:
        assert site.first_law_residual < 1e-9


def test_uncoupled_local_heats_sum_to_total():
    """With no interaction term the spatial breakdown must be exhaustive."""
    cycle, local_H = two_qubit_cycle(coupling=0.0)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    result = cycle.run(rho0)
    breakdown = resolve_stroke(result.strokes[0], DIMS2, local_H)
    assert np.isclose(
        sum(breakdown.local_heat.values()), result.strokes[0].heat, atol=1e-8
    )


def test_coupling_builds_correlation_from_a_product_state():
    cycle, local_H = two_qubit_cycle(coupling=0.35)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    result = cycle.run(rho0)
    breakdown = resolve_stroke(result.strokes[0], DIMS2, local_H)
    assert breakdown.correlation_initial < 1e-10  # product state in
    assert breakdown.correlation_change > 1e-6    # correlated state out


def test_asymmetric_baths_identify_the_dominant_site():
    """A bath on one site only: that site must carry the heat."""
    omega_a, omega_b = 1.0, 1.0
    H_a, H_b = qubit_hamiltonian(omega_a), qubit_hamiltonian(omega_b)
    H = embed(H_a, 0, DIMS2) + embed(H_b, 1, DIMS2)
    c_ops = [embed(op, 1, DIMS2) for op in thermal_bath(0.5, omega_b, 3.0)]
    cycle = Cycle([Stroke("hot_on_b", H, duration=1.5, c_ops=c_ops, temperature=3.0)])

    rho0 = product_state(
        thermal_state(H_a, 0.2), thermal_state(H_b, 0.2)
    )
    breakdown = resolve_stroke(cycle.run(rho0).strokes[0], DIMS2, [H_a, H_b])
    assert breakdown.dominant_site == 1
    assert abs(breakdown.local_heat[0]) < 1e-9
    assert breakdown.local_heat[1] > 1e-6


def test_resolve_cycle_covers_every_stroke():
    cycle, local_H = two_qubit_cycle(coupling=0.0)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    breakdowns = resolve_cycle(cycle.run(rho0), DIMS2, local_H)
    assert len(breakdowns) == len(cycle.strokes)
    assert breakdowns[0].stroke == "contact"


def test_resolve_stroke_rejects_wrong_number_of_local_hamiltonians():
    cycle, local_H = two_qubit_cycle(coupling=0.0)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    result = cycle.run(rho0)
    with pytest.raises(QThermoError):
        resolve_stroke(result.strokes[0], DIMS2, [local_H[0]])


def test_report_runs_and_mentions_every_site():
    cycle, local_H = two_qubit_cycle(coupling=0.35)
    rho0 = product_state(
        thermal_state(qubit_hamiltonian(1.0), 0.3),
        thermal_state(qubit_hamiltonian(1.4), 0.3),
    )
    text = resolve_stroke(cycle.run(rho0).strokes[0], DIMS2, local_H).report()
    assert "per-site breakdown" in text
    assert "interaction energy change" in text
    assert "entropy balance residual" in text
