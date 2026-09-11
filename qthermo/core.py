"""Core thermodynamic quantities for open quantum systems.

Units: hbar = k_B = 1 throughout. Energies and temperatures are therefore in
the same units, and inverse temperature beta = 1 / T.

Sign convention (standard in quantum thermodynamics):
    Q > 0  : heat flows INTO the system from the bath
    W > 0  : work done ON the system by the external driving field
    dU = Q + W
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "internal_energy",
    "von_neumann_entropy",
    "thermal_state",
    "free_energy",
    "heat_work_increments",
    "entropy_production",
]


def _as_matrix(obj) -> np.ndarray:
    """Accept a numpy array or a QuTiP Qobj and return a dense numpy array.

    This is the only place qthermo touches QuTiP, so the package has no hard
    dependency on it: the physics runs on plain arrays.
    """
    if hasattr(obj, "full"):  # qutip.Qobj
        return np.asarray(obj.full(), dtype=complex)
    return np.asarray(obj, dtype=complex)


def internal_energy(rho, H) -> float:
    """Internal energy U = Tr[H rho]."""
    rho, H = _as_matrix(rho), _as_matrix(H)
    return float(np.real(np.trace(H @ rho)))


def von_neumann_entropy(rho, tol: float = 1e-12) -> float:
    """von Neumann entropy S = -Tr[rho ln rho], in nats (k_B = 1)."""
    rho = _as_matrix(rho)
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > tol]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def thermal_state(H, temperature: float):
    """Gibbs state rho = exp(-H/T) / Z."""
    H = _as_matrix(H)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    weights = np.exp(-(eigenvalues - eigenvalues.min()) / temperature)
    weights /= weights.sum()
    return (eigenvectors * weights) @ eigenvectors.conj().T


def free_energy(H, temperature: float) -> float:
    """Equilibrium free energy F = -T ln Z."""
    H = _as_matrix(H)
    eigenvalues = np.linalg.eigvalsh(H)
    shift = eigenvalues.min()
    partition = np.sum(np.exp(-(eigenvalues - shift) / temperature))
    return float(shift - temperature * np.log(partition))


def heat_work_increments(rho_a, rho_b, H_a, H_b) -> tuple[float, float]:
    """Heat and work exchanged over one integration step.

    Uses the midpoint (trapezoidal) split of the Alicki decomposition:

        W = Tr[(H_b - H_a) (rho_a + rho_b) / 2]
        Q = Tr[(H_a + H_b) / 2 (rho_b - rho_a)]

    This particular split is chosen because it satisfies the first law
    *exactly* at every step, not just to first order in dt:

        Q + W = Tr[H_b rho_b] - Tr[H_a rho_a] = dU

    which makes ``first_law_residual`` a genuine test of the solver rather
    than a test of the discretisation scheme.
    """
    rho_a, rho_b = _as_matrix(rho_a), _as_matrix(rho_b)
    H_a, H_b = _as_matrix(H_a), _as_matrix(H_b)

    dH = H_b - H_a
    drho = rho_b - rho_a

    work = float(np.real(np.trace(dH @ (rho_a + rho_b) / 2)))
    heat = float(np.real(np.trace((H_a + H_b) / 2 @ drho)))
    return heat, work


def entropy_production(rho_initial, rho_final, heat: float, temperature: float,
                       tolerance: float = 1e-9) -> float:
    """Irreversible entropy production for a stroke in contact with one bath.

        sigma = Delta S_vN - Q / T

    By Spohn's theorem this is non-negative for Markovian dynamics with a
    thermal fixed point. QuTiP provides ``entropy_vn`` (the state entropy at
    an instant) but not entropy production, which needs the heat flow and
    the bath temperature as well -- this is the one quantity here that is
    not a repackaging of something QuTiP already exposes.
    """
    delta_s = von_neumann_entropy(rho_final) - von_neumann_entropy(rho_initial)
    sigma = float(delta_s - heat / temperature)

    if sigma < -tolerance:
        import warnings
        warnings.warn(
            f"entropy production is negative ({sigma:.3e}), which is "
            "physically impossible for Markovian dynamics with a thermal "
            f"fixed point. The declared bath temperature (T = {temperature}) "
            "is probably inconsistent with the collapse operators -- for "
            "example labelling a zero-temperature amplitude-damping channel "
            "with a finite temperature. Entropy production is only meaningful "
            "relative to the actual bath temperature.",
            RuntimeWarning,
            stacklevel=2,
        )
    return sigma
