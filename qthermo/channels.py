"""Collapse (jump) operators for thermal baths and common noise channels.

Every function returns a list of numpy arrays that can be handed straight to
``qthermo.solver`` -- or, unchanged, to ``qutip.mesolve`` / ``qutip.mcsolve``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigma_x", "sigma_y", "sigma_z", "sigma_plus", "sigma_minus",
    "qubit_hamiltonian",
    "thermal_bath",
    "amplitude_damping",
    "pure_dephasing",
    "bit_flip",
    "mean_occupation",
]

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_plus = np.array([[0, 0], [1, 0]], dtype=complex)   # |1><0|, excitation
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|, relaxation


def qubit_hamiltonian(omega: float) -> np.ndarray:
    """H = (omega / 2) sigma_z, so the gap between |0> and |1> is omega.

    With this convention the excited state |1> is the second basis vector and
    sits at energy -omega/2... which is the wrong way round, so we flip the
    sign: H = -(omega/2) sigma_z puts |1> at +omega/2 and |0> at -omega/2.
    """
    return -0.5 * omega * sigma_z


def mean_occupation(omega: float, temperature: float) -> float:
    """Bose-Einstein occupation of a mode of frequency omega at temperature T."""
    if temperature <= 0:
        return 0.0
    x = omega / temperature
    if x > 700:  # avoid overflow; occupation is numerically zero here
        return 0.0
    return 1.0 / np.expm1(x)


def thermal_bath(gamma: float, omega: float, temperature: float) -> list[np.ndarray]:
    """Emission and absorption operators for a qubit coupled to a thermal bath.

    The rates satisfy detailed balance, so the unique steady state of the
    resulting Lindbladian is the Gibbs state at ``temperature`` -- this is what
    makes the bath genuinely thermal rather than just dissipative.
    """
    n_bar = mean_occupation(omega, temperature)
    return [
        np.sqrt(gamma * (1.0 + n_bar)) * sigma_minus,  # emission into bath
        np.sqrt(gamma * n_bar) * sigma_plus,           # absorption from bath
    ]


def amplitude_damping(gamma: float) -> list[np.ndarray]:
    """Zero-temperature relaxation only (T1-type)."""
    return [np.sqrt(gamma) * sigma_minus]


def pure_dephasing(gamma: float) -> list[np.ndarray]:
    """Pure dephasing (T2-type): destroys coherence, leaves populations alone.

    Because it commutes with a diagonal H, this channel produces *no* heat --
    a structural fact that ``examples/channel_comparison.py`` demonstrates.
    """
    return [np.sqrt(gamma / 2.0) * sigma_z]


def bit_flip(gamma: float) -> list[np.ndarray]:
    """Symmetric bit-flip channel (infinite-temperature-like population mixing)."""
    return [np.sqrt(gamma / 2.0) * sigma_x]
