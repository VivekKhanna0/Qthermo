"""Input validation.

A tool whose selling point is trustworthy numbers has to fail loudly on bad
input rather than silently returning a plausible-looking float. Every check
here corresponds to a physical requirement, and the error messages say which
one was violated rather than surfacing a linear-algebra traceback.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QThermoError",
    "check_density_matrix",
    "check_hermitian",
    "check_operator_shape",
    "check_temperature",
    "check_duration",
]


class QThermoError(ValueError):
    """Raised when an input violates a physical requirement."""


def check_hermitian(operator, name: str = "Hamiltonian", tol: float = 1e-10) -> np.ndarray:
    operator = np.asarray(operator, dtype=complex)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise QThermoError(
            f"{name} must be a square matrix, got shape {operator.shape}"
        )
    deviation = np.max(np.abs(operator - operator.conj().T))
    if deviation > tol:
        raise QThermoError(
            f"{name} is not Hermitian (max |H - H†| = {deviation:.2e}). "
            "Observables and Hamiltonians must be Hermitian, or energies "
            "will come out complex."
        )
    return operator


def check_density_matrix(rho, name: str = "state", tol: float = 1e-8) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise QThermoError(
            f"{name} must be a square density matrix, got shape {rho.shape}"
        )

    trace = np.trace(rho)
    if abs(trace - 1.0) > tol:
        raise QThermoError(
            f"{name} has trace {trace.real:.6f}, not 1. "
            "Normalise it with rho / np.trace(rho)."
        )

    deviation = np.max(np.abs(rho - rho.conj().T))
    if deviation > tol:
        raise QThermoError(
            f"{name} is not Hermitian (max |rho - rho†| = {deviation:.2e})."
        )

    eigenvalues = np.linalg.eigvalsh(rho)
    if eigenvalues.min() < -tol:
        raise QThermoError(
            f"{name} has a negative eigenvalue ({eigenvalues.min():.2e}), so it "
            "is not a valid physical state."
        )
    return rho


def check_operator_shape(operators, dim: int, name: str = "collapse operator"):
    checked = []
    for index, operator in enumerate(operators):
        operator = np.asarray(operator, dtype=complex)
        if operator.shape != (dim, dim):
            raise QThermoError(
                f"{name} {index} has shape {operator.shape}, but the system "
                f"has dimension {dim}. All operators must act on the same "
                "Hilbert space as the state."
            )
        checked.append(operator)
    return checked


def check_temperature(temperature, name: str = "temperature") -> float:
    if temperature is None:
        return None
    temperature = float(temperature)
    if temperature <= 0:
        raise QThermoError(
            f"{name} must be positive, got {temperature}. Entropy production "
            "divides by T, so T = 0 is undefined; use a small positive value "
            "to approach the zero-temperature limit."
        )
    return temperature


def check_duration(duration, name: str = "duration") -> float:
    duration = float(duration)
    if duration <= 0:
        raise QThermoError(
            f"{name} must be positive, got {duration}. A stroke of zero or "
            "negative length has no thermodynamic content."
        )
    return duration
