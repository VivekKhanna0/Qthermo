"""Lindblad integrator with thermodynamic accounting built into the loop.

The point of doing the bookkeeping *inside* the solver rather than after it is
that heat and work are path quantities: they depend on the whole trajectory,
not just on the endpoints. Reconstructing them from a sparse output grid
introduces discretisation error that is easy to mistake for physics.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .core import _as_matrix, heat_work_increments
from .validation import (
    check_density_matrix,
    check_hermitian,
    check_operator_shape,
    check_duration,
)

__all__ = ["lindbladian", "evolve"]


def lindbladian(rho: np.ndarray, H: np.ndarray, c_ops: list[np.ndarray]) -> np.ndarray:
    """Right-hand side of the Lindblad master equation."""
    drho = -1j * (H @ rho - rho @ H)
    for L in c_ops:
        L_dag = L.conj().T
        drho += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
    return drho


def _pack(rho: np.ndarray) -> np.ndarray:
    return np.concatenate([rho.real.ravel(), rho.imag.ravel()])


def _unpack(vector: np.ndarray, dim: int) -> np.ndarray:
    half = dim * dim
    return (vector[:half] + 1j * vector[half:]).reshape(dim, dim)


def evolve(rho0, H, c_ops=None, duration: float = 1.0, steps: int = 200,
           rtol: float = 1e-9, atol: float = 1e-11):
    """Integrate the Lindblad equation and accumulate heat and work.

    Parameters
    ----------
    rho0 : array or Qobj
        Initial density matrix.
    H : array, Qobj, or callable
        Hamiltonian. If callable it is evaluated as ``H(t)``, which is what
        makes driven (work-performing) strokes possible.
    c_ops : list, optional
        Collapse operators. Empty or None gives unitary evolution.
    duration : float
        Stroke duration.
    steps : int
        Number of grid points used for the thermodynamic accumulation.

    Returns
    -------
    dict with keys ``times``, ``states``, ``heat``, ``work``.
    """
    duration = check_duration(duration)
    rho0 = check_density_matrix(_as_matrix(rho0), "initial state")
    dim = rho0.shape[0]
    c_ops = check_operator_shape([_as_matrix(L) for L in (c_ops or [])], dim)

    if callable(H):
        H_of_t = H
        check_hermitian(_as_matrix(H(0.0)), "H(t=0)")
    else:
        _H = check_hermitian(_as_matrix(H))
        if _H.shape[0] != dim:
            from .validation import QThermoError
            raise QThermoError(
                f"Hamiltonian has dimension {_H.shape[0]} but the state has "
                f"dimension {dim}."
            )
        H_of_t = lambda t, _H=_H: _H

    def rhs(t, y):
        rho = _unpack(y, dim)
        return _pack(lindbladian(rho, _as_matrix(H_of_t(t)), c_ops))

    times = np.linspace(0.0, duration, steps + 1)
    solution = solve_ivp(
        rhs, (0.0, duration), _pack(rho0),
        t_eval=times, rtol=rtol, atol=atol, method="DOP853",
    )
    if not solution.success:
        raise RuntimeError(f"integration failed: {solution.message}")

    states = [_unpack(solution.y[:, k], dim) for k in range(len(times))]

    # Accumulate heat and work step by step. Because heat_work_increments uses
    # the exact midpoint split, Q + W equals dU to machine precision here.
    total_heat = 0.0
    total_work = 0.0
    for k in range(len(times) - 1):
        H_a = _as_matrix(H_of_t(times[k]))
        H_b = _as_matrix(H_of_t(times[k + 1]))
        dQ, dW = heat_work_increments(states[k], states[k + 1], H_a, H_b)
        total_heat += dQ
        total_work += dW

    return {
        "times": times,
        "states": states,
        "heat": total_heat,
        "work": total_work,
    }
