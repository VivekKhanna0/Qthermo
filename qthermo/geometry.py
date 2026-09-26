"""Thermodynamic geometry: minimum-dissipation protocols for any driven system.

When a control parameter ``lambda`` of ``H(lambda)`` is changed slowly while
the system stays in contact with a thermalising bath, the state lags behind
the instantaneous Gibbs state ``pi(lambda)`` by

    delta rho = L^+ (d pi / dt) = L^+ (d_j pi) lambda_dot_j

(``L^+`` the Drazin inverse of the instantaneous Liouvillian), and the work
done exceeds the free-energy change by

    W_ex = int lambda_dot_i g_ij(lambda) lambda_dot_j dt,
    g_ij = Tr[ d_i H  L^+ (d_j pi) ].

``g`` is a Riemannian metric on control space (Sivak & Crooks, PRL 108,
190602 (2012); Scandi & Perarnau-Llobet, Quantum 3, 197 (2019); Abiuso et al.,
Entropy 22, 1076 (2020)). Two consequences make it useful:

* along a fixed path in control space, the schedule that minimises the excess
  work runs at constant speed in the metric, and the minimum is ``L^2 / tau``
  with ``L`` the thermodynamic length of the path;
* comparing lengths of different paths ranks protocols before any simulation.

Everything here is computed from the same Liouvillian the simulations use, so a
prediction and a finite-time simulation can be compared directly. The metric
includes coherent (quantum-friction) contributions whenever ``d H`` does not
commute with ``H``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import _as_matrix, thermal_state
from .fluctuations import _drazin_solve
from .steady import _vec, _unvec, liouvillian
from .validation import QThermoError

__all__ = ["friction", "OptimalSchedule", "optimal_schedule", "excess_work"]


def _derivative(H_of, lam, h):
    return (_as_matrix(H_of(lam + h)) - _as_matrix(H_of(lam - h))) / (2 * h)


def friction(H_of, baths_of, temperature: float, lam: float, h: float = 1e-5) -> float:
    """Thermodynamic friction ``g(lambda)`` for a single control parameter.

    Parameters
    ----------
    H_of : callable
        ``H_of(lambda) -> Hamiltonian``.
    baths_of : callable
        ``baths_of(H) -> list of Bath`` (or collapse operators): the
        thermalising dissipation at that Hamiltonian, e.g.
        ``lambda H: [qt.davies_bath(H, qt.sigma_x, T)]``. Its fixed point
        must be the Gibbs state of ``H`` at ``temperature``.
    temperature : float
    lam : float
        Point at which to evaluate the metric.
    """
    H = _as_matrix(H_of(lam))
    dim = H.shape[0]
    pi = thermal_state(H, temperature)
    dH = _derivative(H_of, lam, h)
    dpi = (thermal_state(_as_matrix(H_of(lam + h)), temperature)
           - thermal_state(_as_matrix(H_of(lam - h)), temperature)) / (2 * h)
    L = liouvillian(H, baths_of(H), sparse=False)
    residual = np.max(np.abs(L @ _vec(pi)))
    if residual > 1e-8 * max(np.max(np.abs(L)), 1.0):
        raise QThermoError(
            f"the dissipation at lambda={lam:g} does not have the Gibbs state at "
            f"T={temperature:g} as a fixed point (|L pi| = {residual:.2e}); "
            "thermodynamic geometry needs a thermalising bath at that temperature")
    # The lag solves L x = d pi (to leading order in the driving speed), and
    # the excess power is Tr[dH x] lambda_dot^2.
    lag = _unvec(_drazin_solve(L, _vec(dpi), _vec(pi), dim), dim)
    return float(np.real(np.trace(dH @ lag)))


@dataclass
class OptimalSchedule:
    """Minimum-dissipation schedule along a path ``lambda(s)``, ``s`` in [0, 1]."""

    path: object                 # s -> lambda
    s_grid: np.ndarray
    metric: np.ndarray           # g along the path, including (d lambda / ds)^2
    arc: np.ndarray              # normalised thermodynamic arc length at s_grid

    @property
    def length(self) -> float:
        """Thermodynamic length ``L = int sqrt(g) |d lambda|``."""
        from scipy.integrate import trapezoid
        return float(trapezoid(np.sqrt(np.clip(self.metric, 0, None)), self.s_grid))

    def minimum_excess(self, tau: float) -> float:
        """``L^2 / tau``: the least excess work of any schedule along this path."""
        return self.length ** 2 / tau

    def s_of(self, x):
        """Path coordinate ``s`` reached at fraction ``x = t / tau`` of the time."""
        return np.interp(x, self.arc, self.s_grid)

    def lambda_at(self, t: float, tau: float):
        return self.path(float(self.s_of(np.clip(t / tau, 0.0, 1.0))))

    def hamiltonian(self, H_of, tau: float):
        """``t -> H(lambda(t))`` for a stroke of duration ``tau``."""
        return lambda t: _as_matrix(H_of(self.lambda_at(t, tau)))

    def predicted_excess(self, schedule, tau: float) -> float:
        """Slow-driving excess work of any other schedule ``x -> s`` along the path."""
        x = np.linspace(0.0, 1.0, 4001)
        s = np.array([float(schedule(v)) for v in x])
        ds_dx = np.gradient(s, x)
        g = np.interp(s, self.s_grid, self.metric)
        integrand = g * ds_dx ** 2
        return float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(x)) / tau)


def optimal_schedule(H_of, baths_of, temperature: float, path, points: int = 201,
                     h: float = 1e-5) -> OptimalSchedule:
    """Constant-speed (geodesic) schedule along a path in control space.

    ``path`` is either ``(lambda_start, lambda_end)`` for a single parameter or
    a callable ``s -> lambda`` for a path of any shape in a many-parameter
    space; then ``H_of`` takes whatever ``path`` returns. The friction is
    evaluated along the path direction.
    """
    if not callable(path):
        start, end = path
        def path(s, _a=float(start), _b=float(end)):
            return _a + s * (_b - _a)
    s_grid = np.linspace(0.0, 1.0, points)

    def H_along(s):
        return H_of(path(s))

    g = np.array([friction(H_along, baths_of, temperature, float(s), h=h)
                  for s in s_grid])
    speed = np.sqrt(np.clip(g, 0.0, None))
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(s_grid))])
    if arc[-1] <= 0:
        raise QThermoError("zero thermodynamic length: the path costs nothing to drive")
    return OptimalSchedule(path, s_grid, g, arc / arc[-1])


def excess_work(H_of_t, baths_of, temperature: float, tau: float, rho0=None,
                steps: int = 2000) -> dict:
    """Simulate a driven stroke with instantaneous thermalising baths.

    Returns ``{"work", "delta_F", "dissipation"}`` where
    ``dissipation = T * sigma`` (heat released beyond ``T dS``), the quantity
    the metric predicts. The system starts in the Gibbs state of ``H(0)``.
    """
    from .core import free_energy
    from .cycle import Cycle, Stroke

    H0 = _as_matrix(H_of_t(0.0))
    rho0 = thermal_state(H0, temperature) if rho0 is None else _as_matrix(rho0)

    def baths(t):
        return baths_of(_as_matrix(H_of_t(t)))

    stroke = Stroke("drive", H_of_t, tau, baths, steps=steps)
    result = Cycle([stroke]).run(rho0).strokes[0]
    delta_F = free_energy(_as_matrix(H_of_t(tau)), temperature) - free_energy(H0, temperature)
    return {"work": result.work, "delta_F": delta_F,
            "dissipation": temperature * result.entropy_production,
            "result": result}
