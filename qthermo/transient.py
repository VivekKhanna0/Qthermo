"""Transient thermodynamics: switching a machine on and watching it settle.

``analyze`` gives the steady state; experiments and applications often care
about what happens first. How long until a refrigerator cools its target?
Does the target dip below its steady-state temperature on the way (the
'single-shot cooling' of Mitchison, Woods, Prior & Huber, NJP 17, 115013
(2015))? How much heat flows before the currents settle?

``transient`` integrates the master equation from a chosen initial state and
records, at every output time, each bath's heat current, the cumulative heat
from each bath, and (for a Model) every site's virtual temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .core import _as_matrix
from .network import virtual_temperature
from .solver import evolve
from .subsystems import partial_trace
from .validation import QThermoError

__all__ = ["TransientResult", "transient", "product_thermal_state"]


@dataclass
class TransientResult:
    times: np.ndarray
    states: list
    currents: dict                    # bath -> J_k(t) (into system)
    heat: dict                        # bath -> cumulative Q_k(t)
    energy: np.ndarray                # Tr[E rho(t)]
    virtual_temperatures: np.ndarray | None = None   # (n_sites, n_times)
    site_names: list | None = None

    def energy_balance_residual(self) -> float:
        """``max_t |E(t) - E(0) - sum_k Q_k(t)|`` (quadrature accuracy)."""
        total = sum(self.heat.values())
        return float(np.max(np.abs(self.energy - self.energy[0] - total)))

    def settling_time(self, bath: str, rtol: float = 0.01) -> float:
        """First time after which ``J_bath`` stays within ``rtol`` of its final value."""
        J = self.currents[bath]
        final = J[-1]
        outside = np.abs(J - final) > rtol * max(abs(final), 1e-300)
        if not outside.any():
            return float(self.times[0])
        last = np.nonzero(outside)[0][-1]
        return float(self.times[min(last + 1, len(self.times) - 1)])

    def minimum_temperature(self, site: int = 0) -> tuple[float, float]:
        """``(T*_min, time)`` of a site's virtual temperature."""
        if self.virtual_temperatures is None:
            raise QThermoError("virtual temperatures need a Model (or dims/local_H)")
        T = np.where(self.virtual_temperatures[site] > 0, self.virtual_temperatures[site], np.inf)
        k = int(np.argmin(T))
        return float(T[k]), float(self.times[k])

    def report(self, site: int | None = 0) -> str:
        lines = [f"transient over t in [0, {self.times[-1]:g}]",
                 f"{'bath':<10}{'J(0)':>13}{'J(end)':>13}{'Q total':>13}{'settles by':>13}"]
        for name, J in self.currents.items():
            lines.append(f"{name:<10}{J[0]:>13.4e}{J[-1]:>13.4e}{self.heat[name][-1]:>13.4e}"
                         f"{self.settling_time(name):>13.4g}")
        if site is not None and self.virtual_temperatures is not None:
            T_min, t_min = self.minimum_temperature(site)
            name = self.site_names[site] if self.site_names else f"site {site}"
            lines.append(f"{name}: T* starts {self.virtual_temperatures[site][0]:.4g}, "
                         f"minimum {T_min:.4g} at t = {t_min:.4g}, ends "
                         f"{self.virtual_temperatures[site][-1]:.4g}")
        lines.append(f"energy balance residual: {self.energy_balance_residual():.2e}")
        return "\n".join(lines)


def product_thermal_state(local_H, temperatures) -> np.ndarray:
    """``(x)_i exp(-h_i/T_i)/Z_i``: every site thermal with its own bath, uncorrelated.

    The natural 'machine switched on at t = 0' initial state.
    """
    from .core import thermal_state
    rho = np.array([[1.0 + 0.0j]])
    for h, T in zip(local_H, temperatures):
        rho = np.kron(rho, thermal_state(h, T))
    return rho


def transient(source, rho0, duration: float, steps: int = 400, baths=None,
              energy=None, dims=None, local_H=None, site_names=None) -> TransientResult:
    """Integrate a continuous machine from ``rho0`` and record its thermodynamics.

    Parameters
    ----------
    source : Model or Hamiltonian
        A ``Model`` supplies baths, the consistent energy operator and the
        site structure; with a Hamiltonian pass ``baths`` (and optionally
        ``dims``/``local_H`` for virtual temperatures).
    rho0 : array
        Initial state, e.g. :func:`product_thermal_state`.
    duration, steps : float, int
        Time window and number of output intervals.
    energy : array, optional
        Energy operator for heat currents (defaults as in ``Model.analyze``).
    """
    from .models import Model

    if isinstance(source, Model):
        H, baths = source.H, source.baths
        if energy is None:
            energy = source.energy if source.energy is not None else (
                source.H0 if source.master_equation == "local" else source.H)
        dims = source.dims if dims is None else dims
        local_H = source.local_H if local_H is None else local_H
        site_names = source.site_names if site_names is None else site_names
    else:
        H = _as_matrix(source)
        if baths is None:
            raise QThermoError("transient(H, rho0, ...) needs baths=")
        energy = H if energy is None else _as_matrix(energy)
    baths = list(baths)
    E = _as_matrix(energy)

    c_ops = [L for b in baths for L in b.c_ops]
    out = evolve(rho0, H, c_ops, duration=duration, steps=steps)
    times, states = out["times"], out["states"]
    currents = {b.name: np.array([b.heat_current(r, E) for r in states]) for b in baths}
    heat = {k: cumulative_trapezoid(J, times, initial=0.0) for k, J in currents.items()}
    energy_t = np.array([float(np.real(np.trace(E @ r))) for r in states])

    T_v = None
    if dims is not None and local_H is not None:
        T_v = np.array([[virtual_temperature(partial_trace(r, i, dims), local_H[i])
                         for r in states] for i in range(len(dims))])
    return TransientResult(times, states, currents, heat, energy_t, T_v, site_names)
