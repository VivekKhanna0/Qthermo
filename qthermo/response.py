"""Linear response of multi-terminal machines: Onsager matrix, conductances,
thermal-transistor gain.

For a machine coupled to baths ``k`` at temperatures ``T_k``, the steady-state
heat currents respond to small temperature changes through

    dJ_k = sum_l G_kl dT_l,            G_kl = dJ_k / dT_l      (conductance matrix)

Around equilibrium (all ``T_k = T``) it is natural to use the affinities
``x_l = 1/T - 1/T_l`` (positive when bath ``l`` is hotter), with
``J_k = sum_l L_kl x_l``. Time-reversal symmetry then fixes

    L_kl = L_lk                           Onsager reciprocity
    L >= 0                                second law
    sum_k L_kl = 0                        energy conservation (no work source)

These are exact, so ``response`` doubles as a consistency check of any bath
model. Two derived quantities matter in practice:

* the **coupling** ``q_kl = L_kl / sqrt(L_kk L_ll)``, with ``|q| = 1`` meaning
  tight coupling (the currents are locked together; the machine can reach
  Carnot efficiency in the linear regime -- Kedem & Caplan, Trans. Faraday
  Soc. 61, 1897 (1965));
* the **thermal-transistor gain** ``dJ_out / dJ_control`` when only the
  control bath's temperature is varied (Joulain, Drevillon, Ezzahri &
  Ordonez-Miranda, PRL 116, 200601 (2016)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import QThermoError

__all__ = ["ThermalResponse", "response"]


@dataclass
class ThermalResponse:
    names: list
    temperatures: dict
    conductance: np.ndarray          # G_kl = dJ_k / dT_l
    currents: dict                   # at the reference point

    @property
    def at_equilibrium(self) -> bool:
        T = list(self.temperatures.values())
        return bool(np.allclose(T, T[0], rtol=1e-12))

    @property
    def onsager(self) -> np.ndarray:
        """``L_kl = dJ_k / dx_l`` with ``x_l = 1/T_ref - 1/T_l``: ``T_l^2 G_kl``."""
        T = np.array([self.temperatures[n] for n in self.names])
        return self.conductance * T[None, :] ** 2

    @property
    def reciprocity_residual(self) -> float:
        """``max |L - L^T| / max |L|``: zero at equilibrium (Onsager)."""
        L = self.onsager
        return float(np.max(np.abs(L - L.T)) / max(np.max(np.abs(L)), 1e-300))

    @property
    def conservation_residual(self) -> float:
        """``max_l |sum_k G_kl| / max |G|``: zero when no work is exchanged."""
        G = self.conductance
        return float(np.max(np.abs(G.sum(axis=0))) / max(np.max(np.abs(G)), 1e-300))

    def is_positive_semidefinite(self, tol: float = 1e-9) -> bool:
        L = self.onsager
        sym = 0.5 * (L + L.T)
        return bool(np.min(np.linalg.eigvalsh(sym)) >= -tol * max(np.max(np.abs(L)), 1e-300))

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no bath {name!r}; baths: {self.names}") from None

    def coupling(self, a: str, b: str) -> float:
        """Kedem-Caplan coupling ``q = L_ab / sqrt(L_aa L_bb)``."""
        L = self.onsager
        i, j = self.index(a), self.index(b)
        return float(L[i, j] / np.sqrt(L[i, i] * L[j, j]))

    def amplification(self, control: str, output: str) -> float:
        """Transistor gain ``dJ_output / dJ_control`` from varying ``T_control``.

        ``|gain| > 1``: a small change of the heat injected at the control
        terminal changes the output current by more -- thermal amplification.
        """
        G = self.conductance
        c, o = self.index(control), self.index(output)
        if G[c, c] == 0:
            raise QThermoError(f"the {control!r} current does not respond to its own temperature")
        return float(G[o, c] / G[c, c])

    def report(self) -> str:
        width = max(len(n) for n in self.names) + 2
        lines = ["conductance dJ_k/dT_l (rows k: current, columns l: temperature)",
                 " " * width + "".join(f"{n:>13}" for n in self.names)]
        for i, n in enumerate(self.names):
            lines.append(f"{n:<{width}}" + "".join(f"{v:>13.4e}" for v in self.conductance[i]))
        lines.append(f"energy conservation residual: {self.conservation_residual:.2e}")
        if self.at_equilibrium:
            lines.append(f"Onsager reciprocity residual: {self.reciprocity_residual:.2e}"
                         f"   positive semidefinite: {self.is_positive_semidefinite()}")
        return "\n".join(lines)


def response(build, temperatures: dict, h: float = 1e-4) -> ThermalResponse:
    """Conductance matrix ``dJ_k/dT_l`` of a steady-state machine.

    Parameters
    ----------
    build : callable
        ``build(temperatures: dict) -> Model`` (or anything with
        ``.analyze()`` returning a steady state with ``.currents``).
    temperatures : dict
        Reference temperature of each bath, by bath name. All equal gives the
        Onsager (equilibrium) response.
    h : float
        Relative temperature step for central differences.

    Examples
    --------
    >>> build = lambda T: qt.models.absorption_refrigerator(
    ...     T_c=T["cold"], T_h=T["hot"], T_r=T["room"], master_equation="global")
    >>> r = qt.response(build, {"cold": 1.0, "hot": 1.0, "room": 1.0})
    >>> r.reciprocity_residual      # ~1e-8: Onsager holds
    """
    names = list(temperatures)
    try:
        base = build(dict(temperatures)).analyze()
    except KeyError as exc:
        raise QThermoError(
            f"build() looked up temperature {exc} which is not among the given "
            f"bath names {names}; name every bath the builder uses") from None
    missing = set(names) - set(base.currents)
    if missing:
        raise QThermoError(f"the model has no bath(s) named {sorted(missing)}")
    G = np.zeros((len(names), len(names)))
    for l, name in enumerate(names):
        step = h * temperatures[name]
        up, down = dict(temperatures), dict(temperatures)
        up[name] += step
        down[name] -= step
        J_up = build(up).analyze().currents
        J_down = build(down).analyze().currents
        for k, other in enumerate(names):
            G[k, l] = (J_up[other] - J_down[other]) / (2 * step)
    return ThermalResponse(names, dict(temperatures), G,
                           {n: base.currents[n] for n in names})
