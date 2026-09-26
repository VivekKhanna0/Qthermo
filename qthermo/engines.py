"""Stroke engines with interacting, multi-qubit working media.

The single-qubit Otto cycle has one gap and one answer, ``1 - w_c / w_h``. With
an interacting working medium the spectrum no longer scales uniformly when a
control parameter changes, and the efficiency can be pushed above (or below)
the non-interacting value -- the question behind a sizeable literature on
coupled and many-body Otto engines (Thomas & Johal, PRE 83, 031135 (2011);
Campisi & Fazio, Nat. Commun. 7, 11895 (2016); Watanabe et al., PRL 124,
210603 (2020)).

This module gives two things that are otherwise rewritten per paper:

* :func:`ideal_otto` -- the quasi-static limit for *any* pair of Hamiltonians,
  computed exactly from Gibbs populations. The reference every simulation
  should approach.
* :func:`otto_cycle` -- the corresponding finite-time cycle, with global
  (Davies) baths so thermalisation targets the Gibbs state of the full
  interacting Hamiltonian, and a unitary ramp between the two Hamiltonians.

The simulated cycle is an ordinary :class:`~qthermo.cycle.Cycle`: per-site
resolution, sweeps, Pareto fronts and trajectory unravelling all apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baths import davies_bath
from .core import _as_matrix
from .cycle import Cycle, Stroke
from .validation import QThermoError, check_hermitian

__all__ = ["OttoLimit", "ideal_otto", "otto_cycle", "adiabatic_pairing"]


@dataclass
class OttoLimit:
    """Quasi-static Otto cycle between two Hamiltonians."""

    work: float            # net work done ON the medium per cycle
    heat_hot: float        # heat absorbed from the hot bath
    heat_cold: float       # heat absorbed from the cold bath
    T_cold: float
    T_hot: float
    crossings: bool = False   # level crossings along the driven strokes

    @property
    def is_engine(self) -> bool:
        return self.work < 0 and self.heat_hot > 0

    @property
    def is_refrigerator(self) -> bool:
        return self.work > 0 and self.heat_cold > 0

    @property
    def efficiency(self) -> float:
        if not self.is_engine:
            raise QThermoError("this cycle is not operating as an engine")
        return -self.work / self.heat_hot

    @property
    def cop(self) -> float:
        if not self.is_refrigerator:
            raise QThermoError("this cycle is not operating as a refrigerator")
        return self.heat_cold / self.work

    @property
    def mode(self) -> str:
        """``engine``, ``refrigerator``, ``accelerator``, ``heater`` ..."""
        from .modes import classify
        return classify(self.work, self.heat_hot, self.heat_cold)

    @property
    def carnot_efficiency(self) -> float:
        return 1.0 - self.T_cold / self.T_hot

    def report(self) -> str:
        mode = ("engine" if self.is_engine else
                "refrigerator" if self.is_refrigerator else "heater/accelerator")
        lines = [f"quasi-static Otto cycle ({mode})",
                 f"  work on medium : {self.work:+.6f}",
                 f"  Q from hot     : {self.heat_hot:+.6f}",
                 f"  Q from cold    : {self.heat_cold:+.6f}"]
        if self.is_engine:
            lines.append(f"  efficiency     : {self.efficiency:.6f} "
                         f"(Carnot {self.carnot_efficiency:.6f})")
        if self.is_refrigerator:
            lines.append(f"  COP            : {self.cop:.6f}")
        if self.crossings:
            lines.append("  note: levels cross along the drive; states were "
                         "paired by adiabatic continuation, not energy rank")
        return "\n".join(lines)


def _gibbs_populations(energies, T):
    weights = np.exp(-(energies - energies.min()) / T)
    return weights / weights.sum()


def adiabatic_pairing(H_start, H_end, path=None, points: int = 400,
                      tol: float = 1e-9) -> np.ndarray:
    """Which eigenstate of ``H_end`` each eigenstate of ``H_start`` becomes.

    Follows the eigenvectors along ``path(s)`` (default: the straight line)
    by maximum overlap between neighbouring points. Where the path has exact
    level crossings -- typically between symmetry sectors, which the dynamics
    cannot couple -- a state keeps its identity and changes its energy rank.
    Returns ``perm`` with ``perm[n]`` the energy index in ``H_end`` of the
    continuation of level ``n`` of ``H_start``.

    Degenerate levels are matched within their eigenspace; since Gibbs
    populations are equal on degenerate levels, the ambiguity does not affect
    thermodynamic quantities at the end points.
    """
    from scipy.optimize import linear_sum_assignment

    A, B = _as_matrix(H_start), _as_matrix(H_end)
    if path is None:
        def path(s):
            return A + s * (B - A)
    _, V = np.linalg.eigh(_as_matrix(path(0.0)))
    identity = np.arange(A.shape[0])   # tracked[n] = column of level n now
    tracked = identity.copy()
    for s in np.linspace(0.0, 1.0, points + 1)[1:]:
        _, W = np.linalg.eigh(_as_matrix(path(float(s))))
        overlap = np.abs(V.conj().T @ W) ** 2
        rows, cols = linear_sum_assignment(-overlap)
        step = np.empty_like(identity)
        step[rows] = cols
        tracked = step[tracked]
        V = W
    return tracked


def _has_crossings(perm) -> bool:
    return bool(np.any(perm != np.arange(len(perm))))


def ideal_otto(H_cold, H_hot, T_cold: float, T_hot: float, path=None,
               follow_crossings: bool = True) -> OttoLimit:
    """Exact quasi-static Otto cycle between two Hamiltonians.

    Isochores thermalise fully to the Gibbs state of the current Hamiltonian;
    the driven strokes are quantum adiabatic. Heat and work then follow from
    populations alone:

        Q_h = sum_n E^h_n (p^h_n - p^c_n),   Q_c = sum_n E^c_n (p^c_n - p^h_n)
        W   = -(Q_h + Q_c)

    where ``n`` labels a level *and its adiabatic continuation*. That pairing
    is the subtle part. Pairing levels by energy rank is only right if no
    levels cross between the two Hamiltonians; with a symmetry (parity in an
    Ising chain, total spin in a Heisenberg one) levels from different
    sectors cross exactly and the quasi-static cycle is different. By default
    the pairing is computed by following the eigenstates along ``path(s)``
    (straight line if not given), which is what a slow finite-time ramp
    actually does. ``follow_crossings=False`` gives the naive energy-ordered
    answer, for comparison.
    """
    H_c = check_hermitian(_as_matrix(H_cold), "H_cold")
    H_h = check_hermitian(_as_matrix(H_hot), "H_hot")
    if H_c.shape != H_h.shape:
        raise QThermoError("H_cold and H_hot must have the same dimension")
    E_c = np.linalg.eigvalsh(H_c)
    E_h = np.linalg.eigvalsh(H_h)
    if follow_crossings:
        # level n of H_hot becomes level perm[n] of H_cold on the expansion
        reverse = None if path is None else (lambda s: path(1.0 - s))
        perm = adiabatic_pairing(H_h, H_c, path=reverse)
    else:
        perm = np.arange(len(E_c))
    E_c_paired = E_c[perm]                 # cold energy of hot level n
    p_h = _gibbs_populations(E_h, T_hot)
    p_c = _gibbs_populations(E_c, T_cold)[perm]
    Q_h = float(np.sum(E_h * (p_h - p_c)))
    Q_c = float(np.sum(E_c_paired * (p_c - p_h)))
    result = OttoLimit(work=-(Q_h + Q_c), heat_hot=Q_h, heat_cold=Q_c,
                       T_cold=T_cold, T_hot=T_hot)
    result.crossings = _has_crossings(perm)
    return result


def _require_thermalization(H, baths, label):
    from .steady import steady_state
    try:
        steady_state(H, baths)
    except QThermoError as exc:
        raise QThermoError(
            f"the {label} baths cannot thermalise this working medium: {exc} "
            "Here that means every coupling operator commutes with a symmetry "
            "of the Hamiltonian, so population never moves between its "
            "symmetry sectors. Couple through an operator that breaks the "
            "symmetry (e.g. sigma_z instead of sigma_x for a transverse-field "
            "Ising medium), or pass check_thermalization=False to simulate "
            "the non-ergodic machine deliberately.") from None


def otto_cycle(H_cold, H_hot, T_cold: float, T_hot: float, couplings,
               gamma: float = 0.5, tau_iso: float = 30.0, tau_ramp: float = 5.0,
               steps: int = 300, ramp=None, spectrum=None,
               check_thermalization: bool = True) -> Cycle:
    """Four-stroke Otto cycle for an arbitrary (multi-qubit) working medium.

    Strokes are named ``hot_iso``, ``expand`` (H_hot -> H_cold), ``cold_iso``,
    ``compress`` (H_cold -> H_hot), so ``result.efficiency(("hot_iso",))`` and
    ``result.cop(("cold_iso",))`` work directly.

    Parameters
    ----------
    H_cold, H_hot : array
        Hamiltonians during the cold and hot isochores.
    couplings : list of array
        System operators through which each bath acts, e.g. ``[embed(sigma_x,
        i, dims) for i in range(n)]``. Each gets its own Davies bath (same
        temperature) on the *full* Hamiltonian, so thermalisation reaches the
        interacting Gibbs state. Coupling every site separately avoids the dark
        states a single collective coupling can have.
    tau_ramp : float
        Duration of each unitary stroke. Long ramps approach the adiabatic
        (quasi-static) limit; short ones generate coherence and quantum
        friction.
    ramp : callable, optional
        ``ramp(s)`` mapping [0, 1] to [0, 1]; linear by default.
    spectrum : callable, optional
        Bath spectral function (see :func:`qthermo.baths.ohmic_spectrum`);
        flat with rate ``gamma`` by default. Prefer an Ohmic spectrum when the
        medium has exactly degenerate levels.
    check_thermalization : bool
        Refuse to build a cycle whose baths cannot bring the medium to its
        Gibbs state -- which happens whenever every coupling operator commutes
        with a symmetry of the Hamiltonian (``sigma_x`` couplings on a
        transverse-field Ising medium conserve its parity, for instance). The
        cycle would still run, but its isochores would never thermalise and
        its "efficiency" would describe a different machine.
    """
    H_c = check_hermitian(_as_matrix(H_cold), "H_cold")
    H_h = check_hermitian(_as_matrix(H_hot), "H_hot")
    shape = ramp or (lambda s: s)

    def drive(A, B):
        return lambda t: A + float(shape(t / tau_ramp)) * (B - A)

    def baths(H, T, label):
        return [davies_bath(H, _as_matrix(X), T, gamma=gamma, spectrum=spectrum,
                            name=f"{label}{k}")
                for k, X in enumerate(couplings)]

    if not couplings:
        raise QThermoError("otto_cycle needs at least one bath coupling operator")
    hot_baths, cold_baths = baths(H_h, T_hot, "hot"), baths(H_c, T_cold, "cold")
    if check_thermalization:
        for label, H, bs in (("hot", H_h, hot_baths), ("cold", H_c, cold_baths)):
            _require_thermalization(H, bs, label)
            if H.shape[0] <= 256:
                from .steady import relaxation_time
                t_relax = relaxation_time(H, bs, populations_only=True)
                if tau_iso < 5.0 * t_relax:
                    import warnings
                    warnings.warn(
                        f"the {label} isochore lasts {tau_iso:g} but energy "
                        f"populations of the medium relax on a time scale of "
                        f"{t_relax:.3g}: the working medium will not reach its "
                        "Gibbs state, and the cycle will not approach the "
                        "quasi-static limit. Lengthen tau_iso or raise gamma.",
                        RuntimeWarning, stacklevel=2)
    return Cycle([
        Stroke("hot_iso", H_h, tau_iso, hot_baths, steps=steps),
        Stroke("expand", drive(H_h, H_c), tau_ramp, steps=steps),
        Stroke("cold_iso", H_c, tau_iso, cold_baths, steps=steps),
        Stroke("compress", drive(H_c, H_h), tau_ramp, steps=steps),
    ])
