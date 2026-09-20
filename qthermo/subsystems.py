"""Spatially resolved thermodynamics for multipartite working media.

Phase 1 and 2 answer "what does this cycle do?" and "how does that change with
the noise channel?" -- both totaled over the whole system. This module answers
*where*: which site absorbs the heat, which site produces the entropy, and how
much of the system's behaviour is not attributable to any single site at all.

The last part is the physics that makes this more than a bookkeeping
convenience. Two identities drive the module:

    U_total  =  sum_i U_i  +  U_interaction
    S_total  =  sum_i S_i  -  I_corr

Energy fails to be additive when sites are coupled; entropy fails to be
additive whenever sites are correlated. ``I_corr`` (the total correlation, or
multi-information) is non-negative and vanishes only for product states, so a
per-site breakdown that ignores it will always over-report the system's
entropy. Both residuals are reported explicitly rather than silently dropped.

Units: hbar = k_B = 1, matching the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    _as_matrix,
    heat_work_increments,
    internal_energy,
    von_neumann_entropy,
)
from .validation import QThermoError, check_density_matrix

__all__ = [
    "partial_trace",
    "embed",
    "total_correlation",
    "mutual_information",
    "SiteResult",
    "SubsystemResult",
    "resolve_stroke",
    "resolve_cycle",
]


def _check_dims(dims, total_dim: int) -> list[int]:
    """Validate a subsystem dimension list against the full Hilbert space."""
    dims = [int(d) for d in dims]
    if any(d < 2 for d in dims):
        raise QThermoError(
            f"subsystem dimensions must each be at least 2, got {dims}"
        )
    product = int(np.prod(dims))
    if product != total_dim:
        raise QThermoError(
            f"subsystem dimensions {dims} multiply to {product}, but the state "
            f"has dimension {total_dim}. For n qubits pass dims=[2]*n."
        )
    return dims


def partial_trace(rho, keep, dims) -> np.ndarray:
    """Trace out every subsystem except those in ``keep``.

    Parameters
    ----------
    rho : array or Qobj
        State of the full multipartite system.
    keep : int or sequence of int
        Index (or indices) of the subsystems to retain, in the tensor-product
        ordering implied by ``dims``.
    dims : sequence of int
        Dimension of each subsystem, e.g. ``[2, 2, 2]`` for three qubits.

    Returns
    -------
    ndarray
        Reduced density matrix of the kept subsystems.
    """
    rho = _as_matrix(rho)
    dims = _check_dims(dims, rho.shape[0])
    n = len(dims)

    if isinstance(keep, (int, np.integer)):
        keep = [int(keep)]
    keep = sorted(int(k) for k in keep)
    if any(k < 0 or k >= n for k in keep):
        raise QThermoError(
            f"keep={keep} is out of range for {n} subsystems (valid: 0..{n - 1})"
        )
    if len(set(keep)) != len(keep):
        raise QThermoError(f"keep={keep} contains duplicate subsystem indices")

    traced = [i for i in range(n) if i not in keep]
    if not traced:
        return rho

    # Reshape into a rank-2n tensor with one row index and one column index
    # per subsystem, then contract the row/column pair of each traced site.
    tensor = rho.reshape(dims + dims)
    for offset, site in enumerate(traced):
        # Each contraction removes one row and one column axis, so the axis
        # positions of later sites shift down by the number already removed.
        row_axis = site - offset
        col_axis = site - offset + (n - offset)
        tensor = np.trace(tensor, axis1=row_axis, axis2=col_axis)

    kept_dim = int(np.prod([dims[k] for k in keep]))
    return tensor.reshape(kept_dim, kept_dim)


def embed(operator, site: int, dims) -> np.ndarray:
    """Lift a single-site operator into the full multipartite Hilbert space.

    ``embed(sigma_z, 1, [2, 2, 2])`` returns ``I (x) sigma_z (x) I``. This is
    the bridge between the single-qubit operators in :mod:`qthermo.channels`
    and a multi-qubit ``Stroke``: collapse operators and local Hamiltonian
    terms both need lifting before the solver can use them.
    """
    operator = _as_matrix(operator)
    dims = [int(d) for d in dims]
    if site < 0 or site >= len(dims):
        raise QThermoError(
            f"site={site} is out of range for {len(dims)} subsystems "
            f"(valid: 0..{len(dims) - 1})"
        )
    if operator.shape != (dims[site], dims[site]):
        raise QThermoError(
            f"operator has shape {operator.shape}, but site {site} has "
            f"dimension {dims[site]}"
        )

    result = np.array([[1.0 + 0.0j]])
    for index, dimension in enumerate(dims):
        factor = operator if index == site else np.eye(dimension, dtype=complex)
        result = np.kron(result, factor)
    return result


def total_correlation(rho, dims) -> float:
    """Total correlation (multi-information) ``sum_i S_i - S_total``.

    Non-negative, zero exactly for product states. For a bipartition this is
    the usual quantum mutual information; for more than two sites it measures
    all correlations, classical and quantum, shared across the whole system.
    """
    rho = _as_matrix(rho)
    dims = _check_dims(dims, rho.shape[0])
    local_entropy_sum = sum(
        von_neumann_entropy(partial_trace(rho, i, dims)) for i in range(len(dims))
    )
    return float(local_entropy_sum - von_neumann_entropy(rho))


def mutual_information(rho, site_a: int, site_b: int, dims) -> float:
    """Mutual information ``S_a + S_b - S_ab`` between two specific sites."""
    rho = _as_matrix(rho)
    dims = _check_dims(dims, rho.shape[0])
    if site_a == site_b:
        raise QThermoError("mutual information needs two distinct sites")
    s_a = von_neumann_entropy(partial_trace(rho, site_a, dims))
    s_b = von_neumann_entropy(partial_trace(rho, site_b, dims))
    s_ab = von_neumann_entropy(partial_trace(rho, [site_a, site_b], dims))
    return float(s_a + s_b - s_ab)


@dataclass
class SiteResult:
    """Thermodynamic quantities attributed to one site over one stroke."""

    site: int
    heat: float
    work: float
    delta_U: float
    delta_S: float
    rho_initial: np.ndarray
    rho_final: np.ndarray

    @property
    def first_law_residual(self) -> float:
        """``|dU_i - (Q_i + W_i)|`` for this site alone.

        Uses the same midpoint split as :func:`qthermo.core.heat_work_increments`,
        so it sits at machine precision for the *local* energy balance even when
        the site is coupled to others -- coupling shows up in
        :attr:`SubsystemResult.interaction_energy_change`, not here.
        """
        return abs(self.delta_U - (self.heat + self.work))


@dataclass
class SubsystemResult:
    """Per-site breakdown of one stroke, plus the non-additive remainders."""

    stroke: str
    sites: list[SiteResult]
    total_delta_U: float
    total_delta_S: float
    correlation_initial: float
    correlation_final: float

    def site(self, index: int) -> SiteResult:
        for s in self.sites:
            if s.site == index:
                return s
        raise KeyError(index)

    @property
    def local_heat(self) -> dict:
        return {s.site: s.heat for s in self.sites}

    @property
    def local_work(self) -> dict:
        return {s.site: s.work for s in self.sites}

    @property
    def local_entropy_change(self) -> dict:
        return {s.site: s.delta_S for s in self.sites}

    @property
    def interaction_energy_change(self) -> float:
        """``dU_total - sum_i dU_i``.

        Energy stored in the coupling rather than in any single site. Exactly
        zero for a non-interacting Hamiltonian; non-zero here is a measurement
        of how much of the stroke's energy bookkeeping the per-site picture
        cannot see.
        """
        return float(self.total_delta_U - sum(s.delta_U for s in self.sites))

    @property
    def correlation_change(self) -> float:
        """Change in total correlation over the stroke."""
        return float(self.correlation_final - self.correlation_initial)

    @property
    def entropy_balance_residual(self) -> float:
        """Check of ``dS_total = sum_i dS_i - dI_corr`` to machine precision.

        This identity is exact by construction, so a non-zero value means a
        numerical problem (a non-positive state, a bad partial trace), not
        physics. It is the module's self-test, in the same spirit as
        :attr:`qthermo.cycle.StrokeResult.first_law_residual`.
        """
        local_sum = sum(s.delta_S for s in self.sites)
        return abs(self.total_delta_S - (local_sum - self.correlation_change))

    @property
    def dominant_site(self) -> int:
        """Site with the largest absolute heat exchange."""
        return max(self.sites, key=lambda s: abs(s.heat)).site

    def report(self) -> str:
        lines = [
            f"stroke {self.stroke!r} -- per-site breakdown",
            f"{'site':>6} {'heat':>12} {'work':>12} {'dU':>12} {'dS':>12}",
        ]
        for s in self.sites:
            lines.append(
                f"{s.site:>6} {s.heat:>12.6f} {s.work:>12.6f} "
                f"{s.delta_U:>12.6f} {s.delta_S:>12.6f}"
            )
        lines.append(
            f"{'total':>6} {sum(s.heat for s in self.sites):>12.6f} "
            f"{sum(s.work for s in self.sites):>12.6f} "
            f"{sum(s.delta_U for s in self.sites):>12.6f} "
            f"{sum(s.delta_S for s in self.sites):>12.6f}"
        )
        lines.append(f"interaction energy change : {self.interaction_energy_change:+.6e}")
        lines.append(f"correlation change (dI)   : {self.correlation_change:+.6e}")
        lines.append(f"entropy balance residual  : {self.entropy_balance_residual:.3e}")
        return "\n".join(lines)


def _local_hamiltonian_at(local_H, site: int, time: float, dims) -> np.ndarray:
    """Resolve one site's local Hamiltonian, allowing callables for driving."""
    term = local_H[site]
    if callable(term):
        term = term(time)
    term = _as_matrix(term)
    if term.shape != (dims[site], dims[site]):
        raise QThermoError(
            f"local Hamiltonian for site {site} has shape {term.shape}, "
            f"expected ({dims[site]}, {dims[site]})"
        )
    return term


def resolve_stroke(stroke_result, dims, local_H) -> SubsystemResult:
    """Break one stroke's thermodynamics down site by site.

    Parameters
    ----------
    stroke_result : StrokeResult
        A stroke already run by :meth:`qthermo.cycle.Cycle.run`. The full
        trajectory of states is used, so heat and work are path quantities
        here exactly as they are in the totaled version.
    dims : sequence of int
        Subsystem dimensions, e.g. ``[2, 2, 2]``.
    local_H : sequence
        One local Hamiltonian per site, each acting on that site's space alone
        (*not* embedded). Entries may be callables ``H_i(t)`` for driven sites.
        The interaction part of the full Hamiltonian is deliberately not passed:
        what it contributes is reported as
        :attr:`SubsystemResult.interaction_energy_change`.

    Returns
    -------
    SubsystemResult
    """
    states = stroke_result.states
    times = np.asarray(stroke_result.times, dtype=float)
    if len(states) < 2:
        raise QThermoError(
            f"stroke {stroke_result.name!r} has fewer than two stored states; "
            "nothing to integrate over"
        )

    dims = _check_dims(dims, _as_matrix(states[0]).shape[0])
    n_sites = len(dims)
    if len(local_H) != n_sites:
        raise QThermoError(
            f"got {len(local_H)} local Hamiltonians for {n_sites} subsystems"
        )

    check_density_matrix(states[0], "stroke initial state")

    reduced = [
        [partial_trace(state, site, dims) for state in states]
        for site in range(n_sites)
    ]

    sites = []
    for site in range(n_sites):
        heat = 0.0
        work = 0.0
        for k in range(len(states) - 1):
            H_a = _local_hamiltonian_at(local_H, site, float(times[k]), dims)
            H_b = _local_hamiltonian_at(local_H, site, float(times[k + 1]), dims)
            dq, dw = heat_work_increments(
                reduced[site][k], reduced[site][k + 1], H_a, H_b
            )
            heat += dq
            work += dw

        H_start = _local_hamiltonian_at(local_H, site, float(times[0]), dims)
        H_end = _local_hamiltonian_at(local_H, site, float(times[-1]), dims)
        delta_U = internal_energy(reduced[site][-1], H_end) - internal_energy(
            reduced[site][0], H_start
        )
        delta_S = von_neumann_entropy(reduced[site][-1]) - von_neumann_entropy(
            reduced[site][0]
        )
        sites.append(
            SiteResult(
                site=site,
                heat=heat,
                work=work,
                delta_U=delta_U,
                delta_S=delta_S,
                rho_initial=reduced[site][0],
                rho_final=reduced[site][-1],
            )
        )

    total_delta_S = von_neumann_entropy(states[-1]) - von_neumann_entropy(states[0])

    return SubsystemResult(
        stroke=stroke_result.name,
        sites=sites,
        total_delta_U=float(stroke_result.delta_U),
        total_delta_S=float(total_delta_S),
        correlation_initial=total_correlation(states[0], dims),
        correlation_final=total_correlation(states[-1], dims),
    )


def resolve_cycle(cycle_result, dims, local_H) -> list[SubsystemResult]:
    """Apply :func:`resolve_stroke` to every stroke of a completed cycle."""
    return [
        resolve_stroke(stroke, dims, local_H) for stroke in cycle_result.strokes
    ]
