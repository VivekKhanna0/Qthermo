"""Where the energy goes inside a multi-qubit machine.

Totals say *how much* heat a machine moves. For more than one qubit the
questions that matter are local ones: which bath feeds which site, which bond
carries the current, which qubit is actually cold, and how much correlation or
entanglement the machine is holding. This module computes those from a state
and a model, with every balance closed exactly.

For a Hamiltonian ``H = sum_i h_i + sum_b V_b`` (local terms plus interaction
terms, each ``V_b`` acting on a set of sites), the energy of site ``i`` obeys

    d<h_i>/dt = sum_k J_{k -> i}  +  sum_b J_{b -> i}

    J_{k -> i} = Tr[h_i D_k(rho)]          from bath k
    J_{b -> i} = i Tr(rho [V_b, h_i])      through interaction term b

In the steady state the right-hand side vanishes site by site;
:meth:`HeatFlowMap.site_balance` returns the residual, which should be at
machine precision. Nothing here assumes qubits except the concurrence, and the
virtual temperature uses the two lowest local levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baths import dissipator
from .core import _as_matrix, von_neumann_entropy
from .subsystems import embed, mutual_information, partial_trace
from .validation import QThermoError

__all__ = [
    "HeatFlowMap",
    "heat_flow_map",
    "virtual_temperature",
    "concurrence",
    "negativity",
    "correlation_matrices",
]


def virtual_temperature(rho_site, h_site) -> float:
    """Temperature a site *looks* like it has, from its two lowest levels.

        T_v = (e_1 - e_0) / ln(p_0 / p_1)

    with ``p_k`` the populations of the local energy eigenstates. This is the
    'virtual temperature' of Brunner, Linden, Popescu & Skrzypczyk,
    PRE 85, 051117 (2012): a qubit that is cooled below its bath has
    ``T_v < T_bath``. Negative for a population inversion; ``inf`` when the
    two levels are equally populated; 0 when the excited level is empty.
    """
    rho_site, h_site = _as_matrix(rho_site), _as_matrix(h_site)
    energies, vectors = np.linalg.eigh(h_site)
    gap = float(energies[1] - energies[0])
    if gap <= 1e-14:
        raise QThermoError("the two lowest local levels are degenerate; "
                           "a virtual temperature is not defined")
    p0 = float(np.real(vectors[:, 0].conj() @ rho_site @ vectors[:, 0]))
    p1 = float(np.real(vectors[:, 1].conj() @ rho_site @ vectors[:, 1]))
    if p1 <= 1e-15:
        return 0.0
    if p0 <= 1e-15:
        return -0.0
    ratio = np.log(p0 / p1)
    if abs(ratio) < 1e-15:
        return float("inf")
    return gap / ratio


def concurrence(rho_two_qubits) -> float:
    """Wootters concurrence of a two-qubit state (0 = separable, 1 = Bell)."""
    rho = _as_matrix(rho_two_qubits)
    if rho.shape != (4, 4):
        raise QThermoError(f"concurrence needs a two-qubit state, got {rho.shape}")
    yy = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
                  dtype=complex)
    R = rho @ yy @ rho.conj() @ yy
    lam = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(R)))[::-1], 0, None))
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def negativity(rho, site_a: int, site_b: int, dims) -> float:
    """Negativity of the reduced state of two sites, any local dimension.

    ``(||rho^{T_b}||_1 - 1) / 2``; zero for separable states (and for PPT
    entangled ones, which cannot occur for 2x2 or 2x3).
    """
    pair = partial_trace(rho, [site_a, site_b], dims)
    da, db = dims[min(site_a, site_b)], dims[max(site_a, site_b)]
    t = pair.reshape(da, db, da, db).transpose(0, 3, 2, 1).reshape(da * db, da * db)
    return float((np.sum(np.abs(np.linalg.eigvalsh(t))) - 1.0) / 2.0)


def correlation_matrices(rho, dims) -> dict:
    """Pairwise mutual information, negativity and (qubits) concurrence."""
    rho = _as_matrix(rho)
    n = len(dims)
    mi = np.zeros((n, n))
    neg = np.zeros((n, n))
    conc = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a + 1, n):
            mi[a, b] = mi[b, a] = mutual_information(rho, a, b, dims)
            neg[a, b] = neg[b, a] = max(0.0, negativity(rho, a, b, dims))
            if dims[a] == 2 and dims[b] == 2:
                conc[a, b] = conc[b, a] = concurrence(partial_trace(rho, [a, b], dims))
    return {"mutual_information": mi, "negativity": neg, "concurrence": conc}


@dataclass
class HeatFlowMap:
    """Site-resolved energy flows and correlations of one state."""

    site_names: list
    dims: list
    local_energy: np.ndarray
    virtual_temperature: np.ndarray
    local_entropy: np.ndarray
    bath_to_site: dict                # (bath, site) -> J
    bath_to_interaction: dict         # bath -> Tr[V D_k(rho)]
    term_to_site: dict                # (term key, site) -> J
    bath_temperature: dict            # bath -> T
    bath_sites: dict                  # bath -> tuple of sites it touches
    mutual_information: np.ndarray
    negativity: np.ndarray
    concurrence: np.ndarray
    total_correlation: float
    extras: dict = field(default_factory=dict)

    @property
    def n_sites(self) -> int:
        return len(self.site_names)

    def into_site(self, site: int) -> float:
        """Net energy flow into a site from everything (0 in a steady state)."""
        return (sum(J for (_, s), J in self.bath_to_site.items() if s == site)
                + sum(J for (_, s), J in self.term_to_site.items() if s == site))

    def site_balance(self) -> np.ndarray:
        """Per-site ``d<h_i>/dt`` implied by the flows (zero when stationary)."""
        return np.array([self.into_site(i) for i in range(self.n_sites)])

    def bath_current(self, bath: str) -> float:
        """Total heat from ``bath`` into the sites and the interaction energy."""
        return (sum(J for (b, _), J in self.bath_to_site.items() if b == bath)
                + self.bath_to_interaction.get(bath, 0.0))

    def bond_current(self, term, direction=None) -> float:
        """Energy flow through an interaction term.

        For a two-site term ``(i, j)`` returns the flow from ``i`` to ``j``,
        ``(J_{b->j} - J_{b->i}) / 2`` -- antisymmetrised so that energy
        temporarily stored in the bond does not count as transport.
        """
        key = tuple(term)
        if len(key) != 2 and direction is None:
            raise QThermoError("for many-body terms pass direction=(i, j)")
        i, j = direction if direction is not None else key
        into_j = self.term_to_site.get((key, j), 0.0)
        into_i = self.term_to_site.get((key, i), 0.0)
        return 0.5 * (into_j - into_i)

    def coldest_site(self) -> int:
        T = np.where(self.virtual_temperature > 0, self.virtual_temperature, np.inf)
        return int(np.argmin(T))

    def report(self) -> str:
        lines = [f"{'site':<10}{'T_virtual':>11}{'<h_i>':>12}{'S_i':>9}"
                 f"{'from baths':>14}{'from bonds':>14}", "-" * 70]
        for i, name in enumerate(self.site_names):
            from_baths = sum(J for (_, s), J in self.bath_to_site.items() if s == i)
            from_bonds = sum(J for (_, s), J in self.term_to_site.items() if s == i)
            T = self.virtual_temperature[i]
            T_text = ("inf" if np.isinf(T) else f"{T:.4g}")
            lines.append(f"{name:<10}{T_text:>11}{self.local_energy[i]:>12.5f}"
                         f"{self.local_entropy[i]:>9.4f}{from_baths:>14.4e}"
                         f"{from_bonds:>14.4e}")
        lines.append("-" * 70)
        lines.append("baths: " + ", ".join(
            f"{b} (T={T:g}) -> " + ",".join(self.site_names[s] for s in self.bath_sites[b])
            for b, T in self.bath_temperature.items()))
        for key in {k for (k, _) in self.term_to_site}:
            if len(key) == 2:
                i, j = key
                lines.append(f"bond {self.site_names[i]} -> {self.site_names[j]}: "
                             f"{self.bond_current(key):+.4e}")
        lines.append(f"total correlation: {self.total_correlation:.4e} nats")
        n = self.n_sites
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        if pairs:
            best = max(pairs, key=lambda p: self.mutual_information[p])
            lines.append(f"most correlated pair: {self.site_names[best[0]]}-"
                         f"{self.site_names[best[1]]}, I = "
                         f"{self.mutual_information[best]:.4e}, negativity = "
                         f"{self.negativity[best]:.3e}")
        lines.append(f"max site-balance residual: "
                     f"{np.max(np.abs(self.site_balance())):.2e}")
        if self.extras.get("secular_blind"):
            lines.append(
                "note: this state is diagonal in the eigenbasis of H (global "
                "master equation), so every bond current i<[V_b, h_i]> vanishes "
                "identically and bath heat appears as interaction energy. The "
                "secular approximation discards the coherences that carry "
                "local currents; use local baths (weak inter-site coupling) to "
                "resolve internal transport.")
        return "\n".join(lines)


def heat_flow_map(source, model=None, *, rho=None, dims=None, local_H=None,
                  baths=None, interaction_terms=None, site_names=None) -> HeatFlowMap:
    """Resolve the energy flows of a state site by site.

    Parameters
    ----------
    source : SteadyState, Model, or None
        A ``SteadyState`` from ``Model.analyze()`` carries everything needed.
        Otherwise pass ``model`` and ``rho``, or every keyword explicitly.
    rho, dims, local_H, baths, interaction_terms, site_names
        Explicit overrides. ``interaction_terms`` maps a tuple of site indices
        to the embedded operator ``V_b``; ``local_H`` holds unembedded local
        Hamiltonians.
    """
    from .steady import SteadyState

    if isinstance(source, SteadyState):
        rho = source.rho if rho is None else rho
        baths = source.baths if baths is None else baths
        model = source.model if model is None else model
    elif source is not None:
        model = source
    if model is not None:
        dims = model.dims if dims is None else dims
        local_H = model.local_H if local_H is None else local_H
        baths = model.baths if baths is None else baths
        interaction_terms = (model.interaction_terms if interaction_terms is None
                             else interaction_terms)
        site_names = model.site_names if site_names is None else site_names
    if rho is None or dims is None or local_H is None:
        raise QThermoError("heat_flow_map needs a state, dims and local_H "
                           "(pass a SteadyState from Model.analyze(), or the "
                           "keywords explicitly)")
    rho = _as_matrix(rho)
    baths = list(baths or [])
    interaction_terms = dict(interaction_terms or {})
    n = len(dims)
    site_names = list(site_names or [f"q{i}" for i in range(n)])

    h = [embed(_as_matrix(local_H[i]), i, dims) for i in range(n)]
    reduced = [partial_trace(rho, i, dims) for i in range(n)]

    local_energy = np.array([float(np.real(np.trace(h[i] @ rho))) for i in range(n)])
    local_entropy = np.array([von_neumann_entropy(r) for r in reduced])
    T_v = np.array([virtual_temperature(reduced[i], local_H[i]) for i in range(n)])

    V_total = sum(interaction_terms.values()) if interaction_terms else None
    bath_to_site, bath_to_interaction, bath_T, bath_sites = {}, {}, {}, {}
    for bath in baths:
        D = dissipator(bath.c_ops, rho)
        for i in range(n):
            bath_to_site[(bath.name, i)] = float(np.real(np.trace(h[i] @ D)))
        bath_to_interaction[bath.name] = (
            float(np.real(np.trace(V_total @ D))) if V_total is not None else 0.0)
        bath_T[bath.name] = bath.temperature
        bath_sites[bath.name] = tuple(bath.sites)

    term_to_site = {}
    for key, V in interaction_terms.items():
        V = _as_matrix(V)
        for i in key:
            commutator = V @ h[i] - h[i] @ V
            term_to_site[(tuple(key), i)] = float(np.real(1j * np.trace(rho @ commutator)))

    corr = correlation_matrices(rho, dims)
    total_corr = float(sum(local_entropy) - von_neumann_entropy(rho))
    extras = {}
    if interaction_terms:
        H = sum(h) + V_total
        heat_in = sum(abs(bath_to_interaction[b.name]) for b in baths)
        if (np.max(np.abs(H @ rho - rho @ H)) < 1e-10
                and heat_in > 1e-12
                and all(abs(J) < 1e-12 for J in term_to_site.values())):
            extras["secular_blind"] = True
    return HeatFlowMap(site_names, list(dims), local_energy, T_v, local_entropy,
                       bath_to_site, bath_to_interaction, term_to_site, bath_T,
                       bath_sites, corr["mutual_information"], corr["negativity"],
                       corr["concurrence"], total_corr, extras)
