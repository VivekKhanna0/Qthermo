"""Thermal baths as objects: global (Davies) and local constructions.

``channels.thermal_bath`` builds the two jump operators of a single qubit. For a
multi-qubit working medium that is not enough, and the choice it hides is the
most-argued-about modelling decision in quantum thermodynamics:

* **Local** master equation -- each bath acts through the jump operators of the
  site it touches, as if the other sites were not there. Cheap and intuitive,
  valid when inter-site coupling is weak compared with the bath rates, and
  known to produce thermodynamic inconsistencies otherwise: its fixed point is
  not the Gibbs state of the coupled Hamiltonian, and heat can appear to flow
  from cold to hot (Levy & Kosloff, EPL 107, 20004 (2014)).

* **Global** (Davies) master equation -- jump operators connect eigenstates of
  the *full* Hamiltonian. Thermodynamically consistent by construction (the
  Gibbs state is a fixed point, Spohn's inequality holds), valid when the
  secular approximation is, i.e. when Bohr frequencies are resolved on the
  scale of the bath rates.

Both are provided behind the same :class:`Bath` object, so the same machine can
be run under either and the difference measured rather than argued about --
see :func:`qthermo.steady.compare_master_equations`.

Convention: a bath's heat current is positive when energy flows *into* the
system from that bath, matching ``Q > 0`` elsewhere in the package.
"""

from __future__ import annotations

import numpy as np

from .channels import mean_occupation
from .core import _as_matrix
from .validation import QThermoError, check_hermitian, check_temperature

__all__ = [
    "Bath",
    "davies_bath",
    "local_bath",
    "bohr_decomposition",
    "flat_spectrum",
    "ohmic_spectrum",
    "dissipator",
    "instantaneous_bath",
]


class Bath:
    """A named set of collapse operators coupling the system to one reservoir.

    Parameters
    ----------
    name : str
        Label used in reports and as the key of heat-current dictionaries.
    c_ops : list of ndarray
        Jump operators acting on the full system Hilbert space.
    temperature : float or None
        Reservoir temperature. ``None`` for a non-thermal channel (for example
        pure dephasing noise), which then contributes to the dynamics but not
        to the entropy-production balance.
    kind : str
        ``"global"``, ``"local"`` or ``"custom"``. Informational: reports use it
        to flag results that depend on the local approximation.
    sites : tuple of int
        Sites the bath physically touches, used by per-site plots.
    eigenbasis, eig_ops, energies
        Internal: baths built by :func:`davies_bath` store their jump operators
        as sparse matrices in the eigenbasis of the Hamiltonian (columns of
        ``eigenbasis``), where they are nearly diagonal-free and cheap. The
        dense ``c_ops`` view is built only when something asks for it.
    """

    def __init__(self, name, c_ops=None, temperature=None, kind="custom",
                 sites=(), *, eigenbasis=None, eig_ops=None, energies=None):
        self.name = name
        self.temperature = check_temperature(temperature, f"bath {name!r} temperature")
        self.kind = kind
        self.sites = tuple(int(s) for s in sites)
        self.eigenbasis = eigenbasis
        self.energies = energies
        self.eig_ops = list(eig_ops) if eig_ops is not None else None
        self._c_ops = [_as_matrix(L) for L in c_ops] if c_ops is not None else None
        if not (self._c_ops or self.eig_ops):
            raise QThermoError(
                f"bath {name!r} has no jump operators. If the coupling "
                "operator commutes with the Hamiltonian, no transition is "
                "possible and the bath exchanges no energy with the system."
            )
        if self._c_ops is None and self.eigenbasis is None:
            raise QThermoError("eig_ops need an eigenbasis")

    @property
    def c_ops(self) -> list:
        """Jump operators in the computational basis (built lazily)."""
        if self._c_ops is None:
            V = self.eigenbasis
            self._c_ops = [V @ op.toarray() @ V.conj().T for op in self.eig_ops]
        return self._c_ops

    @property
    def n_ops(self) -> int:
        return len(self.eig_ops) if self.eig_ops is not None else len(self._c_ops)

    @property
    def dim(self) -> int:
        if self.eigenbasis is not None:
            return self.eigenbasis.shape[0]
        return self._c_ops[0].shape[0]

    def dissipator(self, rho) -> np.ndarray:
        """This bath's contribution ``D[rho]`` to the master equation."""
        rho = _as_matrix(rho)
        if self.eig_ops is not None:
            V = self.eigenbasis
            rho_e = V.conj().T @ rho @ V
            jumps, decay = self._eigenbasis_superoperator()
            dim = rho_e.shape[0]
            gained = (jumps @ rho_e.reshape(-1, order="F")).reshape((dim, dim), order="F")
            out = gained - 0.5 * (decay @ rho_e + rho_e @ decay)
            return V @ out @ V.conj().T
        return dissipator(self.c_ops, rho)

    def _eigenbasis_superoperator(self):
        """Cached ``(sum conj(L) (x) L, sum L^dag L)`` in the eigenbasis."""
        if getattr(self, "_super_cache", None) is None:
            from .steady import _jump_sum_sparse
            dim = self.dim
            decay = np.zeros((dim, dim), dtype=complex)
            for L in self.eig_ops:
                decay += (L.conj().T @ L).toarray()
            self._super_cache = (_jump_sum_sparse(self.eig_ops, dim), decay)
        return self._super_cache

    def heat_current(self, rho, H) -> float:
        """Energy flow into the system from this bath, ``Tr[H D(rho)]``."""
        return float(np.real(np.trace(_as_matrix(H) @ self.dissipator(rho))))

    def __repr__(self) -> str:
        T = "--" if self.temperature is None else f"{self.temperature:g}"
        return (f"Bath({self.name!r}, kind={self.kind}, T={T}, "
                f"{self.n_ops} jump operators)")


def dissipator(c_ops, rho) -> np.ndarray:
    """Lindblad dissipator ``sum_L  L rho L^dag - {L^dag L, rho}/2``."""
    rho = _as_matrix(rho)
    out = np.zeros_like(rho)
    for L in c_ops:
        L = _as_matrix(L)
        LdL = L.conj().T @ L
        out += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return out


# --- spectral densities ------------------------------------------------------

def flat_spectrum(gamma: float):
    """Frequency-independent emission rate: ``J(w) = gamma``.

    Matches :func:`qthermo.channels.thermal_bath`, so a Davies bath on a single
    qubit with this spectrum reproduces it exactly.
    """
    def J(omega):
        return gamma * np.ones_like(np.asarray(omega, dtype=float))
    J.zero_frequency = None   # diverges: n(w) ~ T/w with J finite
    return J


def ohmic_spectrum(gamma: float, cutoff: float = np.inf, reference: float = 1.0):
    """Ohmic emission rate ``J(w) = gamma (w / reference) exp(-w / cutoff)``.

    ``reference`` sets the frequency at which the rate equals ``gamma``, so the
    same ``gamma`` means roughly the same coupling strength as ``flat_spectrum``
    for transitions near that frequency.
    """
    def J(omega):
        omega = np.asarray(omega, dtype=float)
        damp = 1.0 if np.isinf(cutoff) else np.exp(-omega / cutoff)
        return gamma * (omega / reference) * damp
    # lim_{w->0} J(w) n(w) = gamma T / reference: finite pure-dephasing rate.
    J.zero_frequency = lambda T: gamma * T / reference
    return J


# --- Davies construction -----------------------------------------------------

def _cluster(values: np.ndarray, tol: float) -> list[np.ndarray]:
    """Group sorted indices whose values agree within ``tol``."""
    order = np.argsort(values)
    groups, current = [], [order[0]]
    for index in order[1:]:
        if abs(values[index] - values[current[-1]]) <= tol:
            current.append(index)
        else:
            groups.append(np.array(current))
            current = [index]
    groups.append(np.array(current))
    return groups


def _bohr_components_eigenbasis(H, A, tol):
    """Bohr components of ``A`` as sparse matrices in the eigenbasis of ``H``.

    Returns ``(energies, V, [(omega, csr), ...])``. Vectorised: every non-zero
    matrix element of ``A`` in the eigenbasis is assigned the Bohr frequency of
    its level pair, frequencies are grouped within ``tol``, and each group
    becomes one sparse operator. Degenerate levels share one energy, so the
    grouping reproduces the projector definition exactly.
    """
    import scipy.sparse as sp

    energies, V = np.linalg.eigh(H)
    width = float(energies[-1] - energies[0])
    if tol is None:
        tol = 1e-9 * max(width, 1.0)
    level_energy = energies.copy()
    for group in _cluster(energies, tol):
        level_energy[group] = np.mean(energies[group])

    A_e = V.conj().T @ A @ V
    scale = np.max(np.abs(A_e)) if A_e.size else 0.0
    if scale == 0.0:
        return energies, V, []
    rows, cols = np.nonzero(np.abs(A_e) > 1e-14 * scale)
    values = A_e[rows, cols]
    omegas = level_energy[cols] - level_energy[rows]   # lowers energy by omega
    order = np.argsort(omegas, kind="stable")
    rows, cols, values, omegas = rows[order], cols[order], values[order], omegas[order]
    breaks = np.nonzero(np.diff(omegas) > tol)[0] + 1
    dim = H.shape[0]
    components = []
    for chunk in np.split(np.arange(len(omegas)), breaks):
        omega = float(np.mean(omegas[chunk]))
        if abs(omega) <= tol:
            omega = 0.0
        op = sp.csr_matrix((values[chunk], (rows[chunk], cols[chunk])),
                           shape=(dim, dim))
        components.append((omega, op))
    return energies, V, components


def bohr_decomposition(H, A, tol: float | None = None) -> dict:
    """Split ``A`` into components ``A(w)`` that lower the energy by ``w``.

        A = sum_w A(w),     A(w) = sum_{e' - e = w}  P(e) A P(e')

    with ``P(e)`` the projector onto the eigenspace of ``H`` with energy ``e``.
    Returns ``{w: A(w)}`` over every Bohr frequency at which ``A`` has a
    non-zero matrix element; ``w > 0`` lowers the energy, ``w < 0`` raises it.

    Degenerate levels and degenerate Bohr frequencies are grouped within
    ``tol`` (default: ``1e-9`` times the spectral width). Grouping is what makes
    the resulting generator the true Davies (secular) generator; frequencies
    closer than the bath rates but further apart than ``tol`` are the regime
    where the secular approximation itself becomes questionable.
    """
    H = check_hermitian(_as_matrix(H))
    A = _as_matrix(A)
    if A.shape != H.shape:
        raise QThermoError(
            f"coupling operator has shape {A.shape} but the Hamiltonian has "
            f"shape {H.shape}")
    _, V, components = _bohr_components_eigenbasis(H, A, tol)
    return {omega: V @ op.toarray() @ V.conj().T for omega, op in components}


def davies_bath(H, coupling, temperature: float, gamma: float = 1.0,
                spectrum=None, name: str = "bath", sites=(),
                zero_frequency_rate: float | None = None,
                tol: float | None = None) -> Bath:
    """Global (Davies) thermal bath for an arbitrary Hamiltonian.

    The system couples to the reservoir through the Hermitian operator
    ``coupling`` (for a qubit chain with a bath on site 0, that is
    ``embed(sigma_x, 0, dims)``). The coupling is decomposed into Bohr
    components of the *full* ``H`` and each component becomes one jump
    operator with the rate fixed by detailed balance:

        gamma(w)  = J(w) (1 + n(w))        w > 0   (emission)
        gamma(-w) = J(w) n(w)                      (absorption)

    ``gamma(-w) / gamma(w) = exp(-w / T)`` exactly, so the Gibbs state of ``H``
    at ``temperature`` is a fixed point for any ``H`` -- coupled, degenerate,
    many-body. The Lamb shift is neglected.

    The jump operators are stored sparse in the eigenbasis of ``H``, so global
    baths on systems of a few hundred levels stay cheap; the solvers use that
    representation directly.

    Parameters
    ----------
    H : array or Qobj
        Full system Hamiltonian, including interactions.
    coupling : array or Qobj
        Hermitian system operator through which the bath acts.
    temperature : float
        Bath temperature.
    gamma : float
        Rate scale; used to build a flat spectrum when ``spectrum`` is None.
    spectrum : callable, optional
        ``J(w)`` for ``w > 0``: the zero-temperature emission rate at Bohr
        frequency ``w``. See :func:`flat_spectrum`, :func:`ohmic_spectrum`.
    zero_frequency_rate : float, optional
        Rate of the ``w = 0`` (energy-conserving, pure dephasing in the
        eigenbasis) component. It exchanges no heat, only damps coherences
        between degenerate-energy states. Defaults to the spectrum's own
        ``w -> 0`` limit when finite, else 0.
    """
    temperature = check_temperature(temperature, f"bath {name!r} temperature")
    if temperature is None:
        raise QThermoError(f"bath {name!r} needs a temperature")
    H = check_hermitian(_as_matrix(H))
    coupling = check_hermitian(_as_matrix(coupling), f"bath {name!r} coupling")
    if coupling.shape != H.shape:
        raise QThermoError(
            f"coupling operator has shape {coupling.shape} but the Hamiltonian "
            f"has shape {H.shape}")
    J = spectrum if spectrum is not None else flat_spectrum(gamma)

    energies, V, components = _bohr_components_eigenbasis(H, coupling, tol)
    eig_ops = []
    for omega, A_omega in components:
        if omega == 0.0:
            rate = zero_frequency_rate
            if rate is None:
                limit = getattr(J, "zero_frequency", None)
                rate = limit(temperature) if callable(limit) else 0.0
        elif omega > 0:
            rate = float(J(omega)) * (1.0 + mean_occupation(omega, temperature))
        else:
            rate = float(J(-omega)) * mean_occupation(-omega, temperature)
        if rate > 0:
            eig_ops.append(np.sqrt(rate) * A_omega)

    return Bath(name, None, temperature, kind="global", sites=tuple(sites),
                eigenbasis=V, eig_ops=eig_ops, energies=energies)


def local_bath(H_site, coupling, temperature: float, site: int, dims,
               gamma: float = 1.0, spectrum=None, name: str | None = None,
               zero_frequency_rate: float | None = None) -> Bath:
    """Local thermal bath: Davies construction on one site, then embedded.

    The jump operators are those the site would have in isolation, lifted to
    the full space. This ignores how inter-site coupling reshapes the
    spectrum -- the defining approximation of the local master equation.
    """
    from .subsystems import embed

    single = davies_bath(H_site, coupling, temperature, gamma=gamma,
                         spectrum=spectrum, name=name or f"bath_{site}",
                         zero_frequency_rate=zero_frequency_rate)
    return Bath(single.name, [embed(L, site, dims) for L in single.c_ops],
                temperature, kind="local", sites=(site,))


def instantaneous_bath(H_of_t, coupling, temperature: float, gamma: float = 1.0,
                       spectrum=None, name: str = "bath", sites=(),
                       cache_size: int = 4096):
    """A bath that follows a time-dependent Hamiltonian.

    Returns ``t -> [Bath]`` giving, at each instant, the Davies bath built on
    the instantaneous ``H(t)``: the adiabatic Markovian master equation of
    Albash, Boixo, Lidar & Zanardi, NJP 14, 123016 (2012). Valid when the
    driving is slow compared with the bath correlation time; it is the
    standard model for finite-time erasure and for driven thermalisation
    strokes. Pass it as a stroke's ``c_ops``.

    The Davies construction is cached per time point, since the ODE solver
    revisits times.
    """
    from functools import lru_cache

    @lru_cache(maxsize=cache_size)
    def at(t: float):
        return [davies_bath(_as_matrix(H_of_t(t)), coupling, temperature,
                            gamma=gamma, spectrum=spectrum, name=name,
                            sites=sites)]

    def baths(t):
        return at(float(t))
    baths.temperature = temperature
    return baths
