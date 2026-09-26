"""Current fluctuations, full counting statistics, and uncertainty relations.

A steady-state machine delivers a mean heat current, but the current is carried
by discrete quanta and fluctuates. How much it fluctuates is constrained by
thermodynamics: for any classical Markov process the **thermodynamic
uncertainty relation** (TUR),

    Q_TUR  =  (D / J^2) * sigma_dot  >=  2,

bounds the relative noise of a current ``J`` (with ``D`` the scaled variance,
``Var[N_t] / t`` for large ``t``) by the entropy production rate
(Barato & Seifert, PRL 114, 158101 (2015); Gingrich et al., PRL 116, 120601
(2016)). Precision costs dissipation. The **kinetic uncertainty relation**
(KUR) bounds the same noise by the dynamical activity ``K`` instead:
``D K / J^2 >= 1`` (Terlizzi & Baiesi, J. Phys. A 52, 02LT03 (2019)).

Quantum coherence can break both, so a value below the bound is a genuine
signature of non-classical transport -- the three-level maser is the standard
example (Kalaee, Wacker & Potts, PRE 104, L012103 (2021)). This module
computes ``J``, ``D``, ``Q_TUR`` and ``Q_KUR`` exactly from the Liouvillian,
without sampling:

    J = Tr[J_sup rho]
    D = sum_j nu_j^2 Tr[L_j rho L_j^dag]  -  2 Tr[J_sup L^D J_sup rho]

with ``J_sup(rho) = sum_j nu_j L_j rho L_j^dag`` the weighted jump superoperator
and ``L^D`` the Drazin inverse of the Liouvillian (Landi, Kewming, Kolodynski &
Potts, PRX Quantum 5, 020201 (2024)). The scaled cumulant generating function
is available too, for the full distribution at long times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .baths import Bath
from .core import _as_matrix
from .steady import _vec, _unvec, liouvillian, steady_state
from .validation import QThermoError, check_hermitian

__all__ = [
    "CurrentStatistics",
    "jump_energy",
    "counting_weights",
    "current_statistics",
    "scaled_cgf",
]


def jump_energy(L, energy, tol: float = 1e-8) -> float:
    """Energy a jump ``L`` deposits in the system, measured with ``energy``.

    Requires ``L`` to be an eigenoperator of the energy superoperator,
    ``[E, L] = e L`` -- true for every Davies jump operator with ``E = H`` and
    for every local jump operator with ``E = H0``. Otherwise a jump does not
    transfer a definite amount of energy and counting it is ambiguous, so this
    raises instead of guessing.
    """
    L, E = _as_matrix(L), _as_matrix(energy)
    commutator = E @ L - L @ E
    norm = np.vdot(L, L).real
    if norm == 0:
        return 0.0
    e = float(np.real(np.vdot(L, commutator) / norm))
    residual = np.linalg.norm(commutator - e * L) / np.sqrt(norm)
    scale = max(np.max(np.abs(np.linalg.eigvalsh(E))), 1.0)
    if residual > tol * scale:
        raise QThermoError(
            f"jump operator does not change the energy by a definite amount "
            f"(residual {residual:.2e}). Measure energy with the operator the "
            "bath was built from: H for global baths, the bare H0 for local ones.")
    return e


def counting_weights(baths, count, energy) -> list[float]:
    """Per-jump weights for a counted current.

    ``count`` maps bath name to what is counted for that bath:

    * ``"energy"`` -- the energy each jump deposits (heat current);
    * ``"quanta"`` -- ``+1`` per absorption, ``-1`` per emission;
    * a number -- the same weight for every jump of that bath;
    * a list -- one weight per jump operator.

    Baths not named are not counted.
    """
    weights = []
    names = {b.name for b in baths}
    unknown = set(count) - names
    if unknown:
        raise QThermoError(f"unknown bath(s) {sorted(unknown)}; baths: {sorted(names)}")
    for bath in baths:
        spec = count.get(bath.name)
        if spec is None:
            weights.extend([0.0] * len(bath.c_ops))
        elif isinstance(spec, str):
            for L in bath.c_ops:
                e = jump_energy(L, energy)
                if spec == "energy":
                    weights.append(e)
                elif spec == "quanta":
                    weights.append(float(np.sign(e)) if abs(e) > 1e-12 else 0.0)
                else:
                    raise QThermoError(f"unknown counting spec {spec!r}")
        elif np.ndim(spec) == 0:
            weights.extend([float(spec)] * len(bath.c_ops))
        else:
            spec = [float(w) for w in spec]
            if len(spec) != len(bath.c_ops):
                raise QThermoError(
                    f"bath {bath.name!r} has {len(bath.c_ops)} jump operators "
                    f"but {len(spec)} weights were given")
            weights.extend(spec)
    return weights


def _jump_superoperators(c_ops, sparse):
    kron = sp.kron if sparse else np.kron
    wrap = sp.csr_matrix if sparse else np.asarray
    return [kron(wrap(L.conj()), wrap(L)) for L in c_ops]


@dataclass
class CurrentStatistics:
    """Mean, noise and uncertainty ratios of one counted current."""

    mean: float                   # J
    noise: float                  # D = lim Var[N_t] / t
    activity: float               # K, total jump rate
    entropy_production_rate: float | None
    label: str = "current"

    @property
    def fano_factor(self) -> float:
        """``D / |J|``: 1 for Poissonian unit-weight counting."""
        return self.noise / abs(self.mean)

    @property
    def relative_noise(self) -> float:
        """``D / J^2``: squared relative uncertainty per unit time."""
        return self.noise / self.mean ** 2

    @property
    def tur_ratio(self) -> float:
        """``(D / J^2) sigma_dot``. At least 2 for classical Markov dynamics."""
        if self.entropy_production_rate is None:
            raise QThermoError("entropy production rate unknown: every "
                               "counted bath needs a temperature")
        return self.relative_noise * self.entropy_production_rate

    @property
    def kur_ratio(self) -> float:
        """``D K / J^2``. At least 1 for classical Markov dynamics."""
        return self.relative_noise * self.activity

    @property
    def violates_tur(self) -> bool:
        return self.tur_ratio < 2.0 - 1e-9

    @property
    def violates_kur(self) -> bool:
        return self.kur_ratio < 1.0 - 1e-9

    def report(self) -> str:
        lines = [f"{self.label}",
                 f"  mean J                 : {self.mean:+.6e}",
                 f"  noise D = Var/t        : {self.noise:.6e}",
                 f"  relative noise D/J^2   : {self.relative_noise:.6e}",
                 f"  activity K             : {self.activity:.6e}"]
        if self.entropy_production_rate is not None:
            lines.append(f"  entropy production     : {self.entropy_production_rate:.6e}")
            lines.append(f"  TUR ratio (>= 2 class.): {self.tur_ratio:.4f}"
                         + ("   <-- VIOLATED: beyond any classical model"
                            if self.violates_tur else ""))
        lines.append(f"  KUR ratio (>= 1 class.): {self.kur_ratio:.4f}"
                     + ("   <-- VIOLATED" if self.violates_kur else ""))
        return "\n".join(lines)


def _resolve(source, baths, energy):
    """Accept a Model, a SteadyState, or explicit (H, baths)."""
    from .models import Model
    from .steady import SteadyState
    if isinstance(source, Model):
        E = energy
        if E is None:
            E = source.energy if source.energy is not None else (
                source.H0 if source.master_equation == "local" else source.H)
        return source.H, source.baths, E
    if isinstance(source, SteadyState):
        return source.H, source.baths, (source.energy if energy is None else energy)
    H = check_hermitian(_as_matrix(source))
    if baths is None:
        raise QThermoError("pass baths=... together with a Hamiltonian")
    return H, list(baths), (H if energy is None else _as_matrix(energy))


def current_statistics(source, count, baths=None, energy=None,
                       label: str | None = None) -> CurrentStatistics:
    """Exact mean and noise of a counted current in the steady state.

    Parameters
    ----------
    source : Model, SteadyState, or Hamiltonian
        With a Hamiltonian, also pass ``baths``.
    count : dict or str
        What to count (see :func:`counting_weights`). A bare bath name means
        ``{name: "energy"}``, the heat current from that bath.
    energy : array, optional
        Energy operator for jump energies and entropy production. Defaults to
        the model's consistent choice (``H0`` for local baths, ``H`` otherwise).

    Examples
    --------
    >>> stats = current_statistics(qt.models.three_level_maser(),
    ...                            {"hot": "energy", "cold": "energy"})
    >>> stats.tur_ratio     # power fluctuations of the maser vs. the TUR
    """
    H, baths, E = _resolve(source, baths, energy)
    if isinstance(count, str):
        count = {count: "energy"}
    label = label or ("counted: " + ", ".join(f"{k} ({v})" for k, v in count.items()))

    c_ops = [L for b in baths for L in b.c_ops]
    weights = np.array(counting_weights(baths, count, E))
    dim = H.shape[0]
    rho = steady_state(H, baths)
    L = liouvillian(H, baths)
    sparse = sp.issparse(L)
    supers = _jump_superoperators(c_ops, sparse)
    r = _vec(rho)

    rates = np.array([float(np.real(np.trace(Lj @ rho @ Lj.conj().T))) for Lj in c_ops])
    mean = float(np.sum(weights * rates))
    second = float(np.sum(weights ** 2 * rates))
    activity = float(np.sum(rates))

    J_sup = sum(w * S for w, S in zip(weights, supers) if w != 0)
    if isinstance(J_sup, int):   # nothing counted
        raise QThermoError("no jump has a non-zero weight; nothing is counted")
    y = J_sup @ r - mean * r                        # (1 - P) J rho, traceless
    x = _drazin_solve(L, y, r, dim)                 # L^D y
    correction = float(np.real(_vec(np.eye(dim)).conj() @ (J_sup @ x)))
    noise = second - 2.0 * correction

    sigma = None
    if all(b.temperature is not None for b in baths):
        sigma = 0.0
        for b in baths:
            sigma -= b.heat_current(rho, E) / b.temperature
    return CurrentStatistics(mean, float(noise), activity, sigma, label)


def _drazin_solve(L, y, rho_vec, dim):
    """Solve ``L x = y`` with ``Tr x = 0`` (the Drazin-inverse solution).

    Uses the bordered system ``[[L, rho], [1^T, 0]] [x; a] = [y; 0]``, which is
    non-singular exactly when the steady state is unique.
    """
    n = dim * dim
    trace_row = _vec(np.eye(dim)).conj()
    if sp.issparse(L):
        border = sp.bmat([[L, sp.csr_matrix(rho_vec.reshape(-1, 1))],
                          [sp.csr_matrix(trace_row.reshape(1, -1)), None]],
                         format="csc")
        solution = spla.spsolve(border, np.concatenate([y, [0.0]]))
    else:
        border = np.zeros((n + 1, n + 1), dtype=complex)
        border[:n, :n] = L
        border[:n, n] = rho_vec
        border[n, :n] = trace_row
        solution = np.linalg.solve(border, np.concatenate([y, [0.0]]))
    return solution[:n]


def scaled_cgf(source, count, s, baths=None, energy=None) -> np.ndarray:
    """Scaled cumulant generating function ``theta(s)`` of the counted current.

        theta(s) = lim_{t->inf} (1/t) ln < exp(s N_t) >

    evaluated as the eigenvalue of the tilted Liouvillian
    ``L + sum_j (exp(s nu_j) - 1) J_j`` with the largest real part. Its
    derivatives at ``s = 0`` are the cumulants (``theta'(0) = J``,
    ``theta''(0) = D``); its Legendre transform is the large-deviation rate
    function of the time-averaged current. Dense; intended for small systems.
    """
    H, baths, E = _resolve(source, baths, energy)
    if isinstance(count, str):
        count = {count: "energy"}
    c_ops = [L for b in baths for L in b.c_ops]
    weights = counting_weights(baths, count, E)
    L0 = liouvillian(H, baths, sparse=False)
    supers = _jump_superoperators(c_ops, sparse=False)
    s_values = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.empty(len(s_values))
    for k, s_k in enumerate(s_values):
        tilted = L0 + sum((np.exp(s_k * w) - 1.0) * S
                          for w, S in zip(weights, supers) if w != 0)
        eigenvalues = np.linalg.eigvals(tilted)
        out[k] = float(np.max(eigenvalues.real))
    return out if np.ndim(s) else out[0]
