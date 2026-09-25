"""Continuous (autonomous) thermal machines in their non-equilibrium steady state.

Stroke-based machines (``qthermo.cycle``) move one working medium between baths
in time. Continuous machines touch several baths *at once* and run in a
steady state: the three-qubit absorption refrigerator, the three-level maser,
qubit chains carrying heat between two reservoirs. For these the objects of
interest are steady-state heat currents, the entropy production *rate*, and
efficiencies built from currents rather than from per-stroke heats.

    J_k       = Tr[H D_k(rho_ss)]                 heat current from bath k
    sigma_dot = - sum_k J_k / T_k   >= 0          (Spohn, for consistent baths)
    sum_k J_k = 0                                 (energy conservation, static H)

Every quantity here is computed from the exact steady state (a linear solve on
the Liouvillian superoperator), not by integrating to long times, so there is
no convergence error to confuse with physics.

The local/global question for multi-qubit machines lives here too:
:func:`compare_master_equations` runs the same machine under both and reports
where they disagree, including second-law violations of the local approach.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .baths import Bath, dissipator
from .core import _as_matrix, von_neumann_entropy
from .validation import QThermoError, check_hermitian

__all__ = [
    "liouvillian",
    "steady_state",
    "SteadyState",
    "analyze",
    "compare_master_equations",
    "MasterEquationComparison",
    "trace_distance",
]


def _vec(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorisation, the convention of ``liouvillian``."""
    return rho.reshape(-1, order="F")


def _unvec(v: np.ndarray, dim: int) -> np.ndarray:
    return v.reshape((dim, dim), order="F")


def _collect_c_ops(baths) -> list[np.ndarray]:
    c_ops = []
    for bath in baths:
        c_ops.extend(bath.c_ops if isinstance(bath, Bath) else [_as_matrix(bath)])
    return c_ops


def _jump_sum_dense(ops, dim) -> np.ndarray:
    """``sum_k conj(A_k) (x) A_k`` as one matrix product."""
    stack = np.array(ops, dtype=complex).reshape(len(ops), dim * dim)
    gram = stack.conj().T @ stack                      # [(a,b), (c,d)]
    return (gram.reshape(dim, dim, dim, dim)
            .transpose(0, 2, 1, 3).reshape(dim * dim, dim * dim))


def _jump_sum_sparse(ops, dim):
    """``sum_k conj(A_k) (x) A_k`` assembled once in COO form."""
    rows, cols, vals = [], [], []
    for op in ops:
        coo = sp.coo_matrix(op)
        r, c, v = coo.row, coo.col, coo.data
        rows.append((r[:, None] * dim + r[None, :]).ravel())
        cols.append((c[:, None] * dim + c[None, :]).ravel())
        vals.append((v.conj()[:, None] * v[None, :]).ravel())
    if not rows:
        return sp.csr_matrix((dim * dim, dim * dim), dtype=complex)
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(dim * dim, dim * dim))


def liouvillian(H, c_ops=(), sparse: bool | None = None):
    """Liouvillian superoperator in column-stacking convention.

    ``vec(d rho / dt) = L @ vec(rho)`` with ``vec`` stacking columns
    (``rho.reshape(-1, order="F")``). ``c_ops`` may contain :class:`Bath`
    objects, bare operators, or both. Returns a sparse matrix when the system
    dimension exceeds 40 (or when ``sparse=True``).
    """
    H = check_hermitian(_as_matrix(H))
    dim = H.shape[0]
    ops = _collect_c_ops(c_ops)
    for op in ops:
        if op.shape != H.shape:
            raise QThermoError(
                f"collapse operator has shape {op.shape}, Hamiltonian has "
                f"shape {H.shape}")
    if sparse is None:
        sparse = dim > 40
    decay = sum((op.conj().T @ op for op in ops), np.zeros_like(H))
    K = -1j * H - 0.5 * decay          # rho -> K rho + rho K^dag + jumps
    if sparse:
        eye = sp.identity(dim, format="csr", dtype=complex)
        K_ = sp.csr_matrix(K)
        L = sp.kron(eye, K_) + sp.kron(K_.conj(), eye)
        if ops:
            L = L + _jump_sum_sparse(ops, dim)
        return sp.csr_matrix(L)
    eye = np.eye(dim)
    L = np.kron(eye, K) + np.kron(K.conj(), eye)
    if ops:
        L = L + _jump_sum_dense(ops, dim)
    return L


def _null_dimension(L_dense: np.ndarray, tol: float) -> tuple[int, np.ndarray]:
    eigenvalues = np.linalg.eigvals(L_dense)
    order = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    scale = max(np.max(np.abs(eigenvalues)), 1.0)
    return int(np.sum(np.abs(eigenvalues) < tol * scale)), eigenvalues


def steady_state(H, baths=(), initial_state=None, check_unique: bool = True,
                 tol: float = 1e-10) -> np.ndarray:
    """Exact steady state of the Lindblad equation.

    Solves ``L rho = 0`` with ``Tr rho = 1`` directly. When the steady state is
    not unique -- a decoherence-free subspace, a conserved quantity, identical
    qubits sharing one bath -- the answer depends on where you start, and
    returning any single state would silently pick one. In that case this
    raises, unless ``initial_state`` is given, in which case it returns the
    steady state that ``initial_state`` actually relaxes to.
    """
    H = check_hermitian(_as_matrix(H))
    dim = H.shape[0]
    L = liouvillian(H, baths)
    dense = not sp.issparse(L)

    trace_row = _vec(np.eye(dim)).conj()
    rhs = np.zeros(dim * dim, dtype=complex)
    rhs[0] = 1.0

    # Replacing one equation of L rho = 0 by the trace condition gives a
    # matrix that is singular exactly when the steady state is not unique,
    # so its conditioning is the uniqueness test -- no eigendecomposition.
    v, singular = _solve_with_trace_row(L, trace_row, rhs, dense)
    degenerate = check_unique and singular
    if degenerate:
        if initial_state is None:
            count = ""
            if dim <= 16:
                null_dim, _ = _null_dimension(L if dense else L.toarray(), tol)
                count = f" ({null_dim} independent stationary states)"
            raise QThermoError(
                f"the steady state is not unique{count}. Something is "
                "conserved -- often identical sites coupled to a common bath, "
                "or a site with no path to any bath. Pass initial_state=... to "
                "get the steady state that state relaxes to.")
        return _relaxed_state(L if dense else L.toarray(), initial_state, dim, tol)

    if singular:                   # uniqueness check disabled: best effort
        A = (L.toarray() if not dense else np.array(L, dtype=complex))
        A[0, :] = trace_row
        v = np.linalg.lstsq(A, rhs, rcond=None)[0]
    rho = _unvec(v, dim)
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho).real


def _solve_with_trace_row(L, trace_row, rhs, dense, rcond_min=1e-13):
    """Solve the trace-constrained steady-state system; flag singularity."""
    import scipy.linalg as sla
    if dense:
        A = np.array(L, dtype=complex)
        A[0, :] = trace_row
        lu, piv = sla.lu_factor(A, check_finite=False)
        anorm = np.max(np.sum(np.abs(A), axis=0))
        rcond, _ = sla.lapack.zgecon(lu, anorm, norm="1")
        if rcond < rcond_min:
            return None, True
        return sla.lu_solve((lu, piv), rhs, check_finite=False), False
    A = sp.lil_matrix(L)
    A[0, :] = trace_row
    try:
        factor = spla.splu(sp.csc_matrix(A))
    except RuntimeError:           # exactly singular
        return None, True
    diag = np.abs(factor.U.diagonal())
    if diag.min() < rcond_min * diag.max():
        return None, True
    return factor.solve(rhs), False


def _relaxed_state(L: np.ndarray, rho0, dim: int, tol: float) -> np.ndarray:
    """``lim_{t->inf} exp(L t) rho0`` via the spectral projector onto ker L."""
    rho0 = _as_matrix(rho0)
    eigenvalues, right = np.linalg.eig(L)
    left = np.linalg.inv(right)
    scale = max(np.max(np.abs(eigenvalues)), 1.0)
    keep = np.abs(eigenvalues) < tol * scale
    projector = right[:, keep] @ left[keep, :]
    rho = _unvec(projector @ _vec(rho0), dim)
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho).real


@dataclass
class SteadyState:
    """Thermodynamics of a machine in its non-equilibrium steady state."""

    H: np.ndarray
    baths: list
    rho: np.ndarray
    currents: dict                  # bath name -> heat current into system
    energy: np.ndarray              # operator used to define the currents
    labels: dict = field(default_factory=dict)
    model: object = None            # the Model this came from, if any

    # --- balance sheet ---------------------------------------------------

    @property
    def total_current(self) -> float:
        """``sum_k J_k``. Zero for a static H when currents use ``H`` itself.

        With a local energy operator (``analyze(..., energy=H0)``) this is
        minus the power needed to keep the inter-site coupling switched on --
        the 'boundary work' of the local master equation.
        """
        return float(sum(self.currents.values()))

    @property
    def work_rate(self) -> float:
        """Work done *on* the system per unit time, ``-sum_k J_k``.

        Zero for an autonomous machine whose currents are measured with ``H``.
        Positive when an external agent pays to run it (the coupling's boundary
        work in the local picture, or a drive).
        """
        return -self.total_current

    @property
    def power_output(self) -> float:
        """Power delivered by the machine, ``sum_k J_k`` (engine convention)."""
        return self.total_current

    @property
    def entropy_production_rate(self) -> float:
        """``-sum_k J_k / T_k`` over thermal baths. Non-negative when consistent."""
        rate = 0.0
        for bath in self.baths:
            if bath.temperature is not None:
                rate -= self.currents[bath.name] / bath.temperature
        return float(rate)

    def current(self, name: str) -> float:
        try:
            return self.currents[name]
        except KeyError:
            raise KeyError(f"no bath named {name!r}; baths: "
                           f"{', '.join(self.currents)}") from None

    def temperature(self, name: str) -> float:
        for bath in self.baths:
            if bath.name == name:
                return bath.temperature
        raise KeyError(name)

    # --- figures of merit -----------------------------------------------

    def cop(self, cold: str, source: str) -> float:
        """Refrigerator COP ``J_cold / J_source``.

        For an absorption refrigerator ``source`` is the hot bath that powers
        it; for a work-driven machine use :attr:`power` instead.
        """
        j_source = self.current(source)
        if j_source <= 0:
            raise QThermoError(
                f"no heat enters from {source!r} (J = {j_source:.3e}); "
                "the machine is not being powered by that bath")
        return self.current(cold) / j_source

    def absorption_carnot_cop(self, cold: str, source: str, sink: str) -> float:
        """Carnot bound for a three-bath absorption refrigerator.

            COP_C = (1 - T_sink / T_source) / (T_sink / T_cold - 1)
        """
        T_c, T_h, T_r = (self.temperature(cold), self.temperature(source),
                         self.temperature(sink))
        return (1 - T_r / T_h) / (T_r / T_c - 1)

    def efficiency(self, hot: str) -> float:
        """Engine efficiency ``P_out / J_hot``."""
        j_hot = self.current(hot)
        if j_hot <= 0:
            raise QThermoError(f"no heat enters from {hot!r}")
        return self.power_output / j_hot

    def is_cooling(self, cold: str) -> bool:
        return self.current(cold) > 0

    # --- state diagnostics ----------------------------------------------

    @property
    def entropy(self) -> float:
        return von_neumann_entropy(self.rho)

    def report(self) -> str:
        lines = [f"{'bath':<14}{'T':>9}{'J (into system)':>18}{'-J/T':>14}",
                 "-" * 55]
        for bath in self.baths:
            J = self.currents[bath.name]
            T = bath.temperature
            flux = "--" if T is None else f"{-J / T:.4e}"
            T_text = "--" if T is None else f"{T:g}"
            lines.append(f"{bath.name:<14}{T_text:>9}{J:>18.6e}{flux:>14}")
        lines.append("-" * 55)
        lines.append(f"{'sum':<14}{'':>9}{self.total_current:>18.3e}"
                     f"{self.entropy_production_rate:>14.4e}")
        lines.append(f"entropy production rate: {self.entropy_production_rate:.4e}"
                     + ("   <-- NEGATIVE: second law violated by this model"
                        if self.entropy_production_rate < -1e-10 else ""))
        scale = max((abs(J) for J in self.currents.values()), default=0.0)
        if (not np.allclose(self.energy, self.H)
                and abs(self.work_rate) > 1e-10 * max(scale, 1e-300)):
            lines.append(f"currents measured with a reference energy operator; "
                         f"work rate on system: {self.work_rate:.4e}")
        kinds = {b.kind for b in self.baths}
        if "local" in kinds:
            lines.append("note: local baths -- valid only for inter-site "
                         "coupling << bath rates; compare with global baths")
        return "\n".join(lines)


def analyze(H, baths, energy=None, initial_state=None) -> SteadyState:
    """Solve for the steady state and compute every bath's heat current.

    Parameters
    ----------
    H : array or Qobj
        Full system Hamiltonian.
    baths : list of Bath
        Reservoirs coupled simultaneously. Names must be distinct.
    energy : array, optional
        Operator used to define heat currents, ``J_k = Tr[energy D_k(rho)]``.
        Defaults to ``H``. With local baths the thermodynamically consistent
        choice is the bare (non-interacting) Hamiltonian; the currents then no
        longer sum to zero and the remainder is the power needed to maintain the
        coupling (De Chiara et al., NJP 20, 113024 (2018)).
    initial_state : array, optional
        Needed only when the steady state is not unique.
    """
    baths = list(baths)
    if not baths:
        raise QThermoError("a steady-state machine needs at least one bath")
    names = [b.name for b in baths]
    if len(set(names)) != len(names):
        raise QThermoError(f"bath names must be distinct, got {names}")
    for bath in baths:
        if not isinstance(bath, Bath):
            raise QThermoError(
                "analyze() takes Bath objects so heat can be attributed to "
                "each reservoir; wrap bare operators with Bath(name, c_ops, T)")

    H = check_hermitian(_as_matrix(H))
    for bath in baths:
        if bath.dim != H.shape[0]:
            raise QThermoError(
                f"bath {bath.name!r} acts on dimension {bath.dim}, the "
                f"Hamiltonian on {H.shape[0]}. Embed single-site operators "
                "with qthermo.embed or build the bath with local_bath().")

    rho = steady_state(H, baths, initial_state=initial_state)
    E = H if energy is None else check_hermitian(_as_matrix(energy), "energy")
    currents = {
        b.name: float(np.real(np.trace(E @ dissipator(b.c_ops, rho))))
        for b in baths
    }
    result = SteadyState(H=H, baths=baths, rho=rho, currents=currents, energy=E)
    if energy is None and result.entropy_production_rate < -1e-9:
        warnings.warn(
            f"negative entropy production rate "
            f"({result.entropy_production_rate:.3e}). The bath model is "
            "thermodynamically inconsistent for this Hamiltonian -- typical of "
            "local baths on strongly coupled sites. Use davies_bath() or pass "
            "the bare Hamiltonian as energy=.",
            RuntimeWarning, stacklevel=2)
    return result


@dataclass
class MasterEquationComparison:
    """The same machine under local and global baths."""

    local: SteadyState
    global_: SteadyState
    trace_distance: float

    def report(self) -> str:
        lines = [f"{'bath':<14}{'J local':>14}{'J global':>14}{'rel. diff':>12}",
                 "-" * 54]
        for name, j_loc in self.local.currents.items():
            j_glob = self.global_.currents[name]
            rel = abs(j_loc - j_glob) / max(abs(j_glob), 1e-300)
            lines.append(f"{name:<14}{j_loc:>14.5e}{j_glob:>14.5e}{rel:>12.2%}")
        lines.append("-" * 54)
        lines.append(f"entropy production rate: local "
                     f"{self.local.entropy_production_rate:.4e}, global "
                     f"{self.global_.entropy_production_rate:.4e}")
        lines.append(f"steady-state trace distance: {self.trace_distance:.3e}")
        flips = [n for n in self.local.currents
                 if np.sign(self.local.currents[n]) != np.sign(self.global_.currents[n])
                 and abs(self.global_.currents[n]) > 1e-12]
        if flips:
            lines.append("heat flows in OPPOSITE directions for: " + ", ".join(flips))
        if self.local.entropy_production_rate < -1e-10:
            lines.append("local model violates the second law here")
        return "\n".join(lines)


def trace_distance(rho, sigma) -> float:
    """``||rho - sigma||_1 / 2``."""
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(_as_matrix(rho) - _as_matrix(sigma)))))


def compare_master_equations(H, local_baths, global_baths) -> MasterEquationComparison:
    """Run one machine under local and global bath models and compare.

    Baths are matched by name, so build both lists with the same names. The
    comparison uses ``H`` as the energy operator for both, which is the
    reference against which the local model's inconsistencies are measured.
    """
    loc_names = sorted(b.name for b in local_baths)
    glob_names = sorted(b.name for b in global_baths)
    if loc_names != glob_names:
        raise QThermoError(
            f"local and global baths must have the same names; got "
            f"{loc_names} and {glob_names}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        local = analyze(H, local_baths)
    global_ = analyze(H, global_baths)
    return MasterEquationComparison(local, global_,
                                    trace_distance(local.rho, global_.rho))
