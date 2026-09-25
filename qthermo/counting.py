"""Exact counting statistics of stroke machines.

``unravel`` samples quantum-jump trajectories to get the distribution of heat
per cycle. This module gets the same distribution *exactly*, without
sampling, by propagating the tilted (counting-field) master equation through
the strokes of a cycle:

    G(chi) = Tr[ P_N(chi) ... P_1(chi) rho_start ],
    P_s(chi) = T exp( int L_s(chi, t) dt ),
    L(chi) = L + sum_j (exp(i chi nu_j) - 1) L_j (x) L_j*

With integer weights (``"quanta"``: +1 per quantum absorbed, -1 per quantum
emitted) ``G`` is a Fourier series and a discrete Fourier transform returns
``P(n)``, the probability of a net exchange of ``n`` quanta in one cycle,
to machine precision.

Over many cycles, the dominant eigenvalue ``Lambda(s)`` of the tilted
one-cycle propagator gives the long-run statistics: mean ``theta'(0)`` and
variance ``theta''(0)`` per cycle, with ``theta = ln Lambda``.

A caution on uncertainty relations: the classical TUR bound of 2 is proven for
*time-independent* Markov dynamics. Periodically driven cycles can go below it
classically, so ``tur_ratio`` here is reported for comparison, not as a
certificate of quantumness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .baths import Bath
from .core import _as_matrix
from .solver import flatten_operators
from .validation import QThermoError

__all__ = ["CycleCounting", "cycle_counting"]


def _vec(rho):
    return rho.reshape(-1, order="F")


def _unvec(v, dim):
    return v.reshape((dim, dim), order="F")


def _jump_sign(L, H, tol=1e-8) -> float:
    """+1 if L raises the energy of H, -1 if it lowers it, 0 otherwise."""
    comm = H @ L - L @ H
    norm = np.vdot(L, L).real
    if norm == 0:
        return 0.0
    e = float(np.real(np.vdot(L, comm) / norm))
    if np.linalg.norm(comm - e * L) > 1e-6 * np.sqrt(norm) * max(1.0, abs(e)):
        raise QThermoError(
            "a counted jump operator does not change the energy by a definite "
            "amount under the stroke's Hamiltonian; counting quanta is ambiguous")
    return 0.0 if abs(e) < tol else float(np.sign(e))


def _stroke_ops(stroke, t):
    items = stroke.c_ops(t) if callable(stroke.c_ops) else stroke.c_ops
    return flatten_operators(items or [])


def _tilted_generator(H, ops, weights, s, dim):
    """Tilted Liouvillian with jump weight exp(s * nu_j) (complex s allowed)."""
    eye = np.eye(dim)
    decay = sum((L.conj().T @ L for L in ops), np.zeros((dim, dim), complex))
    K = -1j * H - 0.5 * decay
    gen = np.kron(eye, K) + np.kron(K.conj(), eye)
    for L, nu in zip(ops, weights):
        gen = gen + np.exp(s * nu) * np.kron(L.conj(), L)
    return gen


def _stroke_propagator(stroke, count_spec, s, dim, substeps):
    """Tilted superoperator propagator of one stroke."""
    H = stroke.H
    constant = not callable(H) and not callable(stroke.c_ops)
    counted = count_spec is not None

    def weights_at(t, H_t, ops):
        if not counted:
            return np.zeros(len(ops))
        if count_spec == "quanta":
            return np.array([_jump_sign(L, H_t) for L in ops])
        return np.full(len(ops), float(count_spec)) * np.array(
            [_jump_sign(L, H_t) for L in ops])

    if constant:
        H_t = _as_matrix(H)
        ops = _stroke_ops(stroke, 0.0)
        gen = _tilted_generator(H_t, ops, weights_at(0.0, H_t, ops), s, dim)
        return expm(gen * stroke.duration)

    dt = stroke.duration / substeps
    P = np.eye(dim * dim, dtype=complex)
    for k in range(substeps):
        t = (k + 0.5) * dt
        H_t = _as_matrix(H(t) if callable(H) else H)
        ops = _stroke_ops(stroke, t)
        gen = _tilted_generator(H_t, ops, weights_at(t, H_t, ops), s, dim)
        P = expm(gen * dt) @ P
    return P


@dataclass
class CycleCounting:
    """Exact statistics of quanta exchanged with the counted strokes."""

    cycle: object
    count: dict
    rho_start: np.ndarray
    substeps: int

    @property
    def dim(self) -> int:
        return self.rho_start.shape[0]

    def propagator(self, s: complex = 0.0) -> np.ndarray:
        """Tilted one-cycle superoperator ``P_N(s) ... P_1(s)``."""
        if not hasattr(self, "_fixed"):
            self._fixed = {}          # uncounted strokes do not depend on s
        P = np.eye(self.dim ** 2, dtype=complex)
        for stroke in self.cycle.strokes:
            spec = self.count.get(stroke.name)
            if spec is None:
                if stroke.name not in self._fixed:
                    self._fixed[stroke.name] = _stroke_propagator(
                        stroke, None, 0.0, self.dim, self.substeps)
                step = self._fixed[stroke.name]
            else:
                step = _stroke_propagator(stroke, spec, s, self.dim, self.substeps)
            P = step @ P
        return P

    # --- one cycle --------------------------------------------------------

    def distribution(self, n_max: int = 16) -> dict:
        """``{n: P(n)}`` for the net quanta exchanged in ONE cycle.

        Starts from ``rho_start`` (the limit-cycle state by default). Computed
        by a DFT of the generating function on ``4 n_max`` points; the
        probability mass outside ``[-n_max, n_max]`` is reported by
        :meth:`truncation_error` and should be negligible.
        """
        points = 4 * n_max
        chis = 2 * np.pi * np.arange(points) / points
        r = _vec(self.rho_start)
        trace = _vec(np.eye(self.dim)).conj()
        G = np.array([trace @ (self.propagator(1j * chi) @ r) for chi in chis])
        coefficients = np.fft.fft(G) / points        # P(n) at index n mod points
        out = {}
        for n in range(-n_max, n_max + 1):
            out[n] = float(np.real(coefficients[n % points]))
        self._last_distribution = out
        return out

    def truncation_error(self, n_max: int = 16) -> float:
        dist = self.distribution(n_max)
        return float(abs(1.0 - sum(dist.values())))

    def probability(self, predicate, n_max: int = 16) -> float:
        """Probability that the net quanta in one cycle satisfy ``predicate``."""
        return float(sum(p for n, p in self.distribution(n_max).items() if predicate(n)))

    def single_cycle_mean(self, n_max: int = 16) -> float:
        return float(sum(n * p for n, p in self.distribution(n_max).items()))

    # --- many cycles -------------------------------------------------------

    def cgf_per_cycle(self, s: float) -> float:
        """``ln Lambda(s)``: scaled cumulant generating function per cycle."""
        eigenvalues = np.linalg.eigvals(self.propagator(s))
        return float(np.log(np.max(np.abs(eigenvalues))))

    def long_run(self, h: float = 1e-3) -> dict:
        """Mean and variance of net quanta per cycle over many cycles."""
        theta = [self.cgf_per_cycle(x) for x in (-h, 0.0, h)]
        mean = (theta[2] - theta[0]) / (2 * h)
        variance = (theta[2] - 2 * theta[1] + theta[0]) / h ** 2
        return {"mean": mean, "variance": variance,
                "fano": variance / abs(mean) if mean else float("inf")}

    def report(self, n_max: int = 8) -> str:
        dist = self.distribution(n_max)
        long = self.long_run()
        counted = ", ".join(f"{k} ({v})" for k, v in self.count.items())
        lines = [f"net quanta exchanged per cycle, counting {counted}",
                 f"{'n':>5}  {'P(n), one cycle':>16}"]
        for n, p in dist.items():
            if p > 1e-6:
                bar = "#" * int(round(40 * p))
                lines.append(f"{n:>5}  {p:>16.6f}  {bar}")
        lines.append(f"one cycle:  mean {self.single_cycle_mean(n_max):+.6f}, "
                     f"P(n <= 0) = {self.probability(lambda n: n <= 0, n_max):.6f}")
        lines.append(f"long run:   mean {long['mean']:+.6f} per cycle, variance "
                     f"{long['variance']:.6f}, Fano {long['fano']:.4f}")
        return "\n".join(lines)


def cycle_counting(cycle, count, rho_start=None, substeps: int = 400) -> CycleCounting:
    """Set up exact counting statistics for a stroke cycle.

    Parameters
    ----------
    cycle : Cycle
    count : dict
        ``{stroke_name: "quanta"}`` counts +1 per quantum absorbed from that
        stroke's bath and -1 per quantum emitted; a number instead of
        ``"quanta"`` scales that (e.g. the quantum's energy). Strokes not
        named are not counted.
    rho_start : array, optional
        State at the start of the cycle. Defaults to the limit-cycle state.
    substeps : int
        Time slices for strokes whose Hamiltonian or baths depend on time
        (midpoint exponential rule, error O(1/substeps^2)).

    Examples
    --------
    >>> stats = qt.cycle_counting(cycle, {"cold_iso": "quanta"})
    >>> stats.probability(lambda n: n <= 0)      # the fridge failed this cycle
    """
    names = {s.name for s in cycle.strokes}
    unknown = set(count) - names
    if unknown:
        raise QThermoError(f"unknown stroke(s) {sorted(unknown)}; strokes: {sorted(names)}")
    for name, spec in count.items():
        if not (spec == "quanta" or np.isscalar(spec)):
            raise QThermoError(f"count[{name!r}] must be 'quanta' or a number")
    if rho_start is None:
        H0 = cycle.strokes[0].H
        dim = _as_matrix(H0(0.0) if callable(H0) else H0).shape[0]
        result, _, _ = cycle.limit_cycle(np.eye(dim, dtype=complex) / dim)
        rho_start = result.strokes[-1].rho_final
    return CycleCounting(cycle, dict(count), _as_matrix(rho_start), substeps)
