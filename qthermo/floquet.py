"""Periodically driven machines: the Floquet-Markov master equation.

A continuous machine powered by a periodic drive -- a laser-driven qubit, a
qubit whose frequency is modulated, a driven many-body system between baths --
has no static Hamiltonian, so ``analyze`` does not apply. Its weak-coupling
thermodynamics lives in the Floquet basis (Kohler, Dittrich & Hanggi, PRE 55,
300 (1997); Breuer & Petruccione, ch. 3; Szczygielski, Gelbwaser-Klimovsky &
Alicki, PRE 87, 012120 (2013); Kosloff, Entropy 15, 2100 (2013)):

* Floquet states ``|u_a(t)>`` (period ``T_d = 2 pi / Omega``) with quasienergies
  ``eps_a``;
* the bath coupling ``A`` decomposed as
  ``<u_a(t)|A|u_b(t)> = sum_q A_ab(q) exp(i q Omega t)``;
* each component drives the jump ``b -> a`` and hands the bath the energy
  ``w = eps_b - eps_a - q Omega``, at the detailed-balance rate
  ``gamma(w) |A_ab(q)|^2`` of that bath's spectrum.

Under the full secular approximation the Floquet populations obey a classical
rate equation. In its steady state

    J_k = - sum_{a,b,q} w_abq gamma_k(w_abq) |A^k_ab(q)|^2 p_b     (heat into system)
    P   = sum_k J_k                                                 (power to the drive)
    sigma_dot = - sum_k J_k / T_k  >= 0

The second law holds sideband by sideband, because every ``gamma_k`` satisfies
the KMS condition at its own temperature. The drive can pump heat, not just
dissipate, because sidebands let a bath exchange quanta ``w != eps_b - eps_a``.

Validity: weak system-bath coupling, bath correlation time short against the
drive period, and quasienergy differences (mod Omega) resolved on the scale of
the rates. The last is checked, with a warning when it fails.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, null_space

from .baths import flat_spectrum
from .channels import mean_occupation
from .core import _as_matrix
from .validation import QThermoError, check_hermitian, check_temperature

__all__ = ["DrivenBath", "FloquetSteadyState", "floquet_states", "floquet_analyze",
           "window_spectrum"]


@dataclass
class DrivenBath:
    """A thermal bath coupled to a periodically driven system.

    ``coupling`` is the (time-independent) Hermitian system operator;
    ``spectrum`` the zero-temperature emission rate ``J(w)``, ``w > 0``, as for
    :func:`qthermo.baths.davies_bath`. Filtered spectra
    (:func:`window_spectrum`) are what turn a driven qubit into an engine or a
    refrigerator.
    """

    name: str
    coupling: np.ndarray
    temperature: float
    gamma: float = 1.0
    spectrum: object = None

    def __post_init__(self):
        self.coupling = check_hermitian(_as_matrix(self.coupling), f"{self.name} coupling")
        self.temperature = check_temperature(self.temperature, f"{self.name} temperature")
        if self.spectrum is None:
            self.spectrum = flat_spectrum(self.gamma)

    def rate(self, w: float) -> float:
        """Detailed-balance rate for handing the bath energy ``w`` (either sign)."""
        if abs(w) < 1e-12:
            return 0.0
        if w > 0:
            return float(self.spectrum(w)) * (1.0 + mean_occupation(w, self.temperature))
        return float(self.spectrum(-w)) * mean_occupation(-w, self.temperature)


def window_spectrum(gamma: float, centre: float, width: float):
    """Box-shaped emission rate: ``gamma`` for ``|w - centre| < width/2``, else 0.

    Lets a bath exchange energy only near ``centre`` -- the spectral filtering
    of Gelbwaser-Klimovsky, Alicki & Kurizki, PRE 87, 012140 (2013).
    """
    def J(w):
        w = np.asarray(w, dtype=float)
        return np.where(np.abs(w - centre) < 0.5 * width, gamma, 0.0)
    J.zero_frequency = None
    return J


def floquet_states(H_of_t, period: float, n_time: int = 256, substeps: int = 8):
    """Quasienergies and Floquet modes over one period.

    Returns ``(eps, modes, times)`` with ``modes[k]`` the matrix whose columns
    are ``|u_a(t_k)>`` at ``times[k]`` (``k = 0..n_time-1``), periodic in time,
    and quasienergies in the zone ``(-Omega/2, Omega/2]``.
    """
    H0 = check_hermitian(_as_matrix(H_of_t(0.0)), "H(0)")
    dim = H0.shape[0]
    dt = period / (n_time * substeps)
    U = np.eye(dim, dtype=complex)
    propagators = [U.copy()]
    for k in range(n_time):
        for j in range(substeps):
            t_mid = (k * substeps + j + 0.5) * dt
            U = expm(-1j * _as_matrix(H_of_t(t_mid)) * dt) @ U
        propagators.append(U.copy())
    U_period = propagators[-1]
    phases, vectors = np.linalg.eig(U_period)
    # orthonormalise (eig of a unitary; degenerate blocks may need it)
    vectors, _ = np.linalg.qr(vectors)
    phases = np.array([np.vdot(v, U_period @ v) for v in vectors.T])
    eps = -np.angle(phases) / period
    times = np.arange(n_time) * period / n_time
    modes = [propagators[k] @ vectors * np.exp(1j * eps * times[k])[None, :]
             for k in range(n_time)]
    return eps, modes, times


@dataclass
class FloquetSteadyState:
    quasienergies: np.ndarray
    populations: np.ndarray
    currents: dict
    temperatures: dict
    period: float
    transitions: list = field(default_factory=list)   # (bath, a, b, q, w, rate)

    @property
    def power(self) -> float:
        """Power delivered TO the drive, ``sum_k J_k``; positive for an engine."""
        return float(sum(self.currents.values()))

    @property
    def entropy_production_rate(self) -> float:
        return float(-sum(J / self.temperatures[k] for k, J in self.currents.items()))

    def current(self, name: str) -> float:
        return self.currents[name]

    def efficiency(self, hot: str) -> float:
        if self.currents[hot] <= 0:
            raise QThermoError(f"no heat enters from {hot!r}")
        return self.power / self.currents[hot]

    def cop(self, cold: str) -> float:
        if self.power >= 0:
            raise QThermoError("the drive is not doing work on the machine")
        return self.currents[cold] / (-self.power)

    def mode(self, hot: str, cold: str) -> str:
        from .modes import classify
        return classify(-self.power, self.currents[hot], self.currents[cold])

    def report(self) -> str:
        lines = [f"{'bath':<10}{'T':>8}{'J (into system)':>18}", "-" * 36]
        for name, J in self.currents.items():
            lines.append(f"{name:<10}{self.temperatures[name]:>8g}{J:>18.6e}")
        lines.append("-" * 36)
        lines.append(f"power delivered to the drive: {self.power:+.6e}")
        lines.append(f"entropy production rate:      {self.entropy_production_rate:.6e}")
        return "\n".join(lines)


def floquet_analyze(H_of_t, period: float, baths, n_time: int = 256,
                    q_max: int | None = None, substeps: int = 8,
                    degeneracy_tol: float | None = None) -> FloquetSteadyState:
    """Steady-state thermodynamics of a periodically driven open system.

    Parameters
    ----------
    H_of_t : callable
        ``H(t)`` with period ``period``.
    baths : list of DrivenBath
    n_time : int
        Time points per period for the Floquet modes and their Fourier series.
    q_max : int, optional
        Largest sideband kept (default ``n_time // 4``).
    """
    baths = list(baths)
    if not baths:
        raise QThermoError("floquet_analyze needs at least one bath")
    names = [b.name for b in baths]
    if len(set(names)) != len(names):
        raise QThermoError(f"bath names must be distinct, got {names}")
    Omega = 2 * np.pi / period
    q_max = n_time // 4 if q_max is None else q_max
    eps, modes, _ = floquet_states(H_of_t, period, n_time, substeps)
    dim = len(eps)

    rates_total = np.zeros((dim, dim))                 # W[a, b]: rate b -> a
    transitions = []
    for bath in baths:
        # <u_a(t)|A|u_b(t)> on the time grid, then its Fourier series
        elements = np.array([m.conj().T @ bath.coupling @ m for m in modes])
        coeffs = np.fft.fft(elements, axis=0) / n_time    # index q -> e^{+i q Omega t}
        for q in range(-q_max, q_max + 1):
            A_q = coeffs[q % n_time]
            weights = np.abs(A_q) ** 2
            for a in range(dim):
                for b in range(dim):
                    if weights[a, b] < 1e-14:
                        continue
                    w = eps[b] - eps[a] - q * Omega
                    r = bath.rate(w) * weights[a, b]
                    if r <= 0:
                        continue
                    transitions.append((bath.name, a, b, q, w, r))
                    if a != b:
                        rates_total[a, b] += r

    # full-secular validity: distinct transitions must not share frequencies
    tol = degeneracy_tol or 1e-3 * max(np.max(rates_total), 1e-12)
    ws = sorted({round(t[4], 12) for t in transitions if t[1] != t[2]})
    close = [(x, y) for x, y in zip(ws, ws[1:]) if 0 < y - x < tol]
    if close:
        warnings.warn(
            f"{len(close)} pairs of Floquet transition frequencies lie closer than "
            "the rates: the full secular approximation behind this rate equation "
            "is not justified there. Results may be unreliable.",
            RuntimeWarning, stacklevel=2)

    generator = rates_total - np.diag(rates_total.sum(axis=0))
    kernel = null_space(generator)
    if kernel.shape[1] != 1:
        raise QThermoError(
            f"the Floquet rate equation has {kernel.shape[1]} stationary states; "
            "some Floquet states are not connected by any bath transition")
    p = np.abs(kernel[:, 0])
    p /= p.sum()

    currents = {name: 0.0 for name in names}
    for name, a, b, q, w, r in transitions:
        currents[name] -= w * r * p[b]
    return FloquetSteadyState(eps, p, currents,
                              {b.name: b.temperature for b in baths}, period,
                              transitions)
