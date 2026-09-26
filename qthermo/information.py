"""Thermodynamics of information: erasure, reset, and their finite-time cost.

Landauer: resetting a bit of unknown value to a fixed value at temperature T
dissipates at least ``T ln 2`` of heat into the environment. More precisely,
for any process driven by a bath at temperature T,

    Q_to_bath  >=  T (S_initial - S_final)                  (Landauer / Clausius)

with equality only in the quasi-static limit. In finite time the excess
dissipation is positive and, for slow protocols, decays as ``1 / tau``
(Proesmans, Ehrich & Bechhoefer, PRL 125, 100602 (2020); Van Vu & Saito,
PRL 128, 140602 (2022)). Every fault-tolerant quantum computer pays this cost
continuously: each round of error correction resets its ancillas, and the heat
of those resets lands on the coldest stage of the fridge.

This module simulates erasure as an actual driven open-system protocol -- a
qubit whose gap is ramped up while it stays in contact with a bath described
by the adiabatic master equation -- so the finite-time cost, the residual
error and the dependence on the ramp shape come out of the dynamics rather
than being assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.integrate import quad

from .baths import Bath, ohmic_spectrum
from .channels import qubit_hamiltonian, sigma_minus, sigma_plus
from .core import von_neumann_entropy
from .cycle import Cycle, Stroke
from .validation import QThermoError

__all__ = [
    "ErasureResult",
    "landauer_erasure",
    "landauer_bound",
    "erasure_schedules",
    "erasure_friction",
    "thermodynamic_length",
    "geodesic_schedule",
    "predicted_excess",
    "FeedbackResult",
    "szilard_engine",
]


def landauer_bound(rho_initial, rho_final, temperature: float) -> float:
    """Minimum heat released to a bath at ``temperature`` by any process
    taking ``rho_initial`` to ``rho_final``: ``T (S_i - S_f)``."""
    return float(temperature * (von_neumann_entropy(rho_initial)
                                - von_neumann_entropy(rho_final)))


def _linear(s):
    return s


def _smoothstep(s):
    return s * s * (3 - 2 * s)


def _exponential(s, k=4.0):
    return np.expm1(k * s) / np.expm1(k)


erasure_schedules = {
    "linear": _linear,
    "smooth": _smoothstep,
    "exponential": _exponential,
}


# --- slow-driving (thermodynamic geometry) ------------------------------------

def erasure_friction(omega: float, temperature: float, gamma: float = 1.0) -> float:
    """Thermodynamic friction ``zeta(omega)`` of the erasure protocol.

    In the slow-driving limit the excess dissipation of a ramp ``omega(t)`` is
    ``W_ex = int zeta(omega) omega_dot^2 dt`` with

        zeta = beta Var(dH/d omega) / Gamma(omega)

    the equilibrium fluctuation of the conjugate force times its relaxation
    time (Sivak & Crooks, PRL 108, 190602 (2012); Scandi & Perarnau-Llobet,
    Quantum 3, 197 (2019)). ``Gamma`` is read off the same Davies bath the
    simulation uses, so theory and dynamics share one model.
    """
    omega = float(omega)
    beta = 1.0 / temperature
    p_excited = 1.0 / (np.exp(omega * beta) + 1.0)
    variance = p_excited * (1.0 - p_excited)         # dH/domega = -sigma_z / 2
    # Total relaxation rate of the ohmic Davies bath used by the simulation,
    # J(w) (1 + 2 n(w)) = gamma (w / T) coth(w / 2T); tests check this
    # against the rates read off davies_bath itself.
    x = omega * beta
    rate = 2.0 * gamma if x < 1e-8 else gamma * x / np.tanh(0.5 * x)
    return beta * variance / rate


def thermodynamic_length(temperature: float = 1.0, omega_min: float = 1e-3,
                         omega_max: float | None = None, gamma: float = 1.0) -> float:
    """``L = int sqrt(zeta) d omega`` along the erasure path.

    The minimum excess dissipation of any protocol of duration ``tau`` is
    ``L^2 / tau`` in the slow-driving limit, attained by the geodesic.
    """
    omega_max = 12.0 * temperature if omega_max is None else omega_max
    value, _ = quad(lambda w: np.sqrt(erasure_friction(w, temperature, gamma)),
                    omega_min, omega_max, limit=200)
    return float(value)


def geodesic_schedule(temperature: float = 1.0, omega_min: float = 1e-3,
                      omega_max: float | None = None, gamma: float = 1.0,
                      grid: int = 2000):
    """The minimum-dissipation ramp: constant speed in the friction metric.

    Returns a schedule ``s -> fraction`` for :func:`landauer_erasure`. The gap
    moves slowly where fluctuations are large and relaxation is slow (near
    ``omega ~ T``), and quickly elsewhere.
    """
    omega_max = 12.0 * temperature if omega_max is None else omega_max
    omegas = np.linspace(omega_min, omega_max, grid)
    speed = np.sqrt([erasure_friction(w, temperature, gamma) for w in omegas])
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1])
                                           * np.diff(omegas))])
    arc /= arc[-1]

    def schedule(s):
        omega = np.interp(s, arc, omegas)
        return (omega - omega_min) / (omega_max - omega_min)
    schedule.length = thermodynamic_length(temperature, omega_min, omega_max, gamma)
    return schedule


def predicted_excess(schedule, tau: float, temperature: float = 1.0,
                     omega_min: float = 1e-3, omega_max: float | None = None,
                     gamma: float = 1.0) -> float:
    """Slow-driving prediction ``(1/tau) int zeta(omega(s)) omega'(s)^2 ds``."""
    omega_max = 12.0 * temperature if omega_max is None else omega_max
    shape = erasure_schedules.get(schedule, schedule) if isinstance(schedule, str) else schedule
    span = omega_max - omega_min
    s = np.linspace(0.0, 1.0, 20001)
    omega = omega_min + span * np.array([float(shape(x)) for x in s])
    derivative = np.gradient(omega, s)
    zeta = np.array([erasure_friction(w, temperature, gamma) for w in omega])
    value = float(np.sum(0.5 * (zeta[1:] * derivative[1:] ** 2
                                + zeta[:-1] * derivative[:-1] ** 2) * np.diff(s)))
    return float(value / tau)


@dataclass
class ErasureResult:
    """Outcome of one finite-time erasure."""

    tau: float
    temperature: float
    heat_to_bath: float
    work: float
    delta_S: float                  # S_final - S_initial of the qubit
    landauer: float                 # T (S_i - S_f)
    error_probability: float        # population left in the excited state
    rho_final: np.ndarray
    stroke: object

    @property
    def excess_heat(self) -> float:
        """Heat released beyond the Landauer bound; >= 0 by the second law."""
        return self.heat_to_bath - self.landauer

    @property
    def entropy_production(self) -> float:
        return self.excess_heat / self.temperature

    @property
    def ideal_bit_bound(self) -> float:
        """``T ln 2``: the bound for a perfect erasure of a full bit."""
        return self.temperature * np.log(2)

    def report(self) -> str:
        return "\n".join([
            f"erasure in time tau = {self.tau:g} at T = {self.temperature:g}",
            f"  heat released to bath : {self.heat_to_bath:.6f}",
            f"  Landauer bound T*dS   : {self.landauer:.6f}"
            f"   (T ln 2 = {self.ideal_bit_bound:.6f})",
            f"  excess dissipation    : {self.excess_heat:.3e}",
            f"  work done on qubit    : {self.work:.6f}",
            f"  residual error p(1)   : {self.error_probability:.3e}",
        ])


def _erasure_bath(omega_of_t, temperature: float, gamma: float):
    """Closed form of ``instantaneous_bath(H, sigma_x, T, ohmic)`` for a qubit.

    Identical operators (checked in the tests against ``davies_bath``), without
    re-diagonalising a 2x2 Hamiltonian at every solver evaluation.
    """
    beta = 1.0 / temperature

    def baths(t):
        w = omega_of_t(t)
        J = gamma * w * beta
        n = 1.0 / np.expm1(w * beta)
        return [Bath("bath", [np.sqrt(J * n) * sigma_plus,
                              np.sqrt(J * (1.0 + n)) * sigma_minus], temperature,
                     kind="global")]
    return baths


def landauer_erasure(tau: float, temperature: float = 1.0,
                     omega_max: float | None = None, omega_min: float = 1e-3,
                     gamma: float = 1.0, schedule="linear", steps: int = 2000,
                     rho_initial=None) -> ErasureResult:
    """Erase a qubit by ramping its gap up while it thermalises.

    The qubit starts maximally mixed (one bit of entropy) with an almost
    closed gap ``omega_min``. The gap is ramped to ``omega_max`` over ``tau``
    while an Ohmic bath at ``temperature`` acts through ``sigma_x`` with rates
    that follow the instantaneous gap. At the end the qubit sits close to its
    ground state; how close sets the residual error.

    Parameters
    ----------
    tau : float
        Duration of the ramp.
    omega_max : float, optional
        Final gap. Defaults to ``12 T``, leaving an error of ~6e-6.
    gamma : float
        Bath coupling: the relaxation rate at gap ``T`` is about ``gamma``.
    schedule : str or callable
        Ramp shape ``s -> fraction``, mapping [0, 1] to [0, 1]:
        ``"linear"``, ``"smooth"``, ``"exponential"``, ``"geodesic"`` (the
        minimum-dissipation protocol, see :func:`geodesic_schedule`) or a
        function.
    rho_initial : array, optional
        Initial state; defaults to the maximally mixed state.
    steps : int
        Grid for the heat/work accounting. The excess dissipation is a small
        difference of large numbers; fast protocols need a fine grid.
    """
    if omega_max is None:
        omega_max = 12.0 * temperature
    if not 0 < omega_min < omega_max:
        raise QThermoError("need 0 < omega_min < omega_max")
    if isinstance(schedule, str) and schedule == "geodesic":
        schedule = geodesic_schedule(temperature, omega_min, omega_max, gamma)
    shape = erasure_schedules.get(schedule, schedule) if isinstance(schedule, str) else schedule
    if not callable(shape):
        raise QThermoError(f"unknown schedule {schedule!r}; options: "
                           f"{sorted(erasure_schedules) + ['geodesic']} or a function")
    if abs(shape(0.0)) > 1e-12 or abs(shape(1.0) - 1.0) > 1e-12:
        raise QThermoError("schedule must map 0 -> 0 and 1 -> 1")

    def omega(t):
        return omega_min + (omega_max - omega_min) * float(shape(min(max(t / tau, 0.0), 1.0)))

    def H(t):
        return qubit_hamiltonian(omega(t))

    bath = _erasure_bath(omega, temperature, gamma)
    stroke = Stroke("erase", H, tau, bath, steps=steps)
    rho0 = np.eye(2, dtype=complex) / 2 if rho_initial is None else rho_initial
    result = Cycle([stroke]).run(rho0).strokes[0]

    rho_f = result.rho_final
    return ErasureResult(
        tau=tau, temperature=temperature,
        heat_to_bath=-result.heat, work=result.work,
        delta_S=von_neumann_entropy(rho_f) - von_neumann_entropy(rho0),
        landauer=landauer_bound(rho0, rho_f, temperature),
        error_probability=float(np.real(rho_f[1, 1])),
        rho_final=rho_f, stroke=result,
    )


# --- measurement and feedback: the Szilard engine --------------------------------

@dataclass
class FeedbackResult:
    """One cycle of a measurement-and-feedback engine."""

    temperature: float
    information: float          # mutual information between outcome and state
    work_extracted: float       # average over outcomes
    outcome_probabilities: np.ndarray
    per_outcome_work: np.ndarray
    tau: float | None

    @property
    def bound(self) -> float:
        """Sagawa-Ueda bound ``T I`` on the extracted work."""
        return self.temperature * self.information

    @property
    def efficiency(self) -> float:
        """Fraction of the information turned into work, ``W / (T I)``."""
        return self.work_extracted / self.bound if self.bound > 0 else float("nan")

    def report(self) -> str:
        tau = "quasi-static" if self.tau is None else f"tau = {self.tau:g}"
        return "\n".join([
            f"Szilard engine at T = {self.temperature:g} ({tau})",
            f"  information gained I     : {self.information:.6f} nats",
            f"  work extracted <W>       : {self.work_extracted:.6f}",
            f"  Sagawa-Ueda bound T I    : {self.bound:.6f}",
            f"  fraction of bound        : {self.efficiency:.4f}",
        ])


def _binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    return float(-np.sum(p * np.log(p)))


def szilard_engine(temperature: float = 1.0, gap: float = 0.0, error: float = 0.0,
                   tau: float | None = None, gamma: float = 1.0, steps: int = 2000,
                   depth: float = 20.0) -> FeedbackResult:
    """Quantum Szilard engine with a qubit memory: measure, then feed back.

    The qubit ``H0 = diag(0, gap)`` starts in its Gibbs state. Its energy is
    measured with error probability ``error`` (a binary symmetric channel).
    Given outcome ``k``, the Hamiltonian is quenched to ``H_k``, whose Gibbs
    state *is* the post-measurement (posterior) state -- so the quench
    dissipates nothing -- and then returned to ``H0`` isothermally while in
    contact with the bath. That is the optimal feedback of Horowitz & Parrondo,
    NJP 13, 123019 (2011): quasi-statically it extracts exactly ``T I``, the
    Sagawa-Ueda bound (PRL 100, 080403 (2008)), where ``I`` is the mutual
    information between outcome and state.

    ``tau=None`` gives the quasi-static result in closed form; a finite ``tau``
    simulates the isothermal return as a driven open-system stroke (Ohmic bath
    following ``H(t)``) and extracts less, by ~1/tau. ``depth`` caps the energy
    (in units of T) of a state the posterior says is never occupied: the
    quasi-static result then falls short of ``T I`` by ~exp(-depth) for a
    perfect measurement, while a shallower quench lowers finite-time friction.
    """
    from .baths import davies_bath
    from .core import free_energy

    T = temperature
    if not 0 <= error < 0.5:
        raise QThermoError("error probability must be in [0, 0.5)")
    E0 = np.array([0.0, float(gap)])
    p = np.exp(-(E0 - E0.min()) / T)
    p /= p.sum()                                       # prior populations
    channel = np.array([[1 - error, error], [error, 1 - error]])   # P(k | state)
    p_k = channel @ p                                  # outcome probabilities
    posterior = (channel * p[None, :]) / p_k[:, None]  # P(state | k), rows k
    information = _binary_entropy(p) - float(np.sum(p_k * [
        _binary_entropy(row) for row in posterior]))

    works = []
    H0 = np.diag(E0).astype(complex)
    for k in range(2):
        q = np.clip(posterior[k], 1e-300, None)
        E_k = -T * np.log(q)
        E_k = np.minimum(E_k - E_k.min(), depth * T)   # Gibbs(H_k) = posterior
        H_k = np.diag(E_k).astype(complex)
        rho = np.diag(posterior[k]).astype(complex)
        quench = float(np.real(np.trace(rho @ (H_k - H0))))
        if tau is None:
            ramp = free_energy(H0, T) - free_energy(H_k, T)
        else:
            H_of_t = lambda t, a=H_k: a + (t / tau) * (H0 - a)

            def baths(t, H_of_t=H_of_t):
                H = H_of_t(t)
                return [davies_bath(H + 1e-9 * np.diag([0.0, 1.0]), sigma_x_matrix(), T,
                                    spectrum=ohmic_spectrum(gamma, reference=T),
                                    name="bath")]
            stroke = Stroke("isothermal", H_of_t, tau, baths, steps=steps)
            ramp = Cycle([stroke]).run(rho).strokes[0].work
        works.append(-(quench + ramp))
    works = np.array(works)
    return FeedbackResult(T, information, float(np.sum(p_k * works)), p_k, works, tau)


def sigma_x_matrix():
    return np.array([[0, 1], [1, 0]], dtype=complex)
