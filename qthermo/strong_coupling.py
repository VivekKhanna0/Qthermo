"""Strong system-bath coupling via the reaction-coordinate mapping.

Weak-coupling master equations assume the bath barely notices the system: heat
currents scale linearly with the coupling, the system relaxes to its own Gibbs
state, and memory effects are absent. None of that survives at strong
coupling, which is where many solid-state machines operate.

The reaction-coordinate (RC) mapping handles this without leaving the Markovian
toolbox. For a bath with a peaked (Brownian / underdamped) spectral density,
the collective bath mode that couples to the system is pulled out and treated
exactly as part of an enlarged system:

    H_ext = H_S  +  Omega a^dag a  +  lam S (a + a^dag)  +  (lam^2 / Omega) S^2

The residual bath then couples weakly to the RC position ``a + a^dag`` and is
treated with a Davies master equation on ``H_ext``. Everything in qthermo that
works on a :class:`~qthermo.models.Model` -- steady states, heat currents,
site-resolved flows, fluctuations -- then works at strong coupling unchanged.

References: Iles-Smith, Lambert & Nazir, PRA 90, 032114 (2014); Strasberg,
Schaller, Lambert & Brandes, NJP 18, 073007 (2016); Nazir & Schaller, in
*Thermodynamics in the Quantum Regime* (Springer, 2018).

Parameters are stated in the package's own rate convention rather than through
a spectral-density formula whose prefactors vary between papers: ``lam`` is the
system-RC coupling, ``Omega`` the RC frequency (the peak of the original
spectral density), ``kappa`` the rate at which the residual bath damps the RC
at its own frequency (the width of the peak). The ``S^2`` counterterm (on by
default) removes the bath-induced renormalisation of the system potential, the
standard choice when comparing against weak coupling.

Truncation: the RC is kept to ``n_levels`` Fock states. Results should be
checked for convergence in ``n_levels`` -- :func:`rc_convergence` does exactly
that -- since strong coupling displaces the RC and populates higher levels.
"""

from __future__ import annotations

import numpy as np

from .baths import davies_bath, ohmic_spectrum
from .core import _as_matrix, thermal_state
from .models import Model
from .subsystems import partial_trace
from .validation import QThermoError, check_hermitian

__all__ = [
    "reaction_coordinate_model",
    "mean_force_state",
    "ultrastrong_limit_state",
    "rc_convergence",
]


def _annihilation(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n, dtype=float)), 1).astype(complex)


def reaction_coordinate_model(H_S, coupling, T: float, lam: float, Omega: float,
                              kappa: float = 0.05, n_levels: int = 12,
                              weak_baths=(), counterterm: bool = True,
                              name: str = "strong") -> Model:
    """Build the enlarged system+RC model for one strongly coupled bath.

    Parameters
    ----------
    H_S : array
        System Hamiltonian.
    coupling : array
        Hermitian system operator ``S`` through which the strong bath couples.
    T : float
        Temperature of the strongly coupled bath.
    lam, Omega, kappa : float
        System-RC coupling, RC frequency, RC damping rate (see module notes).
    n_levels : int
        Fock-space truncation of the RC.
    weak_baths : sequence of dict
        Additional weakly coupled baths acting on the system, each a dict with
        keys ``coupling``, ``T``, ``gamma`` and ``name``. They are built as
        Davies baths on the *enlarged* Hamiltonian, so the model remains
        thermodynamically consistent.
    counterterm : bool
        Include ``(lam^2 / Omega) S^2``.

    Returns
    -------
    Model
        Sites ``["system", "RC"]`` with ``dims = [d_S, n_levels]``. The bath
        named ``name`` acts on the RC; heat currents are measured with the full
        ``H_ext``, as in Strasberg et al. (2016).
    """
    H_S = check_hermitian(_as_matrix(H_S), "system Hamiltonian")
    S = check_hermitian(_as_matrix(coupling), "strong-bath coupling")
    if S.shape != H_S.shape:
        raise QThermoError("coupling and H_S must have the same shape")
    if n_levels < 2:
        raise QThermoError("the reaction coordinate needs at least 2 levels")
    if Omega <= 0:
        raise QThermoError("the RC frequency Omega must be positive")
    d = H_S.shape[0]
    dims = [d, n_levels]
    a = _annihilation(n_levels)
    x = a + a.conj().T
    I_S, I_RC = np.eye(d), np.eye(n_levels)
    H_RC = Omega * a.conj().T @ a

    H0 = np.kron(H_S, I_RC) + np.kron(I_S, H_RC)
    V = lam * np.kron(S, x)
    if counterterm:
        V = V + (lam ** 2 / Omega) * np.kron(S @ S, I_RC)
    H = H0 + V

    # Residual bath: Ohmic in frequency, normalised so the RC decays at rate
    # kappa at its own frequency.
    residual = davies_bath(H, np.kron(I_S, x), T,
                           spectrum=ohmic_spectrum(kappa, reference=Omega),
                           name=name, sites=(1,))
    baths = [residual]
    for spec in weak_baths:
        try:
            coupling_w = np.kron(_as_matrix(spec["coupling"]), I_RC)
            baths.append(davies_bath(H, coupling_w, spec["T"],
                                     gamma=spec.get("gamma", 0.01),
                                     name=spec["name"], sites=(0,)))
        except KeyError as exc:
            raise QThermoError(
                f"weak bath spec is missing {exc}; each needs coupling, T, "
                "gamma and name") from None

    def rebuild(_me):
        raise QThermoError("the RC model is defined with global baths only")

    return Model(
        H=H, H0=H0, dims=dims, local_H=[H_S, H_RC], baths=baths,
        master_equation="global", roles={"strong": name},
        site_names=["system", "RC"], bonds=[(0, 1)],
        interaction_terms={(0, 1): V},
        description=(f"reaction-coordinate model (lam={lam:g}, Omega={Omega:g}, "
                     f"{n_levels} RC levels)"),
    )


def mean_force_state(H_S, coupling, T: float, lam: float, Omega: float,
                     n_levels: int = 30, counterterm: bool = True) -> np.ndarray:
    """Reduced equilibrium state of the system at finite coupling.

    ``Tr_RC exp(-H_ext / T) / Z`` -- the mean-force Gibbs state for a
    single-mode (Brownian, narrow-peak) environment. It differs from the
    system's own Gibbs state at order ``lam^2``; this is the state the
    strongly coupled system actually relaxes to, and quantities such as
    ergotropy or 'temperature' computed against ``exp(-H_S / T)`` are biased
    by the difference.
    """
    H_S = _as_matrix(H_S)
    S = _as_matrix(coupling)
    d = H_S.shape[0]
    a = _annihilation(n_levels)
    x = a + a.conj().T
    H = (np.kron(H_S, np.eye(n_levels)) + Omega * np.kron(np.eye(d), a.conj().T @ a)
         + lam * np.kron(S, x))
    if counterterm:
        H = H + (lam ** 2 / Omega) * np.kron(S @ S, np.eye(n_levels))
    return partial_trace(thermal_state(H, T), 0, [d, n_levels])


def ultrastrong_limit_state(H_S, coupling, T: float) -> np.ndarray:
    """Ultrastrong-coupling limit of the mean-force Gibbs state.

    ``sum_n P_n exp(-P_n H_S P_n / T) P_n / Z`` with ``P_n`` the eigenprojectors
    of the coupling operator: the system becomes diagonal in the basis the bath
    measures, with populations set by the system energy projected onto it.
    Cresser & Anders, PRL 127, 250601 (2021) (with the counterterm included).
    """
    H_S, S = _as_matrix(H_S), _as_matrix(coupling)
    values, vectors = np.linalg.eigh(S)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    groups, current = [], [0]
    for k in range(1, len(values)):
        if abs(values[k] - values[current[-1]]) < 1e-9:
            current.append(k)
        else:
            groups.append(current)
            current = [k]
    groups.append(current)
    blocks = np.zeros_like(H_S)
    for g in groups:
        P = vectors[:, g] @ vectors[:, g].conj().T
        blocks = blocks + P @ H_S @ P
    energies, basis = np.linalg.eigh(blocks)
    weights = np.exp(-(energies - energies.min()) / T)
    rho = (basis * weights) @ basis.conj().T
    return rho / np.trace(rho).real


def rc_convergence(build, levels=(4, 6, 8, 10, 12, 16), observable=None) -> dict:
    """Check a reaction-coordinate result for convergence in the truncation.

    ``build(n_levels) -> Model``. By default tracks every bath's steady-state
    heat current. Returns ``{"levels": [...], "values": array, "converged":
    bool, "relative_change": float}``, comparing the two largest truncations.
    """
    if observable is None:
        def observable(model):
            r = model.analyze()
            return np.array([r.currents[b.name] for b in model.baths])
    values = np.array([np.atleast_1d(observable(build(n))) for n in levels])
    last, previous = values[-1], values[-2]
    change = float(np.max(np.abs(last - previous)) / max(np.max(np.abs(last)), 1e-300))
    return {"levels": list(levels), "values": values,
            "converged": change < 1e-3, "relative_change": change}
