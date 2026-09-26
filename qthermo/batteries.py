"""Quantum batteries: how much work is stored, where, and how fast it arrives.

Ergotropy (``qthermo.core.ergotropy``) is the work a cyclic unitary can extract.
For batteries the finer questions are:

* how much of it sits in **coherence** rather than populations -- the part lost
  first to dephasing (Francica et al., PRL 125, 180603 (2020));
* how much is **locked in correlations** between cells, extractable by a
  global unitary but not by acting on each cell separately (Alicki & Fannes,
  PRE 87, 042123 (2013); Perarnau-Llobet et al., PRX 5, 041011 (2015));
* how much more a **many-copy** (collective) extraction can get per copy -- up
  to the bound set by the thermal state of equal entropy;
* how **fast** a battery charges, and whether charging cells collectively beats
  charging them in parallel (Ferraro et al., PRL 120, 117702 (2018)).

Definitions used (exact, stated so they can be checked):

    incoherent ergotropy  = ergotropy of the state dephased in the energy basis
    coherent ergotropy    = total - incoherent   (>= 0)
    locked ergotropy      = global ergotropy - sum of the cells' ergotropies,
                            for a non-interacting battery Hamiltonian
    asymptotic ergotropy  = U(rho) - U(Gibbs state with the same entropy),
                            the per-copy limit of ergotropy(rho^{(x) N}) / N
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .core import _as_matrix, ergotropy, internal_energy, von_neumann_entropy
from .subsystems import embed, partial_trace
from .validation import QThermoError, check_hermitian

__all__ = [
    "ergotropy_split",
    "local_ergotropies",
    "locked_ergotropy",
    "asymptotic_ergotropy",
    "multi_copy_ergotropy",
    "ChargingResult",
    "charge",
    "dicke_battery",
    "collective_advantage",
]


def ergotropy_split(rho, H) -> dict:
    """Split ergotropy into incoherent and coherent parts.

    Returns ``{"total", "incoherent", "coherent"}``. The incoherent part is the
    ergotropy of the state with all coherence between energy levels removed;
    the coherent part is the remainder, non-negative because dephasing can only
    raise the passive energy.
    """
    rho, H = _as_matrix(rho), check_hermitian(_as_matrix(H))
    energies, basis = np.linalg.eigh(H)
    rho_e = basis.conj().T @ rho @ basis
    # dephase within each energy eigenspace's complement: keep blocks of
    # degenerate levels intact (coherence inside a degenerate block costs no
    # energy and is not 'energetic' coherence)
    mask = np.abs(energies[:, None] - energies[None, :]) < 1e-9 * max(1.0, np.ptp(energies))
    dephased = basis @ np.where(mask, rho_e, 0.0) @ basis.conj().T
    total = ergotropy(rho, H)
    incoherent = ergotropy(dephased, H)
    return {"total": total, "incoherent": incoherent,
            "coherent": max(total - incoherent, 0.0)}


def local_ergotropies(rho, dims, local_H) -> list:
    """Ergotropy of each cell's reduced state (local unitaries only)."""
    rho = _as_matrix(rho)
    if len(local_H) != len(dims):
        raise QThermoError(f"got {len(local_H)} local Hamiltonians for {len(dims)} cells")
    return [ergotropy(partial_trace(rho, i, dims), local_H[i]) for i in range(len(dims))]


def locked_ergotropy(rho, dims, local_H) -> dict:
    """Ergotropy available only to global operations on the battery.

    For ``H = sum_i h_i`` (no interactions between cells), the best local
    unitaries extract ``sum_i W(rho_i)``; a global unitary extracts ``W(rho)``.
    The difference is locked in correlations -- zero for product states.
    """
    rho = _as_matrix(rho)
    H = sum(embed(_as_matrix(h), i, dims) for i, h in enumerate(local_H))
    total = ergotropy(rho, H)
    local = local_ergotropies(rho, dims, local_H)
    return {"global": total, "local": float(sum(local)), "per_cell": local,
            "locked": total - float(sum(local))}


def asymptotic_ergotropy(rho, H) -> float:
    """Per-copy ergotropy in the limit of many copies.

    ``U(rho) - U(G_beta)`` with ``G_beta`` the Gibbs state of ``H`` whose
    entropy equals ``S(rho)``: the most a global unitary can extract per copy
    from ``rho^{(x) N}`` as ``N -> infinity``. Strictly larger than the
    single-copy ergotropy for passive states that are not Gibbs states
    ('activation').
    """
    rho, H = _as_matrix(rho), check_hermitian(_as_matrix(H))
    S = von_neumann_entropy(rho)
    energies = np.linalg.eigvalsh(H)
    S_max = np.log(len(energies))
    if S >= S_max - 1e-12:
        return 0.0
    shifted = energies - energies.min()

    def entropy_at(beta):
        w = np.exp(-beta * shifted)
        p = w / w.sum()
        p = p[p > 1e-300]
        return float(-np.sum(p * np.log(p)))

    if S <= 1e-14:
        target_energy = energies.min()
    else:
        beta = brentq(lambda b: entropy_at(b) - S, 1e-12, 1e4 / max(np.ptp(energies), 1e-12))
        w = np.exp(-beta * shifted)
        target_energy = float(np.sum(energies * w) / w.sum())
    return float(internal_energy(rho, H) - target_energy)


def multi_copy_ergotropy(rho, H, copies: int) -> float:
    """Ergotropy per copy of ``rho^{(x) copies}`` under ``H`` on each copy."""
    rho, H = _as_matrix(rho), _as_matrix(H)
    d = rho.shape[0]
    if d ** copies > 4096:
        raise QThermoError(f"{copies} copies of a {d}-level state is {d ** copies} "
                           "levels; too large for a dense ergotropy")
    big_rho, big_H = rho, H
    for k in range(1, copies):
        big_H = np.kron(big_H, np.eye(d)) + np.kron(np.eye(d ** k), H)
        big_rho = np.kron(big_rho, rho)
    return ergotropy(big_rho, big_H) / copies


# --- charging ---------------------------------------------------------------

@dataclass
class ChargingResult:
    times: np.ndarray
    energy: np.ndarray           # stored energy above the initial value
    ergotropy: np.ndarray | None

    @property
    def average_power(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.times > 0, self.energy / self.times, 0.0)

    def charging_time(self) -> float:
        """Time of the first maximum of stored energy."""
        E = self.energy
        for k in range(1, len(E) - 1):
            if E[k] >= E[k - 1] and E[k] > E[k + 1]:
                return float(self.times[k])
        return float(self.times[int(np.argmax(E))])

    def max_power(self) -> float:
        """Largest average power ``E(t)/t`` over the time window."""
        return float(np.max(self.average_power))

    def report(self) -> str:
        t = self.charging_time()
        k = int(np.argmin(np.abs(self.times - t)))
        lines = [f"charged energy {self.energy[k]:.5f} at t = {t:.4g}",
                 f"max average power {self.max_power():.5e}"]
        if self.ergotropy is not None:
            lines.append(f"ergotropy at that time {self.ergotropy[k]:.5f}")
        return "\n".join(lines)


def charge(H_total, H_battery, state, times, c_ops=None, battery=None) -> ChargingResult:
    """Charge a battery and record stored energy (and ergotropy) against time.

    Parameters
    ----------
    H_total : array
        Full Hamiltonian (battery + charger + coupling), time independent.
    H_battery : array
        The battery Hamiltonian embedded in the full space; stored energy is
        ``Tr[H_battery rho(t)] - Tr[H_battery rho(0)]``.
    state : array
        Initial state vector or density matrix.
    times : array
        Output times (starting at 0).
    c_ops : list, optional
        Collapse operators for an open charging process.
    battery : tuple (keep, dims, h_battery), optional
        If given, the battery's ergotropy is tracked: the reduced state on
        subsystems ``keep`` of the tensor structure ``dims``, with the
        (unembedded) battery Hamiltonian ``h_battery``.
    """
    H = check_hermitian(_as_matrix(H_total))
    HB = _as_matrix(H_battery)
    times = np.asarray(times, dtype=float)
    state = _as_matrix(state)
    if state.ndim == 1:
        state = np.outer(state, state.conj())
    if c_ops:
        from .solver import evolve
        states = []
        rho = state
        states.append(rho)
        for t0, t1 in zip(times[:-1], times[1:]):
            rho = evolve(rho, H, c_ops, duration=t1 - t0, steps=10)["states"][-1]
            states.append(rho)
    else:
        energies, V = np.linalg.eigh(H)
        rho_e = V.conj().T @ state @ V
        phases = energies[:, None] - energies[None, :]
        # Tr[H_B rho(t)] = sum_ij B_ji rho_ij exp(-i (E_i - E_j) t): O(d^2)
        # per time; full states are built only if ergotropy is tracked.
        weights = (V.conj().T @ HB @ V).T * rho_e
        energy = np.array([float(np.real(np.sum(weights * np.exp(-1j * phases * t))))
                           for t in times])
        energy -= energy[0]
        erg = None
        if battery is not None:
            keep, dims, h_b = battery
            erg = np.array([
                ergotropy(partial_trace(V @ (rho_e * np.exp(-1j * phases * t)) @ V.conj().T,
                                        keep, dims), h_b)
                for t in times])
        return ChargingResult(times, energy, erg)
    E0 = float(np.real(np.trace(HB @ states[0])))
    energy = np.array([float(np.real(np.trace(HB @ r))) - E0 for r in states])
    erg = None
    if battery is not None:
        keep, dims, h_b = battery
        erg = np.array([ergotropy(partial_trace(r, keep, dims), h_b) for r in states])
    return ChargingResult(times, energy, erg)


def _spin_operators(j: float):
    m = np.arange(j, -j - 1, -1)
    d = len(m)
    Jz = np.diag(m).astype(complex)
    Jp = np.zeros((d, d), dtype=complex)
    for k in range(1, d):
        Jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return Jz, Jp


def dicke_battery(N: int, g: float = 0.05, omega: float = 1.0,
                  rotating_wave: bool = False, photons: int | None = None,
                  cutoff: int | None = None) -> dict:
    """N two-level cells charged by one cavity mode (Dicke battery).

    Ferraro, Campisi, Andolina, Pellegrini & Polini, PRL 120, 117702 (2018).
    The cells start in their ground state, the cavity in the Fock state with
    ``photons`` quanta (default ``N``: exactly enough to charge every cell).
    The spin part is written in the symmetric (Dicke) subspace, which the
    dynamics never leaves, so N cells cost ``N + 1`` levels, not ``2^N``.

        H = omega (J_z + N/2) + omega a^dag a + g (J_+ + J_-)(a + a^dag)

    ``rotating_wave=True`` keeps only ``g (J_+ a + J_- a^dag)``
    (Tavis-Cummings).

    Returns a dict with ``H``, ``H_battery`` (embedded), ``state`` (initial
    vector), ``dims`` = ``[N + 1, cutoff]`` and ``h_battery`` (unembedded, on
    the symmetric subspace).
    """
    if N < 1:
        raise QThermoError("a battery needs at least one cell")
    j = N / 2
    Jz, Jp = _spin_operators(j)
    Jm = Jp.conj().T
    photons = N if photons is None else photons
    cutoff = cutoff or (photons + N + 6)
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(complex)
    ds = len(Jz)
    Is, Ic = np.eye(ds), np.eye(cutoff)
    h_battery = omega * (Jz + j * Is)
    H_B = np.kron(h_battery, Ic)
    H_C = omega * np.kron(Is, a.conj().T @ a)
    if rotating_wave:
        V = g * (np.kron(Jp, a) + np.kron(Jm, a.conj().T))
    else:
        V = g * np.kron(Jp + Jm, a + a.conj().T)
    spins = np.zeros(ds)
    spins[-1] = 1.0                    # m = -j: every cell in its ground state
    field = np.zeros(cutoff)
    field[photons] = 1.0
    return {"H": H_B + H_C + V, "H_battery": H_B, "state": np.kron(spins, field).astype(complex),
            "dims": [ds, cutoff], "h_battery": h_battery, "N": N}


def collective_advantage(N_values, g: float = 0.05, omega: float = 1.0,
                         rotating_wave: bool = False, points: int = 3000) -> dict:
    """Charging power of the Dicke battery relative to N independent cells.

    For each N, the maximum average power of collective charging is divided by
    ``N`` times the single-cell (N = 1) value -- the parallel scheme, in which
    each cell has its own cavity. Ferraro et al. find this ratio grows as
    ``sqrt(N)`` at large N.
    """
    def best_power(N):
        battery = dicke_battery(N, g, omega, rotating_wave)
        horizon = 1.2 * np.pi / g
        run = charge(battery["H"], battery["H_battery"], battery["state"],
                     np.linspace(0.0, horizon, points))
        return run.max_power()

    single = best_power(1)
    ratios = np.array([best_power(N) / (N * single) for N in N_values])
    N_values = np.asarray(list(N_values), dtype=float)
    slope = float(np.polyfit(np.log(N_values[-3:]), np.log(ratios[-3:]), 1)[0]) \
        if len(N_values) >= 3 else float("nan")
    return {"N": N_values, "advantage": ratios, "large_N_exponent": slope}
