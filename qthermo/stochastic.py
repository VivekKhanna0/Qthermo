"""Trajectory-resolved thermodynamics.

The master equation gives the *average* heat. But a single run of a quantum
thermal machine either emits a quantum into the bath or it does not; the
distribution of outcomes is a different object from its mean, and it is what
fluctuation theorems constrain. This module unravels the same ``Cycle`` into
quantum-jump trajectories so that distribution is available, and provides a
Jarzynski check as a built-in correctness test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .core import _as_matrix, free_energy
from .solver import flatten_operators

__all__ = ["TrajectoryEnsemble", "unravel", "jarzynski_tpm"]


@dataclass
class TrajectoryEnsemble:
    """Per-trajectory heat and work, plus the jump record."""

    heat: np.ndarray          # shape (n_trajectories,), net over the cycle
    work: np.ndarray
    jump_counts: np.ndarray   # shape (n_trajectories, n_operators)
    jump_quanta: list         # per trajectory: list of energy changes
    stroke_heat: dict         # stroke name -> array of shape (n_trajectories,)

    @property
    def mean_heat(self) -> float:
        return float(np.mean(self.heat))

    @property
    def mean_work(self) -> float:
        return float(np.mean(self.work))

    def series(self, which: str) -> np.ndarray:
        """Return a named quantity: 'heat', 'work', or a stroke name."""
        if which in ("heat", "work"):
            return getattr(self, which)
        if which in self.stroke_heat:
            return self.stroke_heat[which]
        raise KeyError(
            f"unknown quantity {which!r}; "
            f"available: heat, work, {', '.join(self.stroke_heat)}"
        )

    def standard_error(self, which: str = "heat") -> float:
        data = self.series(which)
        return float(np.std(data, ddof=1) / np.sqrt(len(data)))

    def histogram(self, which: str = "heat", bins: int = 40):
        return np.histogram(self.series(which), bins=bins)

    def probability_of(self, which: str, predicate) -> float:
        """Fraction of trajectories satisfying a predicate.

        The motivating use: ``probability_of('cold_iso', lambda q: q <= 0)``
        gives the chance that the refrigerator failed to draw any heat out of
        the cold bath on a given run -- a reliability question the averaged
        solution cannot answer, because on average it always succeeds.
        """
        data = self.series(which)
        return float(np.mean(predicate(data)))


def _sample_initial_pure_state(rho: np.ndarray, rng) -> np.ndarray:
    """Draw a pure state from the spectral decomposition of rho."""
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.clip(eigenvalues.real, 0.0, None)
    eigenvalues /= eigenvalues.sum()
    index = rng.choice(len(eigenvalues), p=eigenvalues)
    return eigenvectors[:, index].astype(complex)


def _expectation(psi: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(psi.conj() @ (operator @ psi)))


def _mcwf_stroke(psi, H, c_ops, duration, steps, rng):
    """Monte Carlo wavefunction evolution over one stroke.

    Returns (final state, heat from jumps, jump counts, jump energy quanta).
    """
    dim = psi.shape[0]
    constant_H = not callable(H)
    H_of_t = H if callable(H) else (lambda t, _H=_as_matrix(H): _H)

    if not c_ops:
        # Unitary stroke: no bath, so no jumps and no heat.
        dt = duration / steps
        for k in range(steps):
            H_mid = _as_matrix(H_of_t((k + 0.5) * dt))
            psi = expm(-1j * H_mid * dt) @ psi
        return psi, 0.0, np.zeros(0), []

    if callable(c_ops):
        raise NotImplementedError(
            "trajectory unravelling of time-dependent baths is not supported "
            "yet; use Cycle.run for the averaged dynamics")
    c_ops = flatten_operators(c_ops)
    damping = sum(L.conj().T @ L for L in c_ops)

    dt = duration / steps
    propagator = None
    if constant_H:
        H_eff = _as_matrix(H_of_t(0.0)) - 0.5j * damping
        propagator = expm(-1j * H_eff * dt)

    heat = 0.0
    counts = np.zeros(len(c_ops))
    quanta: list[float] = []
    threshold = rng.random()

    for k in range(steps):
        if propagator is not None:
            psi = propagator @ psi
        else:
            H_eff = _as_matrix(H_of_t((k + 0.5) * dt)) - 0.5j * damping
            psi = expm(-1j * H_eff * dt) @ psi

        norm_squared = float(np.real(psi.conj() @ psi))
        if norm_squared <= threshold:
            H_now = _as_matrix(H_of_t((k + 1) * dt))
            psi_normalised = psi / np.sqrt(norm_squared)
            energy_before = _expectation(psi_normalised, H_now)

            weights = np.array([
                float(np.real(psi_normalised.conj()
                              @ (L.conj().T @ L @ psi_normalised)))
                for L in c_ops
            ])
            weights = np.clip(weights, 0.0, None)
            if weights.sum() <= 0:
                threshold = rng.random()
                continue
            choice = rng.choice(len(c_ops), p=weights / weights.sum())

            psi = c_ops[choice] @ psi_normalised
            psi /= np.linalg.norm(psi)

            energy_after = _expectation(psi, H_now)
            delta = energy_after - energy_before
            heat += delta
            quanta.append(delta)
            counts[choice] += 1
            threshold = rng.random()

    psi /= np.linalg.norm(psi)
    return psi, heat, counts, quanta


def unravel(cycle, rho0, trajectories: int = 500, steps: int = 400, seed: int | None = None
            ) -> TrajectoryEnsemble:
    """Run a ``Cycle`` as an ensemble of quantum-jump trajectories.

    Heat is accumulated as the energy carried by individual jumps -- the
    physical picture in which the bath exchanges discrete quanta -- and work is
    recovered as the first-law residual, so ``dU = Q + W`` holds exactly on
    every single trajectory.
    """
    rng = np.random.default_rng(seed)
    rho0 = _as_matrix(rho0)

    heats = np.zeros(trajectories)
    works = np.zeros(trajectories)
    all_counts = []
    all_quanta = []
    stroke_heat = {s.name: np.zeros(trajectories) for s in cycle.strokes}

    for index in range(trajectories):
        psi = _sample_initial_pure_state(rho0, rng)
        total_heat = 0.0
        total_work = 0.0
        counts_per_stroke = []
        quanta_per_trajectory: list[float] = []

        for stroke in cycle.strokes:
            H_initial = (stroke.H(0.0) if callable(stroke.H) else stroke.H)
            H_final = (stroke.H(stroke.duration)
                       if callable(stroke.H) else stroke.H)

            energy_before = _expectation(psi, _as_matrix(H_initial))
            psi, heat, counts, quanta = _mcwf_stroke(
                psi, stroke.H, stroke.c_ops, stroke.duration,
                max(steps // max(len(cycle.strokes), 1), 20), rng,
            )
            energy_after = _expectation(psi, _as_matrix(H_final))

            total_heat += heat
            total_work += (energy_after - energy_before) - heat
            stroke_heat[stroke.name][index] += heat
            counts_per_stroke.append(counts)
            quanta_per_trajectory.extend(quanta)

        heats[index] = total_heat
        works[index] = total_work
        all_counts.append(np.concatenate(counts_per_stroke)
                          if counts_per_stroke else np.zeros(0))
        all_quanta.append(quanta_per_trajectory)

    width = max((c.size for c in all_counts), default=0)
    padded = np.zeros((trajectories, width))
    for i, c in enumerate(all_counts):
        padded[i, :c.size] = c

    return TrajectoryEnsemble(heat=heats, work=works,
                              jump_counts=padded, jump_quanta=all_quanta,
                              stroke_heat=stroke_heat)


def _propagator(H_of_t, duration: float, dim: int, steps: int = 400) -> np.ndarray:
    """Time-ordered propagator U(tau, 0) for a driven closed system."""
    U = np.eye(dim, dtype=complex)
    dt = duration / steps
    for k in range(steps):
        H_mid = _as_matrix(H_of_t((k + 0.5) * dt))
        U = expm(-1j * H_mid * dt) @ U
    return U


def jarzynski_tpm(H_of_t, duration: float, temperature: float, dim: int,
                  steps: int = 400, shots: int | None = None,
                  seed: int | None = None) -> dict:
    """Verify the Jarzynski equality under the two-point measurement protocol.

    The system starts in equilibrium at ``temperature`` with respect to
    ``H_of_t(0)``, its energy is measured, it is driven unitarily for
    ``duration``, and its energy is measured again. Work is the difference of
    the two outcomes. Jarzynski's equality then requires

        <exp(-W / T)> = exp(-dF / T)

    with dF the equilibrium free energy difference -- an identity that holds
    arbitrarily far from equilibrium, which is exactly why it makes a good
    self-test: it should hold for *any* driving protocol you write, so a
    violation points at a bug rather than at physics.

    Returns a dict with the two sides of the identity, the residual, the
    average work, and (when ``shots`` is given) a sampled work distribution.
    """
    beta = 1.0 / temperature
    H_initial = _as_matrix(H_of_t(0.0))
    H_final = _as_matrix(H_of_t(duration))

    E_initial, V_initial = np.linalg.eigh(H_initial)
    E_final, V_final = np.linalg.eigh(H_final)

    shift = E_initial.min()
    weights = np.exp(-beta * (E_initial - shift))
    p_initial = weights / weights.sum()

    U = _propagator(H_of_t, duration, dim, steps)
    # transition[i, f] = |<f| U |i>|^2
    amplitudes = V_final.conj().T @ U @ V_initial
    transition = np.abs(amplitudes.T) ** 2

    work_matrix = E_final[None, :] - E_initial[:, None]
    joint = p_initial[:, None] * transition

    exponential_average = float(np.sum(joint * np.exp(-beta * work_matrix)))
    mean_work = float(np.sum(joint * work_matrix))

    delta_F = free_energy(H_final, temperature) - free_energy(H_initial, temperature)
    predicted = float(np.exp(-beta * delta_F))

    result = {
        "exp_average": exponential_average,
        "predicted": predicted,
        "residual": abs(exponential_average - predicted),
        "mean_work": mean_work,
        "delta_F": delta_F,
        "dissipated_work": mean_work - delta_F,
    }

    if shots:
        rng = np.random.default_rng(seed)
        flat = joint.ravel() / joint.sum()
        draws = rng.choice(flat.size, size=shots, p=flat)
        result["work_samples"] = work_matrix.ravel()[draws]

    return result
