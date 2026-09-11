"""The ``Stroke`` / ``Cycle`` abstraction.

A thermodynamic cycle is a sequence of strokes; each stroke is an evolution of
the same working medium under some Hamiltonian, optionally coupled to a bath.
Chaining them -- feeding each stroke's final state into the next -- plus the
per-stroke energy bookkeeping is the part every paper currently rewrites by
hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import (
    _as_matrix,
    entropy_production,
    internal_energy,
    von_neumann_entropy,
)
from .solver import evolve
from .validation import QThermoError, check_duration, check_temperature

__all__ = ["Stroke", "StrokeResult", "CycleResult", "Cycle"]


@dataclass
class Stroke:
    """One leg of a thermodynamic cycle.

    Parameters
    ----------
    name : str
        Label used in reports.
    H : array, Qobj, or callable
        Hamiltonian, constant or ``H(t)``.
    c_ops : list
        Collapse operators. Empty means an isolated (unitary) stroke.
    duration : float
        Stroke duration.
    temperature : float, optional
        Bath temperature. Required to report entropy production; a stroke with
        no bath does not need one.
    """

    name: str
    H: object
    duration: float
    c_ops: list = field(default_factory=list)
    temperature: float | None = None
    steps: int = 200

    def __post_init__(self):
        self.duration = check_duration(self.duration, f"stroke {self.name!r} duration")
        self.temperature = check_temperature(
            self.temperature, f"stroke {self.name!r} temperature"
        )
        if self.temperature is None and self.c_ops:
            # Legal, but the stroke will not report entropy production.
            pass
        if self.temperature is not None and not self.c_ops:
            raise QThermoError(
                f"stroke {self.name!r} has a temperature but no collapse "
                "operators. A stroke with no bath exchanges no heat, so its "
                "entropy production is not defined by a bath temperature."
            )
        if self.steps < 10:
            raise QThermoError(
                f"stroke {self.name!r} has steps={self.steps}; at least 10 are "
                "needed for the heat/work accumulation to be meaningful."
            )


@dataclass
class StrokeResult:
    name: str
    heat: float
    work: float
    delta_U: float
    entropy_production: float | None
    rho_initial: np.ndarray
    rho_final: np.ndarray
    times: np.ndarray
    states: list

    @property
    def first_law_residual(self) -> float:
        """|dU - (Q + W)|. Should be at machine-precision level."""
        return abs(self.delta_U - (self.heat + self.work))


@dataclass
class CycleResult:
    strokes: list[StrokeResult]

    def stroke(self, name: str) -> StrokeResult:
        for s in self.strokes:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def heat(self) -> dict:
        return {s.name: s.heat for s in self.strokes}

    @property
    def work(self) -> dict:
        return {s.name: s.work for s in self.strokes}

    @property
    def net_work(self) -> float:
        """Total work done on the system over the cycle."""
        return sum(s.work for s in self.strokes)

    @property
    def total_entropy_production(self) -> float:
        return sum(s.entropy_production or 0.0 for s in self.strokes)

    def heat_from(self, *stroke_names: str) -> float:
        """Net heat absorbed by the system across the named strokes."""
        return sum(self.stroke(name).heat for name in stroke_names)

    def cop(self, cold_strokes: tuple[str, ...]) -> float:
        """Coefficient of performance of a refrigerator.

        COP = (heat drawn from the cold bath) / (net work input).
        """
        q_cold = self.heat_from(*cold_strokes)
        w_net = self.net_work
        if w_net <= 0:
            raise ValueError(
                "net work is not positive: this cycle is not consuming work, "
                "so it is not operating as a refrigerator"
            )
        return q_cold / w_net

    def efficiency(self, hot_strokes: tuple[str, ...]) -> float:
        """Heat-engine efficiency = (work extracted) / (heat absorbed from hot bath)."""
        q_hot = self.heat_from(*hot_strokes)
        if q_hot <= 0:
            raise ValueError("no net heat absorbed from the hot bath")
        return -self.net_work / q_hot

    def cooling_power(self, cold_strokes: tuple[str, ...], cycle_time: float) -> float:
        return self.heat_from(*cold_strokes) / cycle_time

    def figure_of_merit(self, cold_strokes: tuple[str, ...], cycle_time: float) -> float:
        """Power x COP -- the power/efficiency trade-off metric."""
        return self.cooling_power(cold_strokes, cycle_time) * self.cop(cold_strokes)

    @property
    def max_first_law_residual(self) -> float:
        return max(s.first_law_residual for s in self.strokes)

    def report(self) -> str:
        lines = [
            f"{'stroke':<14}{'Q':>13}{'W':>13}{'dU':>13}{'sigma':>13}",
            "-" * 66,
        ]
        for s in self.strokes:
            sigma = "--" if s.entropy_production is None else f"{s.entropy_production:.3e}"
            lines.append(
                f"{s.name:<14}{s.heat:>13.3e}{s.work:>13.3e}"
                f"{s.delta_U:>13.3e}{sigma:>13}"
            )
        lines.append("-" * 66)
        lines.append(f"{'net':<14}{sum(s.heat for s in self.strokes):>13.3e}"
                     f"{self.net_work:>13.3e}")
        lines.append(f"max |dU - (Q+W)| = {self.max_first_law_residual:.2e}")
        return "\n".join(lines)


class Cycle:
    """A sequence of strokes applied repeatedly to one working medium."""

    def __init__(self, strokes: list[Stroke]):
        if not strokes:
            raise ValueError("a cycle needs at least one stroke")
        self.strokes = strokes

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.strokes)

    def run(self, rho0, cycles: int = 1) -> CycleResult:
        """Run the cycle, returning the results of the final pass.

        Running more than one cycle lets the medium settle into its limit
        cycle -- the periodic steady state -- which is the regime almost all
        reported figures of merit implicitly assume.
        """
        rho = _as_matrix(rho0)
        results: list[StrokeResult] = []

        for pass_index in range(cycles):
            results = []
            for stroke in self.strokes:
                out = evolve(
                    rho, stroke.H, stroke.c_ops,
                    duration=stroke.duration, steps=stroke.steps,
                )
                rho_final = out["states"][-1]

                H_initial = stroke.H(0.0) if callable(stroke.H) else stroke.H
                H_final = (stroke.H(stroke.duration)
                           if callable(stroke.H) else stroke.H)
                delta_U = (internal_energy(rho_final, H_final)
                           - internal_energy(rho, H_initial))

                sigma = None
                if stroke.temperature is not None:
                    sigma = entropy_production(
                        rho, rho_final, out["heat"], stroke.temperature
                    )

                results.append(StrokeResult(
                    name=stroke.name,
                    heat=out["heat"],
                    work=out["work"],
                    delta_U=delta_U,
                    entropy_production=sigma,
                    rho_initial=rho,
                    rho_final=rho_final,
                    times=out["times"],
                    states=out["states"],
                ))
                rho = rho_final

        return CycleResult(results)

    def limit_cycle(self, rho0, max_cycles: int = 60, tol: float = 1e-10):
        """Iterate until the state at the start of the cycle stops changing.

        Returns ``(CycleResult, cycles_used, converged)``.
        """
        rho = _as_matrix(rho0)
        for n in range(1, max_cycles + 1):
            result = self.run(rho, cycles=1)
            rho_next = result.strokes[-1].rho_final
            drift = np.max(np.abs(rho_next - rho))
            rho = rho_next
            if drift < tol:
                return self.run(rho, cycles=1), n, True
        return self.run(rho, cycles=1), max_cycles, False
