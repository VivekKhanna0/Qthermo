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
from scipy.integrate import trapezoid

from .core import (
    _as_matrix,
    entropy_production,
    internal_energy,
    von_neumann_entropy,
)
from .baths import Bath
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
    c_ops : list or callable
        Collapse operators, or :class:`~qthermo.baths.Bath` objects, or a
        callable ``t -> list`` of either for dissipation that follows a driven
        Hamiltonian (see :func:`qthermo.baths.instantaneous_bath`). Empty means
        an isolated (unitary) stroke. With ``Bath`` objects the heat exchanged
        with each bath is reported separately in ``StrokeResult.heat_by_bath``
        and entropy production uses each bath's own temperature.
    duration : float
        Stroke duration.
    temperature : float, optional
        Bath temperature for bare collapse operators. Required to report
        entropy production; a stroke with no bath does not need one. Leave it
        unset when passing ``Bath`` objects, which carry their temperatures.
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
        if self.c_ops is None:
            self.c_ops = []
        sample = self.c_ops(0.0) if callable(self.c_ops) else self.c_ops
        self._has_baths = bool(sample) and all(
            isinstance(item, Bath) for item in sample)
        if sample and not self._has_baths and any(
                isinstance(item, Bath) for item in sample):
            raise QThermoError(
                f"stroke {self.name!r} mixes Bath objects and bare collapse "
                "operators; wrap the bare operators in a Bath so every "
                "channel's heat can be attributed")
        if self._has_baths and self.temperature is not None:
            raise QThermoError(
                f"stroke {self.name!r} has Bath objects and a temperature; "
                "baths carry their own temperatures, so drop temperature=")
        if self.temperature is not None and not sample:
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

    @property
    def dissipative(self) -> bool:
        return bool(self.c_ops(0.0) if callable(self.c_ops) else self.c_ops)

    def baths_at(self, t: float) -> list:
        """The ``Bath`` objects active at time ``t`` (empty for bare c_ops)."""
        if not self._has_baths:
            return []
        return list(self.c_ops(t) if callable(self.c_ops) else self.c_ops)


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
    heat_by_bath: dict = field(default_factory=dict)

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

    def mode(self, hot_strokes=("hot_iso",), cold_strokes=("cold_iso",)) -> str:
        """Operation mode (see :mod:`qthermo.modes`) from net work and heats."""
        from .modes import classify
        return classify(self.net_work, self.heat_from(*hot_strokes),
                        self.heat_from(*cold_strokes))

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


def _heat_by_bath(stroke, out, total_heat) -> dict:
    """Heat from each bath, ``int Tr[H(t) D_k(t)[rho(t)]] dt``.

    With one bath this is the stroke's exact heat. With several, each bath's
    share is integrated with the trapezoid rule on the stored grid and the
    shares are rescaled so they sum to the exact total -- the rescaling is
    O(dt^2) and keeps the first law exact per stroke.
    """
    times, states = out["times"], out["states"]
    names = [b.name for b in stroke.baths_at(0.0)]
    if len(names) == 1:
        return {names[0]: float(total_heat)}
    H = stroke.H
    currents = {name: np.empty(len(times)) for name in names}
    for index, (t, state) in enumerate(zip(times, states)):
        H_t = _as_matrix(H(t) if callable(H) else H)
        for bath in stroke.baths_at(float(t)):
            currents[bath.name][index] = bath.heat_current(state, H_t)
    raw = {n: float(trapezoid(c, times)) for n, c in currents.items()}
    raw_total = sum(raw.values())
    if abs(raw_total) > 1e-14 and abs(total_heat - raw_total) < 1e-2 * abs(raw_total):
        # distribute the O(dt^2) quadrature error proportionally
        return {n: q + (total_heat - raw_total) * abs(q) / sum(abs(v) for v in raw.values())
                for n, q in raw.items()}
    return raw


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
                heat_by_bath = {}
                if stroke.temperature is not None:
                    sigma = entropy_production(
                        rho, rho_final, out["heat"], stroke.temperature
                    )
                elif stroke._has_baths:
                    heat_by_bath = _heat_by_bath(stroke, out, out["heat"])
                    temps = {b.name: b.temperature for b in stroke.baths_at(0.0)}
                    if all(T is not None for T in temps.values()):
                        sigma = float(
                            von_neumann_entropy(rho_final)
                            - von_neumann_entropy(rho)
                            - sum(q / temps[k] for k, q in heat_by_bath.items()))

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
                    heat_by_bath=heat_by_bath,
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
