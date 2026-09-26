"""Parameter sweeps, joint scans, and channel comparison.

Finding where a thermal machine performs best means running the same cycle
many times with one thing changed. That loop -- build, run to the limit cycle,
extract a figure of merit, record, repeat, then find the optimum -- is written
by hand in essentially every paper that reports an optimised quantum thermal
machine. This module is that loop.

The 2D scan exists because the standard practice is sequential optimisation:
tune one parameter, fix it, tune the next. That misses interactions between
parameters, and the only reason it is standard is that the nested loop is
tedious to write. It is three lines here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import _as_matrix
from .validation import QThermoError

__all__ = ["SweepResult", "ScanResult", "ParetoFront", "sweep", "scan_2d",
           "compare_channels", "pareto_front"]


def _default_initial_state(cycle):
    """Maximally mixed state of the right dimension.

    The limit cycle is independent of where you start for the Markovian
    dynamics supported here, so this only affects how many passes convergence
    takes, not the answer.
    """
    H = cycle.strokes[0].H
    H = _as_matrix(H(0.0) if callable(H) else H)
    dim = H.shape[0]
    return np.eye(dim, dtype=complex) / dim


@dataclass
class SweepResult:
    """Metric values over a one-dimensional parameter sweep."""

    parameter: str
    values: np.ndarray
    metrics: np.ndarray
    metric_name: str

    def optimum(self, mode: str = "max") -> tuple[float, float]:
        """Return ``(parameter_value, metric)`` at the best point."""
        finite = np.isfinite(self.metrics)
        if not finite.any():
            raise QThermoError(
                f"every point in the sweep over {self.parameter!r} failed to "
                "produce a finite metric; check that the cycle operates as the "
                "machine you expect across this range"
            )
        index = (np.nanargmax(np.where(finite, self.metrics, -np.inf))
                 if mode == "max"
                 else np.nanargmin(np.where(finite, self.metrics, np.inf)))
        return float(self.values[index]), float(self.metrics[index])

    def improvement_over(self, baseline_value: float) -> float:
        """Percentage gain of the optimum over the metric at a chosen point."""
        index = int(np.argmin(np.abs(self.values - baseline_value)))
        baseline = self.metrics[index]
        if baseline == 0:
            raise QThermoError("baseline metric is zero; relative gain undefined")
        best = self.optimum()[1]
        return 100.0 * (best - baseline) / abs(baseline)

    def plateau_width(self, tolerance: float = 0.05) -> float:
        """Width of the region within ``tolerance`` of the optimum.

        A machine tuned to a knife-edge optimum will not hold that setting on
        real hardware, where parameters drift. A wide plateau means the tuning
        is robust; a narrow one is a warning.
        """
        best_value, best_metric = self.optimum()
        if best_metric == 0:
            raise QThermoError("optimum metric is zero; plateau undefined")
        within = np.abs(self.metrics - best_metric) <= tolerance * abs(best_metric)
        within &= np.isfinite(self.metrics)
        if not within.any():
            return 0.0
        selected = self.values[within]
        return float(selected.max() - selected.min())

    def sensitivity(self, tolerance: float = 0.05) -> float:
        """Plateau width as a fraction of the swept range.

        Near 1 means the metric barely depends on this parameter; near 0 means
        performance is sharply peaked and the parameter must be held precisely.
        """
        span = float(self.values.max() - self.values.min())
        if span == 0:
            raise QThermoError("sweep covers a single point")
        return self.plateau_width(tolerance) / span

    def report(self, top: int = 5) -> str:
        best_value, best_metric = self.optimum()
        lines = [
            f"sweep over {self.parameter}  ({len(self.values)} points)",
            f"{'value':>14}{self.metric_name:>16}",
            "-" * 30,
        ]
        order = np.argsort(-np.where(np.isfinite(self.metrics),
                                     self.metrics, -np.inf))[:top]
        for index in sorted(order):
            marker = "  <-- best" if self.values[index] == best_value else ""
            lines.append(f"{self.values[index]:>14.4f}"
                         f"{self.metrics[index]:>16.5e}{marker}")
        lines.append("-" * 30)
        lines.append(f"optimum: {self.parameter} = {best_value:.4f}, "
                     f"{self.metric_name} = {best_metric:.5e}")
        finite_values = self.values[np.isfinite(self.metrics)]
        if best_value in (finite_values.min(), finite_values.max()):
            lines.append("optimum lies at the edge of the swept range: the true "
                         "optimum may be outside it -- extend the range before "
                         "reading anything into sensitivity")
            return "\n".join(lines)
        try:
            fraction = self.sensitivity()
            verdict = ("robust -- the metric barely depends on this parameter"
                       if fraction > 0.5 else
                       "sharply peaked -- this parameter must be held precisely"
                       if fraction < 0.15 else "moderately sensitive")
            lines.append(f"within 5% of optimum over {100*fraction:.0f}% of the "
                         f"swept range ({verdict})")
        except QThermoError:
            pass
        return "\n".join(lines)


@dataclass
class ScanResult:
    """Metric surface over a two-dimensional joint scan."""

    parameters: tuple[str, str]
    x_values: np.ndarray
    y_values: np.ndarray
    grid: np.ndarray  # shape (len(y_values), len(x_values))
    metric_name: str

    def optimum(self) -> tuple[float, float, float]:
        """Return ``(x, y, metric)`` at the best point on the surface."""
        masked = np.where(np.isfinite(self.grid), self.grid, -np.inf)
        row, column = np.unravel_index(np.argmax(masked), masked.shape)
        return (float(self.x_values[column]), float(self.y_values[row]),
                float(self.grid[row, column]))

    def sequential_optimum(self) -> tuple[float, float, float]:
        """The result of the standard tune-one-then-the-other procedure.

        Optimise x with y held at its midpoint, fix that x, then optimise y.
        Comparing this against ``optimum`` shows what sequential tuning misses.
        """
        row = len(self.y_values) // 2
        column = int(np.argmax(np.where(np.isfinite(self.grid[row]),
                                        self.grid[row], -np.inf)))
        best_row = int(np.argmax(np.where(np.isfinite(self.grid[:, column]),
                                          self.grid[:, column], -np.inf)))
        return (float(self.x_values[column]), float(self.y_values[best_row]),
                float(self.grid[best_row, column]))

    def sequential_gap(self) -> float:
        """Percentage the joint optimum beats the sequential one by."""
        joint = self.optimum()[2]
        sequential = self.sequential_optimum()[2]
        if sequential == 0:
            raise QThermoError("sequential optimum is zero; gap undefined")
        return 100.0 * (joint - sequential) / abs(sequential)


def _run_machine(machine, initial_state):
    """Evaluate whatever a builder returned.

    * ``Cycle`` -> its limit cycle;
    * ``Model`` (continuous machine) -> its steady state;
    * anything else (e.g. an ``OttoLimit``) is used as the result itself.
    Returns ``(result, machine)`` for the metric.
    """
    from .cycle import Cycle
    from .models import Model
    if isinstance(machine, Cycle):
        state = initial_state if initial_state is not None else _default_initial_state(machine)
        result, _, _ = machine.limit_cycle(state)
        return result
    if isinstance(machine, Model):
        return machine.analyze()
    return machine


def sweep(build_cycle, values, metric, initial_state=None,
          parameter: str = "parameter", metric_name: str = "metric",
          on_error: str = "nan") -> SweepResult:
    """Run one cycle per parameter value and record a metric.

    Parameters
    ----------
    build_cycle : callable
        ``build_cycle(value)`` returning a ``Cycle`` (run to its limit cycle),
        a ``Model`` (solved for its steady state), or any result object such
        as an ``OttoLimit``, which is passed to ``metric`` as is.
    values : sequence
        Parameter values to try.
    metric : callable
        ``metric(result, machine) -> float``, e.g.
        ``lambda r, c: r.cop(("cold_iso",))`` for cycles or
        ``lambda r, m: r.current("cold")`` for models.
    on_error : {"nan", "raise"}
        What to do when a point fails -- for instance when a cycle stops
        operating as a refrigerator partway through the range, which is
        informative rather than a bug.
    """
    values = np.asarray(list(values), dtype=float)
    metrics = np.empty(len(values))

    for index, value in enumerate(values):
        cycle = build_cycle(value)
        try:
            result = _run_machine(cycle, initial_state)
            metrics[index] = metric(result, cycle)
        except Exception:
            if on_error == "raise":
                raise
            metrics[index] = np.nan

    return SweepResult(parameter, values, metrics, metric_name)


def scan_2d(build_cycle, x_values, y_values, metric, initial_state=None,
            parameters: tuple[str, str] = ("x", "y"),
            metric_name: str = "metric", on_error: str = "nan") -> ScanResult:
    """Joint scan over two parameters.

    ``build_cycle(x, y)`` returns a ``Cycle``, ``Model`` or result object, as
    for :func:`sweep`. Cost is the product of the two grid sizes, so keep grids
    modest.
    """
    x_values = np.asarray(list(x_values), dtype=float)
    y_values = np.asarray(list(y_values), dtype=float)
    grid = np.empty((len(y_values), len(x_values)))

    for row, y in enumerate(y_values):
        for column, x in enumerate(x_values):
            cycle = build_cycle(x, y)
            try:
                result = _run_machine(cycle, initial_state)
                grid[row, column] = metric(result, cycle)
            except Exception:
                if on_error == "raise":
                    raise
                grid[row, column] = np.nan

    return ScanResult(parameters, x_values, y_values, grid, metric_name)


def compare_channels(build_cycle, channels: dict, metrics: dict,
                     initial_state=None) -> dict:
    """Run the same cycle under different noise channels.

    Parameters
    ----------
    build_cycle : callable
        ``build_cycle(c_ops) -> Cycle``.
    channels : dict
        Name -> list of collapse operators.
    metrics : dict
        Name -> ``metric(cycle_result, cycle) -> float``.

    Returns
    -------
    dict mapping channel name to a dict of metric values.
    """
    table = {}
    for channel_name, c_ops in channels.items():
        cycle = build_cycle(c_ops)
        state = (initial_state if initial_state is not None
                 else _default_initial_state(cycle))
        result, _, _ = cycle.limit_cycle(state)
        row = {}
        for metric_name, metric in metrics.items():
            try:
                row[metric_name] = metric(result, cycle)
            except Exception:
                row[metric_name] = np.nan
        table[channel_name] = row
    return table


@dataclass
class ParetoFront:
    """Cooling power against efficiency over a swept control parameter.

    A thermal machine run infinitely slowly reaches its best efficiency and
    delivers no power; run fast it delivers power at a worse efficiency.
    Neither endpoint is the useful operating point, which is why the trade-off
    curve rather than either number alone is what gets reported.
    """

    values: np.ndarray
    power: np.ndarray
    efficiency: np.ndarray
    parameter: str

    def knee(self) -> tuple[float, float, float]:
        """The point maximising the product of power and efficiency.

        Returns ``(parameter, power, efficiency)``.
        """
        product = self.power * self.efficiency
        index = int(np.nanargmax(np.where(np.isfinite(product), product, -np.inf)))
        return (float(self.values[index]), float(self.power[index]),
                float(self.efficiency[index]))

    def report(self) -> str:
        best_value, best_power, best_efficiency = self.knee()
        lines = [f"{self.parameter:>12}{'power':>14}{'COP':>12}", "-" * 38]
        for value, power, efficiency in zip(self.values, self.power, self.efficiency):
            marker = " <--" if value == best_value else ""
            lines.append(f"{value:>12.3f}{power:>14.4e}{efficiency:>12.4f}{marker}")
        lines.append("-" * 38)
        lines.append(f"best power x COP at {self.parameter} = {best_value:.3f}")
        return "\n".join(lines)


def pareto_front(build_cycle, values, cold_strokes, initial_state=None,
                 parameter: str = "parameter") -> ParetoFront:
    """Trace cooling power against coefficient of performance.

    ``build_cycle(value) -> Cycle``. The swept parameter is usually a duration
    or a coupling strength -- anything that moves the machine between the slow,
    efficient limit and the fast, powerful one.
    """
    values = np.asarray(list(values), dtype=float)
    power = np.empty(len(values))
    efficiency = np.empty(len(values))

    for index, value in enumerate(values):
        cycle = build_cycle(value)
        state = (initial_state if initial_state is not None
                 else _default_initial_state(cycle))
        try:
            result, _, _ = cycle.limit_cycle(state)
            power[index] = result.cooling_power(cold_strokes, cycle.duration)
            efficiency[index] = result.cop(cold_strokes)
        except Exception:
            power[index] = np.nan
            efficiency[index] = np.nan

    return ParetoFront(values, power, efficiency, parameter)
