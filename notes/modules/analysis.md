---
title: analysis.py
---

Parameter sweeps, 2D joint scans, Pareto fronts, channel comparison — the loop
(build cycle, run to limit cycle, extract a metric, repeat, find optimum) that
every paper reporting an optimized thermal machine writes by hand.

- `_default_initial_state(cycle)` — maximally mixed state of the right dimension; only affects convergence speed, not the limit-cycle answer
- `SweepResult` (dataclass: `parameter, values, metrics, metric_name`) —
  - `.optimum(mode='max')` — raises `QThermoError` if every point is non-finite
  - `.improvement_over(baseline_value)`
  - `.plateau_width(tolerance=0.05)`
  - `.sensitivity(tolerance=0.05)` — fraction of range within tolerance of optimum; answers whether the optimum is a knife-edge or a wide plateau (matters for hardware drift)
  - `.report(top=5)`
- `ScanResult` — `.optimum()`, `.sequential_optimum()` (tune-one-then-other), `.sequential_gap()` (difference between joint and sequential optimum — a real result when it's zero, not a failure)
- `sweep(build_cycle, values, metric, initial_state=None, ...)` — runs the 1D sweep, returns `SweepResult`
- `scan_2d(build_cycle, x_values, y_values, metric, initial_state=None, ...)` — 2D joint scan, returns `ScanResult`
- `compare_channels(build_cycle, channels: dict, metrics: dict, ...)`
- `ParetoFront` — `.knee()` (parameter maximizing power x COP), `.report()`
- `pareto_front(build_cycle, values, cold_strokes, initial_state=None, ...)`

Depends on: `core.py`, `validation.py` (QThermoError).
