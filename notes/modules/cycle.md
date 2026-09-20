---
title: cycle.py
---

The main user-facing objects. A cycle is a sequence of strokes; chaining them
(each stroke's final state feeds the next) plus per-stroke energy bookkeeping
is the part every paper rewrites by hand.

- `Stroke` (dataclass) — one leg: `name, H, duration, c_ops=[], temperature=None, steps=200`. `__post_init__` validates: temperature set but no `c_ops` → raises `QThermoError` (see gotchas/stroke-temperature-requires-bath.md); `steps < 10` → raises.
- `StrokeResult` (dataclass) — `heat, work, delta_U, entropy_production, rho_initial, rho_final, times, states`. `.first_law_residual` property = `|dU - (Q+W)|`, should sit at machine precision.
- `CycleResult` (dataclass, wraps `list[StrokeResult]`) —
  - `.stroke(name)` — lookup by name
  - `.heat` / `.work` — dicts keyed by stroke name
  - `.net_work` — sum of work over all strokes
  - `.total_entropy_production`
  - `.heat_from(*stroke_names)` — net heat over named strokes
  - `.cop(cold_strokes)` — coefficient of performance; raises `ValueError` if net work isn't positive (not operating as a refrigerator)
  - `.efficiency(hot_strokes)` — heat-engine efficiency; raises if no net heat absorbed
  - `.cooling_power(cold_strokes, cycle_time)`
  - `.figure_of_merit(cold_strokes, cycle_time)` — power × COP
  - `.max_first_law_residual`
  - `.report() -> str` — the table format shown in the README
- `Cycle` — `.duration` (sum of stroke durations); `.run(rho0, cycles=1) -> CycleResult`; `.limit_cycle(rho0, max_cycles=60, tol=1e-10) -> (CycleResult, cycles_used, converged)` — iterates until the state at cycle start stops changing (periodic steady state).

Depends on: `core.py` (entropy_production, internal_energy), `solver.py` (evolve), `validation.py`.
