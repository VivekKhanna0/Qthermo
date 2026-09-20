---
title: plotting.py
---

Figure generation. Lazily imported — matplotlib is optional, so importing
`qthermo` doesn't require it. Accessed via `qthermo.__getattr__` in
`__init__.py`, not a direct top-level import.

- `_require_matplotlib()` — raises a clear error if matplotlib isn't installed, instead of a bare `ImportError`
- `_gap_and_population(states, H_of_t, times)` — helper turning raw states into the energy-gap/population coordinates used for the "quantum P-V diagram"
- `plot_cycle(result, cycle=None, ax=None, annotate=True)` — the cycle diagram (gap vs. population; enclosed area = work)
- `plot_sweep(sweep_result, ax=None, mark_optimum=True)`
- `plot_scan(scan_result, ax=None, log_x=False, log_y=False)`
- `plot_distribution(ensemble, which, deterministic=None, ...)` — trajectory histogram vs. the deterministic (master-equation) value
- `plot_dashboard(result, cycle, sweep_result=None, ensemble=None, ...)` — combined 2x2 figure, what `examples/full_demo.py` calls to produce the four README figures

Depends on: `matplotlib` (optional, imported lazily inside each function's first call path).
