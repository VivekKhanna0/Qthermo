---
title: steady.py
---

Continuous machines in their non-equilibrium steady state.

## Functions
- `liouvillian(H, baths, sparse=None)` — column-stacking superoperator; dense up to dim 40.
- `steady_state(H, baths, initial_state=None)` — trace-row-replaced linear solve. Uniqueness test = LAPACK rcond of that matrix (no eigendecomposition). Non-unique -> raises unless `initial_state` given. Shared-eigenbasis fast path when all baths are Davies of the same H.
- `analyze(H, baths, energy=None)` -> `SteadyState` — per-bath `currents` with `J_k = Tr[E D_k(rho)]`.
- `compare_master_equations(H, local, global)` -> side-by-side report, flags opposite flows / sigma < 0.
- `relaxation_time(H, baths, populations_only=False)`.

## SteadyState
`.currents`, `.current(name)`, `.entropy_production_rate`, `.total_current`, `.work_rate` (= -sum J, boundary work under H0 accounting), `.power_output`, `.cop(cold, source)`, `.absorption_carnot_cop`, `.efficiency(hot)`, `.report()`, `.model`.

See gotchas/local-master-equation.md. Tests: `tests/test_steady.py`.
