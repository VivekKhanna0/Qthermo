---
title: solver.py
---

The Lindblad integrator, with heat/work accumulated inside the integration
loop rather than reconstructed after — because heat and work are path
quantities, not endpoint quantities.

- `lindbladian(rho, H, c_ops) -> ndarray` — right-hand side of the master equation
- `_pack(rho)` / `_unpack(vector, dim)` — flatten a complex density matrix to a real vector and back, for `scipy.integrate.solve_ivp` (which wants real state vectors)
- `evolve(rho0, H, c_ops=None, duration=1.0, steps=200, rtol=1e-9, atol=1e-11) -> dict` — integrates one stroke, returns `{times, states, heat, work}`. `H` can be callable (`H(t)`) for driven strokes. Uses `solve_ivp` with method `"DOP853"`.

Validates inputs via `validation.py` (`check_duration`, `check_density_matrix`,
`check_hermitian`, `check_operator_shape`) before integrating.

Depended on by: `cycle.py` (`Cycle.run` calls `evolve` once per stroke).
