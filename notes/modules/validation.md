---
title: validation.py
---

Input validation. Every check corresponds to a physical requirement; error
messages name which one was violated instead of surfacing a raw linear-algebra
traceback.

- `QThermoError(ValueError)` — the one exception type the package raises for bad input
- `check_hermitian(operator, name="Hamiltonian", tol=1e-10) -> ndarray` — square + `|H - H†| <= tol`
- `check_density_matrix(rho, name="state", tol=1e-8) -> ndarray` — square, trace 1, Hermitian, no eigenvalue below `-tol`
- `check_operator_shape(operators, dim, name="collapse operator") -> list` — every operator must be `(dim, dim)`
- `check_temperature(temperature, name="temperature") -> float | None` — `None` passes through; must be `> 0` otherwise (division by T in entropy production)
- `check_duration(duration, name="duration") -> float` — must be `> 0`

Used by `solver.py` and `cycle.py` (`Stroke.__post_init__`) before any physics runs.
