---
title: core.py
---

Energy, entropy, ergotropy, and thermal states. Units: hbar = k_B = 1.
Sign convention: Q > 0 heat into system, W > 0 work done on system, dU = Q + W.

- `_as_matrix(obj)` — accepts numpy array or QuTiP `Qobj`, returns dense complex array. The only place this module touches QuTiP — everything else is plain arrays.
- `internal_energy(rho, H) -> float` — U = Tr[H rho]
- `von_neumann_entropy(rho, tol=1e-12) -> float` — S = -Tr[rho ln rho], nats
- `thermal_state(H, temperature) -> ndarray` — Gibbs state exp(-H/T)/Z
- `free_energy(H, temperature) -> float` — F = -T ln Z
- `heat_work_increments(rho_a, rho_b, H_a, H_b) -> (heat, work)` — one integration step, midpoint split. See decisions/heat-work-split.md.
- `entropy_production(rho_initial, rho_final, heat, temperature, tolerance=1e-9) -> float` — sigma = dS - Q/T. Warns (doesn't raise) if negative. See gotchas/entropy-production-sign-warning.md.
- `passive_state(rho, H) -> ndarray` — state reachable from rho by unitary evolution that minimizes energy (pairs largest population with lowest energy level)
- `ergotropy(rho, H) -> float` — max work extractable by cyclic unitary; zero for thermal states

Depended on by: `solver.py` (heat_work_increments), `cycle.py` (entropy_production, internal_energy), `stochastic.py` (free_energy).
