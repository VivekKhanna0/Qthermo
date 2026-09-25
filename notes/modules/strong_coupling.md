---
title: strong_coupling.py
---

Reaction-coordinate mapping.

- `reaction_coordinate_model(H_S, S, T, lam, Omega, kappa, n_levels, weak_baths=...)` -> `Model` (sites system, RC; global baths only).
- `mean_force_state`, `ultrastrong_limit_state` (Cresser-Anders).
- `rc_convergence(build, levels)` — always report this for strong coupling.

Parameters in the package's rate convention (kappa = RC decay rate at Omega),
not a spectral-density formula. Tests: `tests/test_strong_coupling.py`.
