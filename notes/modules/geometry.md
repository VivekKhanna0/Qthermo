---
title: geometry.py
---

Slow-driving thermodynamic geometry for any H(lambda) and thermalising bath.
- `friction(H_of, baths_of, T, lam)` = Tr[dH L^+(d pi)] (sign: positive).
- `optimal_schedule(H_of, baths_of, T, path)` -> `OptimalSchedule` (`length`, `minimum_excess(tau)`, `s_of`, `hamiltonian(H_of, tau)`, `predicted_excess(schedule, tau)`).
- `excess_work(H_of_t, baths_of, T, tau)` simulates with the adiabatic ME, reports T*sigma.
Raises if the baths' fixed point is not Gibbs at T. Tests: `tests/test_geometry.py`.
