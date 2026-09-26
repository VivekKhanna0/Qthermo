---
title: floquet.py
---

Floquet-Markov (full secular) master equation for periodically driven machines.

- `floquet_analyze(H_of_t, period, [DrivenBath...], n_time, q_max)` -> `FloquetSteadyState` (`currents`, `power` = power TO the drive, `entropy_production_rate`, `efficiency`, `cop`, `mode`, `transitions` list of (bath, a, b, q, w, rate)).
- `DrivenBath(name, coupling, T, gamma|spectrum)`, `window_spectrum(gamma, centre, width)`.
- `floquet_states` — quasienergies + periodic modes on a time grid.

Sideband index q is shifted by quasienergy folding: compare weights by transition
frequency w = eps_b - eps_a - q Omega, not by q. Tests: `tests/test_floquet.py`.
