---
title: engines.py
---

Otto cycles with interacting working media.

- `ideal_otto(H_c, H_h, T_c, T_h, path=None, follow_crossings=True)` -> `OttoLimit`; pairs levels by adiabatic continuation (`adiabatic_pairing`).
- `otto_cycle(H_c, H_h, T_c, T_h, couplings, ..., spectrum=None)` — Davies bath per coupling; refuses non-thermalising baths; warns on short isochores.

See gotchas/symmetry-blocked-thermalisation.md. Tests: `tests/test_engines.py`.
