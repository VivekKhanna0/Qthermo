---
title: fluctuations.py
---

Exact current statistics from the Liouvillian (no sampling).

- `current_statistics(model|steady|H, count, baths=None, energy=None)` -> `CurrentStatistics` (`mean`, `noise` = Var/t, `activity`, `tur_ratio`, `kur_ratio`, `violates_tur`, `.report()`).
- `count`: bath name, or `{bath: "energy"|"quanta"|weight|[weights]}`.
- `scaled_cgf(source, count, s)` — tilted-generator eigenvalue.
- `jump_energy(L, E)` — refuses operators that are not eigenoperators of E.

Noise via Drazin inverse through a bordered solve. Cross-checked against
finite differences of `scaled_cgf` and an independent classical rate model.
Tests: `tests/test_fluctuations.py`.
