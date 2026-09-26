---
title: models.py
---

Canonical machines returning a `Model` (H, H0, dims, local_H, baths, roles,
bonds, interaction_terms, energy, rebuild).

- `absorption_refrigerator(...)` — LPS three-qubit fridge; sites cold/hot/room.
- `three_level_maser(...)` — SSDB maser, rotating frame, `energy=H0`.
- `spin_chain(n, ...)` — XXZ chain between left/right baths.
- `two_qubit_heat_valve(...)` — minimal Levy-Kosloff system.

`Model.analyze()` picks the consistent energy operator (H0 for local baths).
`Model.compare()` rebuilds under the other master equation.
