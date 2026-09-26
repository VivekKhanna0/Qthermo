---
title: information.py
---

Finite-time Landauer erasure.

- `landauer_erasure(tau, temperature, schedule=...)` -> `ErasureResult` (`heat_to_bath`, `landauer`, `excess_heat`, `error_probability`).
- `erasure_friction`, `thermodynamic_length`, `geodesic_schedule`, `predicted_excess`.

Uses a closed-form copy of the ohmic Davies qubit bath (tested equal to
`davies_bath`) for speed. Needs a fine heat grid (default steps=2000): the
excess is a small difference of large numbers.
Tests: `tests/test_information.py`.
