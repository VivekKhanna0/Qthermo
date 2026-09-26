---
title: response.py
---

`response(build(dict of T) -> Model, {bath: T})` -> `ThermalResponse`:
`conductance` G = dJ/dT, `onsager` L = T^2 G, `reciprocity_residual`,
`conservation_residual`, `is_positive_semidefinite`, `coupling(a, b)`,
`amplification(control, output)`. Central differences (h = 1e-4 relative),
one steady-state solve per point. Tests: `tests/test_response.py`.
