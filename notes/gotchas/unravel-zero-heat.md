---
title: unravel-zero-heat
---

MCWF heats are sums of jump energies (differences of expectation values):
trajectories whose jumps cancel carry ~1e-16, not 0. Before snapping, `q <= 0`
predicates misclassified them and the README claimed the Otto fridge "fails
about half the time" (0.525); the exact value (cycle_counting) is 0.7955.
`unravel` now snaps |q| < 1e-9 * scale to 0; regression test in
tests/test_counting.py.
