---
title: heat-work-split
---

`core.heat_work_increments` uses the midpoint (trapezoidal) split of the
Alicki decomposition:

    W = Tr[(H_b - H_a)(rho_a + rho_b)/2]
    Q = Tr[(H_a + H_b)/2 (rho_b - rho_a)]

This specific split satisfies the first law exactly at every integration
step, not just to first order in dt — `Q + W = Tr[H_b rho_b] - Tr[H_a rho_a] = dU`
holds to machine precision. That's deliberate: it makes `first_law_residual`
a genuine test of the solver's correctness rather than a measurement of the
discretization scheme's error. Source: `core.py` docstring for
`heat_work_increments`.

Consequence: if this split is ever changed for performance or generality,
`first_law_residual` stops being trustworthy as a solver self-test and needs
a different validation strategy.
