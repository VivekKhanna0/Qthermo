---
title: local-entropy-does-not-sum
---

Summing `SubsystemResult.local_entropy_change` across sites does NOT give the
system's entropy change. It overcounts by exactly the correlation built during
the stroke:

    sum_i dS_i  -  dI_corr  =  dS_total

In `examples/subsystem_demo.py` the sites sum to +1.1514 nats while the true
total is +1.1363 — a 0.0151 nat gap that is precisely
`correlation_change`. The gap is physics, not error: correlations carry
entropy that belongs to no individual site.

Same trap on the energy side: `local_heat` summed over sites equals the
stroke's total heat *only* when the Hamiltonian has no interaction term. With
coupling, the difference is `interaction_energy_change`.

If you need a number that adds up, use the totals from `StrokeResult`
(`.heat`, `.work`) and treat the per-site figures as an attribution, not a
partition. `entropy_balance_residual` is the check that the identity above
holds — it should be at machine precision; anything larger means a numerical
problem upstream (non-positive state, bad dims).
