---
title: local-hamiltonian-api
---

`resolve_stroke(stroke_result, dims, local_H)` takes one *unembedded*
Hamiltonian per site and deliberately has no argument for the interaction term.

The alternative was to accept the full Hamiltonian and try to decompose it
into local + interaction parts automatically. Rejected: that decomposition
isn't unique (you can always shift a constant, or reassign a term that acts
non-trivially on two sites), so the function would be silently picking one
convention on the user's behalf and reporting numbers that depend on it.

Instead the user states which local Hamiltonians they mean, and whatever the
full `H` contains beyond them is reported as
`SubsystemResult.interaction_energy_change` — a measured residual, not an
assumption. Exactly zero for a genuinely non-interacting Hamiltonian
(`tests/test_subsystems.py::test_uncoupled_sites_have_no_interaction_energy`),
non-zero and explicit otherwise.

Consequence: a caller who passes the wrong `local_H` doesn't get a wrong
answer silently — they get a large `interaction_energy_change` that flags it.
