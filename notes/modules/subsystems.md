---
title: subsystems.py
---

Phase 3: spatial resolution. Phases 1-2 answer "what does this cycle do" and
"how does that change with the noise channel", both totaled over the whole
system. This answers *where*, and how much isn't attributable to any site.

Two identities drive the module:

    U_total = sum_i U_i + U_interaction
    S_total = sum_i S_i - I_corr

Energy isn't additive under coupling; entropy isn't additive under
correlation. Both residuals are reported, not dropped.

## Functions
- `partial_trace(rho, keep, dims) -> ndarray` — `keep` is an int or list of site indices; `dims` is e.g. `[2,2,2]`. Reshape-to-rank-2n then contract; axis bookkeeping accounts for axes already removed.
- `embed(operator, site, dims) -> ndarray` — lifts a single-site operator to the full space (`I ⊗ sigma_z ⊗ I`). The bridge between single-qubit ops in `channels.py` and a multi-qubit `Stroke` — both collapse operators and local H terms need lifting.
- `total_correlation(rho, dims) -> float` — multi-information `sum_i S_i - S_total`. Non-negative, zero iff product state.
- `mutual_information(rho, site_a, site_b, dims) -> float` — pairwise version.
- `resolve_stroke(stroke_result, dims, local_H) -> SubsystemResult` — the main entry. `local_H` is one *unembedded* Hamiltonian per site; entries may be callables `H_i(t)`. Interaction terms are deliberately NOT passed — their contribution surfaces as `interaction_energy_change`.
- `resolve_cycle(cycle_result, dims, local_H) -> list[SubsystemResult]`

## Dataclasses
- `SiteResult` — `site, heat, work, delta_U, delta_S, rho_initial, rho_final`; `.first_law_residual` (machine precision per site, even when coupled).
- `SubsystemResult` — `.site(i)`, `.local_heat`, `.local_work`, `.local_entropy_change` (all dicts keyed by site index), `.interaction_energy_change`, `.correlation_change`, `.entropy_balance_residual` (module self-test, exact by construction — non-zero means numerics broke, not physics), `.dominant_site`, `.report()`.

## Reuse
Per-site heat/work uses `core.heat_work_increments` on the *reduced* state and
local Hamiltonian — same midpoint split, so local first-law residuals stay at
machine precision. See decisions/heat-work-split.md.

Tests: `tests/test_subsystems.py`, 19 tests. Demo: `examples/subsystem_demo.py`.
