---
title: network.py
---

Site-resolved energy flows of a state.

- `heat_flow_map(steady_or_model, ...)` -> `HeatFlowMap`: `bath_to_site`, `term_to_site`, `bath_to_interaction`, `virtual_temperature`, `mutual_information`, `negativity`, `concurrence`, `.bond_current(term)`, `.site_balance()` (should be ~1e-17 in steady state), `.report()`.
- `virtual_temperature`, `concurrence`, `negativity`, `correlation_matrices`.

Plots in plotting.py: `plot_heat_network`, `plot_correlations`, `plot_machine`.
See gotchas/secular-blind-currents.md. Tests: `tests/test_network.py`.
