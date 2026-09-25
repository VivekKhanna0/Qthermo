# Changelog

## Unreleased

- **Your own machine:** `build_model(local_H, interactions, baths, ...)`
  builds a `Model` from site Hamiltonians, couplings and bath specifications.
  `compare`, `audit`, `report`, `response`, `current_statistics` and the
  plots all work on it, and it can be rebuilt under local or global baths.
- **Shareable results:** `report(model, path)` writes a single HTML file with
  the audit, the per-bath balance sheet, a heat-flow figure, the fluctuation
  ratios and the raw data.
- **Interactive pages:** `explorer(build, values, parameter, path)` writes an
  HTML page with a slider over one parameter. It shows the heat-flow network,
  currents, the entropy production rate and virtual temperatures, and can put
  the local and global master equations side by side.
- **Szilard engine:** `szilard_engine` models feedback with measurement errors
  and checks the result against the Sagawa–Ueda bound `W <= kT I`.

## 0.5.0

- **Periodically driven machines:** `floquet_analyze`, `DrivenBath` and
  `window_spectrum`, a Floquet–Markov master equation with per-bath currents
  and power to the drive. It reduces exactly to Davies without a drive,
  reproduces the Bessel sideband weights, and gives the tight-coupling
  efficiency and COP of the modulated-qubit machine.
- **Transients:** `transient` and `product_thermal_state` give currents,
  cumulative heat and virtual temperatures after switching a machine on. This
  reproduces single-shot cooling below the steady state.
- **Linear response:** `response` gives the conductance and Onsager matrices,
  reciprocity and conservation checks, the Kedem–Caplan coupling, and
  transistor gain. New model: `models.thermal_transistor`.
- **Thermodynamic geometry:** `friction`, `optimal_schedule` and
  `excess_work`, giving minimum-dissipation schedules for any H(λ).
- **Larger systems:** ILU-preconditioned GMRES for steady states above 128
  levels. An 8-qubit chain goes from 158 s and 6 GB to 26 s and 0.6 GB.
- **Lab units:** `LabUnits` converts to and from GHz, mK, seconds and watts.
- **New examples:** the thermal diode tutorial notebook, driven machine,
  transient cooling, thermal transistor, transport scaling, optimal protocol
  and multi-qubit cycle.

## 0.4.0

Aimed at research use: multi-qubit machines, trustworthy bath models, exact
fluctuations, and checks for known modelling traps. Every item below is backed
by tests and, where a published or closed-form result exists, by a row in
`python -m qthermo.benchmarks`.

**Bath models and continuous machines**
- `davies_bath` (global, detailed balance for any H; stored sparse in the
  eigenbasis, so global baths scale to a few hundred levels), `local_bath`,
  `instantaneous_bath` (baths that follow a driven H).
- `analyze` / `steady_state`: exact steady states (uniqueness detected), per-bath
  heat currents, entropy production rate, COP/efficiency and Carnot bounds,
  `compare_master_equations`, `relaxation_time`.
- `qthermo.models`: absorption refrigerator, three-level maser, XXZ chain,
  two-qubit heat valve. Each can be rebuilt under local or global baths.

**Diagnostics**
- `audit`: runs every check in one call and reports it in plain language.
- Warnings or refusals for:
  - non-unique steady states;
  - baths that cannot thermalise because of a symmetry;
  - degenerate levels whose zero-frequency rate is undefined;
  - isochores shorter than the relaxation time;
  - bond currents that vanish identically under the secular approximation;
  - bath temperatures inconsistent with the jump rates.

**Where the heat goes**
- `heat_flow_map`, `plot_machine`: site-resolved flows, virtual temperatures,
  mutual information, negativity, concurrence.
- `site_dynamics`, `plot_site_dynamics`, `plot_bloch_paths`: per-qubit views
  through a cycle.

**Fluctuations**
- `current_statistics`: exact noise, TUR and KUR ratios. `scaled_cgf`: the
  full counting statistics.
- `cycle_counting`: exact single-cycle heat distributions of stroke machines,
  plus long-run statistics.

**Stroke machines**
- Strokes take `Bath` objects (several per stroke, heat per bath) and
  time-dependent baths.
- `ideal_otto`: quasi-static limit for any Hamiltonian pair, with pairing by
  adiabatic continuation. `otto_cycle`: finite-time interacting Otto cycles.
- `classify`, `mode_map`, `plot_mode_map`: engine, refrigerator, accelerator
  and heater regions.

**Beyond weak coupling, information, batteries, protocols**
- `reaction_coordinate_model`, `mean_force_state`, `ultrastrong_limit_state`,
  `rc_convergence`.
- `landauer_erasure`, `geodesic_schedule`, `thermodynamic_length`.
- `friction`, `optimal_schedule`: minimum-dissipation protocols for any H(λ).
- `qthermo.batteries`: coherent, locked and asymptotic ergotropy; Dicke
  charging and its collective advantage.

**Infrastructure**
- CI on Python 3.10 and 3.12, with QuTiP cross-validation and every example
  script run.
- `python -m qthermo.benchmarks`.
- `export`, `save` and `load` (JSON with the version recorded).
- `docs/physics.md`: exact definitions. `examples/tutorial.ipynb`.
- `unravel` vectorised across trajectories (15× faster).

**Fixes**
- `unravel`: trajectories whose jumps cancelled carried heat of about
  +1e-16 instead of 0, so predicates like `q <= 0` misclassified them. The
  earlier README claim that the Otto fridge "draws no heat about half the
  time" was this artifact; the exact value is 0.7955.

## 0.2.0

Single-qubit stroke cycles, trajectory unravelling, sweeps, Pareto fronts,
subsystem resolution.
