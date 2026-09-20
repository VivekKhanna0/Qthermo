---
title: stochastic.py
---

Trajectory-resolved thermodynamics via quantum-jump (MCWF) unravelling. The
master equation gives the average heat; a single run either emits a quantum
or doesn't, and that distribution is what fluctuation theorems actually
constrain.

- `TrajectoryEnsemble` (dataclass) — `heat, work` (shape `(n_trajectories,)`), `jump_counts`, `jump_quanta`, `stroke_heat` (dict: stroke name -> per-trajectory array).
  - `.mean_heat` / `.mean_work`
  - `.series(which)` — `'heat'`, `'work'`, or a stroke name
  - `.standard_error(which='heat')`
  - `.histogram(which='heat', bins=40)`
  - `.probability_of(which, predicate)` — e.g. `probability_of('cold_iso', lambda q: q <= 0)`, the reliability question the master equation can't answer
- `_sample_initial_pure_state(rho, rng)` — draws a pure state from rho's spectral decomposition
- `_mcwf_stroke(psi, H, c_ops, duration, steps, rng)` — one stroke of Monte Carlo wavefunction evolution; returns `(final_state, heat_from_jumps, jump_counts, jump_quanta)`. No `c_ops` → pure unitary propagation, zero jumps.
- `unravel(cycle, rho0, trajectories=500, steps=400, seed=None) -> TrajectoryEnsemble` — runs a `Cycle` as an ensemble of jump trajectories. Heat = energy carried by individual jumps; work = first-law residual, so dU = Q+W exactly on every single trajectory.
- `_propagator(H_of_t, duration, dim, steps=400)` — time-ordered unitary propagator for a driven closed system
- `jarzynski_tpm(H_of_t, duration, temperature, dim, steps=400, shots=None, seed=None) -> dict` — two-point-measurement Jarzynski equality check; returns `exp_average, predicted, residual, mean_work, delta_F, dissipated_work` (+ `work_samples` if `shots` given). Used as a correctness self-test, not just a physics result.

Depends on: `core.py` (free_energy).
