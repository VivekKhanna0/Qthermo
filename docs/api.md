# API reference

Generated from the docstrings by `python docs/build_api.py`. Every name
below is importable as `qthermo.<name>` unless marked with its module.
Definitions of the physics are in [physics.md](physics.md).

## Machines and baths

### `qthermo.models`

Canonical quantum thermal machines, built in one call.

- **`class Model(H: 'np.ndarray', H0: 'np.ndarray', dims: 'list', local_H: 'list', baths: 'list', master_equation: 'str', r...)`**  
  A thermal machine: Hamiltonian, reference energies, baths, structure.
- **`models.absorption_refrigerator(omega_c: 'float' = 1.0, omega_h: 'float' = 3.0, T_c: 'float' = 1.0, T_h: 'float' = 4.0, T_r: 'float' = 1.5...)`**  
  Three-qubit absorption refrigerator.
- **`models.three_level_maser(omega_c: 'float' = 1.0, omega_h: 'float' = 3.0, T_c: 'float' = 1.0, T_h: 'float' = 10.0, drive: 'float' = ...)`**  
  Scovil--Schulz-DuBois three-level maser as a continuous heat engine.
- **`models.spin_chain(n: 'int' = 3, omega=1.0, J: 'float' = 0.2, delta: 'float' = 1.0, T_left: 'float' = 2.0, T_right: 'float' =...)`**  
  XXZ chain of ``n`` qubits between a left and a right thermal bath.
- **`models.two_qubit_heat_valve(omega_1: 'float' = 1.0, omega_2: 'float' = 0.6, g: 'float' = 0.6, coupling: 'str' = 'xx', T_hot: 'float' =...)`**  
  Two coupled qubits, hot bath on the first, cold bath on the second.
- **`models.thermal_transistor(T_L: 'float' = 1.0, T_M: 'float' = 0.2, T_R: 'float' = 0.2, omega: 'float' = 1.0, zz_left: 'float' = 1.0, ...)`**  
  Three qubits with Ising (zz) couplings as a quantum thermal transistor.
- **`build_model(local_H, interactions=None, baths=(), master_equation: 'str' = 'global', site_names=None, description: 'st...)`**  
  Build a :class:`Model` for your own multi-site machine.

### `qthermo.baths`

Thermal baths as objects: global (Davies) and local constructions.

- **`class Bath(name, c_ops=None, temperature=None, kind='custom', sites=(), *, eigenbasis=None, eig_ops=None, energies=No...)`**  
  A named set of collapse operators coupling the system to one reservoir.
- **`davies_bath(H, coupling, temperature: 'float', gamma: 'float' = 1.0, spectrum=None, name: 'str' = 'bath', sites=(), ze...)`**  
  Global (Davies) thermal bath for an arbitrary Hamiltonian.
- **`local_bath(H_site, coupling, temperature: 'float', site: 'int', dims, gamma: 'float' = 1.0, spectrum=None, name: 'str...)`**  
  Local thermal bath: Davies construction on one site, then embedded.
- **`bohr_decomposition(H, A, tol: 'float | None' = None) -> 'dict'`**  
  Split ``A`` into components ``A(w)`` that lower the energy by ``w``.
- **`flat_spectrum(gamma: 'float')`**  
  Frequency-independent emission rate: ``J(w) = gamma``.
- **`ohmic_spectrum(gamma: 'float', cutoff: 'float' = inf, reference: 'float' = 1.0)`**  
  Ohmic emission rate ``J(w) = gamma (w / reference) exp(-w / cutoff)``.
- **`dissipator(c_ops, rho) -> 'np.ndarray'`**  
  Lindblad dissipator ``sum_L  L rho L^dag - {L^dag L, rho}/2``.
- **`instantaneous_bath(H_of_t, coupling, temperature: 'float', gamma: 'float' = 1.0, spectrum=None, name: 'str' = 'bath', sites=(...)`**  
  A bath that follows a time-dependent Hamiltonian.

### `qthermo.steady`

Continuous (autonomous) thermal machines in their non-equilibrium steady state.

- **`liouvillian(H, c_ops=(), sparse: 'bool | None' = None)`**  
  Liouvillian superoperator in column-stacking convention.
- **`steady_state(H, baths=(), initial_state=None, check_unique: 'bool' = True, tol: 'float' = 1e-10) -> 'np.ndarray'`**  
  Exact steady state of the Lindblad equation.
- **`class SteadyState(H: 'np.ndarray', baths: 'list', rho: 'np.ndarray', currents: 'dict', energy: 'np.ndarray', labels: 'dict' ...)`**  
  Thermodynamics of a machine in its non-equilibrium steady state.
- **`analyze(H, baths, energy=None, initial_state=None) -> 'SteadyState'`**  
  Solve for the steady state and compute every bath's heat current.
- **`compare_master_equations(H, local_baths, global_baths) -> 'MasterEquationComparison'`**  
  Run one machine under local and global bath models and compare.
- **`class MasterEquationComparison(local: 'SteadyState', global_: 'SteadyState', trace_distance: 'float') -> None`**  
  The same machine under local and global baths.
- **`trace_distance(rho, sigma) -> 'float'`**  
  ``||rho - sigma||_1 / 2``.
- **`relaxation_time(H, baths, populations_only: 'bool' = False) -> 'float'`**  
  Slowest relaxation time ``1 / gap`` of the dynamics.

## Diagnostics

### `qthermo.audit`

One call that checks a thermal-machine model for the known traps.

- **`class Finding(severity: 'str', check: 'str', summary: 'str', detail: 'str' = '') -> None`**  
  Finding(severity: 'str', check: 'str', summary: 'str', detail: 'str' = '')
- **`class AuditReport(findings: 'list' = <factory>, steady: 'object' = None) -> None`**  
  AuditReport(findings: 'list' = <factory>, steady: 'object' = None)
- **`audit(source, baths=None, energy=None, fluctuations: 'bool' = True, compare: 'bool' = True) -> 'AuditReport'`**  
  Check a steady-state thermal machine for known modelling traps.

## Where the heat goes

### `qthermo.network`

Where the energy goes inside a multi-qubit machine.

- **`class HeatFlowMap(site_names: 'list', dims: 'list', local_energy: 'np.ndarray', virtual_temperature: 'np.ndarray', local_ent...)`**  
  Site-resolved energy flows and correlations of one state.
- **`heat_flow_map(source, model=None, *, rho=None, dims=None, local_H=None, baths=None, interaction_terms=None, site_names=N...)`**  
  Resolve the energy flows of a state site by site.
- **`virtual_temperature(rho_site, h_site) -> 'float'`**  
  Temperature a site *looks* like it has, from its two lowest levels.
- **`concurrence(rho_two_qubits) -> 'float'`**  
  Wootters concurrence of a two-qubit state (0 = separable, 1 = Bell).
- **`negativity(rho, site_a: 'int', site_b: 'int', dims) -> 'float'`**  
  Negativity of the reduced state of two sites, any local dimension.
- **`correlation_matrices(rho, dims) -> 'dict'`**  
  Pairwise mutual information, negativity and (qubits) concurrence.
- **`site_dynamics(cycle_result, dims, local_H) -> 'dict'`**  
  Per-site quantities along every stored state of a cycle.

### `qthermo.subsystems`

Spatially resolved thermodynamics for multipartite working media.

- **`partial_trace(rho, keep, dims) -> 'np.ndarray'`**  
  Trace out every subsystem except those in ``keep``.
- **`embed(operator, site: 'int', dims) -> 'np.ndarray'`**  
  Lift a single-site operator into the full multipartite Hilbert space.
- **`total_correlation(rho, dims) -> 'float'`**  
  Total correlation (multi-information) ``sum_i S_i - S_total``.
- **`mutual_information(rho, site_a: 'int', site_b: 'int', dims) -> 'float'`**  
  Mutual information ``S_a + S_b - S_ab`` between two specific sites.
- **`class SiteResult(site: 'int', heat: 'float', work: 'float', delta_U: 'float', delta_S: 'float', rho_initial: 'np.ndarray', ...)`**  
  Thermodynamic quantities attributed to one site over one stroke.
- **`class SubsystemResult(stroke: 'str', sites: 'list[SiteResult]', total_delta_U: 'float', total_delta_S: 'float', correlation_init...)`**  
  Per-site breakdown of one stroke, plus the non-additive remainders.
- **`resolve_stroke(stroke_result, dims, local_H) -> 'SubsystemResult'`**  
  Break one stroke's thermodynamics down site by site.
- **`resolve_cycle(cycle_result, dims, local_H) -> 'list[SubsystemResult]'`**  
  Apply :func:`resolve_stroke` to every stroke of a completed cycle.

## Fluctuations and statistics

### `qthermo.fluctuations`

Current fluctuations, full counting statistics, and uncertainty relations.

- **`class CurrentStatistics(mean: 'float', noise: 'float', activity: 'float', entropy_production_rate: 'float | None', label: 'str' = ...)`**  
  Mean, noise and uncertainty ratios of one counted current.
- **`jump_energy(L, energy, tol: 'float' = 1e-08) -> 'float'`**  
  Energy a jump ``L`` deposits in the system, measured with ``energy``.
- **`fluctuations.counting_weights(baths, count, energy) -> 'list[float]'`**  
  Per-jump weights for a counted current.
- **`current_statistics(source, count, baths=None, energy=None, label: 'str | None' = None) -> 'CurrentStatistics'`**  
  Exact mean and noise of a counted current in the steady state.
- **`scaled_cgf(source, count, s, baths=None, energy=None) -> 'np.ndarray'`**  
  Scaled cumulant generating function ``theta(s)`` of the counted current.

### `qthermo.counting`

Exact counting statistics of stroke machines.

- **`class CycleCounting(cycle: 'object', count: 'dict', rho_start: 'np.ndarray', substeps: 'int') -> None`**  
  Exact statistics of quanta exchanged with the counted strokes.
- **`cycle_counting(cycle, count, rho_start=None, substeps: 'int' = 400) -> 'CycleCounting'`**  
  Set up exact counting statistics for a stroke cycle.

### `qthermo.stochastic`

Trajectory-resolved thermodynamics.

- **`class TrajectoryEnsemble(heat: 'np.ndarray', work: 'np.ndarray', jump_counts: 'np.ndarray', jump_quanta: 'list', stroke_heat: 'dict...)`**  
  Per-trajectory heat and work, plus the jump record.
- **`unravel(cycle, rho0, trajectories: 'int' = 500, steps: 'int' = 400, seed: 'int | None' = None) -> 'TrajectoryEnsem...)`**  
  Run a ``Cycle`` as an ensemble of quantum-jump trajectories.
- **`jarzynski_tpm(H_of_t, duration: 'float', temperature: 'float', dim: 'int', steps: 'int' = 400, shots: 'int | None' = Non...)`**  
  Verify the Jarzynski equality under the two-point measurement protocol.

## Stroke machines

### `qthermo.cycle`

The ``Stroke`` / ``Cycle`` abstraction.

- **`class Stroke(name: 'str', H: 'object', duration: 'float', c_ops: 'list' = <factory>, temperature: 'float | None' = None...)`**  
  One leg of a thermodynamic cycle.
- **`class StrokeResult(name: 'str', heat: 'float', work: 'float', delta_U: 'float', entropy_production: 'float | None', rho_initi...)`**  
  StrokeResult(name: 'str', heat: 'float', work: 'float', delta_U: 'float', entropy_production: 'float | None', rho_initial: 'np.ndarray', rho_final: 'np.ndarray', times: 'np.ndarray', states: 'list', heat_by_bath: 'dict' = <factory>)
- **`class CycleResult(strokes: 'list[StrokeResult]') -> None`**  
  CycleResult(strokes: 'list[StrokeResult]')
- **`class Cycle(strokes: 'list[Stroke]')`**  
  A sequence of strokes applied repeatedly to one working medium.

### `qthermo.engines`

Stroke engines with interacting, multi-qubit working media.

- **`class OttoLimit(work: 'float', heat_hot: 'float', heat_cold: 'float', T_cold: 'float', T_hot: 'float', crossings: 'bool' =...)`**  
  Quasi-static Otto cycle between two Hamiltonians.
- **`ideal_otto(H_cold, H_hot, T_cold: 'float', T_hot: 'float', path=None, follow_crossings: 'bool' = True) -> 'OttoLimit'`**  
  Exact quasi-static Otto cycle between two Hamiltonians.
- **`otto_cycle(H_cold, H_hot, T_cold: 'float', T_hot: 'float', couplings, gamma: 'float' = 0.5, tau_iso: 'float' = 30.0, ...)`**  
  Four-stroke Otto cycle for an arbitrary (multi-qubit) working medium.
- **`adiabatic_pairing(H_start, H_end, path=None, points: 'int' = 400, tol: 'float' = 1e-09) -> 'np.ndarray'`**  
  Which eigenstate of ``H_end`` each eigenstate of ``H_start`` becomes.

### `qthermo.modes`

Operation modes of two-reservoir thermal machines.

- **`classify(work: 'float', heat_hot: 'float', heat_cold: 'float', tol: 'float' = 1e-12) -> 'str'`**  
  Operation mode from the signs of work and the two heats.
- **`modes.mode_code(mode: 'str') -> 'int'`**  
  
- **`mode_map(build, x_values, y_values, parameters=('x', 'y'), **kwargs)`**  
  Grid of operation modes: ``build(x, y)`` returns anything with a ``.mode()`` or ``.mode`` (an ``OttoLimit``, a ``Model``, a ``Cycle`` whose strokes are named ``hot_iso``/``cold_iso``).

## Driven, transient and linear response

### `qthermo.floquet`

Periodically driven machines: the Floquet-Markov master equation.

- **`class DrivenBath(name: 'str', coupling: 'np.ndarray', temperature: 'float', gamma: 'float' = 1.0, spectrum: 'object' = None...)`**  
  A thermal bath coupled to a periodically driven system.
- **`class FloquetSteadyState(quasienergies: 'np.ndarray', populations: 'np.ndarray', currents: 'dict', temperatures: 'dict', period: 'f...)`**  
  FloquetSteadyState(quasienergies: 'np.ndarray', populations: 'np.ndarray', currents: 'dict', temperatures: 'dict', period: 'float', transitions: 'list' = <factory>)
- **`floquet_states(H_of_t, period: 'float', n_time: 'int' = 256, substeps: 'int' = 8)`**  
  Quasienergies and Floquet modes over one period.
- **`floquet_analyze(H_of_t, period: 'float', baths, n_time: 'int' = 256, q_max: 'int | None' = None, substeps: 'int' = 8, dege...)`**  
  Steady-state thermodynamics of a periodically driven open system.
- **`window_spectrum(gamma: 'float', centre: 'float', width: 'float')`**  
  Box-shaped emission rate: ``gamma`` for ``|w - centre| < width/2``, else 0.

### `qthermo.transient`

Transient thermodynamics: switching a machine on and watching it settle.

- **`class TransientResult(times: 'np.ndarray', states: 'list', currents: 'dict', heat: 'dict', energy: 'np.ndarray', virtual_tempera...)`**  
  TransientResult(times: 'np.ndarray', states: 'list', currents: 'dict', heat: 'dict', energy: 'np.ndarray', virtual_temperatures: 'np.ndarray | None' = None, site_names: 'list | None' = None)
- **`transient(source, rho0, duration: 'float', steps: 'int' = 400, baths=None, energy=None, dims=None, local_H=None, sit...)`**  
  Integrate a continuous machine from ``rho0`` and record its thermodynamics.
- **`product_thermal_state(local_H, temperatures) -> 'np.ndarray'`**  
  ``(x)_i exp(-h_i/T_i)/Z_i``: every site thermal with its own bath, uncorrelated.

### `qthermo.response`

Linear response of multi-terminal machines: Onsager matrix, conductances, thermal-transistor gain.

- **`class ThermalResponse(names: 'list', temperatures: 'dict', conductance: 'np.ndarray', currents: 'dict') -> None`**  
  ThermalResponse(names: 'list', temperatures: 'dict', conductance: 'np.ndarray', currents: 'dict')
- **`response(build, temperatures: 'dict', h: 'float' = 0.0001) -> 'ThermalResponse'`**  
  Conductance matrix ``dJ_k/dT_l`` of a steady-state machine.

## Strong coupling

### `qthermo.strong_coupling`

Strong system-bath coupling via the reaction-coordinate mapping.

- **`reaction_coordinate_model(H_S, coupling, T: 'float', lam: 'float', Omega: 'float', kappa: 'float' = 0.05, n_levels: 'int' = 12, weak...)`**  
  Build the enlarged system+RC model for one strongly coupled bath.
- **`mean_force_state(H_S, coupling, T: 'float', lam: 'float', Omega: 'float', n_levels: 'int' = 30, counterterm: 'bool' = True)...)`**  
  Reduced equilibrium state of the system at finite coupling.
- **`ultrastrong_limit_state(H_S, coupling, T: 'float') -> 'np.ndarray'`**  
  Ultrastrong-coupling limit of the mean-force Gibbs state.
- **`rc_convergence(build, levels=(4, 6, 8, 10, 12, 16), observable=None) -> 'dict'`**  
  Check a reaction-coordinate result for convergence in the truncation.

## Information and protocols

### `qthermo.information`

Thermodynamics of information: erasure, reset, and their finite-time cost.

- **`class ErasureResult(tau: 'float', temperature: 'float', heat_to_bath: 'float', work: 'float', delta_S: 'float', landauer: 'flo...)`**  
  Outcome of one finite-time erasure.
- **`landauer_erasure(tau: 'float', temperature: 'float' = 1.0, omega_max: 'float | None' = None, omega_min: 'float' = 0.001, ga...)`**  
  Erase a qubit by ramping its gap up while it thermalises.
- **`landauer_bound(rho_initial, rho_final, temperature: 'float') -> 'float'`**  
  Minimum heat released to a bath at ``temperature`` by any process taking ``rho_initial`` to ``rho_final``: ``T (S_i - S_f)``.
- **`information.erasure_friction(omega: 'float', temperature: 'float', gamma: 'float' = 1.0) -> 'float'`**  
  Thermodynamic friction ``zeta(omega)`` of the erasure protocol.
- **`thermodynamic_length(temperature: 'float' = 1.0, omega_min: 'float' = 0.001, omega_max: 'float | None' = None, gamma: 'float' =...)`**  
  ``L = int sqrt(zeta) d omega`` along the erasure path.
- **`geodesic_schedule(temperature: 'float' = 1.0, omega_min: 'float' = 0.001, omega_max: 'float | None' = None, gamma: 'float' =...)`**  
  The minimum-dissipation ramp: constant speed in the friction metric.
- **`predicted_excess(schedule, tau: 'float', temperature: 'float' = 1.0, omega_min: 'float' = 0.001, omega_max: 'float | None' ...)`**  
  Slow-driving prediction ``(1/tau) int zeta(omega(s)) omega'(s)^2 ds``.
- **`class FeedbackResult(temperature: 'float', information: 'float', work_extracted: 'float', outcome_probabilities: 'np.ndarray', ...)`**  
  One cycle of a measurement-and-feedback engine.
- **`szilard_engine(temperature: 'float' = 1.0, gap: 'float' = 0.0, error: 'float' = 0.0, tau: 'float | None' = None, gamma: '...)`**  
  Quantum Szilard engine with a qubit memory: measure, then feed back.

### `qthermo.geometry`

Thermodynamic geometry: minimum-dissipation protocols for any driven system.

- **`friction(H_of, baths_of, temperature: 'float', lam: 'float', h: 'float' = 1e-05) -> 'float'`**  
  Thermodynamic friction ``g(lambda)`` for a single control parameter.
- **`class OptimalSchedule(path: 'object', s_grid: 'np.ndarray', metric: 'np.ndarray', arc: 'np.ndarray') -> None`**  
  Minimum-dissipation schedule along a path ``lambda(s)``, ``s`` in [0, 1].
- **`optimal_schedule(H_of, baths_of, temperature: 'float', path, points: 'int' = 201, h: 'float' = 1e-05) -> 'OptimalSchedule'`**  
  Constant-speed (geodesic) schedule along a path in control space.
- **`excess_work(H_of_t, baths_of, temperature: 'float', tau: 'float', rho0=None, steps: 'int' = 2000) -> 'dict'`**  
  Simulate a driven stroke with instantaneous thermalising baths.

## Quantum batteries

### `qthermo.batteries`

Quantum batteries: how much work is stored, where, and how fast it arrives.

- **`ergotropy_split(rho, H) -> 'dict'`**  
  Split ergotropy into incoherent and coherent parts.
- **`batteries.local_ergotropies(rho, dims, local_H) -> 'list'`**  
  Ergotropy of each cell's reduced state (local unitaries only).
- **`locked_ergotropy(rho, dims, local_H) -> 'dict'`**  
  Ergotropy available only to global operations on the battery.
- **`batteries.asymptotic_ergotropy(rho, H) -> 'float'`**  
  Per-copy ergotropy in the limit of many copies.
- **`batteries.multi_copy_ergotropy(rho, H, copies: 'int') -> 'float'`**  
  Ergotropy per copy of ``rho^{(x) copies}`` under ``H`` on each copy.
- **`class batteries.ChargingResult(times: 'np.ndarray', energy: 'np.ndarray', ergotropy: 'np.ndarray | None') -> None`**  
  ChargingResult(times: 'np.ndarray', energy: 'np.ndarray', ergotropy: 'np.ndarray | None')
- **`batteries.charge(H_total, H_battery, state, times, c_ops=None, battery=None) -> 'ChargingResult'`**  
  Charge a battery and record stored energy (and ergotropy) against time.
- **`batteries.dicke_battery(N: 'int', g: 'float' = 0.05, omega: 'float' = 1.0, rotating_wave: 'bool' = False, photons: 'int | None' = ...)`**  
  N two-level cells charged by one cavity mode (Dicke battery).
- **`batteries.collective_advantage(N_values, g: 'float' = 0.05, omega: 'float' = 1.0, rotating_wave: 'bool' = False, points: 'int' = 3000) ->...)`**  
  Charging power of the Dicke battery relative to N independent cells.

## Core quantities, channels, solver

### `qthermo.core`

Core thermodynamic quantities for open quantum systems.

- **`internal_energy(rho, H) -> 'float'`**  
  Internal energy U = Tr[H rho].
- **`passive_state(rho, H)`**  
  The passive state reachable from ``rho`` by unitary evolution.
- **`ergotropy(rho, H) -> 'float'`**  
  Maximum work extractable from ``rho`` by a cyclic unitary process.
- **`von_neumann_entropy(rho, tol: 'float' = 1e-12) -> 'float'`**  
  von Neumann entropy S = -Tr[rho ln rho], in nats (k_B = 1).
- **`thermal_state(H, temperature: 'float')`**  
  Gibbs state rho = exp(-H/T) / Z.
- **`free_energy(H, temperature: 'float') -> 'float'`**  
  Equilibrium free energy F = -T ln Z.
- **`heat_work_increments(rho_a, rho_b, H_a, H_b) -> 'tuple[float, float]'`**  
  Heat and work exchanged over one integration step.
- **`entropy_production(rho_initial, rho_final, heat: 'float', temperature: 'float', tolerance: 'float' = 1e-09) -> 'float'`**  
  Irreversible entropy production for a stroke in contact with one bath.

### `qthermo.channels`

Collapse (jump) operators for thermal baths and common noise channels.

- **`qubit_hamiltonian(omega: 'float') -> 'np.ndarray'`**  
  H = (omega / 2) sigma_z, so the gap between |0> and |1> is omega.
- **`thermal_bath(gamma: 'float', omega: 'float', temperature: 'float') -> 'list[np.ndarray]'`**  
  Emission and absorption operators for a qubit coupled to a thermal bath.
- **`amplitude_damping(gamma: 'float') -> 'list[np.ndarray]'`**  
  Zero-temperature relaxation only (T1-type).
- **`pure_dephasing(gamma: 'float') -> 'list[np.ndarray]'`**  
  Pure dephasing (T2-type): destroys coherence, leaves populations alone.
- **`bit_flip(gamma: 'float') -> 'list[np.ndarray]'`**  
  Symmetric bit-flip channel (infinite-temperature-like population mixing).
- **`mean_occupation(omega: 'float', temperature: 'float') -> 'float'`**  
  Bose-Einstein occupation of a mode of frequency omega at temperature T.

### `qthermo.solver`

Lindblad integrator with thermodynamic accounting built into the loop.

- **`lindbladian(rho: 'np.ndarray', H: 'np.ndarray', c_ops: 'list[np.ndarray]') -> 'np.ndarray'`**  
  Right-hand side of the Lindblad master equation.
- **`evolve(rho0, H, c_ops=None, duration: 'float' = 1.0, steps: 'int' = 200, rtol: 'float' = 1e-09, atol: 'float' = 1...)`**  
  Integrate the Lindblad equation and accumulate heat and work.
- **`solver.flatten_operators(items) -> 'list[np.ndarray]'`**  
  Expand a mix of operators and Bath objects into a flat operator list.

## Analysis and plotting

### `qthermo.analysis`

Parameter sweeps, joint scans, and channel comparison.

- **`class SweepResult(parameter: 'str', values: 'np.ndarray', metrics: 'np.ndarray', metric_name: 'str') -> None`**  
  Metric values over a one-dimensional parameter sweep.
- **`class ScanResult(parameters: 'tuple[str, str]', x_values: 'np.ndarray', y_values: 'np.ndarray', grid: 'np.ndarray', metric_...)`**  
  Metric surface over a two-dimensional joint scan.
- **`class analysis.ParetoFront(values: 'np.ndarray', power: 'np.ndarray', efficiency: 'np.ndarray', parameter: 'str') -> None`**  
  Cooling power against efficiency over a swept control parameter.
- **`sweep(build_cycle, values, metric, initial_state=None, parameter: 'str' = 'parameter', metric_name: 'str' = 'met...)`**  
  Run one cycle per parameter value and record a metric.
- **`scan_2d(build_cycle, x_values, y_values, metric, initial_state=None, parameters: 'tuple[str, str]' = ('x', 'y'), m...)`**  
  Joint scan over two parameters.
- **`compare_channels(build_cycle, channels: 'dict', metrics: 'dict', initial_state=None) -> 'dict'`**  
  Run the same cycle under different noise channels.
- **`analysis.pareto_front(build_cycle, values, cold_strokes, initial_state=None, parameter: 'str' = 'parameter') -> 'ParetoFront'`**  
  Trace cooling power against coefficient of performance.

### `qthermo.plotting`

Plotting.

- **`plotting.plot_cycle(result, cycle=None, ax=None, annotate: 'bool' = True)`**  
  Energy gap against excited-state population -- the quantum P-V diagram.
- **`plotting.plot_sweep(sweep_result, ax=None, mark_optimum: 'bool' = True)`**  
  Metric against a swept parameter, with the optimum marked.
- **`plotting.plot_scan(scan_result, ax=None, log_x: 'bool' = False, log_y: 'bool' = False)`**  
  Heatmap of a two-parameter scan, with joint and sequential optima marked.
- **`plotting.plot_distribution(ensemble, which: 'str', deterministic: 'float | None' = None, ax=None, bins: 'int' = 30, exact: 'dict | No...)`**  
  Trajectory histogram with the master-equation mean marked.
- **`plotting.plot_dashboard(result, cycle, sweep_result=None, ensemble=None, deterministic=None, figsize=(12, 4.2))`**  
  One figure summarising a machine: cycle diagram, sweep, distribution.
- **`plotting.plot_heat_network(flow_map, ax=None, cmap='coolwarm', show_values=True, title=None)`**  
  Draw a machine as a network: baths, sites, and the heat between them.
- **`plotting.plot_correlations(flow_map, ax=None, quantity='mutual_information')`**  
  Matrix of pairwise correlations between sites.
- **`plotting.plot_machine(steady, figsize=(11, 8))`**  
  One-figure summary of a continuous machine.
- **`plotting.plot_mode_map(scan_result, ax=None, log_x: 'bool' = False, log_y: 'bool' = False)`**  
  Categorical map of operation modes from :func:`qthermo.modes.mode_map`.
- **`plotting.plot_site_dynamics(dynamics, site_names=None, figsize=(10, 7.5), uniform_strokes: 'bool' = True)`**  
  Per-qubit energy, virtual temperature and pairwise correlations through a cycle. Takes the dict from :func:`qthermo.network.site_dynamics`.
- **`plotting.plot_bloch_paths(dynamics, site_names=None, plane=('x', 'z'), figsize=None)`**  
  Each qubit's reduced Bloch vector through the cycle, one disk per qubit.

## Output, units, validation

### `qthermo.export`

Plain-data export of results, for saving, sharing and plotting elsewhere.

- **`export(obj, include_states: 'bool' = False)`**  
  Convert any qthermo result (or container of results) to plain data.
- **`save(obj, path, include_states: 'bool' = False, note: 'str | None' = None) -> 'None'`**  
  Write ``export(obj)`` to a JSON file with version and timestamp.
- **`load(path) -> 'dict'`**  
  Read a file written by :func:`save` (as plain data).

### `qthermo.report`

A self-contained HTML report of a continuous machine, for sharing.

- **`report(model, path=None, title: 'str | None' = None, fluctuations: 'bool' = True) -> 'str'`**  
  Write (and return) a self-contained HTML report of a ``Model``.

### `qthermo.units`

Converting between laboratory units and qthermo's dimensionless units.

- **`class LabUnits(reference_GHz: 'float') -> None`**  
  Unit system with energy unit ``E0 = h * reference_GHz * 1e9``.

### `qthermo.validation`

Input validation.

- **`class QThermoError`**  
  Raised when an input violates a physical requirement.
- **`validation.check_density_matrix(rho, name: 'str' = 'state', tol: 'float' = 1e-08) -> 'np.ndarray'`**  
  
- **`validation.check_hermitian(operator, name: 'str' = 'Hamiltonian', tol: 'float' = 1e-10) -> 'np.ndarray'`**  
  
- **`validation.check_operator_shape(operators, dim: 'int', name: 'str' = 'collapse operator')`**  
  
- **`validation.check_temperature(temperature, name: 'str' = 'temperature') -> 'float'`**  
  
- **`validation.check_duration(duration, name: 'str' = 'duration') -> 'float'`**  
  

### `qthermo.benchmarks`

Benchmarks against results fixed independently of this implementation.

- **`class benchmarks.Benchmark(name: 'str', reference: 'str', function: 'Callable', tolerance: 'float', kind: 'str', relative: 'bool', sl...)`**  
  Benchmark(name: 'str', reference: 'str', function: 'Callable', tolerance: 'float', kind: 'str', relative: 'bool', slow: 'bool', group: 'str')
- **`class benchmarks.BenchmarkOutcome(benchmark: 'Benchmark', computed: 'float', expected: 'float', passed: 'bool', seconds: 'float', error: 'st...)`**  
  BenchmarkOutcome(benchmark: 'Benchmark', computed: 'float', expected: 'float', passed: 'bool', seconds: 'float', error: 'str | None' = None)
- **`benchmarks.benchmark(name: 'str', reference: 'str', tolerance: 'float' = 1e-06, kind: 'str' = 'equal', relative: 'bool' = False...)`**  
  Register a function returning ``(computed, expected)``.
- **`benchmarks.run_benchmarks(quick: 'bool' = False, group: 'str | None' = None, stream=<_io.TextIOWrapper name='<stdout>' mode='w' enco...)`**  
  Run every registered benchmark and print a table of outcomes.
