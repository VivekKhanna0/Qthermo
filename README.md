# qthermo

Thermodynamic bookkeeping for open quantum systems — heat, work, entropy
production, currents and their fluctuations — with the modelling traps checked
for you.

You write down a Hamiltonian and the baths. `qthermo` gives you per-bath heat
currents, entropy production, efficiencies against their Carnot bounds,
site-resolved energy flows, exact current fluctuations and uncertainty-relation
ratios, for stroke machines and continuous machines, from one qubit to a few
hundred levels. It also tells you when the model you wrote down cannot mean
what you think it means.

```bash
git clone https://github.com/VivekKhanna0/Qthermo.git && cd Qthermo
pip install -e ".[plot]"          # numpy, scipy; matplotlib for figures
python -m qthermo.benchmarks      # 26 checks against published results, ~10 s
```

**New here?** [`examples/tutorial.ipynb`](examples/tutorial.ipynb) takes one
research question (a two-qubit thermal diode) from a Hamiltonian to publishable
numbers, including the checks a referee would ask about. It renders on GitHub
with all outputs.

## Thirty seconds

```python
import qthermo as qt

fridge = qt.models.absorption_refrigerator(T_c=1.0, T_h=6.0, T_r=1.5)
steady = fridge.analyze()
print(steady.report())
```
```
bath                  T   J (into system)          -J/T
-------------------------------------------------------
cold                  1      6.963227e-04   -6.9632e-04
hot                   6      2.088968e-03   -3.4816e-04
room                1.5     -2.785291e-03    1.8569e-03
-------------------------------------------------------
sum                             1.301e-18    8.1238e-04
entropy production rate: 8.1238e-04
note: local baths -- valid only for inter-site coupling << bath rates; compare with global baths
```
```python
steady.cop("cold", "hot")                                  # 0.3333 = w_c / w_h exactly
steady.absorption_carnot_cop("cold", "hot", "room")        # 1.5
print(fridge.compare().report())       # would "you used a local master equation" change anything?
```
```
bath                 J local      J global   rel. diff
------------------------------------------------------
cold             6.96323e-04   7.53796e-04       7.62%
hot              2.08897e-03   2.27304e-03       8.10%
room            -2.78529e-03  -3.02684e-03       7.98%
------------------------------------------------------
entropy production rate: local 8.1238e-04, global 8.8526e-04
steady-state trace distance: 1.040e-02
```
```python
qt.plot_machine(steady)      # the figure below
```

<img src="examples/figures/absorption_fridge.png" width="620" alt="heat-flow network of the absorption refrigerator">

Baths (squares) and qubits (circles) share one temperature colour scale; each
qubit is coloured by its **virtual temperature**. The cold qubit is colder
than every bath, which nothing passive can do. Arrow widths are heat
currents, and the internal balances close to ~1e-17.

## Your own model

Nothing above is special-cased. A machine is a Hamiltonian plus named baths:

```python
dims = [2, 2]
h0, h1 = qt.qubit_hamiltonian(1.0), qt.qubit_hamiltonian(0.6)
V = 0.3 * qt.embed(qt.sigma_x, 0, dims) @ qt.embed(qt.sigma_x, 1, dims)
H = qt.embed(h0, 0, dims) + qt.embed(h1, 1, dims) + V

hot  = qt.davies_bath(H, qt.embed(qt.sigma_x, 0, dims), temperature=2.0, gamma=0.1, name="hot")
cold = qt.davies_bath(H, qt.embed(qt.sigma_x, 1, dims), temperature=1.0, gamma=0.1, name="cold")

result = qt.analyze(H, [hot, cold])                           # exact steady state, per-bath currents
stats  = qt.current_statistics(H, "hot", baths=[hot, cold])   # exact noise, TUR / KUR ratios
flows  = qt.heat_flow_map(result, dims=dims, local_H=[h0, h1],
                          interaction_terms={(0, 1): V})   # who heats whom, virtual temperatures
```

`davies_bath` builds the *global* (secular, detailed-balance) master equation
for any Hamiltonian: coupled, degenerate, many-body. `local_bath` builds the
local one. Both return the same `Bath` object, so every tool accepts either.
QuTiP `Qobj`s are accepted anywhere an array is.

## I want to...

| ...do this | call |
|---|---|
| get currents, entropy production, COP/efficiency of a continuous machine | `qt.analyze(H, baths)` or `model.analyze()` |
| check my model for known modelling mistakes | `qt.audit(model)` or `qt.audit(H, baths)` |
| build baths for a coupled / many-body system | `qt.davies_bath` (global), `qt.local_bath` (local) |
| know whether local vs global master equations matter | `model.compare()` |
| see where heat flows inside a multi-qubit machine | `qt.heat_flow_map(steady)`, `qt.plot_machine(steady)` |
| follow each qubit (energy, T*, entanglement) through a cycle | `qt.site_dynamics(result, dims, local_H)`, `qt.plot_site_dynamics` |
| get the noise of a current and the TUR/KUR ratios | `qt.current_statistics(model, "bath")` |
| simulate a stroke cycle and its limit cycle | `qt.Cycle([qt.Stroke(...), ...]).limit_cycle(rho0)` |
| know the quasi-static limit of an interacting Otto engine | `qt.ideal_otto(H_c, H_h, T_c, T_h)` |
| get the exact heat distribution of one cycle | `qt.cycle_counting(cycle, {"stroke": "quanta"})` |
| map engine / fridge / heater / accelerator regions | `qt.mode_map(build, xs, ys)`, `qt.plot_mode_map` |
| scan any machine over parameters | `qt.sweep`, `qt.scan_2d`, `qt.pareto_front` |
| go beyond weak coupling | `qt.reaction_coordinate_model`, `qt.rc_convergence` |
| cost of erasure and the optimal protocol | `qt.landauer_erasure`, `qt.geodesic_schedule` |
| the least-dissipative schedule for *my* drive | `qt.optimal_schedule(H_of, baths_of, T, path)` |
| a machine powered by a periodic drive | `qt.floquet_analyze(H_of_t, period, [qt.DrivenBath(...)])` |
| analyse a quantum battery | `qt.batteries.ergotropy_split`, `locked_ergotropy`, `dicke_battery` |
| use lab numbers (GHz, mK, µs, W) | `lab = qt.LabUnits(5.0)`; `lab.temperature(20)`, `lab.rate(1/T1)`, `lab.to_watts(J)` |
| save results with provenance | `qt.save(result, "file.json")` |
| confirm the package is right | `python -m qthermo.benchmarks` |

## What it catches

Most of the value is here. Each of these is a mistake that produces a plausible
number with no error message, and each is detected and explained. `qt.audit`
runs every check in one call:

```python
print(qt.audit(qt.models.two_qubit_heat_valve(master_equation="local")))
```
```
[ERROR] local vs global: the two master equations disagree on the direction of heat flow for hot, cold
          Inter-site coupling is too strong for the local approximation. Use the
          global model.
[info]  relaxation time: 2.72 (populations)
[ok]    steady state: unique
[ok]    second law: entropy production rate 1.937e-02 >= 0
[ok]    internal currents: site energy balances close to 3.5e-17
[ok]    uncertainty relation: 'hot': TUR ratio 9.626 >= 2
[ok]    uncertainty relation: 'cold': TUR ratio 5.117 >= 2
------------------------------------------------------------
1 error(s), 0 warning(s), 6 other checks
```

| Trap | What happens | What `qthermo` does |
|---|---|---|
| Local master equation on coupled sites | Heat can flow cold → hot; σ < 0 (Levy & Kosloff 2014) | Warns on σ < 0; `model.compare()` runs local vs global side by side; bare-Hamiltonian accounting with explicit boundary work (De Chiara et al. 2018) |
| Non-unique steady state | Identical qubits on one bath, dark states, conserved quantities: "the" steady state doesn't exist | Detected from the conditioning of the linear solve; raises unless you give an initial state, then returns the state it actually relaxes to |
| Baths that respect a symmetry of H | e.g. σx couplings on an Ising medium conserve parity; isochores never thermalise and the "efficiency" describes a different machine | `otto_cycle` refuses and says which symmetry-breaking coupling to use |
| Level crossings in an interacting Otto engine | Energy-rank pairing of levels gives the wrong quasi-static cycle | `ideal_otto` follows adiabatic continuation along the actual drive |
| Degenerate levels + flat spectral density | The zero-frequency channel has an undefined rate; thermalisation silently takes 1000× longer | Warns; `relaxation_time()` exposes the time scale; `otto_cycle` checks τ_iso against it |
| Global ME and local currents | The secular steady state is diagonal in H, so every bond current i⟨[V,h_i]⟩ is identically zero | `heat_flow_map` says so instead of printing zeros |
| Reaction-coordinate truncation | Strong coupling displaces the RC; too few levels gives unconverged currents | `rc_convergence()` reports the change between truncations |
| Temperature inconsistent with the channel | σ < 0 from a mislabelled bath | Warning naming the likely cause |
| Bath temperature inconsistent with its rates | A bath labelled T=2 whose jump rates encode T=3 gives a meaningless σ | `audit` infers the detailed-balance temperature from the jump operators and compares |
| Invalid input | Non-Hermitian H, unnormalised ρ, wrong dimensions | `QThermoError` naming the violated requirement, never a LinAlg traceback |

## What it does

| Area | Entry points | Validated against |
|---|---|---|
| **Stroke machines** (Otto, arbitrary cycles) | `Stroke`, `Cycle`, `limit_cycle` | Otto COP/efficiency limits; first law to 1e-16 by construction |
| **Interacting working media** | `ideal_otto`, `otto_cycle`, `resolve_stroke` | Exact quasi-static limit (simulation agrees to 1e-8); coupling-enhanced efficiency (Thomas & Johal 2011) |
| **Continuous machines** | `analyze`, `steady_state`, `models.*` | LPS fridge COP = ω_c/ω_h exactly; SSDB maser η = 1 − ω_c/ω_h; Carnot bounds |
| **Bath models** | `davies_bath`, `local_bath`, `instantaneous_bath` | Gibbs fixed point for random H to 1e-16; Levy–Kosloff violation reproduced and resolved |
| **Where the heat goes** | `heat_flow_map`, `plot_machine`, virtual temperatures, concurrence, negativity | Site balances close to 1e-17; Werner-state entanglement threshold |
| **Fluctuations** | `current_statistics`, `scaled_cgf` | Exact vs tilted-generator vs independent classical FCS (1e-10); Gallavotti–Cohen symmetry to 1e-15; TUR holds for all classical machines, violated by the maser as published |
| **Strong coupling** | `reaction_coordinate_model`, `mean_force_state` | O(λ²) weak-coupling limit; Cresser–Anders ultrastrong limit; heat-current turnover |
| **Information** | `landauer_erasure`, `geodesic_schedule`, `thermodynamic_length` | → T ln 2; excess ∝ 1/τ matching slow-driving theory to <1%; geodesic attains L²/τ |
| **Periodically driven machines** | `floquet_analyze`, `DrivenBath`, `window_spectrum` | Undriven limit = static Davies (1e-14); Bessel sideband weights; tight-coupling efficiency and COP of the modulated-qubit machine to 1e-7; σ ≥ 0 for random drives |
| **Optimal protocols** (any H(λ), any thermalising bath) | `friction`, `optimal_schedule`, `excess_work` | Friction metric reproduces the erasure closed form to 1e-11; non-commuting drive: simulated excess within 0.2% of L²/τ, 34% below a linear ramp |
| **Quantum batteries** | `batteries.ergotropy_split`, `locked_ergotropy`, `asymptotic_ergotropy`, `dicke_battery`, `collective_advantage` | Dicke √N power advantage (exponent 0.499); locked ergotropy of a Bell pair; activation of passive states |
| **Per-cycle statistics** | `cycle_counting` (exact), `unravel` (sampled) | Exact P(n) vs independent classical telegraph model (1e-12); vs trajectories; Jarzynski to 1e-16 |
| **Operation modes** | `classify`, `mode_map`, `plot_mode_map`, `.mode()` on every result | Quasi-static qubit Otto boundary ω_c/ω_h = T_c/T_h reproduced exactly |
| **Optimisation** | `sweep`, `scan_2d`, `pareto_front` (cycles *and* continuous models) | Interior optimum located; sequential vs joint tuning |

## Figures

Each comes from one script in `examples/`, and each script prints the numbers
behind its figure.

<table>
<tr>
<td width="50%"><img src="examples/figures/local_vs_global.png" alt="local vs global master equation"></td>
<td width="50%"><img src="examples/figures/tur.png" alt="TUR violation in the maser"></td>
</tr>
<tr>
<td><b>When the local master equation breaks</b> (<code>local_vs_global.py</code>).
Past g ≈ 0.55 the local model moves heat from cold to hot. The global model
never violates the second law, and counting boundary work repairs the local
bookkeeping but not its currents.</td>
<td><b>Beyond any classical machine</b> (<code>uncertainty_relation.py</code>).
The maser's power fluctuates less than the TUR allows any classical Markov
process to at the same dissipation. The result is computed exactly, with no
sampling.</td>
</tr>
<tr>
<td width="50%"><img src="examples/figures/strong_coupling.png" alt="strong coupling turnover"></td>
<td width="50%"><img src="examples/figures/landauer.png" alt="finite-time Landauer erasure"></td>
</tr>
<tr>
<td><b>What weak coupling misses</b> (<code>strong_coupling.py</code>).
The heat current peaks and falls, where weak-coupling theory predicts λ²
growth forever. The equilibrium state moves from Gibbs to the ultrastrong limit.</td>
<td><b>The price of forgetting</b> (<code>landauer.py</code>). The excess
dissipation of erasing a bit depends on the protocol. The geodesic ramp
reaches the thermodynamic-length bound L²/τ, 65% below a linear ramp.</td>
</tr>
<tr>
<td width="50%"><img src="examples/figures/coupled_otto.png" alt="coupled Otto engine"></td>
<td width="50%"><img src="examples/figures/spin_chain.png" alt="spin chain heat transport"></td>
</tr>
<tr>
<td><b>Interacting working medium</b> (<code>coupled_otto.py</code>). Heisenberg
coupling lifts a two-qubit Otto engine above 1 − B_c/B_h. The finite-time
simulation (dots) matches the exact quasi-static limit (line).</td>
<td><b>Transport through a chain</b> (<code>absorption_refrigerator.py</code>).
The same current crosses every bond, and virtual temperatures fall
monotonically from the hot end to the cold end.</td>
</tr>
<tr>
<td width="50%"><img src="examples/figures/operation_modes.png" alt="operation mode maps"></td>
<td width="50%"><img src="examples/figures/distribution.png" alt="per-cycle heat distribution"></td>
</tr>
<tr>
<td><b>Operation modes</b> (<code>operation_modes.py</code>). The quasi-static
qubit Otto cycle splits exactly at ω_c/ω_h = T_c/T_h. Fast, non-commuting
ramps (quantum friction) open up accelerator and heater regions.</td>
<td><b>One cycle, not the average</b> (<code>full_demo.py</code>). Sampled
quantum-jump trajectories against the exact counting-statistics distribution
(diamonds). The master-equation mean is a value no single cycle produces.</td>
</tr>
<tr>
<td colspan="2"><img src="examples/figures/multiqubit_cycle.png" alt="two-qubit engine cycle, site by site"></td>
</tr>
<tr>
<td colspan="2"><b>Inside a two-qubit engine</b> (<code>multiqubit_cycle.py</code>).
A transverse-field Ising pair run as an Otto engine, site by site and stroke by
stroke. The cold bath drives the pair into an <i>entangled</i> Gibbs state
(concurrence 0.45), the fast compression ramp partly unwinds it, and the hot
bath destroys it. Fast ramps cost efficiency: 0.515 against 0.648 quasi-static.</td>
</tr>
<tr>
<td colspan="2"><img src="examples/figures/optimal_protocol.png" alt="optimal driving protocol"></td>
</tr>
<tr>
<td colspan="2"><b>Optimal protocols for any drive</b> (<code>optimal_protocol.py</code>).
The slow-driving friction metric for a qubit whose field grows and tilts (a
non-commuting drive). The constant-speed schedule dissipates 34% less than a
linear ramp, and full finite-time simulations (dots) land on the predictions
(dashed).</td>
</tr>
<tr>
<td colspan="2"><img src="examples/figures/transport_scaling.png" alt="heat transport scaling in spin chains"></td>
</tr>
<tr>
<td colspan="2"><b>Ballistic or not</b> (<code>transport_scaling.py</code>).
Boundary-driven XXZ chains up to 7 spins (128 levels, seconds each). The XX
chain's current is independent of length to 12 digits, and its interior
temperature profile is flat (ballistic). The zz term makes the current fall
roughly as 1/N and a gradient build up.</td>
</tr>
<tr>
<td colspan="2"><img src="examples/figures/driven_machine.png" alt="Floquet heat machine"></td>
</tr>
<tr>
<td colspan="2"><b>A periodically driven machine</b> (<code>driven_machine.py</code>). A
frequency-modulated qubit with spectrally filtered baths, solved with the
Floquet–Markov master equation. It runs as an engine with η = 1 − (ω₀−Ω)/(ω₀+Ω)
and switches to a refrigerator with COP (ω₀−Ω)/2Ω exactly where predicted.
Power follows the first Bessel sideband until a higher one opens a
short-circuit channel.</td>
</tr>
<tr>
<td colspan="2"><img src="examples/figures/quantum_battery.png" alt="Dicke quantum battery"></td>
</tr>
<tr>
<td colspan="2"><b>Quantum batteries</b> (<code>quantum_battery.py</code>). N cells charged
through one cavity charge faster per cell than N separate ones, with a power
advantage growing as √N (Ferraro et al. 2018; fitted exponent 0.499). The same
script splits stored work into population and coherence parts, shows the
coherent part lost to dephasing, and shows work locked in correlations or
unlocked by many-copy operations.</td>
</tr>
</table>

## Verification

```bash
python -m qthermo.benchmarks     # computed value next to the published / analytic one
python -m pytest                 # 120+ tests
```

Every benchmark is a number fixed independently of this code: a closed-form
limit, an exact identity, or a published bound or violation. A failure means
the physics is wrong, not that an interface changed. CI runs both on every
push.

```
[continuous machines]
ok  3-qubit absorption fridge COP (local ME, tight coupling)         0.333333     0.333333  ==
      w_c/w_h; Linden, Popescu & Skrzypczyk, PRL 105, 130401 (2010)
ok  three-level maser efficiency                                     0.666667     0.666667  ==
      1 - w_c/w_h; Scovil & Schulz-DuBois, PRL 2, 262 (1959)
ok  local ME, detuned XX qubits: heat current from hot bath       -0.00044776            0  <=
      < 0, i.e. cold -> hot; Levy & Kosloff, EPL 107, 20004 (2014)
[fluctuations]
ok  TUR ratio, three-level maser power (coherent drive)               1.96697            2  <=
      < 2 possible; Kalaee, Wacker & Potts, PRE 104, L012103 (2021)
ok  Gallavotti-Cohen symmetry of heat FCS, 3-qubit chain          1.84228e-15            0  ==
[information]
ok  geodesic erasure protocol: excess heat x tau                     0.890737     0.886534  ==
      = L^2 (thermodynamic length); Scandi & Perarnau-Llobet, Quantum 3, 197 (2019)
...
26/26 passed in 10.0 s
```

The exact definitions of every computed quantity (sign conventions, which
energy operator defines heat under local baths, rate conventions of the
spectral densities, the FCS formulas) are in [docs/physics.md](docs/physics.md).

## Stroke machines and trajectories

The original core is still here and unchanged: a cycle is a list of strokes,
and whether it is a fridge or an engine follows from the parameters.

```python
H_cold, H_hot = qt.qubit_hamiltonian(1.0), qt.qubit_hamiltonian(1.5)
ramp = lambda a, b, tau: (lambda t: qt.qubit_hamiltonian(a + (t/tau)*(b-a)))

cycle = qt.Cycle([
    qt.Stroke("cold_iso", H_cold, 12.0, qt.thermal_bath(1.0, 1.0, 1.0), temperature=1.0),
    qt.Stroke("compress", ramp(1.0, 1.5, 0.02), 0.02),
    qt.Stroke("hot_iso",  H_hot,  12.0, qt.thermal_bath(1.0, 1.5, 1.3), temperature=1.3),
    qt.Stroke("expand",   ramp(1.5, 1.0, 0.02), 0.02),
])
result, passes, converged = cycle.limit_cycle(qt.thermal_state(H_cold, 1.0))
result.cop(("cold_iso",))

counts = qt.cycle_counting(cycle, {"cold_iso": "quanta"})   # exact, no sampling
counts.probability(lambda n: n <= 0)     # 0.7955: 80% of cycles extract nothing
print(counts.report())

ensemble = qt.unravel(cycle, result.strokes[-1].rho_final, trajectories=6000)
ensemble.probability_of("cold_iso", lambda q: q <= 0)   # ~0.79, sampled
```
```
net quanta exchanged per cycle, counting cold_iso (quanta)
    n   P(n), one cycle
   -1          0.175299  #######
    0          0.620249  #########################
    1          0.204453  ########
one cycle:  mean +0.029154, P(n <= 0) = 0.795547
long run:   mean +0.029154 per cycle, variance 0.378901, Fano 12.9965
```

On average this refrigerator cools. In 80% of individual cycles it draws
nothing from the cold bath, a number the master equation cannot give.
`cycle_counting` computes the single-cycle distribution *exactly*, by
propagating the counting-field master equation through the strokes, and the
long-run noise from its dominant eigenvalue. `unravel` samples the same thing
with quantum-jump trajectories. (Earlier versions of this README said "about
half": that was floating-point round-off in the sampled heats misclassifying
zero-heat trajectories, found by comparing against the exact result, and now
fixed.)

Heat and work use the midpoint Alicki split, which satisfies `dU = Q + W` to
machine precision at every step. That makes the first-law residual a test of
the solver rather than of the discretisation. Strokes also accept `Bath`
objects, including several at once and baths that follow a driven Hamiltonian
(`instantaneous_bath`), and then report heat per bath. See
`examples/full_demo.py` for sweeps, joint scans, Pareto fronts and trajectory
distributions.

## Scope and limitations

- **Markovian** master equations (Lindblad form). Non-Markovian and
  strong-coupling effects enter through the reaction-coordinate mapping, which
  is exact for a single-peaked (Brownian) spectral density and approximate
  otherwise. No HEOM, no Redfield (non-GKLS) equations.
- **Lamb shifts are neglected** in the Davies construction.
- **Size:** dense or sparse matrices, no tensor networks. Global-bath steady
  states run to a few hundred levels (the 200-level RC model takes ~20 s);
  current fluctuations use dense superoperators and suit a few dozen levels.
- **Periodically driven machines** use the full-secular Floquet–Markov
  equation. It is valid when quasienergy differences are resolved on the scale
  of the bath rates, and warns when they are not.
- Simulation only. Nothing here has been checked against hardware data.

## Feedback wanted

This is being shared with researchers in quantum thermodynamics to find out
what is actually useful. The most valuable replies are specific: *"I would
use this if it did X"*, *"this number disagrees with Ref. Y"*, *"the local/global
check would have saved me a week"*. Please open an issue.

## Units and conventions

`ħ = k_B = 1`, with an energy unit of your choice; `qt.LabUnits(f0_GHz)`
converts to and from GHz, mK, seconds and watts for the unit `E0 = h f0`.
`Q > 0` and `J > 0` mean energy flowing **into** the system;
`W > 0` means work done **on** it. `qubit_hamiltonian(ω) = −(ω/2)σ_z`, so
|1⟩ is the excited state.

## References

Implemented quantities are standard; the package does not introduce new
physics. Each benchmark cites its source (run `python -m qthermo.benchmarks`).
Main ones: Alicki (1979) heat/work split; Spohn (1978) entropy production;
Davies (1974) weak-coupling generator; Levy & Kosloff, EPL 107, 20004 (2014);
De Chiara et al., NJP 20, 113024 (2018); Linden, Popescu & Skrzypczyk, PRL 105,
130401 (2010); Brunner et al., PRE 85, 051117 (2012); Scovil & Schulz-DuBois,
PRL 2, 262 (1959); Barato & Seifert, PRL 114, 158101 (2015); Kalaee, Wacker &
Potts, PRE 104, L012103 (2021); Landi et al., PRX Quantum 5, 020201 (2024);
Strasberg et al., NJP 18, 073007 (2016); Cresser & Anders, PRL 127, 250601
(2021); Scandi & Perarnau-Llobet, Quantum 3, 197 (2019); Thomas & Johal, PRE
83, 031135 (2011); Jarzynski, PRL 78, 2690 (1997).

## License

MIT
