# qthermo

Thermodynamic analysis for open quantum systems.

Define a thermodynamic cycle as a sequence of strokes and get heat, work,
entropy production, COP and figure of merit — deterministically from the
master equation, or trajectory-resolved with full distributions.

**Status: early prototype.** Single-qubit working media, Markovian baths. The
physics below is validated against analytic limits; everything beyond that is
roadmap, not a claim.

## Why

Quantum thermodynamics has no shared implementation of its own core
quantities. QuTiP provides `entropy_vn` — the von Neumann entropy of a state at
an instant — but heat, work, entropy *production*, COP and figures of merit are
re-derived by hand in each new paper, along with the stroke-chaining
bookkeeping that connects them. `qthermo` packages that layer, in the same
spirit as Mitiq packaging error-mitigation techniques that groups were each
reimplementing.

The part that is more than packaging is the stochastic layer. The master
equation returns the *average* heat. A single run of a quantum thermal machine
either emits a quantum into the bath or it does not, and the distribution over
runs is a different object from its mean — which is where reliability lives,
and what fluctuation theorems constrain.

## Install

```bash
git clone <repo> && cd qthermo
pip install -e .
```

Requires numpy and scipy. QuTiP is optional: `Qobj` inputs are accepted
anywhere an array is, so existing QuTiP models work unchanged.

## Example

```python
import qthermo as qt
from qthermo.cycle import Cycle, Stroke

H_cold = qt.qubit_hamiltonian(1.0)
H_hot  = qt.qubit_hamiltonian(1.5)
ramp   = lambda a, b, tau: (lambda t: qt.qubit_hamiltonian(a + (t/tau)*(b-a)))

cycle = Cycle([
    Stroke("cold_iso", H_cold, 12.0, qt.thermal_bath(1.0, 1.0, 1.0), temperature=1.0),
    Stroke("compress", ramp(1.0, 1.5, 0.02), 0.02),
    Stroke("hot_iso",  H_hot,  12.0, qt.thermal_bath(1.0, 1.5, 1.3), temperature=1.3),
    Stroke("expand",   ramp(1.5, 1.0, 0.02), 0.02),
])

result, passes, converged = cycle.limit_cycle(
    qt.thermal_state(H_cold, 1.0)
)
print(result.report())
print(result.cop(("cold_iso",)))
```

```
stroke                    Q            W           dU        sigma
------------------------------------------------------------------
cold_iso          2.915e-02    0.000e+00    2.915e-02    2.214e-03
compress          0.000e+00   -1.155e-01   -1.155e-01           --
hot_iso          -4.373e-02    0.000e+00   -4.373e-02    2.271e-03
expand            0.000e+00    1.301e-01    1.301e-01           --
------------------------------------------------------------------
max |dU - (Q+W)| = 3.33e-16
```

### Trajectories

```python
ensemble = qt.unravel(cycle, rho_limit, trajectories=6000)
ensemble.probability_of("cold_iso", lambda q: q <= 0)   # -> 0.525
```

On average this refrigerator cools. On any individual run it draws no heat at
all out of the cold bath about half the time. That number is not recoverable
from the master equation, which reports only the mean.

## Validation

Every check below is a number thermodynamics fixes independently of this
implementation, so a failure means the physics is wrong rather than that an
interface changed.

| Check | Result |
|---|---|
| First law, `dU = Q + W`, per stroke | residual `3.3e-16` |
| Thermal fixed point (bath drives state to Gibbs) | `< 1e-6` |
| Entropy production non-negative (Spohn) | holds on all dissipative strokes |
| Otto COP → `ω_c/(ω_h−ω_c)` on full thermalisation | `2.0000` vs `2.0000` |
| COP below Carnot bound | `2.00 < 3.33` |
| Jarzynski equality, `⟨e^{−W/T}⟩ = e^{−ΔF/T}` | residual `4.4e-16` |
| `⟨W⟩ ≥ ΔF` (second law) | holds |
| Trajectory average reproduces master equation | agrees within `1.3σ` at N=6000 |
| Pure dephasing produces exactly zero heat | `0.00000` |
| Otto engine efficiency -> `1 - ω_low/ω_high` | `0.6000` vs `0.6000` |
| Invalid input raises `QThermoError` rather than returning a number | 6 cases |
| Sweep locates an interior optimum | passes |
| Inconsistent bath temperature warns instead of silently returning σ < 0 | passes |

Run them with `python tests/test_physics.py`.

The first-law residual is at machine precision by construction: heat and work
are accumulated with a midpoint split chosen so that `Q + W = dU` exactly at
every integration step, which makes the residual a test of the solver rather
than of the discretisation scheme.

## Engines and refrigerators are the same code

Nothing in `qthermo` knows what a refrigerator is. A cycle is a sequence of
strokes; whether it consumes work to move heat or absorbs heat to produce work
follows from the parameters, not from a different code path.

```
python examples/otto_refrigerator.py   # COP 2.0000, ideal Otto COP 2.0000
python examples/heat_engine.py         # efficiency 0.6000, ideal Otto 0.6000
```

## Failing loudly

Physical requirements are checked on input, with errors that name the violated
requirement rather than surfacing a linear-algebra traceback:

```python
>>> qt.evolve(2 * rho, H, [], 1.0)
QThermoError: initial state has trace 2.000000, not 1.
              Normalise it with rho / np.trace(rho).

>>> Stroke("s", H, 1.0, [], temperature=1.0)
QThermoError: stroke 's' has a temperature but no collapse operators.
              A stroke with no bath exchanges no heat, so its entropy
              production is not defined by a bath temperature.
```

Non-Hermitian Hamiltonians, negative durations, dimension mismatches between
state and collapse operators, non-positive states, and zero bath temperatures
are all rejected at the boundary.

## Figures

`python examples/full_demo.py` runs the whole workflow on one machine and
writes four figures to `examples/figures/`:

| figure | what it shows |
|---|---|
| `cycle_diagram.png` | energy gap against population -- the quantum P-V diagram. Isochores vertical, driven strokes horizontal, enclosed area is the work. |
| `sweep.png` | figure of merit against a swept parameter, optimum marked |
| `scan.png` | joint two-parameter scan, with joint and sequential optima |
| `distribution.png` | trajectory histogram against the master-equation mean |

The distribution figure is the one worth looking at: heat arrives in discrete
quanta, so the histogram has separated peaks at one emission, no exchange, and
one absorption. The mean value the master equation reports falls between them
-- it is a number no individual run ever produces.

Plotting needs matplotlib, which is an optional dependency; the numerical API
does not require it.

## Sweeps and joint scans

Finding where a machine performs best means running the same cycle many times
with one thing changed. That loop is rewritten by hand in essentially every
paper reporting an optimised quantum thermal machine.

```python
from qthermo.analysis import sweep, scan_2d

result = sweep(build_cycle, np.linspace(1.1, 2.6, 16),
               lambda r, c: r.figure_of_merit(("cold_iso",), c.duration))
result.optimum()            # (2.30, 3.945e-03)
result.improvement_over(1.5)  # +62.7 %
result.sensitivity()          # fraction of the range within 5% of optimum
```

`sensitivity` answers a question that matters for hardware: a knife-edge
optimum will not survive parameter drift on a real device, while a wide
plateau will. The sweep report states this in words.

`scan_2d` searches two parameters jointly and can compare that against the
sequential procedure everyone actually uses -- tune one, fix it, tune the next:

```python
scan.optimum()             # joint search over the full grid
scan.sequential_optimum()  # tune-one-then-the-other
scan.sequential_gap()      # +0.00 % for the single-qubit Otto refrigerator
```

For this system the gap is zero: sequential tuning finds the joint optimum.
That is a result rather than a disappointment. Sequential optimisation is
standard practice because it is cheap, and it is usually assumed to be safe
rather than checked -- the nested loop that checks it is tedious to write by
hand, and is one call here.

## Noise channels

`python examples/channel_comparison.py`

```
channel                            Q          dS   |coherence|   dP(excited)
----------------------------------------------------------------------------
amplitude damping           -0.51877    -0.04225       0.15727      -0.51877
pure dephasing               0.00000     0.27175       0.07066       0.00000
bit flip                    -0.11972     0.26296       0.17548      -0.11972
thermal bath (T=0.8)        -0.40343     0.17801       0.08272      -0.40343
```

Pure dephasing exchanges no energy: it destroys coherence while leaving
populations untouched, so it is structurally invisible to any heat-based
observable while still producing entropy. This is why `qthermo` reports
entropy production alongside heat rather than treating heat as the whole
thermodynamic story.

## Limitations

- Single-qubit working media. Multi-qubit systems are supported by the solver
  but untested and unvalidated.
- Markovian, weak-coupling baths only. No non-Markovian dynamics, no HEOM.
- Dense matrices, so this will not scale past a handful of qubits.
- The Jarzynski check uses the two-point measurement protocol on a closed
  driven system, which is exact, rather than an open-system fluctuation
  theorem.
- Simulation only. Nothing here has been run on hardware.

## Roadmap

1. Multi-qubit working media, with per-qubit resolution of heat and entropy
   production rather than system totals.
2. Non-Markovian baths.
3. Validation against published results in the thermodynamics-of-error-
   correction literature.

## Units

`ħ = k_B = 1`. Energies and temperatures share units; `β = 1/T`.

## References

The quantities implemented here are standard; the package does not introduce
new physics.

- Alicki, *J. Phys. A* **12**, L103 (1979) — the heat/work split for driven
  open systems.
- Spohn, *J. Math. Phys.* **19**, 1227 (1978) — non-negativity of entropy
  production.
- Jarzynski, *Phys. Rev. Lett.* **78**, 2690 (1997); Tasaki, arXiv:cond-mat/0009244
  (2000) — the equality and its quantum two-point-measurement form.
- Kosloff & Rezek, *Entropy* **19**, 136 (2017) — the quantum Otto cycle.

## License

MIT
