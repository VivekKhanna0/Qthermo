# qthermo

**Heat, work and efficiency for quantum machines, with the physics checked for you.**

Tiny machines built from a few qubits can act as engines and refrigerators.
Working out how much heat they move, and whether the model you used is even
allowed by the laws of thermodynamics, usually means hand-writing hundreds of
lines of simulation code for every paper. qthermo does that part. You describe
the machine; it gives you the heat flows, the efficiency, and a warning when
the model breaks physics.

[![tests](https://github.com/VivekKhanna0/Qthermo/actions/workflows/tests.yml/badge.svg)](https://github.com/VivekKhanna0/Qthermo/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qthermo)](https://pypi.org/project/qthermo/)

```bash
pip install "qthermo[plot]"        # Python 3.10 or newer
```

## Try it: reproduce a published result

Pick a paper and run it. Each one rebuilds the paper's setup from scratch and
prints qthermo's numbers next to the paper's.

```bash
python -m qthermo                  # list the papers
python -m qthermo fridge           # run one
python -m qthermo all              # run all five (~15 s)
```

| name | paper | what it shows |
|---|---|---|
| `fridge` | Linden, Popescu & Skrzypczyk, PRL 105, 130401 (2010) | The smallest possible refrigerator: three qubits, cooled by heat alone. Its efficiency is exactly ω_c/ω_h. |
| `maser` | Scovil & Schulz-DuBois, PRL 2, 262 (1959) | The first quantum heat engine. Its efficiency is exactly 1 − ω_c/ω_h. |
| `szilard` | Sagawa & Ueda, PRL 100, 080403 (2008) | Work from information: a measurement worth *I* bits of information yields at most *T·I* of work, even when the measurement makes errors. |
| `cold-to-hot` | Levy & Kosloff, EPL 107, 20004 (2014) | A standard shortcut (the "local master equation") can make heat flow from cold to hot, breaking the second law. qthermo catches it. |
| `local-vs-global` | Hofer et al., NJP 19, 123037 (2017) | Which of the standard models gives the right heat current? Each one is graded against the exact answer. |

What "reproduced" means here: the first four papers state their result as a
formula, and qthermo matches it to machine precision. For the last one,
qthermo computes the exact answer independently and reproduces the paper's
finding (which approximation fails, and where), not its plotted figure.

The same thing from Python, with a figure:

```python
import qthermo as qt

r = qt.papers.local_vs_global(detuning=0.1)
print(r.report())
r.plot()
```

<img src="https://raw.githubusercontent.com/VivekKhanna0/Qthermo/main/examples/figures/local_vs_global_exact.png" width="720" alt="local, global and Redfield heat currents against the exact result">

Each of the two textbook approximations fails somewhere: the *global* one for
resonant oscillators at weak coupling (top, off by up to 25×), the *local* one
for detuned oscillators (bottom, 10% off). The *Redfield* model is within 0.1%
of exact in both.

## Build your own machine

Describe the parts, how they are coupled, and which heat bath touches which
part:

```python
machine = qt.build_model(
    local_H=[qt.qubit_hamiltonian(1.0), qt.qubit_hamiltonian(0.6)],   # two qubits
    interactions={(0, 1): (0.3 * qt.sigma_x, qt.sigma_x)},            # coupled
    baths=[dict(name="hot", site=0, coupling=qt.sigma_x, T=2.0, gamma=0.1),
           dict(name="cold", site=1, coupling=qt.sigma_x, T=1.0, gamma=0.1)])

steady = machine.analyze()        # heat flow from every bath, efficiency, entropy
print(steady.report())
print(qt.audit(machine))          # does the model break any law of thermodynamics?
qt.plot_machine(steady)           # where the heat goes
```

Then, depending on what you need:

| to... | use |
|---|---|
| compare the standard models on *your* machine | `machine.compare()`, `machine.rebuild("redfield")` |
| scan a parameter and find the best setting | `qt.sweep`, `qt.scan_2d` |
| send someone the result and the checks | `qt.report(machine, "machine.html")` |
| make an interactive page with a slider | `qt.explorer(build, values, path="page.html")` |
| engines that run in cycles, like a car engine | `qt.Cycle`, `qt.otto_cycle` |
| how noisy the heat flow is | `qt.current_statistics(machine, "hot")` |
| the cost of erasing a bit, and the cheapest way to do it | `qt.landauer_erasure`, `qt.geodesic_schedule` |
| strong coupling to the environment | `qt.reaction_coordinate_model` |
| storing energy in qubits (quantum batteries) | `qt.batteries` |
| real lab units (GHz, mK, µs) | `qt.LabUnits` |

Every function is listed in the [API reference](https://github.com/VivekKhanna0/Qthermo/blob/main/docs/api.md),
the [guide](https://github.com/VivekKhanna0/Qthermo/blob/main/docs/guide.md) walks
through all of them with figures, and the
[tutorial notebook](https://github.com/VivekKhanna0/Qthermo/blob/main/examples/tutorial.ipynb)
takes one research question from a Hamiltonian to publishable numbers.

## Proof that it works

```bash
python -m qthermo.benchmarks      # 44 checks against published results, ~40 s
```

- **44 benchmarks**, each comparing a computed number with a published or
  exact result, with the source named next to it.
- **About 270 automated tests**, run on every change, including randomised
  checks that the laws of thermodynamics hold for random machines.
- **Agrees with QuTiP**, the standard quantum simulation library, to 1e-16
  for steady states and 1e-10 for time evolution.
- Every code example in this README and the guide is run automatically, so
  none of them can silently go stale.

The exact definitions behind every number are in
[docs/physics.md](https://github.com/VivekKhanna0/Qthermo/blob/main/docs/physics.md).

## Limits

- Weak coupling to the environment by default (Markovian master equations);
  strong coupling only through the reaction-coordinate mapping.
- Up to about 8 qubits (a few hundred energy levels) on a laptop.
- The Redfield model gives steady states only, and is not guaranteed to stay
  physical at strong bath coupling.
- Simulation only: nothing here has been compared with experimental data yet.

## Feedback and where this could go

qthermo is new and built by one person, and the most useful thing you can do
is tell me where it falls short: a result it gets wrong, a paper it should
reproduce, or a feature that would make it useful in your work. Please
[open an issue](https://github.com/VivekKhanna0/Qthermo/issues).

Directions under consideration:

- more reproduced papers, each graded against an exact or published result
- a multi-parameter optimiser ("find the settings that maximise cooling")
- stronger coupling to the environment (HEOM or TEMPO backends)
- presets for specific hardware (superconducting qubits, trapped ions)

## Citing and license

If you use qthermo in research, please cite it using
[CITATION.cff](https://github.com/VivekKhanna0/Qthermo/blob/main/CITATION.cff).
MIT license. Changes are listed in the
[changelog](https://github.com/VivekKhanna0/Qthermo/blob/main/CHANGELOG.md).
