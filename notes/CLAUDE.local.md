# qthermo — vault pointer

Python package (`qthermo/`, ~76K) giving thermodynamic bookkeeping — heat, work,
entropy production, COP, ergotropy — for open quantum systems evolved with a
Lindblad master equation. Sits on top of numpy/scipy; QuTiP is optional
(anything with `.full()` is accepted as a state/Hamiltonian). Public API surface
is re-exported flat from `qthermo/__init__.py`. 17 physics tests in
`tests/test_physics.py`. Entry points for a human: `examples/full_demo.py`
runs the whole workflow; `README.md` has the pitch and validation table.

## Read before working on:
- modules/core.md — energy, entropy, ergotropy, thermal states (`core.py`)
- modules/solver.md — the Lindblad integrator (`solver.py`)
- modules/cycle.md — Stroke/Cycle/CycleResult, the main user-facing objects (`cycle.py`)
- modules/channels.md — Pauli ops, thermal bath, noise channels (`channels.py`)
- modules/stochastic.md — trajectory unravelling, Jarzynski check (`stochastic.py`)
- modules/analysis.md — sweeps, 2D scans, Pareto fronts (`analysis.py`)
- modules/plotting.md — lazily-imported matplotlib figures (`plotting.py`)
- modules/validation.md — input checks, `QThermoError` (`validation.py`)
- decisions/heat-work-split.md — why the midpoint split makes the first law exact
- decisions/qutip-optional.md — how QuTiP stays an optional dependency
- decisions/qubit-hamiltonian-sign.md — the sign flip in `qubit_hamiltonian`
- gotchas/stroke-temperature-requires-bath.md — `Stroke` validation trap
- gotchas/entropy-production-sign-warning.md — what a negative sigma means

## When adding a feature
Read the one or two module files that actually touch it, not the whole repo.
`core.py` and `solver.py` are the physics floor — anything new in `cycle.py`,
`stochastic.py`, or `analysis.py` builds on those without needing to re-derive them.
