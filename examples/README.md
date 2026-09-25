# Examples

Each script runs on its own (`python examples/<name>.py`), prints the numbers
behind its figure, and writes the figure to `examples/figures/`. CI runs all
of them on every push.

| script | question it answers | runtime |
|---|---|---|
| [`tutorial.ipynb`](tutorial.ipynb) | Start here: a two-qubit thermal diode from Hamiltonian to publishable numbers, with the checks a referee would ask for | notebook |
| [`absorption_refrigerator.py`](absorption_refrigerator.py) | How does the three-qubit absorption fridge cool, where does the heat go, and does the master equation matter? | ~2 s |
| [`local_vs_global.py`](local_vs_global.py) | When does the local master equation violate the second law? | ~2 s |
| [`uncertainty_relation.py`](uncertainty_relation.py) | Can a quantum machine be more precise than the TUR allows? | ~1 s |
| [`strong_coupling.py`](strong_coupling.py) | What does weak-coupling theory miss at strong coupling? | ~40 s |
| [`landauer.py`](landauer.py) | What does erasing a bit cost in finite time, and what is the cheapest protocol? | ~30 s |
| [`szilard_engine.py`](szilard_engine.py) | Work from information: measurement, feedback and the Sagawa-Ueda bound | ~15 s |
| [`optimal_protocol.py`](optimal_protocol.py) | The least-dissipative schedule for an arbitrary (non-commuting) drive | ~70 s |
| [`coupled_otto.py`](coupled_otto.py) | Does coupling two qubits improve an Otto engine? | ~6 s |
| [`multiqubit_cycle.py`](multiqubit_cycle.py) | Inside a two-qubit engine: energy, temperature, entanglement stroke by stroke | ~5 s |
| [`operation_modes.py`](operation_modes.py) | Where is a cycle an engine, fridge, accelerator or heater? | ~60 s |
| [`transient_cooling.py`](transient_cooling.py) | Switching a refrigerator on: transient cooling below the steady state | ~10 s |
| [`thermal_transistor.py`](thermal_transistor.py) | Thermal amplification in three zz-coupled qubits; Onsager checks | ~5 s |
| [`driven_machine.py`](driven_machine.py) | A periodically driven qubit as engine and refrigerator (Floquet) | ~5 s |
| [`transport_scaling.py`](transport_scaling.py) | Ballistic vs graded heat transport through spin chains | ~15 s |
| [`quantum_battery.py`](quantum_battery.py) | Collective charging advantage; coherent and locked ergotropy | ~20 s |
| [`otto_refrigerator.py`](otto_refrigerator.py) | The single-qubit Otto fridge: averages, trajectories, exact distribution | ~2 s |
| [`heat_engine.py`](heat_engine.py) | The same code as an engine | ~1 s |
| [`full_demo.py`](full_demo.py) | Sweeps, joint scans, Pareto fronts and reliability for the Otto fridge | ~20 s |
| [`channel_comparison.py`](channel_comparison.py) | Heat and entropy under different noise channels | ~1 s |
| [`subsystem_demo.py`](subsystem_demo.py) | Per-site heat and correlation entropy in a coupled pair | ~1 s |
| [`optimisation.py`](optimisation.py) | Parameter optimisation of a stroke machine | ~20 s |
