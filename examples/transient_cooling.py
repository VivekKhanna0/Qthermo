"""Switching on a quantum refrigerator: faster, and colder, than steady state.

The three-qubit absorption refrigerator is switched on at t = 0 with every
qubit thermal with its own bath. When the internal coupling g is large compared
with the bath rates gamma, the cold qubit's temperature oscillates and dips
well below the value it eventually settles at -- 'single-shot cooling'
(Mitchison, Woods, Prior & Huber, NJP 17, 115013 (2015)). An application
that can use the qubit at the right moment gets it colder than the steady
state ever will.

    python examples/transient_cooling.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
runs = {}
for gamma in (0.002, 0.01, 0.05):
    fridge = qt.models.absorption_refrigerator(g=0.05, gamma=gamma, T_c=1.0, T_h=6.0, T_r=1.5)
    start = qt.product_thermal_state(fridge.local_H, [1.0, 6.0, 1.5])
    run = qt.transient(fridge, start, duration=8.0 / gamma, steps=3000)
    T_steady = qt.heat_flow_map(fridge.analyze()).virtual_temperature[0]
    runs[gamma] = (run, T_steady)
    T_min, t_min = run.minimum_temperature(0)
    print(f"g/gamma = {0.05 / gamma:>4.0f}: cold qubit T* minimum {T_min:.4f} at t = {t_min:.0f} "
          f"(gamma t = {gamma * t_min:.2f}); steady state {T_steady:.4f}; "
          f"cold current settles by gamma t = {gamma * run.settling_time('cold'):.2f}")

print()
print(runs[0.002][0].report())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
colours = {0.002: "#1B4F72", 0.01: "#2E86C1", 0.05: "#AED6F1"}
for gamma, (run, T_steady) in runs.items():
    label = f"g/γ = {0.05 / gamma:.0f}"
    axes[0].plot(gamma * run.times, run.virtual_temperatures[0], lw=2,
                 color=colours[gamma], label=label)
    axes[0].axhline(T_steady, color=colours[gamma], ls=":", lw=1)
    axes[1].plot(gamma * run.times, run.currents["cold"] / gamma, lw=2,
                 color=colours[gamma], label=label)
axes[0].axhline(1.0, color="0.4", ls="--", lw=1, label="cold bath")
axes[0].set_xlabel(r"$\gamma t$")
axes[0].set_ylabel("cold qubit virtual temperature $T^*$")
axes[0].set_title("Transient undershoot (dotted: steady state)", fontsize=10)
axes[0].legend(fontsize=8, frameon=False)
axes[0].grid(alpha=0.25)
axes[1].set_xlabel(r"$\gamma t$")
axes[1].set_ylabel(r"heat drawn from cold bath $/\gamma$")
axes[1].set_title("Cooling current after switch-on", fontsize=10)
axes[1].legend(fontsize=8, frameon=False)
axes[1].grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "transient_cooling.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'transient_cooling.png'}")
