"""The price of forgetting, and how to pay less of it.

Erasing a bit at temperature T releases at least T ln 2 of heat. In finite time
it releases more, and the excess depends on *how* the erasure is done. For
slow protocols the excess is (1/tau) times a protocol-dependent constant; the
smallest possible constant is the squared thermodynamic length L^2, reached by
the geodesic protocol (Scandi & Perarnau-Llobet, Quantum 3, 197 (2019)).

This script simulates erasure as a driven open-system process under four ramp
shapes and compares each with the slow-driving prediction.

    python examples/landauer.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
T = 1.0
taus = np.geomspace(2.0, 200.0, 9)
schedules = ["linear", "smooth", "exponential", "geodesic"]
L2 = qt.thermodynamic_length(T) ** 2

print(f"Landauer bound T ln 2 = {T * np.log(2):.5f}; thermodynamic length^2 = {L2:.4f}")
print(f"{'schedule':<12}{'predicted tau*excess':>22}{'simulated (tau=200)':>22}")
results = {}
for name in schedules:
    shape = qt.geodesic_schedule(T) if name == "geodesic" else name
    runs = [qt.landauer_erasure(tau, temperature=T, schedule=shape) for tau in taus]
    results[name] = np.array([r.excess_heat for r in runs])
    predicted = qt.predicted_excess(shape, 1.0, T)
    print(f"{name:<12}{predicted:>22.4f}{results[name][-1] * taus[-1]:>22.4f}")

slowest = qt.landauer_erasure(taus[-1], temperature=T, schedule="geodesic")
print()
print(slowest.report())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

colours = {"linear": "#C0392B", "smooth": "#D68910", "exponential": "#7D3C98",
           "geodesic": "#1B4F72"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
for name in schedules:
    ax.loglog(taus, results[name], "o-", ms=4, lw=1.8, color=colours[name], label=name)
ax.loglog(taus, L2 / taus, "k--", lw=1.2, label=r"minimum $L^2/\tau$")
ax.set_xlabel(r"erasure time $\tau$")
ax.set_ylabel(r"heat released beyond $T\,\Delta S$")
ax.set_title("Finite-time cost of erasing one bit", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")

ax = axes[1]
s = np.linspace(0, 1, 200)
geodesic = qt.geodesic_schedule(T)
for name in schedules:
    shape = geodesic if name == "geodesic" else qt.information.erasure_schedules[name]
    ax.plot(s, [12.0 * T * float(shape(x)) for x in s], lw=2, color=colours[name],
            label=name)
ax.set_xlabel(r"time $t/\tau$")
ax.set_ylabel(r"qubit gap $\omega(t)$")
ax.set_title("Protocols (geodesic slows down where it is costly)", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "landauer.png", dpi=130, bbox_inches="tight")
print(f"\nfigure written to {FIGURES / 'landauer.png'}")
