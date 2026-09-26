"""A three-qubit quantum thermal transistor, and the linear-response toolkit.

Qubits L (emitter), M (base) and R (collector), coupled by zz interactions,
each touch their own bath (Joulain et al., PRL 116, 200601 (2016)). Changing
the base temperature changes the heat the base absorbs -- and changes the heat
reaching the collector by several times more: thermal amplification.

The same `response` call gives the full conductance matrix. At equilibrium it
must be symmetric (Onsager reciprocity) and positive semidefinite (second
law), which is checked, not assumed.

    python examples/thermal_transistor.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"


def device(T):
    return qt.models.thermal_transistor(T_L=T["L"], T_M=T["M"], T_R=T["R"])


print(qt.response(device, {"L": 0.5, "M": 0.5, "R": 0.5}).report())

base_temperatures = np.linspace(0.08, 1.0, 30)
gain, currents = [], []
for T_M in base_temperatures:
    r = qt.response(device, {"L": 1.0, "M": T_M, "R": 0.2})
    gain.append(-r.amplification("M", "R"))       # heat delivered to R per heat into M
    currents.append([r.currents[n] for n in ("L", "M", "R")])
gain, currents = np.array(gain), np.array(currents)
k = int(np.argmax(gain))
print(f"\nlargest gain {gain[k]:.2f} at T_M = {base_temperatures[k]:.2f}: "
      f"each extra unit of heat into the base sends {gain[k]:.2f} more to the collector")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
for i, (name, colour) in enumerate((("L (emitter)", "#C0392B"), ("M (base)", "#7D3C98"),
                                    ("R (collector)", "#1B4F72"))):
    ax.plot(base_temperatures, currents[:, i], lw=2, color=colour, label=name)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xlabel("base temperature $T_M$")
ax.set_ylabel("heat current into the device")
ax.set_title("Currents ($T_L = 1$, $T_R = 0.2$)", fontsize=10)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(base_temperatures, gain, "o-", ms=4, lw=2, color="#1B4F72")
ax.axhline(1.0, color="#C0392B", ls="--", lw=1.2, label="gain = 1")
ax.set_xlabel("base temperature $T_M$")
ax.set_ylabel(r"gain $|\partial J_R / \partial J_M|$")
ax.set_title("Thermal amplification", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "thermal_transistor.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'thermal_transistor.png'}")
