"""Inside a two-qubit engine cycle: who holds the energy, and when.

A transverse-field Ising pair is run through an Otto cycle: thermalise with
a hot bath at strong field, ramp the field down, thermalise with a cold bath,
ramp it back up. The ramps are fast and do not commute with the Ising
coupling, so they build correlations -- including entanglement -- that the
baths then destroy. The totals (efficiency, work) hide all of this; the site
view shows it stroke by stroke.

    python examples/multiqubit_cycle.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
D = [2, 2]
E, X, Z = qt.embed, qt.sigma_x, qt.sigma_z
J = 1.0


def ising(h):
    return -J * E(Z, 0, D) @ E(Z, 1, D) - h * (E(X, 0, D) + E(X, 1, D))


h_cold, h_hot, T_cold, T_hot, tau_ramp = 0.8, 2.5, 0.3, 3.0, 0.3
cycle = qt.otto_cycle(ising(h_cold), ising(h_hot), T_cold, T_hot,
                      couplings=[E(Z, 0, D), E(Z, 1, D)], gamma=1.0,
                      tau_iso=25.0, tau_ramp=tau_ramp, steps=250)
result, passes, converged = cycle.limit_cycle(np.eye(4) / 4)
print(result.report())
print(f"mode: {result.mode()}, limit cycle after {passes} passes")
ideal = qt.ideal_otto(ising(h_cold), ising(h_hot), T_cold, T_hot)
print(f"efficiency {result.efficiency(('hot_iso',)):.4f} in finite time vs "
      f"{ideal.efficiency:.4f} quasi-static: fast ramps cost work (quantum friction)")

# Site view: each qubit's local term is its transverse field, which the
# ramps drive; the Ising bond is the interaction.
ramp = lambda a, b: (lambda t: -(a + (t / tau_ramp) * (b - a)) * X)
local = {"hot_iso": [-h_hot * X] * 2, "expand": [ramp(h_hot, h_cold)] * 2,
         "cold_iso": [-h_cold * X] * 2, "compress": [ramp(h_cold, h_hot)] * 2}
dyn = qt.site_dynamics(result, D, local)
C = dyn["concurrence"][(0, 1)]
edges, names = dyn["stroke_edges"], dyn["stroke_names"]
k = int(np.nanargmax(C))
where = names[min(int(np.searchsorted(edges, dyn["time"][k], side="right") - 1), 3)]
print(f"\nentanglement: concurrence peaks at {C[k]:.3f} during '{where}'")
for name, a, b in zip(names, edges[:-1], edges[1:]):
    mask = (dyn["time"] >= a) & (dyn["time"] <= b)
    print(f"  {name:<9} concurrence {np.nanmin(C[mask]):.3f} .. {np.nanmax(C[mask]):.3f}")

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figures)")

FIGURES.mkdir(exist_ok=True)
fig = qt.plot_site_dynamics(dyn, site_names=["qubit A", "qubit B"], figsize=(10, 7))
fig.savefig(FIGURES / "multiqubit_cycle.png", dpi=130, bbox_inches="tight")
print(f"figures written to {FIGURES}")
