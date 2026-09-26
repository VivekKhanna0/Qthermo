"""Does coupling the qubits of an Otto engine make it better?

Two qubits with Heisenberg coupling J as the working medium, driven between
fields B_c and B_h. Without coupling the efficiency is 1 - B_c/B_h. With
antiferromagnetic coupling the spectrum does not scale uniformly with the field
and the efficiency can exceed that value (Thomas & Johal, PRE 83, 031135
(2011)). The simulated finite-time cycle (global baths, one per qubit) is
compared point by point with the exact quasi-static limit, and the per-qubit
breakdown shows where the heat goes.

    python examples/coupled_otto.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
D = [2, 2]
E, X, Y, Z = qt.embed, qt.sigma_x, qt.sigma_y, qt.sigma_z
B_C, B_H, T_C, T_H = 2.0, 4.0, 0.5, 4.0
COUPLINGS = [E(X, 0, D), E(X, 1, D)]


def heisenberg(B, J):
    return (-0.5 * B * (E(Z, 0, D) + E(Z, 1, D))
            + J * sum(E(s, 0, D) @ E(s, 1, D) for s in (X, Y, Z)))


J_line = np.linspace(0.0, 0.8, 81)
ideal = []
for J in J_line:
    lim = qt.ideal_otto(heisenberg(B_C, J), heisenberg(B_H, J), T_C, T_H)
    ideal.append(lim.efficiency if lim.is_engine else np.nan)
ideal = np.array(ideal)

# (B_c = 4J, i.e. J = 0.5, makes two levels exactly degenerate: with a flat
# spectral density the cold isochore then cannot thermalise -- qthermo warns.)
J_sim = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.45])
simulated = []
for J in J_sim:
    cycle = qt.otto_cycle(heisenberg(B_C, J), heisenberg(B_H, J), T_C, T_H,
                          COUPLINGS, tau_iso=30.0, tau_ramp=1.0)
    result, _, _ = cycle.limit_cycle(np.eye(4) / 4)
    try:
        simulated.append(result.efficiency(("hot_iso",)))
    except ValueError:
        simulated.append(np.nan)
simulated = np.array(simulated)

print(f"{'J':>6}{'simulated':>12}{'quasi-static':>14}")
for J, eta in zip(J_sim, simulated):
    ref = ideal[np.argmin(np.abs(J_line - J))]
    print(f"{J:>6.2f}{eta:>12.5f}{ref:>14.5f}")
best = int(np.nanargmax(ideal))
print(f"\nbest efficiency {ideal[best]:.4f} at J = {J_line[best]:.2f} "
      f"(uncoupled: {1 - B_C / B_H:.4f}, Carnot: {1 - T_C / T_H:.4f})")

# Where does the heat go? Per-qubit breakdown of the hot isochore at J = 0.2.
cycle = qt.otto_cycle(heisenberg(B_C, 0.2), heisenberg(B_H, 0.2), T_C, T_H,
                      COUPLINGS, tau_iso=30.0, tau_ramp=1.0)
result, _, _ = cycle.limit_cycle(np.eye(4) / 4)
local_H = [qt.qubit_hamiltonian(B_H), qt.qubit_hamiltonian(B_H)]
print()
print(qt.resolve_stroke(result.stroke("hot_iso"), D, local_H).report())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(J_line, ideal, "-", lw=2.2, color="#1B4F72", label="quasi-static (exact)")
ax.plot(J_sim, simulated, "o", ms=7, color="#C0392B", label="finite-time simulation")
ax.axhline(1 - B_C / B_H, color="0.4", ls="--", lw=1.2, label="uncoupled qubits")
ax.set_xlabel("Heisenberg coupling J")
ax.set_ylabel("efficiency")
ax.set_title("Two-qubit Otto engine", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "coupled_otto.png", dpi=130, bbox_inches="tight")
print(f"\nfigure written to {FIGURES / 'coupled_otto.png'}")
