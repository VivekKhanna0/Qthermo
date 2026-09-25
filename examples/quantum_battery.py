"""Quantum batteries: collective charging and where the work is stored.

1. Charging N two-level cells through one shared cavity (Dicke battery) is
   faster than charging each with its own cavity; the power advantage grows
   as sqrt(N) (Ferraro et al., PRL 120, 117702 (2018)).
2. The work a state holds splits into a part in populations and a part in
   coherence -- the latter is what dephasing destroys first.
3. Correlations between cells can lock ergotropy away from any local
   operation, and many-copy operations unlock work from states no single-copy
   unitary can touch.

    python examples/quantum_battery.py
"""

import pathlib

import numpy as np

import qthermo as qt
from qthermo import batteries as qb

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"

# 1. collective advantage
N_values = [1, 2, 3, 4, 6, 8, 10, 12]
advantage = qb.collective_advantage(N_values)
print("collective charging power / (N x single-cell power):")
for N, a in zip(advantage["N"], advantage["advantage"]):
    print(f"  N = {int(N):>2}: {a:.3f}   (sqrt(N) = {np.sqrt(N):.3f})")
print(f"large-N exponent: {advantage['large_N_exponent']:.3f}  (published: 1/2)")

curves = {}
for N in (1, 4, 10):
    b = qb.dicke_battery(N)
    run = qb.charge(b["H"], b["H_battery"], b["state"], np.linspace(0, 45, 1500))
    curves[N] = run

# 2. coherent vs incoherent ergotropy during charging of 4 cells
b = qb.dicke_battery(4)
times = np.linspace(0, 45, 300)
run = qb.charge(b["H"], b["H_battery"], b["state"], times,
                battery=(0, b["dims"], b["h_battery"]))
print(f"\n4-cell battery: {run.report()}")

# a cell charged into a superposition, then left to dephase: the coherent
# part of its ergotropy disappears, the population part survives
cell = qt.qubit_hamiltonian(1.0)
theta = 2.2                                    # partially inverted, with coherence
psi = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
out = qt.evolve(np.outer(psi, psi.conj()), cell, qt.pure_dephasing(0.5), duration=8.0, steps=4)
print("\ncell under pure dephasing:   t   total   incoherent   coherent")
for t, r in zip(out["times"], out["states"]):
    split = qb.ergotropy_split(r, cell)
    print(f"{t:>30.1f} {split['total']:>7.4f} {split['incoherent']:>11.4f} {split['coherent']:>10.4f}")

# 3. locked and activated ergotropy
bell = np.zeros((4, 4), dtype=complex)
bell[0, 0] = bell[0, 3] = bell[3, 0] = bell[3, 3] = 0.5
H = qt.qubit_hamiltonian(1.0)
print("\nBell pair of cells:", {k: round(v, 4) if np.isscalar(v) else v
                                for k, v in qb.locked_ergotropy(bell, [2, 2], [H, H]).items()})
Hq = np.diag([0.0, 1.0, 2.0]).astype(complex)
rho = np.diag([0.5, 0.4, 0.1]).astype(complex)
print("passive qutrit, ergotropy per copy for 1..6 copies:",
      [round(qb.multi_copy_ergotropy(rho, Hq, k), 4) for k in range(1, 7)],
      f"-> limit {qb.asymptotic_ergotropy(rho, Hq):.4f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
for N, run in curves.items():
    ax.plot(run.times, run.energy / N, lw=2, label=f"N = {N}")
ax.set_xlabel("time")
ax.set_ylabel("stored energy per cell")
ax.set_title("Dicke battery: collective charging", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.loglog(advantage["N"], advantage["advantage"], "o-", lw=2, color="#1B4F72",
          label="collective / parallel power")
Ns = np.array(N_values, dtype=float)
ax.loglog(Ns, np.sqrt(Ns) * advantage["advantage"][-1] / np.sqrt(Ns[-1]), "k--",
          lw=1.2, label=r"$\propto\sqrt{N}$")
ax.set_xlabel("number of cells N")
ax.set_ylabel("power advantage")
ax.set_title("Collective advantage", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "quantum_battery.png", dpi=130, bbox_inches="tight")
print(f"\nfigure written to {FIGURES / 'quantum_battery.png'}")
