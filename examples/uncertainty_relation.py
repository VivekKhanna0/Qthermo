"""Is this machine's output more precise than any classical machine allows?

The thermodynamic uncertainty relation (TUR) says that for classical Markov
dynamics the relative noise of any current times the entropy production rate
is at least 2. A value below 2 certifies that coherence is doing something no
classical rate model can. The three-level (SSDB) maser is the standard system
where this happens (Kalaee, Wacker & Potts, PRE 104, L012103 (2021)).

This script scans the coherent drive strength and reports the TUR ratio of the
maser's power output, computed exactly from the Liouvillian (no sampling).

    python examples/uncertainty_relation.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
POWER = {"hot": "energy", "cold": "energy"}

drives = np.geomspace(2e-3, 2.0, 40)
ratios, kur = [], []
for drive in drives:
    maser = qt.models.three_level_maser(omega_c=1.0, omega_h=3.0, T_c=1.0, T_h=200.0,
                                        drive=drive, gamma_c=0.05, gamma_h=0.001)
    stats = qt.current_statistics(maser, POWER)
    ratios.append(stats.tur_ratio)
    kur.append(stats.kur_ratio)
ratios, kur = np.array(ratios), np.array(kur)

best = int(np.argmin(ratios))
print(f"minimum TUR ratio {ratios[best]:.4f} at drive = {drives[best]:.3g} "
      f"(classical bound: 2)")
print(f"violation window: drive in [{drives[ratios < 2].min():.3g}, "
      f"{drives[ratios < 2].max():.3g}]")
best_maser = qt.models.three_level_maser(1.0, 3.0, 1.0, 200.0, drives[best], 0.05, 0.001)
print()
print(qt.current_statistics(best_maser, POWER, label="maser power at optimum").report())

# Control: incoherent machines cannot go below 2.
chain = qt.models.spin_chain(3, master_equation="global")
print()
print(qt.current_statistics(chain, "left", label="3-qubit chain, global ME (classical)").report())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogx(drives, ratios, "-", lw=2.2, color="#1B4F72")
ax.axhline(2.0, color="#C0392B", ls="--", lw=1.4, label="classical bound (TUR)")
ax.fill_between(drives, ratios, 2.0, where=ratios < 2, color="#1B4F72", alpha=0.2,
                label="beyond any classical model")
ax.set_xlabel("coherent drive strength")
ax.set_ylabel(r"$(D/J^2)\,\dot\sigma$")
ax.set_ylim(1.9, 3.0)
ax.set_title("Three-level maser: precision vs dissipation", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "tur.png", dpi=130, bbox_inches="tight")
print(f"\nfigure written to {FIGURES / 'tur.png'}")
