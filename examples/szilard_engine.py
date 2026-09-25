"""Turning information into work: a quantum Szilard engine.

A qubit memory in equilibrium is measured; the outcome is used to extract work
from the bath. Sagawa & Ueda (PRL 100, 080403 (2008)) bound the average work by
T times the mutual information I the measurement gained, and the optimal
feedback protocol (Horowitz & Parrondo, NJP 13, 123019 (2011)) reaches that
bound exactly -- even for an imperfect measurement. In finite time the
isothermal feedback stroke dissipates, and the engine falls short by ~1/tau.

    python examples/szilard_engine.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
T = 1.0

print(qt.szilard_engine(T).report())
errors = np.linspace(0.0, 0.49, 25)
quasi = [qt.szilard_engine(T, error=e) for e in errors]
print(f"\nwith errors, W / (T I) ranges over "
      f"[{min(r.efficiency for r in quasi[:-1]):.10f}, {max(r.efficiency for r in quasi[:-1]):.10f}]")

taus = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
finite = np.array([qt.szilard_engine(T, tau=tau, steps=1500, depth=10.0).work_extracted
                   for tau in taus])
limit = qt.szilard_engine(T, depth=10.0).work_extracted
for tau, w in zip(taus, finite):
    print(f"tau = {tau:>5.0f}: W = {w:.4f}  (shortfall x tau = {(limit - w) * tau:.3f})")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.plot(errors, [r.bound for r in quasi], "-", lw=2.5, color="#C0392B",
        label="Sagawa-Ueda bound $T\\,I$")
ax.plot(errors, [r.work_extracted for r in quasi], "o", ms=5, color="#1B4F72",
        label="optimal feedback (quasi-static)")
ax.axhline(T * np.log(2), color="0.5", ls=":", lw=1, label=r"$T\ln 2$")
ax.set_xlabel("measurement error probability")
ax.set_ylabel("work extracted per cycle")
ax.set_title("Information is worth exactly T I", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.semilogx(taus, finite, "o-", lw=2, color="#1B4F72", label="finite-time feedback")
ax.axhline(limit, color="#C0392B", ls="--", lw=1.4, label="quasi-static")
ax.set_xlabel(r"duration of the feedback stroke $\tau$")
ax.set_ylabel("work extracted per cycle")
ax.set_title("Finite time costs ~1/τ", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "szilard_engine.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'szilard_engine.png'}")
