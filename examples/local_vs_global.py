"""When does the local master equation break? One figure, one answer.

Two detuned qubits with XX coupling g, hot bath on one, cold bath on the
other. The local master equation (bath jump operators of each isolated qubit)
is standard in quantum transport and quantum-machine papers. It predicts heat
flowing from the cold bath into the hot one once g is no longer small -- a
second-law violation (Levy & Kosloff, EPL 107, 20004 (2014)). The global
(Davies) master equation never does; counting the local model's boundary work
repairs its bookkeeping but not its predicted currents.

    python examples/local_vs_global.py
"""

import pathlib
import warnings

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
couplings = np.linspace(0.0, 1.2, 25)

rows = []
for g in couplings:
    local = qt.models.two_qubit_heat_valve(g=g, gamma=0.1, master_equation="local")
    glob = qt.models.two_qubit_heat_valve(g=g, gamma=0.1, master_equation="global")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_local = local.analyze(energy=None)          # heat measured with full H
    r_local_H0 = local.analyze()                      # bare-H heat + boundary work
    r_glob = glob.analyze()
    rows.append((g, r_local.current("hot"), r_glob.current("hot"),
                 r_local.entropy_production_rate, r_glob.entropy_production_rate,
                 r_local_H0.entropy_production_rate))
rows = np.array(rows)

print(f"{'g':>6}{'J_hot local':>14}{'J_hot global':>14}{'sigma local':>14}{'sigma global':>14}")
for row in rows[::4]:
    print(f"{row[0]:>6.2f}{row[1]:>14.3e}{row[2]:>14.3e}{row[3]:>14.3e}{row[4]:>14.3e}")
violating = rows[rows[:, 3] < -1e-12, 0]
if len(violating):
    print(f"\nlocal ME violates the second law for g >= {violating.min():.2f} "
          f"(bath rate gamma = 0.1)")
print(f"global ME: min entropy production rate {rows[:, 4].min():.2e} (never negative)")
print(f"local ME + boundary work: min {rows[:, 5].min():.2e} (never negative)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.plot(rows[:, 0], rows[:, 2], "-", lw=2.2, color="#1B4F72", label="global (Davies)")
ax.plot(rows[:, 0], rows[:, 1], "--", lw=2.2, color="#C0392B", label="local")
ax.axhline(0, color="0.4", lw=0.8)
ax.fill_between(rows[:, 0], rows[:, 1], 0, where=rows[:, 1] < 0, color="#C0392B",
                alpha=0.15, label="heat flows cold -> hot")
ax.set_xlabel("inter-qubit coupling g")
ax.set_ylabel("heat current from hot bath")
ax.set_title("Same machine, two master equations", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(rows[:, 0], rows[:, 4], "-", lw=2.2, color="#1B4F72", label="global")
ax.plot(rows[:, 0], rows[:, 3], "--", lw=2.2, color="#C0392B", label="local, heat via full H")
ax.plot(rows[:, 0], rows[:, 5], ":", lw=2.2, color="#7D3C98", label="local, heat via bare H + boundary work")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xlabel("inter-qubit coupling g")
ax.set_ylabel("entropy production rate")
ax.set_title("Second law", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "local_vs_global.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'local_vs_global.png'}")
