"""Ballistic or not? Heat transport through spin chains of increasing length.

Boundary-driven XXZ chains with a hot bath on the first spin and a cold bath on
the last, in the local (boundary-driven) master equation that is standard in
the spin-transport literature. The XX chain (Delta = 0) is ballistic: its
current does not depend on length at all, and its interior temperature profile
is flat (cf. Karevski & Platini, PRL 102, 207207 (2009)). The Ising (zz) term
makes the current fall with length (roughly as 1/N over the lengths reachable
here, Fourier-like) and a temperature gradient build up across the chain.

The audit line at the end is a reminder that these numbers belong to the local
model: with J comparable to the bath rate, a global treatment gives different
currents. The qualitative ballistic/graded contrast is what this model is used
for.

    python examples/transport_scaling.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
lengths = [2, 3, 4, 5, 6, 7]
deltas = [0.0, 0.5, 1.0, 2.0]


def chain(n, delta):
    return qt.models.spin_chain(n, J=0.5, delta=delta, T_left=5.0, T_right=0.5,
                                gamma=0.5, master_equation="local")


currents = {d: [] for d in deltas}
profiles = {}
for d in deltas:
    for n in lengths:
        steady = chain(n, d).analyze()
        currents[d].append(steady.current("left"))
        if n == lengths[-1]:
            profiles[d] = qt.heat_flow_map(steady).virtual_temperature

print(f"{'Delta':>6}" + "".join(f"{'N=' + str(n):>12}" for n in lengths))
for d in deltas:
    print(f"{d:>6}" + "".join(f"{J:>12.4e}" for J in currents[d]))
ratio = currents[0.0][-1] / currents[0.0][0]
print(f"\nXX chain: J(N={lengths[-1]}) / J(N=2) = {ratio:.12f}  (ballistic: exactly 1)")
slope = np.polyfit(np.log(lengths[-3:]), np.log(currents[2.0][-3:]), 1)[0]
print(f"Delta = 2: J ~ N^{slope:.2f} over the last three lengths")
print("\nvirtual temperature profile, N = 7:")
for d in deltas:
    print(f"  Delta = {d}: " + "  ".join(f"{T:.2f}" for T in profiles[d]))
print("\naudit of the N = 4 XX chain: "
      + qt.audit(chain(4, 0.0), fluctuations=False).report().splitlines()[0].strip())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
for d in deltas:
    ax.plot(lengths, currents[d], "o-", lw=2, label=rf"$\Delta = {d:g}$")
ax.set_xlabel("chain length N")
ax.set_ylabel("heat current")
ax.set_yscale("log")
ax.set_title("Current against length", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")

ax = axes[1]
for d in deltas:
    ax.plot(range(1, lengths[-1] + 1), profiles[d], "o-", lw=2, label=rf"$\Delta = {d:g}$")
ax.axhline(5.0, color="#C0392B", ls=":", lw=1)
ax.axhline(0.5, color="#2E86C1", ls=":", lw=1)
ax.set_xlabel("site")
ax.set_ylabel(r"virtual temperature $T^*$")
ax.set_title(f"Temperature profile, N = {lengths[-1]}", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "transport_scaling.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'transport_scaling.png'}")
