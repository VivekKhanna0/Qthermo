"""What weak-coupling theory gets wrong about heat transport.

A qubit strongly coupled to a hot bath (reaction-coordinate mapping) and weakly
to a cold one. Weak-coupling master equations predict a heat current that grows
as lam^2 without limit. The reaction-coordinate model shows it peaking and then
falling -- the 'turnover' familiar from spin-boson transport -- and the qubit's
equilibrium state drifting away from its Gibbs state toward the ultrastrong
limit of Cresser & Anders (PRL 127, 250601 (2021)).

    python examples/strong_coupling.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
H_S, S = qt.qubit_hamiltonian(1.0), qt.sigma_x


def machine(lam, n_levels=22):
    return qt.reaction_coordinate_model(
        H_S, S, T=2.0, lam=lam, Omega=2.0, kappa=0.1, n_levels=n_levels,
        name="hot", weak_baths=[dict(coupling=S, T=0.5, gamma=0.05, name="cold")])


print("convergence in the RC truncation at the strongest coupling:")
check = qt.rc_convergence(lambda n: machine(2.4, n), levels=(14, 18, 22, 26))
print(f"  relative change between {check['levels'][-2]} and {check['levels'][-1]} "
      f"levels: {check['relative_change']:.2e} (converged: {check['converged']})")

lams = np.linspace(0.02, 2.6, 30)
currents = np.array([machine(l).analyze().current("hot") for l in lams])
weak = currents[0] * (lams / lams[0]) ** 2
peak = int(np.argmax(currents))
print(f"\nheat current peaks at lam = {lams[peak]:.2f}; at lam = {lams[-1]:.1f} it is "
      f"{currents[-1] / currents[peak]:.0%} of the peak, while weak-coupling "
      f"scaling predicts {weak[-1] / currents[peak]:.0f}x the peak")

T = 0.5
gibbs, limit = qt.thermal_state(H_S, T), qt.ultrastrong_limit_state(H_S, S, T)
lam_eq = np.geomspace(0.01, 5.0, 30)
d_gibbs = [qt.trace_distance(qt.mean_force_state(H_S, S, T, l, 2.0, 60), gibbs) for l in lam_eq]
d_ultra = [qt.trace_distance(qt.mean_force_state(H_S, S, T, l, 2.0, 60), limit) for l in lam_eq]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.plot(lams, currents, "-", lw=2.2, color="#1B4F72", label="reaction coordinate")
ax.plot(lams, weak, "--", lw=1.6, color="#C0392B", label=r"weak coupling $\propto\lambda^2$")
ax.set_ylim(0, 1.4 * currents.max())
ax.set_xlabel(r"system-bath coupling $\lambda$")
ax.set_ylabel("heat current from hot bath")
ax.set_title("Turnover at strong coupling", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.loglog(lam_eq, d_gibbs, "-", lw=2.2, color="#1B4F72", label="distance to Gibbs state")
ax.loglog(lam_eq, d_ultra, "-", lw=2.2, color="#7D3C98", label="distance to ultrastrong limit")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("trace distance")
ax.set_title("Equilibrium state of a strongly coupled qubit", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "strong_coupling.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'strong_coupling.png'}")
