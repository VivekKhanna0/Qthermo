"""The least-dissipative way to drive a quantum system, for any control path.

A qubit whose field grows and tilts (the drive does not commute with itself at
different times) is driven in contact with a bath. Slow-driving linear
response gives a friction metric g(lambda) along the path; running at constant
speed in that metric minimises the dissipated work, to L^2/tau with L the
thermodynamic length. The script compares that prediction with full
finite-time simulations of the linear and the optimal schedules.

    python examples/optimal_protocol.py
"""

import pathlib

import numpy as np

import qthermo as qt
from qthermo.baths import ohmic_spectrum
from qthermo.geometry import excess_work, optimal_schedule

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
T = 0.7


def H_of(theta):
    """Field grows and rotates towards x as theta goes 0 -> 2."""
    return qt.qubit_hamiltonian(1.0 + theta) + 0.8 * theta * qt.sigma_x


def baths_of(H):
    return [qt.davies_bath(H, qt.sigma_x + qt.sigma_z, T,
                           spectrum=ohmic_spectrum(0.5, reference=1.0))]


schedule = optimal_schedule(H_of, baths_of, T, (0.0, 2.0), points=101)
print(f"thermodynamic length L = {schedule.length:.4f}; "
      f"minimum excess work = {schedule.length ** 2:.4f} / tau")
print(f"linear ramp predicted  = {schedule.predicted_excess(lambda x: x, 1.0):.4f} / tau")

taus = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
rows = []
for tau in taus:
    lin = excess_work(lambda t, tau=tau: H_of(2.0 * t / tau), baths_of, T, tau,
                      steps=1500)["dissipation"]
    opt = excess_work(schedule.hamiltonian(H_of, tau), baths_of, T, tau,
                      steps=1500)["dissipation"]
    rows.append((tau, lin, opt))
    print(f"tau = {tau:>5.0f}:  linear {lin * tau:.4f}/tau   optimal {opt * tau:.4f}/tau")
rows = np.array(rows)
saving = 1 - rows[-1, 2] / rows[-1, 1]
print(f"optimal schedule dissipates {saving:.0%} less at tau = {taus[-1]:.0f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
ax = axes[0]
ax.plot(2.0 * schedule.s_grid, schedule.metric / 4.0, lw=2, color="0.3")
ax.set_xlabel(r"control $\theta$")
ax.set_ylabel(r"friction $g(\theta)$")
ax.set_title("Friction along the path", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[1]
x = np.linspace(0, 1, 200)
ax.plot(x, 2.0 * x, lw=2, color="#C0392B", label="linear")
ax.plot(x, [2.0 * float(schedule.s_of(v)) for v in x], lw=2, color="#1B4F72",
        label="optimal: slow where friction is high")
ax.set_xlabel(r"time $t/\tau$")
ax.set_ylabel(r"control $\theta(t)$")
ax.set_title("Schedules", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25)

ax = axes[2]
ax.loglog(taus, rows[:, 1], "o", color="#C0392B", label="linear, simulated")
ax.loglog(taus, rows[:, 2], "o", color="#1B4F72", label="optimal, simulated")
ax.loglog(taus, schedule.predicted_excess(lambda v: v, 1.0) / taus, "--", color="#C0392B",
          label="linear, predicted")
ax.loglog(taus, schedule.length ** 2 / taus, "--", color="#1B4F72",
          label=r"minimum $L^2/\tau$")
ax.set_xlabel(r"protocol duration $\tau$")
ax.set_ylabel(r"dissipated work $T\sigma$")
ax.set_title("Prediction vs finite-time simulation", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "optimal_protocol.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'optimal_protocol.png'}")
