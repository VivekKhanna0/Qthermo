"""A periodically driven qubit as a heat engine and a refrigerator.

The 'minimal universal quantum heat machine' (Gelbwaser-Klimovsky, Alicki &
Kurizki, PRE 87, 012140 (2013)): a qubit whose frequency is modulated,
w(t) = w0 + lam sin(Omega t), between a hot bath that only absorbs/emits near
w0 + Omega and a cold bath that only does so near w0 - Omega. The modulation
creates Floquet sidebands with Bessel-function weights; each exchange moves
w0 + Omega with the hot bath and w0 - Omega with the cold one, so

    engine:        efficiency = 1 - (w0 - Omega)/(w0 + Omega)
    refrigerator:  COP = (w0 - Omega) / (2 Omega)

and the machine switches mode when T_c/T_h crosses (w0 - Omega)/(w0 + Omega).
Power follows J_1(lam/Omega)^2 and peaks at lam/Omega = 1.841 -- until the
modulation is strong enough that a higher sideband opens a second channel into
a bath window. Here, with weight J_5^2, the qubit is *excited* while emitting
w0 - Omega into the cold bath, paid for by five drive quanta: drive work leaks
straight into the cold bath and short-circuits the engine.

    python examples/driven_machine.py
"""

import pathlib

import numpy as np
from scipy.special import jv

import qthermo as qt
from qthermo.floquet import DrivenBath, floquet_analyze, window_spectrum

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
W0, OM = 3.0, 1.0


def machine(T_h, T_c, lam=0.6):
    H = lambda t: qt.qubit_hamiltonian(W0 + lam * np.sin(OM * t))
    hot = DrivenBath("hot", qt.sigma_x, T_h, spectrum=window_spectrum(0.05, W0 + OM, 0.8))
    cold = DrivenBath("cold", qt.sigma_x, T_c, spectrum=window_spectrum(0.05, W0 - OM, 0.8))
    return floquet_analyze(H, 2 * np.pi / OM, [hot, cold], n_time=128)


print(machine(4.0, 0.5).report())
ratios = np.linspace(0.05, 0.95, 37)
performance, modes = [], []
for x in ratios:
    r = machine(2.0, 2.0 * x)
    m = r.mode("hot", "cold")
    modes.append(m)
    performance.append(r.efficiency("hot") if m == "engine" else
                       r.cop("cold") if m == "refrigerator" else np.nan)
switch = ratios[[m == "refrigerator" for m in modes].index(True)]
print(f"\nengine below T_c/T_h = {(W0 - OM) / (W0 + OM):.3f}, refrigerator above "
      f"(first refrigerator point at {switch:.3f})")

lams = np.linspace(0.1, 3.5, 35)
power = np.array([machine(4.0, 0.5, lam=l).power for l in lams])
print(f"power peaks at lam/Omega = {lams[int(np.argmax(power))]:.2f} "
      f"(J_1 maximum at 1.841)")
print(f"at lam/Omega = 3.5: J_1^2 = {jv(1, 3.5) ** 2:.4f}, J_5^2 = {jv(5, 3.5) ** 2:.4f} -- "
      "the short-circuit channel is no longer negligible")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("(matplotlib not installed: skipping figure)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
perf = np.array(performance)
engine = np.array([m == "engine" for m in modes])
ax.plot(ratios[engine], perf[engine], "o", color="#1B4F72", label="efficiency (engine)")
ax.plot(ratios[~engine], perf[~engine], "s", color="#2E86C1", label="COP (refrigerator)")
x = np.linspace(0.02, 0.98, 200)
ax.plot(x, 1 - x, "--", color="0.5", lw=1, label="Carnot efficiency")
ax.plot(x, x / (1 - x), ":", color="0.5", lw=1, label="Carnot COP")
ax.axvline((W0 - OM) / (W0 + OM), color="#C0392B", lw=1, label="predicted switch")
ax.set_ylim(0, 2.0)
ax.set_xlabel(r"$T_c / T_h$")
ax.set_ylabel("efficiency or COP")
ax.set_title("Mode switch of a Floquet machine", fontsize=10)
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(lams, power, "o", color="#1B4F72", label="Floquet master equation")
bessel = jv(1, lams / OM) ** 2
ax.plot(lams, bessel * power.max() / bessel.max(), "-", color="#C0392B",
        label=r"$\propto J_1(\lambda/\Omega)^2$")
ax.set_xlabel(r"modulation amplitude $\lambda/\Omega$")
ax.set_ylabel("power delivered")
ax.axvspan(2.6, 3.5, color="0.92", zorder=0)
ax.annotate("higher sidebands\nopen (J$_5$)", (3.05, power.max() * 0.75), ha="center",
            fontsize=8, color="0.35")
ax.set_title("Power follows the first sideband", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "driven_machine.png", dpi=130, bbox_inches="tight")
print(f"figure written to {FIGURES / 'driven_machine.png'}")
