"""Which master equation gives the right heat current? A 10-line reproduction.

Two coupled oscillators between a hot and a cold bath: the standard test case
for the local-versus-global question (Hofer et al., NJP 19, 123037 (2017)).
The model is quadratic, so the exact current is known at any coupling and each
master equation can be graded against it.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qthermo as qt

resonant = qt.papers.local_vs_global(detuning=0.0)
detuned = qt.papers.local_vs_global(detuning=0.1)
print(resonant.report(), "\n")
print(detuned.report())

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
resonant.plot(axes=axes[0])
detuned.plot(axes=axes[1])
axes[0, 0].set_title("resonant oscillators", fontsize=10)
axes[0, 1].set_title("resonant: global fails at weak coupling", fontsize=10)
axes[1, 0].set_title("detuned by 0.1", fontsize=10)
axes[1, 1].set_title("detuned: local is ~10% off at every g", fontsize=10)
fig.tight_layout()
fig.savefig("examples/figures/local_vs_global_exact.png", dpi=130)
