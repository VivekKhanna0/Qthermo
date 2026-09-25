"""Build and execute examples/tutorial.ipynb.

    python docs/build_tutorial.py

Requires nbformat, nbclient and ipykernel. The notebook is committed with its
outputs so it renders on GitHub; rerun this after changing the package.
"""

import pathlib

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "tutorial.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md("""# qthermo tutorial: a two-qubit quantum thermal diode

This notebook takes one research question from a Hamiltonian to publishable
numbers, using the checks a referee would ask about along the way.

**The question.** Two coupled qubits with different frequencies sit between a
hot and a cold bath. Does heat flow more easily one way than the other (a
*thermal diode*), and what controls the effect?

Every cell runs in seconds. Conventions: $\\hbar = k_B = 1$; heat currents are
positive when they flow *into* the system."""),
    code("""import numpy as np
import matplotlib.pyplot as plt
import qthermo as qt"""),

    md("""## 1. The machine

$H = \\tfrac{\\omega_1}{2}(-\\sigma_z^1) + \\tfrac{\\omega_2}{2}(-\\sigma_z^2)
+ \\tfrac{J}{2}\\,(\\sigma_x^1\\sigma_x^2 + \\sigma_y^1\\sigma_y^2 + \\Delta\\,\\sigma_z^1\\sigma_z^2)$,
with a bath on each qubit coupling through $\\sigma_x$. Since $J \\gg \\gamma$ we use
**global** (Davies) baths, whose jump operators connect eigenstates of the full
coupled $H$. `qt.models.spin_chain` builds exactly this; the cell after shows
the same thing by hand, which is how you would write your own model."""),
    code("""diode = qt.models.spin_chain(2, omega=[1.0, 2.0], J=0.2, delta=3.0,
                             T_left=2.0, T_right=0.5, gamma=0.01,
                             master_equation="global")
print(diode.description)
print(diode.baths)"""),
    code("""dims = [2, 2]
E, X, Y, Z = qt.embed, qt.sigma_x, qt.sigma_y, qt.sigma_z
H = (E(qt.qubit_hamiltonian(1.0), 0, dims) + E(qt.qubit_hamiltonian(2.0), 1, dims)
     + 0.1 * (E(X, 0, dims) @ E(X, 1, dims) + E(Y, 0, dims) @ E(Y, 1, dims)
              + 3.0 * E(Z, 0, dims) @ E(Z, 1, dims)))
left = qt.davies_bath(H, E(X, 0, dims), temperature=2.0, gamma=0.01, name="left")
right = qt.davies_bath(H, E(X, 1, dims), temperature=0.5, gamma=0.01, name="right")
print("same Hamiltonian as the model:", np.allclose(H, diode.H))"""),

    md("""## 2. Audit before trusting any number

`qt.audit` runs every consistency check: steady-state uniqueness, relaxation
time, second law, detailed balance of each bath against its declared
temperature, local-vs-global agreement, whether internal currents are
resolvable, and uncertainty-relation ratios."""),
    code("""print(qt.audit(diode))"""),
    md("""Two things to take from this. The local and global master equations
disagree here, which is expected at $J = 20\\gamma$, and the global one is the
right choice. And under the global model the bond current between the qubits
cannot be resolved (the secular steady state is diagonal in $H$), so only the
per-bath totals below are meaningful."""),

    md("""## 3. Forward and reverse currents

Swap the temperatures and compare the magnitudes of the heat currents. The
rectification $R = (|J_\\to| - |J_\\leftarrow|)/(|J_\\to| + |J_\\leftarrow|)$ is zero for
a symmetric (linear) conductor."""),
    code("""def diode_currents(J=0.2, delta=3.0, gamma=0.01, master_equation="global"):
    common = dict(omega=[1.0, 2.0], J=J, delta=delta, gamma=gamma,
                  master_equation=master_equation)
    forward = qt.models.spin_chain(2, T_left=2.0, T_right=0.5, **common).analyze()
    reverse = qt.models.spin_chain(2, T_left=0.5, T_right=2.0, **common).analyze()
    return forward.current("left"), reverse.current("right")

def rectification(**kwargs):
    f, r = diode_currents(**kwargs)
    return (abs(f) - abs(r)) / (abs(f) + abs(r))

f, r = diode_currents()
print(f"hot bath on qubit 1 (w=1): J = {f:.4e}")
print(f"hot bath on qubit 2 (w=2): J = {r:.4e}")
print(f"rectification R = {rectification():+.3f}")"""),
    md("""Heat flows about 18% more easily when the hot bath touches the
higher-frequency qubit ($R < 0$ with this labelling). The steady state behind these numbers is exact (a
linear solve, not long-time integration), and energy is conserved to machine
precision:"""),
    code("""steady = diode.analyze()
print(steady.report())"""),

    md("""## 4. What controls it: a parameter sweep

`qt.sweep` accepts any builder that returns a model, a cycle or a result, so
scanning a steady-state machine is one call. Here: rectification against the
$ZZ$ anisotropy $\\Delta$, the nonlinearity of the coupling."""),
    code("""deltas = np.linspace(0.0, 4.0, 21)
scan = qt.sweep(lambda d: d, deltas, lambda d, _: abs(rectification(delta=d)),
                parameter="Delta", metric_name="|R|")
print(scan.report())
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(scan.values, scan.metrics, "o-")
ax.set_xlabel(r"anisotropy $\\Delta$"); ax.set_ylabel(r"rectification $|R|$")
ax.grid(alpha=0.3); plt.show()"""),
    md("""With an isotropic XX coupling ($\\Delta = 0$) the diode barely works.
The $ZZ$ term, which makes the two-qubit spectrum anharmonic, is what drives
the rectification."""),

    md("""## 5. Would a different master equation change the conclusion?

The referee question. Sweep the bath coupling $\\gamma$ at fixed $J$ and compare
local and global predictions. The local model is justified for
$\\gamma \\gg J$ and the global one for $\\gamma \\ll J$."""),
    code("""gammas = np.geomspace(1e-3, 3.0, 12)
local = [rectification(gamma=g, master_equation="local") for g in gammas]
glob = [rectification(gamma=g, master_equation="global") for g in gammas]
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.semilogx(gammas, glob, "o-", label="global (valid for gamma << J)")
ax.semilogx(gammas, local, "s--", label="local (valid for gamma >> J)")
ax.axvline(0.2, color="0.5", ls=":", label="J")
ax.set_xlabel(r"bath coupling $\\gamma$"); ax.set_ylabel("rectification R")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.show()"""),
    md("""Read this carefully, because it changes the physics conclusion:

* The global prediction does not depend on $\\gamma$ (every rate scales with it).
* Where the global model is valid ($\\gamma \\ll J$), the local model predicts
  rectification in the **opposite direction**.
* Only for $\\gamma \\gtrsim J$ does the local model agree on the sign, with a
  smaller magnitude. There the global model's secular approximation is the one
  that fails.

So the direction of this diode is a property of the master equation unless the
regime is stated. Near $\\gamma \\approx J$ neither model is controlled, and a
partially secular or Redfield treatment would be needed. A paper on this device
should say which regime it is in."""),

    md("""## 6. Fluctuations: how reliable is the diode?

The mean current is not the whole story. `current_statistics` gives the exact
noise of the heat current, and the thermodynamic uncertainty relation ratio
$(D/J^2)\\,\\dot\\sigma$, which is at least 2 for any classical Markov process."""),
    code("""for label, T_left, T_right in (("forward", 2.0, 0.5), ("reverse", 0.5, 2.0)):
    m = qt.models.spin_chain(2, omega=[1.0, 2.0], J=0.2, delta=3.0, gamma=0.01,
                             T_left=T_left, T_right=T_right, master_equation="global")
    s = qt.current_statistics(m, "left", label=label)
    print(f"{label:>8}: J = {s.mean:+.3e}, Fano = {s.fano_factor:.3f}, "
          f"TUR ratio = {s.tur_ratio:.3f}")"""),
    md("""Both directions respect the classical bound, as they must: global Davies
dynamics with non-degenerate Bohr frequencies is a classical rate process.
A ratio below 2 would signal coherent transport; see
`examples/uncertainty_relation.py` for a machine where that happens."""),

    md("""## 7. Where next

| to study... | start from |
|---|---|
| refrigerators and virtual temperatures | `examples/absorption_refrigerator.py` |
| when local master equations fail | `examples/local_vs_global.py` |
| precision vs dissipation (TUR) | `examples/uncertainty_relation.py` |
| strong system-bath coupling | `examples/strong_coupling.py` |
| erasure and optimal protocols | `examples/landauer.py` |
| interacting Otto engines | `examples/coupled_otto.py` |
| operation-mode diagrams | `examples/operation_modes.py` |
| quantum batteries | `examples/quantum_battery.py` |

Every quantity above is defined precisely in `docs/physics.md`, and
`python -m qthermo.benchmarks` checks the package against published results."""),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, OUT)
print(f"wrote {OUT}")
