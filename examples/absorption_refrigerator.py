"""The three-qubit absorption refrigerator, end to end.

Linden, Popescu & Skrzypczyk, PRL 105, 130401 (2010): three qubits, three
baths, no work input. Heat flowing from a hot bath into a room-temperature bath
drags heat out of a cold bath along with it.

This script shows the three things you would want to know about it:

1. Does it cool, and how close to Carnot? (steady-state currents)
2. Where does the heat actually go inside it? (site-resolved flows and the
   cold qubit's virtual temperature)
3. Would a reviewer's objection -- "you used a local master equation" --
   change the answer? (local vs global comparison)

    python examples/absorption_refrigerator.py
"""

import pathlib

import numpy as np

import qthermo as qt

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"

fridge = qt.models.absorption_refrigerator(
    omega_c=1.0, omega_h=3.0, T_c=1.0, T_h=6.0, T_r=1.5, g=0.05, gamma=0.01)

steady = fridge.analyze()
print("1. STEADY STATE (local master equation)")
print(steady.report())
cop = steady.cop("cold", "hot")
carnot = steady.absorption_carnot_cop("cold", "hot", "room")
print(f"COP = {cop:.6f}   (tight-coupling prediction w_c/w_h = {1/3:.6f})")
print(f"Carnot bound = {carnot:.4f}, so the machine runs at "
      f"{cop / carnot:.1%} of Carnot")
print()

print("2. INSIDE THE MACHINE")
flows = qt.heat_flow_map(steady)
print(flows.report())
print(f"-> the cold qubit sits at T* = {flows.virtual_temperature[0]:.4f}, "
      f"below the coldest bath (T_c = 1.0)")
print()

print("3. LOCAL vs GLOBAL MASTER EQUATION")
print(fridge.compare().report())
print()

print("4. WHERE DOES IT STOP COOLING?")
print(f"{'T_hot':>8}{'J_cold':>14}{'COP':>10}{'Carnot':>10}{'T*_cold':>10}")
for T_h in (1.6, 2.0, 3.0, 6.0, 12.0, 50.0):
    m = qt.models.absorption_refrigerator(T_h=T_h, T_c=1.0, T_r=1.5)
    r = m.analyze()
    T_star = qt.heat_flow_map(r).virtual_temperature[0]
    cop_text = f"{r.cop('cold', 'hot'):.4f}" if r.current("hot") > 0 else "--"
    print(f"{T_h:>8.1f}{r.current('cold'):>14.4e}{cop_text:>10}"
          f"{r.absorption_carnot_cop('cold', 'hot', 'room'):>10.4f}{T_star:>10.4f}")
print("(J_cold > 0 means heat is extracted from the cold bath)")

try:
    import matplotlib
    matplotlib.use("Agg")
    FIGURES.mkdir(exist_ok=True)
    fig = qt.plot_machine(steady)
    fig.savefig(FIGURES / "absorption_fridge.png", dpi=130, bbox_inches="tight")
    chain = qt.models.spin_chain(5, J=0.3, T_left=3.0, T_right=0.5,
                                 master_equation="local")
    fig = qt.plot_machine(chain.analyze())
    fig.savefig(FIGURES / "spin_chain.png", dpi=130, bbox_inches="tight")
    print(f"\nfigures written to {FIGURES}")
except ImportError:
    print("\n(matplotlib not installed: skipping figures)")
