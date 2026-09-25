"""Write interactive explorer pages (open the .html files in any browser).

    python examples/explorer.py

- heat_valve_explorer.html: drag the inter-qubit coupling and switch between
  local and global master equations; watch the local model reverse the heat
  flow (Levy & Kosloff 2014).
- fridge_explorer.html: drag the hot-bath temperature of the three-qubit
  absorption refrigerator.
"""

import pathlib

import numpy as np

import qthermo as qt

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
qt.explorer(lambda g: qt.models.two_qubit_heat_valve(g=g, gamma=0.1), np.linspace(0, 1.2, 41),
            parameter="inter-qubit coupling g", title="Heat Valve Explorer",
            path=OUT / "heat_valve_explorer.html")
qt.explorer(lambda T: qt.models.absorption_refrigerator(T_h=T), np.linspace(1.2, 12, 30),
            parameter="hot bath temperature T_h", title="Absorption Fridge Explorer",
            path=OUT / "fridge_explorer.html")
print(f"wrote {OUT / 'heat_valve_explorer.html'} and {OUT / 'fridge_explorer.html'}")
