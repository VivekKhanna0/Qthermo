"""The same four primitives, run as an engine instead of a refrigerator.

Nothing in qthermo knows what a "refrigerator" is. A cycle is a sequence of
strokes; whether it consumes work to move heat (refrigerator) or absorbs heat
to produce work (engine) is a consequence of the parameters, not of a
different code path. Swapping which bath sits at which energy gap flips the
machine over, and the same functions report efficiency instead of COP.

Run with:  python examples/heat_engine.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qthermo as qt
from qthermo.cycle import Cycle, Stroke

OMEGA_LOW, OMEGA_HIGH = 1.0, 2.5
T_COLD, T_HOT = 0.5, 4.0
GAMMA, TAU_ISO, TAU_DRIVE = 1.0, 25.0, 0.02


def ramp(a, b, tau):
    return lambda t: qt.qubit_hamiltonian(a + (t / tau) * (b - a))


def main():
    # Engine: absorb heat at the LARGE gap from the hot bath, dump it at the
    # SMALL gap into the cold bath. (The refrigerator does the reverse.)
    cycle = Cycle([
        Stroke("hot_iso", qt.qubit_hamiltonian(OMEGA_HIGH), TAU_ISO,
               qt.thermal_bath(GAMMA, OMEGA_HIGH, T_HOT), temperature=T_HOT),
        Stroke("expand", ramp(OMEGA_HIGH, OMEGA_LOW, TAU_DRIVE), TAU_DRIVE),
        Stroke("cold_iso", qt.qubit_hamiltonian(OMEGA_LOW), TAU_ISO,
               qt.thermal_bath(GAMMA, OMEGA_LOW, T_COLD), temperature=T_COLD),
        Stroke("compress", ramp(OMEGA_LOW, OMEGA_HIGH, TAU_DRIVE), TAU_DRIVE),
    ])

    result, passes, converged = cycle.limit_cycle(
        qt.thermal_state(qt.qubit_hamiltonian(OMEGA_HIGH), T_HOT)
    )

    print(f"limit cycle reached after {passes} cycles (converged={converged})\n")
    print(result.report())

    efficiency = result.efficiency(("hot_iso",))
    otto_ideal = 1 - OMEGA_LOW / OMEGA_HIGH
    carnot = 1 - T_COLD / T_HOT

    print()
    print(f"  work extracted           {-result.net_work:.5f}")
    print(f"  efficiency               {efficiency:.4f}")
    print(f"  ideal Otto efficiency    {otto_ideal:.4f}   (1 - omega_low/omega_high)")
    print(f"  Carnot bound             {carnot:.4f}")
    print(f"  entropy production       {result.total_entropy_production:.4e}")

    assert efficiency < carnot + 1e-9, "efficiency exceeded the Carnot bound"
    print("\n  Carnot bound respected.")
    print("\nSame Stroke/Cycle objects as the refrigerator example -- only the")
    print("parameters changed. That is what makes this a library rather than")
    print("a script that simulates one machine.")


if __name__ == "__main__":
    main()
