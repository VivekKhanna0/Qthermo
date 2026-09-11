"""Optimising a quantum refrigerator: sweeps, joint scans, channel comparison.

This is the workflow that gets rewritten by hand in every paper reporting an
optimised quantum thermal machine.

Run with:  python examples/optimisation.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qthermo as qt
from qthermo.analysis import compare_channels, scan_2d, sweep
from qthermo.cycle import Cycle, Stroke

OMEGA_COLD = 1.0
T_COLD, T_HOT = 1.0, 1.3
TAU_DRIVE = 0.02


def ramp(a, b, tau):
    return lambda t: qt.qubit_hamiltonian(a + (t / tau) * (b - a))


def make_cycle(omega_hot=1.5, gamma=1.0, tau_iso=12.0, cold_ops_override=None):
    """Otto refrigerator with the knobs exposed.

    ``cold_ops_override`` replaces the cold bath only, so the hot bath still
    provides a temperature gradient and the cycle still runs.
    """
    cold_ops = (cold_ops_override if cold_ops_override is not None
                else qt.thermal_bath(gamma, OMEGA_COLD, T_COLD))
    hot_ops = qt.thermal_bath(gamma, omega_hot, T_HOT)
    return Cycle([
        Stroke("cold_iso", qt.qubit_hamiltonian(OMEGA_COLD), tau_iso,
               cold_ops, temperature=T_COLD),
        Stroke("compress", ramp(OMEGA_COLD, omega_hot, TAU_DRIVE), TAU_DRIVE),
        Stroke("hot_iso", qt.qubit_hamiltonian(omega_hot), tau_iso,
               hot_ops, temperature=T_HOT),
        Stroke("expand", ramp(omega_hot, OMEGA_COLD, TAU_DRIVE), TAU_DRIVE),
    ])


def figure_of_merit(result, cycle):
    return result.figure_of_merit(("cold_iso",), cycle.duration)


def main():
    print("=" * 68)
    print("1. ONE-DIMENSIONAL SWEEP  (frequency ratio)")
    print("=" * 68)
    print()

    ratios = np.linspace(1.1, 2.6, 16)
    frequency_sweep = sweep(
        lambda r: make_cycle(omega_hot=OMEGA_COLD * r),
        ratios, figure_of_merit,
        parameter="frequency ratio", metric_name="FOM",
    )
    print(frequency_sweep.report())
    print(f"\ngain over r = 1.5: {frequency_sweep.improvement_over(1.5):+.1f}%")

    print()
    print("=" * 68)
    print("2. JOINT SCAN  (frequency ratio x coupling strength)")
    print("=" * 68)
    print()

    couplings = np.logspace(-1.0, 0.6, 9)
    scan = scan_2d(
        lambda r, g: make_cycle(omega_hot=OMEGA_COLD * r, gamma=g),
        np.linspace(1.1, 2.6, 9), couplings, figure_of_merit,
        parameters=("frequency ratio", "coupling"), metric_name="FOM",
    )

    joint_r, joint_g, joint_fom = scan.optimum()
    seq_r, seq_g, seq_fom = scan.sequential_optimum()

    print(f"  joint optimum       r = {joint_r:.3f}, gamma = {joint_g:.3f}"
          f"  ->  FOM = {joint_fom:.4e}")
    print(f"  sequential optimum  r = {seq_r:.3f}, gamma = {seq_g:.3f}"
          f"  ->  FOM = {seq_fom:.4e}")
    print(f"\n  joint search beats sequential tuning by {scan.sequential_gap():+.2f}%")
    print()
    print("  For this system the gap is zero: tuning the frequency ratio and")
    print("  then the coupling finds the same optimum as searching both at")
    print("  once. That is a result, not a disappointment -- sequential")
    print("  optimisation is standard practice precisely because it is cheap,")
    print("  and it is usually assumed rather than checked. The nested loop")
    print("  that checks it is tedious to write by hand and is one call here,")
    print("  so the assumption can be tested per system instead of trusted.")

    print()
    print("=" * 68)
    print("3. CHANNEL COMPARISON")
    print("=" * 68)
    print()

    channels = {
        "thermal (cold T)": qt.thermal_bath(1.0, OMEGA_COLD, T_COLD),
        "amplitude damping": qt.amplitude_damping(1.0),
        "pure dephasing": qt.pure_dephasing(1.0),
        "bit flip": qt.bit_flip(1.0),
    }
    metrics = {
        "Q_cold": lambda r, c: r.stroke("cold_iso").heat,
        "W_net": lambda r, c: r.net_work,
        "sigma": lambda r, c: r.total_entropy_production,
    }
    table = compare_channels(
        lambda c_ops: make_cycle(cold_ops_override=c_ops), channels, metrics
    )

    print(f"{'channel':<22}{'Q_cold':>13}{'W_net':>13}{'sigma':>13}")
    print("-" * 61)
    for name, row in table.items():
        print(f"{name:<22}{row['Q_cold']:>13.5f}"
              f"{row['W_net']:>13.5f}{row['sigma']:>13.5f}")

    print()
    print("Reading the table:")
    print("  * pure dephasing exchanges no heat at all -- structurally flat,")
    print("    because it moves coherence but never population.")
    print("  * bit flip behaves like an infinite-temperature bath: it pumps")
    print("    population upward, so Q_cold is large and positive.")
    print("  * amplitude damping is a ZERO-temperature channel, but the stroke")
    print("    is still labelled T = 1.0. That mismatch makes the reported")
    print("    entropy production come out negative, which is impossible --")
    print("    and qthermo raises a RuntimeWarning saying so. Entropy")
    print("    production is only defined relative to the bath the channel")
    print("    actually represents.")


if __name__ == "__main__":
    main()
