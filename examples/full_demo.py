"""Full workflow on one machine, with figures.

This is the demo: define a quantum Otto refrigerator once, then analyse it
every way qthermo supports -- energy ledger, optimisation, robustness,
trajectory statistics, and a consistency self-test -- and save the figures.

Run with:  python examples/full_demo.py
Figures are written to  examples/figures/
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qthermo as qt
from qthermo.analysis import scan_2d, sweep
from qthermo.cycle import Cycle, Stroke
from qthermo.plotting import (
    plot_cycle,
    plot_distribution,
    plot_scan,
    plot_sweep,
)

OMEGA_COLD, OMEGA_HOT = 1.0, 1.5
T_COLD, T_HOT = 1.0, 1.3
GAMMA, TAU_ISO, TAU_DRIVE = 1.0, 12.0, 0.02

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"


def ramp(a, b, tau):
    return lambda t: qt.qubit_hamiltonian(a + (t / tau) * (b - a))


def build(omega_hot=OMEGA_HOT, gamma=GAMMA, tau_iso=TAU_ISO):
    return Cycle([
        Stroke("cold_iso", qt.qubit_hamiltonian(OMEGA_COLD), tau_iso,
               qt.thermal_bath(gamma, OMEGA_COLD, T_COLD), temperature=T_COLD),
        Stroke("compress", ramp(OMEGA_COLD, omega_hot, TAU_DRIVE), TAU_DRIVE),
        Stroke("hot_iso", qt.qubit_hamiltonian(omega_hot), tau_iso,
               qt.thermal_bath(gamma, omega_hot, T_HOT), temperature=T_HOT),
        Stroke("expand", ramp(omega_hot, OMEGA_COLD, TAU_DRIVE), TAU_DRIVE),
    ])


def figure_of_merit(result, cycle):
    return result.figure_of_merit(("cold_iso",), cycle.duration)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    cycle = build()

    # ---- 1. energy ledger ------------------------------------------------
    print("=" * 68)
    print("1. ENERGY LEDGER")
    print("=" * 68)
    result, passes, converged = cycle.limit_cycle(
        qt.thermal_state(qt.qubit_hamiltonian(OMEGA_COLD), T_COLD)
    )
    print(f"\nlimit cycle after {passes} passes (converged={converged})\n")
    print(result.report())

    cop = result.cop(("cold_iso",))
    print(f"\n  COP {cop:.4f}   ideal Otto "
          f"{OMEGA_COLD/(OMEGA_HOT-OMEGA_COLD):.4f}   "
          f"Carnot {T_COLD/(T_HOT-T_COLD):.4f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_cycle(result, cycle, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "cycle_diagram.png", dpi=160)
    plt.close(fig)

    # ---- 2. optimisation -------------------------------------------------
    print()
    print("=" * 68)
    print("2. OPTIMISATION AND ROBUSTNESS")
    print("=" * 68)
    print()

    frequency_sweep = sweep(
        lambda w: build(omega_hot=w), np.linspace(1.1, 2.6, 16),
        figure_of_merit, parameter="omega_hot", metric_name="FOM",
    )
    print(frequency_sweep.report())

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_sweep(frequency_sweep, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "sweep.png", dpi=160)
    plt.close(fig)

    scan = scan_2d(
        lambda w, g: build(omega_hot=w, gamma=g),
        np.linspace(1.1, 2.6, 9), np.logspace(-1.0, 0.6, 9),
        figure_of_merit, parameters=("omega_hot", "coupling"),
        metric_name="FOM",
    )
    print(f"\n  joint optimum      {scan.optimum()[0]:.3f}, "
          f"{scan.optimum()[1]:.3f} -> {scan.optimum()[2]:.4e}")
    print(f"  sequential optimum {scan.sequential_optimum()[0]:.3f}, "
          f"{scan.sequential_optimum()[1]:.3f} -> "
          f"{scan.sequential_optimum()[2]:.4e}")
    print(f"  gap {scan.sequential_gap():+.2f}%  "
          "(sequential tuning is adequate for this system)")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    plot_scan(scan, ax=ax, log_y=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "scan.png", dpi=160)
    plt.close(fig)

    # ---- 3. trajectories -------------------------------------------------
    print()
    print("=" * 68)
    print("3. RELIABILITY")
    print("=" * 68)

    ensemble = qt.unravel(cycle, result.strokes[-1].rho_final,
                          trajectories=2500, steps=1200, seed=5)
    cold_deterministic = result.stroke("cold_iso").heat
    failure = ensemble.probability_of("cold_iso", lambda q: q <= 0)

    print(f"\n  master equation, cold stroke   {cold_deterministic:+.5f}")
    print(f"  trajectory mean                "
          f"{ensemble.series('cold_iso').mean():+.5f} "
          f"+/- {ensemble.standard_error('cold_iso'):.5f}")
    exact = qt.cycle_counting(cycle, {"cold_iso": "quanta"},
                              rho_start=result.strokes[-1].rho_final)
    print(f"\n  P(no heat drawn from cold bath) = {failure:.3f} sampled, "
          f"{exact.probability(lambda n: n <= 0):.4f} exact")
    print("  The machine cools on average and fails on most individual runs.")
    print()
    print("  The distribution is discrete: the qubit either absorbs one quantum")
    print("  from the cold bath, emits one, or exchanges nothing. The mean the")
    print("  master equation reports is a value no single run ever produces.")

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_distribution(ensemble, "cold_iso", cold_deterministic, ax=ax,
                      exact=exact.distribution(4), quantum=OMEGA_COLD)
    fig.tight_layout()
    fig.savefig(FIGURES / "distribution.png", dpi=160)
    plt.close(fig)

    # ---- 4. self-test ----------------------------------------------------
    print()
    print("=" * 68)
    print("4. CONSISTENCY SELF-TEST")
    print("=" * 68)

    jz = qt.jarzynski_tpm(ramp(OMEGA_COLD, OMEGA_HOT, 0.5), 0.5, T_COLD,
                          dim=2, steps=800)
    print(f"\n  <exp(-W/T)> = {jz['exp_average']:.10f}")
    print(f"  exp(-dF/T)  = {jz['predicted']:.10f}")
    print(f"  residual      {jz['residual']:.2e}")
    print(f"  first law     max |dU - (Q+W)| = "
          f"{result.max_first_law_residual:.2e}")

    print()
    print("=" * 68)
    print(f"figures written to {FIGURES}")
    for path in sorted(FIGURES.glob("*.png")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
