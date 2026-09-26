"""Four-stroke quantum Otto refrigerator: the full qthermo workflow.

Run with:  python examples/otto_refrigerator.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qthermo as qt
from qthermo.cycle import Cycle, Stroke

OMEGA_COLD, OMEGA_HOT = 1.0, 1.5
T_COLD, T_HOT = 1.0, 1.3
GAMMA = 1.0
TAU_ISO, TAU_DRIVE = 12.0, 0.02


def build_cycle():
    H_cold = qt.qubit_hamiltonian(OMEGA_COLD)
    H_hot = qt.qubit_hamiltonian(OMEGA_HOT)

    def ramp(a, b, tau):
        return lambda t: qt.qubit_hamiltonian(a + (t / tau) * (b - a))

    return Cycle([
        Stroke("cold_iso", H_cold, TAU_ISO,
               qt.thermal_bath(GAMMA, OMEGA_COLD, T_COLD), temperature=T_COLD),
        Stroke("compress", ramp(OMEGA_COLD, OMEGA_HOT, TAU_DRIVE), TAU_DRIVE),
        Stroke("hot_iso", H_hot, TAU_ISO,
               qt.thermal_bath(GAMMA, OMEGA_HOT, T_HOT), temperature=T_HOT),
        Stroke("expand", ramp(OMEGA_HOT, OMEGA_COLD, TAU_DRIVE), TAU_DRIVE),
    ])


def main():
    cycle = build_cycle()
    rho0 = qt.thermal_state(qt.qubit_hamiltonian(OMEGA_COLD), T_COLD)

    print("=" * 66)
    print("1. DETERMINISTIC LAYER  (Lindblad master equation)")
    print("=" * 66)

    result, passes, converged = cycle.limit_cycle(rho0)
    print(f"limit cycle reached after {passes} cycles (converged={converged})\n")
    print(result.report())

    cop = result.cop(("cold_iso",))
    ideal = OMEGA_COLD / (OMEGA_HOT - OMEGA_COLD)
    carnot = T_COLD / (T_HOT - T_COLD)
    fom = result.figure_of_merit(("cold_iso",), cycle.duration)

    print()
    print(f"  COP                      {cop:.4f}")
    print(f"  ideal Otto COP           {ideal:.4f}   (omega_c / (omega_h - omega_c))")
    print(f"  Carnot bound             {carnot:.4f}")
    print(f"  cooling power            {result.cooling_power(('cold_iso',), cycle.duration):.4e}")
    print(f"  figure of merit          {fom:.4e}")
    print(f"  total entropy production {result.total_entropy_production:.4e}")

    print()
    print("=" * 66)
    print("2. STOCHASTIC LAYER  (quantum-jump trajectories)")
    print("=" * 66)

    rho_limit = result.strokes[-1].rho_final
    ensemble = qt.unravel(cycle, rho_limit, trajectories=3000, steps=1500, seed=11)

    cold_deterministic = result.stroke("cold_iso").heat
    cold_trajectories = ensemble.series("cold_iso")

    print(f"\n  cold stroke, master equation  {cold_deterministic:+.5f}")
    print(f"  cold stroke, trajectory mean  {cold_trajectories.mean():+.5f} "
          f"+/- {ensemble.standard_error('cold_iso'):.5f}")
    print("  (agreement here is the check that the unravelling is faithful)")

    failure = ensemble.probability_of("cold_iso", lambda q: q <= 0)
    exact = qt.cycle_counting(cycle, {"cold_iso": "quanta"}, rho_start=rho_limit)
    exact_failure = exact.probability(lambda n: n <= 0)
    print(f"\n  spread (std dev)              {np.std(cold_trajectories):.4f}")
    print(f"  P(no heat drawn from cold bath) = {failure:.3f} (trajectories), "
          f"{exact_failure:.4f} (exact)")
    print(f"  -- on average this cycle cools. On any individual run it fails to")
    print(f"     extract heat from the cold bath {exact_failure:.0%} of the time. The")
    print("     master equation reports only the mean and cannot show this.")

    counts, edges = ensemble.histogram("cold_iso", bins=12)
    peak = counts.max()
    print("\n  heat distribution:")
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(40 * count / peak)
        print(f"   {lo:+.3f}..{hi:+.3f} | {bar}")

    print()
    print("=" * 66)
    print("3. SELF-TEST  (Jarzynski equality on the compression stroke)")
    print("=" * 66)

    def driven(t):
        return qt.qubit_hamiltonian(OMEGA_COLD + (t / 0.5) * (OMEGA_HOT - OMEGA_COLD))

    jz = qt.jarzynski_tpm(driven, duration=0.5, temperature=T_COLD, dim=2, steps=800)
    print(f"\n  <exp(-W/T)>              {jz['exp_average']:.10f}")
    print(f"  exp(-dF/T)               {jz['predicted']:.10f}")
    print(f"  residual                 {jz['residual']:.2e}")
    print(f"  <W>                      {jz['mean_work']:.6f}")
    print(f"  dF                       {jz['delta_F']:.6f}")
    print(f"  dissipated work <W> - dF {jz['dissipated_work']:.6f}  (>= 0 required)")


if __name__ == "__main__":
    main()
