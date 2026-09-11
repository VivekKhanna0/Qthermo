"""Comparing noise channels through their thermodynamic signature.

Different noise channels leave different thermodynamic fingerprints, and the
difference is structural rather than incidental:

  * amplitude damping moves population down the energy ladder, so it exchanges
    energy with the bath and shows up in the heat;
  * pure dephasing destroys coherence while leaving populations untouched, so
    it exchanges no energy at all and is invisible to any heat-based
    observable -- but it still produces entropy;
  * bit flip mixes populations symmetrically, behaving like an
    infinite-temperature bath.

Run with:  python examples/channel_comparison.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qthermo as qt

OMEGA = 1.0
GAMMA = 0.4
DURATION = 4.0


def main():
    H = qt.qubit_hamiltonian(OMEGA)

    # A state with both excited population and coherence, so that every
    # channel has something it could act on.
    rho0 = np.array([[0.35, 0.35], [0.35, 0.65]], dtype=complex)
    rho0 = rho0 / np.trace(rho0)

    channels = {
        "amplitude damping": qt.amplitude_damping(GAMMA),
        "pure dephasing": qt.pure_dephasing(GAMMA),
        "bit flip": qt.bit_flip(GAMMA),
        "thermal bath (T=0.8)": qt.thermal_bath(GAMMA, OMEGA, 0.8),
    }

    header = f"{'channel':<24}{'Q':>12}{'dS':>12}{'|coherence|':>14}{'dP(excited)':>14}"
    print(header)
    print("-" * len(header))

    for name, c_ops in channels.items():
        out = qt.evolve(rho0, H, c_ops, duration=DURATION, steps=400)
        rho_final = out["states"][-1]

        delta_s = (qt.von_neumann_entropy(rho_final)
                   - qt.von_neumann_entropy(rho0))
        coherence = abs(rho_final[0, 1])
        delta_population = float(np.real(rho_final[1, 1] - rho0[1, 1]))

        print(f"{name:<24}{out['heat']:>12.5f}{delta_s:>12.5f}"
              f"{coherence:>14.5f}{delta_population:>14.5f}")

    print()
    print("Pure dephasing is the interesting row: zero heat to numerical")
    print("precision, coherence driven to zero, populations unchanged. Any")
    print("diagnostic built on heat alone is structurally blind to it, which is")
    print("why qthermo reports entropy production alongside heat rather than")
    print("treating heat as the whole thermodynamic story.")


if __name__ == "__main__":
    main()
