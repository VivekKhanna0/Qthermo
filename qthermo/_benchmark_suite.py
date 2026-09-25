"""The benchmark suite run by ``python -m qthermo.benchmarks``.

Grouped by what is being tested. Each entry states the reference it is
compared against, so the printed table doubles as a validation record.
"""

from __future__ import annotations

import numpy as np

from .benchmarks import benchmark
from .channels import qubit_hamiltonian, sigma_x, thermal_bath
from .core import ergotropy, thermal_state
from .cycle import Cycle, Stroke
from .solver import evolve
from .stochastic import jarzynski_tpm


def _ramp(a, b, tau):
    return lambda t: qubit_hamiltonian(a + (t / tau) * (b - a))


def _otto(omega_c, omega_h, T_c, T_h, tau_iso, gamma=1.0, tau_drive=0.02,
          engine=False):
    strokes = [
        Stroke("cold_iso", qubit_hamiltonian(omega_c), tau_iso,
               thermal_bath(gamma, omega_c, T_c), temperature=T_c),
        Stroke("compress", _ramp(omega_c, omega_h, tau_drive), tau_drive),
        Stroke("hot_iso", qubit_hamiltonian(omega_h), tau_iso,
               thermal_bath(gamma, omega_h, T_h), temperature=T_h),
        Stroke("expand", _ramp(omega_h, omega_c, tau_drive), tau_drive),
    ]
    return Cycle(strokes)


# --- stroke-based machines --------------------------------------------------

@benchmark("single-qubit Otto fridge COP, full thermalisation",
           "w_c/(w_h - w_c); Kosloff & Rezek, Entropy 19, 136 (2017)",
           tolerance=2e-3, relative=True, group="stroke machines")
def _otto_cop():
    cycle = _otto(1.0, 1.5, 1.0, 1.3, tau_iso=40.0)
    result, _, _ = cycle.limit_cycle(thermal_state(qubit_hamiltonian(1.0), 1.0))
    return result.cop(("cold_iso",)), 1.0 / 0.5


@benchmark("single-qubit Otto engine efficiency, full thermalisation",
           "1 - w_c/w_h; Kosloff & Rezek, Entropy 19, 136 (2017)",
           tolerance=2e-3, relative=True, group="stroke machines")
def _otto_efficiency():
    omega_l, omega_h = 1.0, 2.5
    cycle = _otto(omega_l, omega_h, 0.5, 4.0, tau_iso=25.0)
    result, _, _ = cycle.limit_cycle(thermal_state(qubit_hamiltonian(omega_l), 0.5))
    return -result.net_work / result.heat_from("hot_iso"), 1 - omega_l / omega_h


@benchmark("Otto cycle first-law residual max|dU - Q - W|",
           "exact identity of the midpoint Alicki split (should be ~1e-16)",
           tolerance=1e-12, group="stroke machines")
def _first_law():
    cycle = _otto(1.0, 1.5, 1.0, 1.3, tau_iso=8.0)
    result, _, _ = cycle.limit_cycle(thermal_state(qubit_hamiltonian(1.0), 1.0))
    return result.max_first_law_residual, 0.0


# --- equilibrium and fluctuation identities --------------------------------

@benchmark("thermal bath relaxes qubit to Gibbs state, max|rho - rho_G|",
           "detailed balance => unique Gibbs fixed point (Davies 1974)",
           tolerance=1e-6, group="identities")
def _gibbs_fixed_point():
    H = qubit_hamiltonian(1.0)
    out = evolve(thermal_state(H, 5.0), H, thermal_bath(1.0, 1.0, 0.7),
                 duration=60.0)
    return np.max(np.abs(out["states"][-1] - thermal_state(H, 0.7))), 0.0


@benchmark("Jarzynski <exp(-W/T)> / exp(-dF/T), driven qubit (TPM)",
           "= 1; Jarzynski PRL 78, 2690 (1997); Tasaki cond-mat/0009244",
           tolerance=1e-8, group="identities")
def _jarzynski():
    def H_of_t(t):
        return qubit_hamiltonian(1.0 + 2.0 * t) + 0.7 * sigma_x
    out = jarzynski_tpm(H_of_t, duration=1.0, temperature=0.8, dim=2, steps=600)
    return out["exp_average"] / out["predicted"], 1.0


@benchmark("ergotropy of a fully inverted qubit",
           "= w; Allahverdyan, Balian & Nieuwenhuizen, EPL 67, 565 (2004)",
           tolerance=1e-12, group="identities")
def _ergotropy_inverted():
    H = qubit_hamiltonian(1.7)
    return ergotropy(np.diag([0.0, 1.0]).astype(complex), H), 1.7
