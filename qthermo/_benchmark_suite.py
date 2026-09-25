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


# --- continuous machines and bath models -----------------------------------

@benchmark("Davies bath: steady state of random 8-level H vs Gibbs",
           "KMS detailed balance => Gibbs fixed point; Spohn, Rep. Math. Phys. 10, 189 (1976)",
           tolerance=1e-10, group="continuous machines")
def _davies_gibbs():
    from .baths import davies_bath
    from .steady import steady_state
    rng = np.random.default_rng(8)
    m = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    H, A = (m + m.conj().T) / 2, (a + a.conj().T) / 2
    rho = steady_state(H, [davies_bath(H, A, 0.9)])
    return np.max(np.abs(rho - thermal_state(H, 0.9))), 0.0


@benchmark("3-qubit absorption fridge COP (local ME, tight coupling)",
           "w_c/w_h; Linden, Popescu & Skrzypczyk, PRL 105, 130401 (2010)",
           tolerance=1e-10, group="continuous machines")
def _lps_cop():
    from .models import absorption_refrigerator
    r = absorption_refrigerator(omega_c=1.0, omega_h=3.0).analyze()
    return r.cop("cold", "hot"), 1.0 / 3.0


@benchmark("3-qubit absorption fridge COP (global ME) below Carnot",
           "(1-T_r/T_h)/(T_r/T_c-1); Correa et al., Sci. Rep. 4, 3949 (2014)",
           kind="upper", tolerance=0.0, group="continuous machines")
def _lps_carnot():
    from .models import absorption_refrigerator
    r = absorption_refrigerator(master_equation="global").analyze()
    return r.cop("cold", "hot"), r.absorption_carnot_cop("cold", "hot", "room")


@benchmark("three-level maser efficiency",
           "1 - w_c/w_h; Scovil & Schulz-DuBois, PRL 2, 262 (1959)",
           tolerance=1e-10, group="continuous machines")
def _ssdb():
    from .models import three_level_maser
    r = three_level_maser(omega_c=1.0, omega_h=3.0).analyze()
    return r.efficiency("hot"), 2.0 / 3.0


@benchmark("local ME, detuned XX qubits: heat current from hot bath",
           "< 0, i.e. cold -> hot; Levy & Kosloff, EPL 107, 20004 (2014)",
           kind="upper", tolerance=0.0, group="continuous machines")
def _levy_kosloff():
    import warnings
    from .models import two_qubit_heat_valve
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = two_qubit_heat_valve(master_equation="local").analyze(energy=None)
    return r.current("hot"), 0.0


@benchmark("same machine, global ME: entropy production rate",
           ">= 0; Spohn's inequality for Davies generators",
           kind="lower", tolerance=0.0, group="continuous machines")
def _levy_kosloff_global():
    from .models import two_qubit_heat_valve
    r = two_qubit_heat_valve(master_equation="global").analyze()
    return r.entropy_production_rate, 0.0


@benchmark("same machine, local ME with bare-H heat: entropy production",
           ">= 0 once boundary work is counted; De Chiara et al., NJP 20, 113024 (2018)",
           kind="lower", tolerance=0.0, group="continuous machines")
def _boundary_work():
    from .models import two_qubit_heat_valve
    r = two_qubit_heat_valve(master_equation="local").analyze()
    return r.entropy_production_rate, 0.0
