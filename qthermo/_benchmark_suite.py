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


# --- fluctuations -----------------------------------------------------------

def _qubit_two_baths():
    from .baths import davies_bath
    H = qubit_hamiltonian(1.0)
    return H, [davies_bath(H, sigma_x, 3.0, gamma=0.3, name="hot"),
               davies_bath(H, sigma_x, 0.5, gamma=0.7, name="cold")]


@benchmark("qubit between two baths: current noise D, exact vs classical FCS",
           "two-state jump process, tilted-generator eigenvalue (independent code)",
           tolerance=1e-6, relative=True, group="fluctuations")
def _fcs_classical():
    from .fluctuations import current_statistics
    H, baths = _qubit_two_baths()
    n = lambda T: 1 / np.expm1(1 / T)
    up_h, dn_h, up_c, dn_c = 0.3 * n(3), 0.3 * (1 + n(3)), 0.7 * n(.5), 0.7 * (1 + n(.5))

    def theta(s):
        M = np.array([[-(up_h + up_c), dn_h * np.exp(-s) + dn_c],
                      [up_h * np.exp(s) + up_c, -(dn_h + dn_c)]])
        return np.max(np.linalg.eigvals(M).real)
    h = 1e-4
    D = (theta(h) - 2 * theta(0) + theta(-h)) / h ** 2
    return current_statistics(H, {"hot": "quanta"}, baths=baths).noise, D


@benchmark("TUR ratio, classical two-bath qubit",
           ">= 2; Barato & Seifert, PRL 114, 158101 (2015)",
           kind="lower", tolerance=0.0, group="fluctuations")
def _tur_classical():
    from .fluctuations import current_statistics
    H, baths = _qubit_two_baths()
    return current_statistics(H, {"hot": "energy"}, baths=baths).tur_ratio, 2.0


@benchmark("TUR ratio, three-level maser power (coherent drive)",
           "< 2 possible; Kalaee, Wacker & Potts, PRE 104, L012103 (2021)",
           kind="upper", tolerance=0.0, group="fluctuations")
def _tur_maser():
    from .fluctuations import current_statistics
    from .models import three_level_maser
    m = three_level_maser(1.0, 3.0, T_c=1.0, T_h=200.0, drive=0.05,
                          gamma_c=0.05, gamma_h=0.001)
    return current_statistics(m, {"hot": "energy", "cold": "energy"}).tur_ratio, 2.0


# --- strong coupling ----------------------------------------------------------

@benchmark("mean-force state: weak-coupling correction scaling d(2lam)/d(lam)",
           "= 4: O(lam^2) deviation from Gibbs; Cresser & Anders, PRL 127, 250601 (2021)",
           tolerance=0.02, relative=True, group="strong coupling")
def _mfgs_weak():
    from .steady import trace_distance
    from .strong_coupling import mean_force_state
    H, G = qubit_hamiltonian(1.0), thermal_state(qubit_hamiltonian(1.0), 0.5)
    d1 = trace_distance(mean_force_state(H, sigma_x, 0.5, 0.01, 2.0, 40), G)
    d2 = trace_distance(mean_force_state(H, sigma_x, 0.5, 0.02, 2.0, 40), G)
    return d2 / d1, 4.0


@benchmark("mean-force state at lam=4 vs ultrastrong limit (trace distance)",
           "-> 0; Cresser & Anders, PRL 127, 250601 (2021)",
           kind="upper", tolerance=0.02, group="strong coupling")
def _mfgs_ultrastrong():
    from .steady import trace_distance
    from .strong_coupling import mean_force_state, ultrastrong_limit_state
    H = qubit_hamiltonian(1.0)
    return trace_distance(mean_force_state(H, sigma_x, 0.5, 4.0, 2.0, 60),
                          ultrastrong_limit_state(H, sigma_x, 0.5)), 0.0


@benchmark("reaction-coordinate model: steady state vs Gibbs of H_ext",
           "Davies fixed point of the mapped model; Strasberg et al., NJP 18, 073007 (2016)",
           tolerance=1e-10, group="strong coupling")
def _rc_gibbs():
    from .strong_coupling import reaction_coordinate_model
    m = reaction_coordinate_model(qubit_hamiltonian(1.0), sigma_x, 0.5, lam=0.5,
                                  Omega=2.0, n_levels=10)
    return np.max(np.abs(m.analyze().rho - thermal_state(m.H, 0.5))), 0.0


# --- information thermodynamics ----------------------------------------------

@benchmark("slow erasure of one bit: heat released / (T ln 2)",
           "-> 1; Landauer, IBM J. Res. Dev. 5, 183 (1961); Reeb & Wolf, NJP 16, 103011 (2014)",
           tolerance=0.01, relative=True, slow=True, group="information")
def _landauer_slow():
    from .information import landauer_erasure
    r = landauer_erasure(200.0, temperature=1.0, schedule="geodesic")
    return r.heat_to_bath / np.log(2), 1.0


@benchmark("finite-time erasure: excess heat x tau, linear ramp",
           "slow-driving prediction int zeta w'^2; Sivak & Crooks, PRL 108, 190602 (2012)",
           tolerance=0.01, relative=True, slow=True, group="information")
def _landauer_linear():
    from .information import landauer_erasure, predicted_excess
    return (landauer_erasure(160.0).excess_heat * 160.0,
            predicted_excess("linear", 1.0))


@benchmark("geodesic erasure protocol: excess heat x tau",
           "= L^2 (thermodynamic length); Scandi & Perarnau-Llobet, Quantum 3, 197 (2019)",
           tolerance=0.01, relative=True, slow=True, group="information")
def _landauer_geodesic():
    from .information import landauer_erasure, thermodynamic_length
    return (landauer_erasure(160.0, schedule="geodesic").excess_heat * 160.0,
            thermodynamic_length() ** 2)


@benchmark("fast erasure (tau=1): heat released vs Landauer bound",
           ">= T (S_i - S_f) at any speed (Clausius)",
           kind="lower", tolerance=0.0, group="information")
def _landauer_fast():
    from .information import landauer_erasure
    r = landauer_erasure(1.0, steps=800)
    return r.heat_to_bath, r.landauer


# --- interacting working media -------------------------------------------------

def _heisenberg(B, J):
    from .subsystems import embed
    from .channels import sigma_y, sigma_z
    D = [2, 2]
    return (-0.5 * B * (embed(sigma_z, 0, D) + embed(sigma_z, 1, D))
            + J * sum(embed(s, 0, D) @ embed(s, 1, D) for s in (sigma_x, sigma_y, sigma_z)))


@benchmark("2-qubit Heisenberg Otto: simulated vs quasi-static efficiency",
           "exact population formula (ideal_otto); coupling-enhanced, Thomas & Johal PRE 83, 031135 (2011)",
           tolerance=1e-6, group="interacting media")
def _heisenberg_otto():
    from .engines import ideal_otto, otto_cycle
    from .subsystems import embed
    D = [2, 2]
    cycle = otto_cycle(_heisenberg(2, 0.2), _heisenberg(4, 0.2), 0.5, 4.0,
                       [embed(sigma_x, 0, D), embed(sigma_x, 1, D)],
                       tau_iso=30, tau_ramp=1.0)
    result, _, _ = cycle.limit_cycle(np.eye(4) / 4)
    return (result.efficiency(("hot_iso",)),
            ideal_otto(_heisenberg(2, 0.2), _heisenberg(4, 0.2), 0.5, 4.0).efficiency)


@benchmark("2-qubit Heisenberg Otto efficiency vs uncoupled 1 - B_c/B_h",
           "> 0.5 for antiferromagnetic J; Thomas & Johal, PRE 83, 031135 (2011)",
           kind="lower", tolerance=0.0, group="interacting media")
def _heisenberg_enhancement():
    from .engines import ideal_otto
    return ideal_otto(_heisenberg(2, 0.2), _heisenberg(4, 0.2), 0.5, 4.0).efficiency, 0.5


@benchmark("Gallavotti-Cohen symmetry of heat FCS, 3-qubit chain: max|theta(s)-theta(-s-A)|",
           "= 0 exactly; Gallavotti & Cohen PRL 74, 2694 (1995); Esposito, Harbola & Mukamel RMP 81, 1665 (2009)",
           tolerance=1e-12, group="fluctuations")
def _gallavotti_cohen():
    from .fluctuations import scaled_cgf
    from .models import spin_chain
    m = spin_chain(3, T_left=2.0, T_right=0.5, master_equation="global")
    s = np.linspace(-1.0, 0.9, 7)
    A = 1 / 0.5 - 1 / 2.0
    return np.max(np.abs(scaled_cgf(m, {"left": "energy"}, s)
                         - scaled_cgf(m, {"left": "energy"}, -s - A))), 0.0


@benchmark("exact single-cycle counting: P(1 emission) of a decaying qubit",
           "= 1 - exp(-gamma tau) (exponential decay)",
           tolerance=1e-12, group="fluctuations")
def _counting_decay():
    from .channels import amplitude_damping
    from .counting import cycle_counting
    cycle = Cycle([Stroke("decay", qubit_hamiltonian(1.0), 1.3, amplitude_damping(0.7))])
    dist = cycle_counting(cycle, {"decay": "quanta"},
                          rho_start=np.diag([0.0, 1.0]).astype(complex)).distribution(4)
    return dist[-1], 1 - np.exp(-0.7 * 1.3)


# --- quantum batteries ----------------------------------------------------------

@benchmark("Dicke battery: large-N exponent of collective power advantage",
           "= 1/2 (sqrt N); Ferraro et al., PRL 120, 117702 (2018)",
           tolerance=0.05, slow=True, group="batteries")
def _dicke_sqrt_n():
    from .batteries import collective_advantage
    return collective_advantage([1, 2, 4, 6, 8, 10])["large_N_exponent"], 0.5


@benchmark("Bell pair of cells: ergotropy locked away from local unitaries",
           "= w (all of it); Alicki & Fannes, PRE 87, 042123 (2013)",
           tolerance=1e-12, group="batteries")
def _locked_bell():
    from .batteries import locked_ergotropy
    bell = np.zeros((4, 4), dtype=complex)
    bell[0, 0] = bell[0, 3] = bell[3, 0] = bell[3, 3] = 0.5
    H = qubit_hamiltonian(1.0)
    return locked_ergotropy(bell, [2, 2], [H, H])["locked"], 1.0


@benchmark("coherent ergotropy of |+> (qubit gap w = 1)",
           "= w/2, all in coherence; Francica et al., PRL 125, 180603 (2020)",
           tolerance=1e-12, group="batteries")
def _coherent_plus():
    from .batteries import ergotropy_split
    return ergotropy_split(np.full((2, 2), 0.5, dtype=complex),
                           qubit_hamiltonian(1.0))["coherent"], 0.5


@benchmark("general friction metric vs erasure closed form, omega = 1",
           "linear-response friction beta Var/Gamma; Sivak & Crooks, PRL 108, 190602 (2012)",
           tolerance=1e-6, relative=True, group="information")
def _friction_general():
    from .baths import davies_bath, ohmic_spectrum
    from .geometry import friction
    from .information import erasure_friction
    bath = lambda H: [davies_bath(H, sigma_x, 1.0, spectrum=ohmic_spectrum(1.0, reference=1.0))]
    return friction(qubit_hamiltonian, bath, 1.0, 1.0), erasure_friction(1.0, 1.0, 1.0)


@benchmark("optimal schedule, non-commuting qubit drive: simulated / predicted L^2",
           "-> 1 in the slow limit; Scandi & Perarnau-Llobet, Quantum 3, 197 (2019)",
           tolerance=0.02, relative=True, slow=True, group="information")
def _geodesic_general():
    from .baths import davies_bath, ohmic_spectrum
    from .geometry import excess_work, optimal_schedule
    T = 0.7
    H_of = lambda th: qubit_hamiltonian(1.0 + th) + 0.8 * th * sigma_x
    from .channels import sigma_z
    baths_of = lambda H: [davies_bath(H, sigma_x + sigma_z, T,
                                      spectrum=ohmic_spectrum(0.5, reference=1.0))]
    sched = optimal_schedule(H_of, baths_of, T, (0.0, 2.0), points=61)
    tau = 120.0
    sim = excess_work(sched.hamiltonian(H_of, tau), baths_of, T, tau, steps=1200)["dissipation"]
    return sim / sched.minimum_excess(tau), 1.0


# --- periodically driven machines ---------------------------------------------

def _floquet_machine(T_h, T_c, lam=0.6):
    from .floquet import DrivenBath, floquet_analyze, window_spectrum
    w0, om = 3.0, 1.0
    H = lambda t: qubit_hamiltonian(w0 + lam * np.sin(om * t))
    return floquet_analyze(H, 2 * np.pi / om, [
        DrivenBath("hot", sigma_x, T_h, spectrum=window_spectrum(0.05, w0 + om, 0.8)),
        DrivenBath("cold", sigma_x, T_c, spectrum=window_spectrum(0.05, w0 - om, 0.8))],
        n_time=128)


@benchmark("modulated qubit engine (Floquet): efficiency",
           "1 - (w0-W)/(w0+W) by tight coupling; Gelbwaser-Klimovsky, Alicki & Kurizki, PRE 87, 012140 (2013)",
           tolerance=1e-6, group="driven machines")
def _floquet_engine():
    return _floquet_machine(4.0, 0.5).efficiency("hot"), 0.5


@benchmark("modulated qubit refrigerator (Floquet): COP",
           "(w0-W)/(2W) by tight coupling; same reference",
           tolerance=1e-5, group="driven machines")
def _floquet_fridge():
    return _floquet_machine(1.2, 1.0).cop("cold"), 1.0


@benchmark("undriven limit of the Floquet master equation: |J_floquet - J_davies|",
           "= 0: reduces to the static Davies generator",
           tolerance=1e-12, group="driven machines")
def _floquet_static():
    from .baths import davies_bath
    from .floquet import DrivenBath, floquet_analyze
    from .steady import analyze
    H = qubit_hamiltonian(1.3)
    f = floquet_analyze(lambda t: H, 2 * np.pi / 5.0,
                        [DrivenBath("hot", sigma_x, 2.0, gamma=0.3),
                         DrivenBath("cold", sigma_x, 0.5, gamma=0.7)], n_time=64)
    s = analyze(H, [davies_bath(H, sigma_x, 2.0, gamma=0.3, name="hot"),
                    davies_bath(H, sigma_x, 0.5, gamma=0.7, name="cold")])
    return abs(f.current("hot") - s.current("hot")), 0.0


@benchmark("boundary-driven XX chain: J(N=6) / J(N=2)",
           "= 1 exactly (ballistic transport); cf. Karevski & Platini, PRL 102, 207207 (2009)",
           tolerance=1e-10, group="continuous machines")
def _xx_ballistic():
    from .models import spin_chain
    J = [spin_chain(n, J=0.5, delta=0.0, T_left=5.0, T_right=0.5, gamma=0.5,
                    master_equation="local").analyze().current("left") for n in (2, 6)]
    return J[1] / J[0], 1.0


@benchmark("absorption fridge switched on (g/gamma = 25): min T*_cold - steady T*_cold",
           "< 0: transient 'single-shot' cooling; Mitchison et al., NJP 17, 115013 (2015)",
           kind="upper", tolerance=0.0, group="continuous machines")
def _single_shot():
    from .models import absorption_refrigerator
    from .network import heat_flow_map
    from .transient import product_thermal_state, transient
    m = absorption_refrigerator(g=0.05, gamma=0.002, T_c=1.0, T_h=6.0, T_r=1.5)
    run = transient(m, product_thermal_state(m.local_H, [1.0, 6.0, 1.5]),
                    duration=4000.0, steps=800)
    return run.minimum_temperature(0)[0] - heat_flow_map(m.analyze()).virtual_temperature[0], 0.0
