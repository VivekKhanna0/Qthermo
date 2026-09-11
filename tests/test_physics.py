"""Validation against analytically known results.

These are not unit tests of the code's internal consistency: each one checks a
number that thermodynamics fixes independently of the implementation, so a
failure means the physics is wrong, not that a refactor changed an interface.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qthermo as qt
from qthermo.cycle import Cycle, Stroke


def otto_refrigerator(omega_cold=1.0, omega_hot=1.5, T_cold=1.0, T_hot=1.3,
                      gamma=1.0, tau_iso=8.0, tau_drive=0.02):
    """Standard four-stroke Otto refrigerator on a single qubit."""
    H_cold = qt.qubit_hamiltonian(omega_cold)
    H_hot = qt.qubit_hamiltonian(omega_hot)

    def ramp_up(t):
        s = t / tau_drive
        return qt.qubit_hamiltonian(omega_cold + s * (omega_hot - omega_cold))

    def ramp_down(t):
        s = t / tau_drive
        return qt.qubit_hamiltonian(omega_hot + s * (omega_cold - omega_hot))

    return Cycle([
        Stroke("cold_iso", H_cold, tau_iso,
               qt.thermal_bath(gamma, omega_cold, T_cold), temperature=T_cold),
        Stroke("compress", ramp_up, tau_drive),
        Stroke("hot_iso", H_hot, tau_iso,
               qt.thermal_bath(gamma, omega_hot, T_hot), temperature=T_hot),
        Stroke("expand", ramp_down, tau_drive),
    ])


def test_first_law():
    """dU = Q + W must hold on every stroke to machine precision."""
    cycle = otto_refrigerator()
    result, _, _ = cycle.limit_cycle(qt.thermal_state(qt.qubit_hamiltonian(1.0), 1.0))
    assert result.max_first_law_residual < 1e-9, result.max_first_law_residual


def test_thermal_fixed_point():
    """A qubit left in contact with one bath must relax to the Gibbs state."""
    omega, T = 1.0, 0.7
    H = qt.qubit_hamiltonian(omega)
    out = qt.evolve(qt.thermal_state(H, 5.0), H,
                    qt.thermal_bath(1.0, omega, T), duration=60.0)
    expected = qt.thermal_state(H, T)
    assert np.max(np.abs(out["states"][-1] - expected)) < 1e-6


def test_entropy_production_non_negative():
    """Spohn's theorem: entropy production is non-negative for each stroke."""
    cycle = otto_refrigerator()
    result, _, _ = cycle.limit_cycle(qt.thermal_state(qt.qubit_hamiltonian(1.0), 1.0))
    for stroke in result.strokes:
        if stroke.entropy_production is not None:
            assert stroke.entropy_production > -1e-9, (stroke.name, stroke.entropy_production)


def test_otto_cop_approaches_ideal():
    """With full thermalisation the Otto COP tends to omega_c / (omega_h - omega_c)."""
    omega_cold, omega_hot = 1.0, 1.5
    cycle = otto_refrigerator(omega_cold, omega_hot, tau_iso=40.0, gamma=1.0)
    result, _, _ = cycle.limit_cycle(qt.thermal_state(qt.qubit_hamiltonian(omega_cold), 1.0))
    cop = result.cop(("cold_iso",))
    ideal = omega_cold / (omega_hot - omega_cold)
    assert abs(cop - ideal) / ideal < 0.02, (cop, ideal)


def test_cop_below_carnot():
    """The refrigerator must not beat the Carnot bound."""
    T_cold, T_hot = 1.0, 1.3
    cycle = otto_refrigerator(T_cold=T_cold, T_hot=T_hot, tau_iso=40.0)
    result, _, _ = cycle.limit_cycle(qt.thermal_state(qt.qubit_hamiltonian(1.0), 1.0))
    carnot = T_cold / (T_hot - T_cold)
    assert result.cop(("cold_iso",)) < carnot


def test_dephasing_produces_no_heat():
    """Pure dephasing commutes with a diagonal H, so it exchanges no energy."""
    omega = 1.0
    H = qt.qubit_hamiltonian(omega)
    rho0 = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)  # coherent superposition
    out = qt.evolve(rho0, H, qt.pure_dephasing(1.0), duration=5.0)
    assert abs(out["heat"]) < 1e-9, out["heat"]
    # but it does destroy coherence
    assert abs(out["states"][-1][0, 1]) < 1e-2


def test_amplitude_damping_produces_heat():
    """Contrast: relaxation does exchange energy."""
    omega = 1.0
    H = qt.qubit_hamiltonian(omega)
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)  # excited state
    out = qt.evolve(rho0, H, qt.amplitude_damping(1.0), duration=10.0)
    assert out["heat"] < -0.9 * omega, out["heat"]


def test_jarzynski_equality():
    """<exp(-W/T)> = exp(-dF/T) for an arbitrary unitary drive."""
    def H_of_t(t):
        omega = 1.0 + 2.0 * (t / 1.0)
        return qt.qubit_hamiltonian(omega) + 0.7 * qt.sigma_x

    out = qt.jarzynski_tpm(H_of_t, duration=1.0, temperature=0.8, dim=2, steps=600)
    assert out["residual"] < 1e-6, out


def test_jarzynski_second_law():
    """<W> >= dF: the dissipated work is non-negative."""
    def H_of_t(t):
        return qt.qubit_hamiltonian(1.0 + 3.0 * t) + 0.5 * qt.sigma_x

    out = qt.jarzynski_tpm(H_of_t, duration=0.3, temperature=0.6, dim=2)
    assert out["dissipated_work"] > -1e-9, out


def test_trajectories_reproduce_master_equation():
    """Averaging quantum-jump trajectories must reproduce the Lindblad heat."""
    cycle = otto_refrigerator(tau_iso=6.0, gamma=1.0)
    rho0 = qt.thermal_state(qt.qubit_hamiltonian(1.0), 1.0)
    deterministic, _, _ = cycle.limit_cycle(rho0)
    rho_start = deterministic.strokes[-1].rho_final

    ensemble = qt.unravel(cycle, rho_start, trajectories=4000, steps=1200, seed=7)
    deterministic_total = sum(s.heat for s in deterministic.strokes)

    error = ensemble.standard_error("heat")
    assert abs(ensemble.mean_heat - deterministic_total) < 5 * error + 0.02, (
        ensemble.mean_heat, deterministic_total, error
    )


def test_heat_engine_efficiency():
    """The same primitives, run as an engine, hit the ideal Otto efficiency."""
    omega_low, omega_high = 1.0, 2.5
    T_cold, T_hot = 0.5, 4.0
    ramp = lambda a, b, tau: (lambda t: qt.qubit_hamiltonian(a + (t/tau)*(b-a)))
    cycle = Cycle([
        Stroke("hot_iso", qt.qubit_hamiltonian(omega_high), 25.0,
               qt.thermal_bath(1.0, omega_high, T_hot), temperature=T_hot),
        Stroke("expand", ramp(omega_high, omega_low, 0.02), 0.02),
        Stroke("cold_iso", qt.qubit_hamiltonian(omega_low), 25.0,
               qt.thermal_bath(1.0, omega_low, T_cold), temperature=T_cold),
        Stroke("compress", ramp(omega_low, omega_high, 0.02), 0.02),
    ])
    result, _, _ = cycle.limit_cycle(
        qt.thermal_state(qt.qubit_hamiltonian(omega_high), T_hot))
    efficiency = result.efficiency(("hot_iso",))
    assert abs(efficiency - (1 - omega_low/omega_high)) < 0.01, efficiency
    assert efficiency < 1 - T_cold/T_hot


def test_invalid_inputs_raise():
    """Bad input must fail loudly rather than return a plausible number."""
    H = qt.qubit_hamiltonian(1.0)
    good = qt.thermal_state(H, 1.0)
    bad_cases = [
        lambda: qt.evolve(good, np.array([[0, 1], [0, 0]], dtype=complex), [], 1.0),
        lambda: qt.evolve(2 * good, H, [], 1.0),
        lambda: qt.evolve(good, H, [], -1.0),
        lambda: qt.evolve(good, H, [np.eye(3)], 1.0),
        lambda: Stroke("s", H, 1.0, qt.amplitude_damping(1.0), temperature=0.0),
        lambda: Stroke("s", H, 1.0, [], temperature=1.0),
    ]
    for index, case in enumerate(bad_cases):
        try:
            case()
        except qt.QThermoError:
            continue
        raise AssertionError(f"case {index} did not raise QThermoError")


def test_sweep_finds_optimum():
    """A sweep must locate a genuine interior maximum of the figure of merit."""
    from qthermo.analysis import sweep

    def build(omega_hot):
        ramp = lambda a, b, tau: (lambda t: qt.qubit_hamiltonian(a + (t/tau)*(b-a)))
        return Cycle([
            Stroke("cold_iso", qt.qubit_hamiltonian(1.0), 12.0,
                   qt.thermal_bath(1.0, 1.0, 1.0), temperature=1.0),
            Stroke("compress", ramp(1.0, omega_hot, 0.02), 0.02),
            Stroke("hot_iso", qt.qubit_hamiltonian(omega_hot), 12.0,
                   qt.thermal_bath(1.0, omega_hot, 1.3), temperature=1.3),
            Stroke("expand", ramp(omega_hot, 1.0, 0.02), 0.02),
        ])

    out = sweep(build, np.linspace(1.1, 2.6, 10),
                lambda r, c: r.figure_of_merit(("cold_iso",), c.duration),
                parameter="omega_hot", metric_name="FOM")
    best_value, best_metric = out.optimum()
    assert np.isfinite(best_metric) and best_metric > 0
    # the optimum should be interior, not at a boundary
    assert out.values[0] < best_value < out.values[-1], best_value


def test_negative_entropy_production_warns():
    """A temperature inconsistent with the channel must warn, not pass silently."""
    import warnings

    rho_initial = qt.thermal_state(qt.qubit_hamiltonian(1.0), 2.0)
    rho_final = np.array([[1, 0], [0, 0]], dtype=complex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qt.entropy_production(rho_initial, rho_final, heat=0.5, temperature=1.0)
    assert any(issubclass(x.category, RuntimeWarning) for x in caught)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print("all passed" if failures == 0 else f"{failures} failing")
