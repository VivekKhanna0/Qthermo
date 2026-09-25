"""Landauer erasure, finite-time cost, geodesic protocols, bath-resolved strokes."""

import numpy as np
import pytest

import qthermo as qt
from qthermo.baths import davies_bath, ohmic_spectrum
from qthermo.information import (
    _erasure_bath,
    geodesic_schedule,
    landauer_erasure,
    predicted_excess,
    thermodynamic_length,
)


def test_erasure_bath_is_the_davies_bath():
    for omega in (0.01, 0.7, 5.0):
        fast = _erasure_bath(lambda t: omega, 0.8, 1.3)(0.0)[0].c_ops
        slow = davies_bath(qt.qubit_hamiltonian(omega), qt.sigma_x, 0.8,
                           spectrum=ohmic_spectrum(1.3, reference=0.8)).c_ops
        assert all(np.allclose(a, b) for a, b in zip(fast, slow))


@pytest.mark.parametrize("schedule", ["linear", "smooth", "exponential", "geodesic"])
def test_landauer_bound_holds_at_every_speed(schedule):
    for tau in (1.0, 10.0):
        r = landauer_erasure(tau, schedule=schedule, steps=800)
        assert r.heat_to_bath >= r.landauer - 1e-12
        assert r.entropy_production >= -1e-12


def test_slow_erasure_approaches_T_ln_2():
    for T in (1.0, 0.4):
        r = landauer_erasure(200.0, temperature=T, schedule="geodesic")
        assert r.error_probability < 3e-5   # e^-12 plus finite-time lag
        assert np.isclose(r.heat_to_bath, T * np.log(2), rtol=0.01)


def test_excess_dissipation_decays_as_one_over_tau():
    ex = [landauer_erasure(tau).excess_heat * tau for tau in (80.0, 160.0)]
    assert np.isclose(ex[0], ex[1], rtol=0.01)
    assert np.isclose(ex[1], predicted_excess("linear", 1.0), rtol=0.01)


def test_geodesic_protocol_attains_the_thermodynamic_length_bound():
    tau = 160.0
    length_squared = thermodynamic_length() ** 2
    geodesic = landauer_erasure(tau, schedule="geodesic").excess_heat
    assert np.isclose(geodesic * tau, length_squared, rtol=0.01)
    for other in ("linear", "smooth", "exponential"):
        assert landauer_erasure(tau, schedule=other).excess_heat > geodesic


def test_thermodynamic_length_scales_with_temperature():
    # zeta depends on omega/T only up to a 1/T prefactor, so L^2 is linear in T
    # when the ramp end points scale with T.
    assert np.isclose(thermodynamic_length(0.5, 5e-4, 6.0) ** 2,
                      0.5 * thermodynamic_length(1.0, 1e-3, 12.0) ** 2, rtol=1e-6)


def test_geodesic_schedule_is_monotone_with_fixed_end_points():
    shape = geodesic_schedule()
    s = np.linspace(0, 1, 101)
    values = np.array([shape(x) for x in s])
    assert np.isclose(values[0], 0.0) and np.isclose(values[-1], 1.0)
    assert np.all(np.diff(values) > 0)


def test_bad_schedule_rejected():
    with pytest.raises(qt.QThermoError):
        landauer_erasure(1.0, schedule=lambda s: 0.5 * s)
    with pytest.raises(qt.QThermoError):
        landauer_erasure(1.0, schedule="nope")


# --- strokes with Bath objects ------------------------------------------------

def two_bath_stroke(T_h=3.0, T_c=0.5):
    H = qt.qubit_hamiltonian(1.0)
    return qt.Stroke("contact", H, 6.0, [
        davies_bath(H, qt.sigma_x, T_h, gamma=0.4, name="hot"),
        davies_bath(H, qt.sigma_x, T_c, gamma=0.6, name="cold"),
    ], steps=600)


def test_multi_bath_stroke_attributes_heat_to_each_bath():
    stroke = two_bath_stroke()
    result = qt.Cycle([stroke]).run(np.eye(2) / 2).strokes[0]
    assert set(result.heat_by_bath) == {"hot", "cold"}
    assert np.isclose(sum(result.heat_by_bath.values()), result.heat, atol=1e-12)
    assert result.first_law_residual < 1e-12
    assert result.entropy_production > 0


def test_multi_bath_stroke_steady_currents_match_steady_state_solver():
    # Start in the steady state: heat per bath over the stroke must equal
    # current x duration.
    stroke = two_bath_stroke()
    baths = stroke.c_ops
    rho_ss = qt.steady_state(stroke.H, baths)
    result = qt.Cycle([stroke]).run(rho_ss).strokes[0]
    currents = qt.analyze(stroke.H, baths).currents
    for name in ("hot", "cold"):
        assert np.isclose(result.heat_by_bath[name], currents[name] * 6.0, rtol=1e-6)


def test_stroke_rejects_mixed_or_doubly_specified_baths():
    H = qt.qubit_hamiltonian(1.0)
    bath = davies_bath(H, qt.sigma_x, 1.0)
    with pytest.raises(qt.QThermoError, match="mixes"):
        qt.Stroke("s", H, 1.0, [bath, qt.sigma_minus])
    with pytest.raises(qt.QThermoError, match="own temperatures"):
        qt.Stroke("s", H, 1.0, [bath], temperature=1.0)


# --- measurement and feedback -----------------------------------------------------

def test_szilard_engine_extracts_T_ln_2_from_a_perfect_measurement():
    r = qt.szilard_engine(temperature=0.8)
    assert np.isclose(r.information, np.log(2))
    assert np.isclose(r.work_extracted, 0.8 * np.log(2), rtol=1e-8)


@pytest.mark.parametrize("error", [0.02, 0.1, 0.3])
def test_optimal_feedback_saturates_sagawa_ueda_with_errors(error):
    """Horowitz & Parrondo (2011): W = T I exactly, I = ln 2 - H(error)."""
    r = qt.szilard_engine(temperature=1.3, error=error)
    H = -(error * np.log(error) + (1 - error) * np.log(1 - error))
    assert np.isclose(r.information, np.log(2) - H, atol=1e-12)
    assert np.isclose(r.work_extracted, r.bound, rtol=1e-10)


def test_non_degenerate_memory_still_saturates_the_bound():
    r = qt.szilard_engine(temperature=0.7, gap=0.5, error=0.1)
    assert 0 < r.information < np.log(2)
    assert np.isclose(r.work_extracted, r.bound, rtol=1e-10)


def test_finite_time_feedback_stays_below_the_bound_and_approaches_it():
    works = [qt.szilard_engine(temperature=1.0, tau=tau, steps=1200, depth=10.0).work_extracted
             for tau in (20.0, 80.0)]
    bound = qt.szilard_engine(temperature=1.0, depth=10.0).work_extracted
    assert works[0] < works[1] < bound
    assert (bound - works[1]) * 80 < 1.3 * (bound - works[0]) * 20   # ~1/tau
