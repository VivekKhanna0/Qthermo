"""Floquet-Markov thermodynamics of periodically driven machines."""

import numpy as np
import pytest
from scipy.special import jv

import qthermo as qt
from qthermo.floquet import DrivenBath, floquet_analyze, floquet_states, window_spectrum

W0, OM = 3.0, 1.0


def modulated(lam):
    return lambda t: qt.qubit_hamiltonian(W0 + lam * np.sin(OM * t))


def machine(T_h, T_c, lam=0.6):
    hot = DrivenBath("hot", qt.sigma_x, T_h, spectrum=window_spectrum(0.05, W0 + OM, 0.8))
    cold = DrivenBath("cold", qt.sigma_x, T_c, spectrum=window_spectrum(0.05, W0 - OM, 0.8))
    return floquet_analyze(modulated(lam), 2 * np.pi / OM, [hot, cold], n_time=128)


def test_undriven_limit_reproduces_static_davies_currents():
    H = qt.qubit_hamiltonian(1.3)
    floquet = floquet_analyze(lambda t: H, 2 * np.pi / 5.0,
                              [DrivenBath("hot", qt.sigma_x, 2.0, gamma=0.3),
                               DrivenBath("cold", qt.sigma_x, 0.5, gamma=0.7)], n_time=64)
    static = qt.analyze(H, [qt.davies_bath(H, qt.sigma_x, 2.0, gamma=0.3, name="hot"),
                            qt.davies_bath(H, qt.sigma_x, 0.5, gamma=0.7, name="cold")])
    for name in ("hot", "cold"):
        assert np.isclose(floquet.current(name), static.current(name), rtol=1e-10)
    assert abs(floquet.power) < 1e-12


def test_sideband_weights_are_bessel_functions():
    lam = 0.6
    eps, modes, _ = floquet_states(modulated(lam), 2 * np.pi / OM, 128)
    elements = np.array([m.conj().T @ qt.sigma_x @ m for m in modes])
    coeffs = np.fft.fft(elements, axis=0) / 128
    for q in range(-6, 7):
        w = eps[1] - eps[0] - q * OM
        if w <= 0:
            continue
        k = int(round((w - W0) / OM))
        assert np.isclose(abs(coeffs[q % 128][0, 1]) ** 2, jv(k, lam / OM) ** 2, atol=1e-6)


def test_tight_coupling_engine_and_refrigerator():
    """Gelbwaser-Klimovsky, Alicki & Kurizki, PRE 87, 012140 (2013)."""
    engine = machine(4.0, 0.5)
    assert engine.mode("hot", "cold") == "engine"
    assert np.isclose(engine.efficiency("hot"), 1 - (W0 - OM) / (W0 + OM), atol=1e-6)
    fridge = machine(1.2, 1.0)
    assert fridge.mode("hot", "cold") == "refrigerator"
    assert np.isclose(fridge.cop("cold"), (W0 - OM) / (2 * OM), atol=1e-5)
    assert fridge.cop("cold") < 1.0 / (1.2 / 1.0 - 1.0)            # Carnot


def test_power_peaks_at_the_maximum_of_the_first_bessel_function():
    ratios = np.linspace(1.2, 2.5, 27)
    power = [machine(4.0, 0.5, lam=x * OM).power for x in ratios]
    assert abs(ratios[int(np.argmax(power))] - 1.8412) < 0.05


def test_second_law_for_random_drives():
    rng = np.random.default_rng(4)
    for _ in range(6):
        a, b, c = rng.uniform(0.2, 1.5, 3)
        H = lambda t, a=a, b=b, c=c: (qt.qubit_hamiltonian(2.0 + a * np.cos(1.3 * t))
                                      + b * np.sin(1.3 * t) * qt.sigma_x
                                      + c * np.cos(2.6 * t) * qt.sigma_y)
        baths = [DrivenBath("hot", qt.sigma_x, rng.uniform(1, 5), gamma=0.1),
                 DrivenBath("cold", qt.sigma_z + qt.sigma_x, rng.uniform(0.2, 1), gamma=0.1)]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = floquet_analyze(H, 2 * np.pi / 1.3, baths, n_time=96)
        assert r.entropy_production_rate > -1e-12
        assert np.isclose(np.sum(r.populations), 1.0)
