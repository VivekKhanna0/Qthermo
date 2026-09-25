"""Operation-mode classification and parameter maps."""

import numpy as np

import qthermo as qt


def test_classification_of_sign_patterns():
    assert qt.classify(-1.0, 3.0, -2.0) == "engine"
    assert qt.classify(1.0, -3.0, 2.0) == "refrigerator"
    assert qt.classify(1.0, 2.0, -3.0) == "accelerator"
    assert qt.classify(1.0, -0.4, -0.6) == "heater"
    assert qt.classify(0.0, 0.0, 0.0) == "idle"
    assert qt.classify(0.0, -1.0, 1.0) == "forbidden"     # cold -> hot, no work
    assert qt.classify(-1.0, 0.0, 1.0) == "forbidden"     # work from one bath


def test_quasi_static_qubit_otto_boundary_is_exact():
    ratios = np.linspace(0.05, 0.95, 13)
    scan = qt.mode_map(
        lambda x, y: qt.ideal_otto(qt.qubit_hamiltonian(x), qt.qubit_hamiltonian(1.0), y, 1.0),
        ratios, ratios)
    X, Y = np.meshgrid(scan.x_values, scan.y_values)
    assert np.array_equal(scan.grid == qt.MODES.index("engine"), X > Y + 1e-12)
    assert np.array_equal(scan.grid == qt.MODES.index("refrigerator"), X < Y - 1e-12)


def test_finite_time_friction_produces_accelerator_and_heater():
    def build(w, t):
        H = lambda x: qt.qubit_hamiltonian(x) + 0.4 * qt.sigma_x
        return qt.otto_cycle(H(w), H(1.0), t, 1.0, [qt.sigma_x, qt.sigma_z],
                             gamma=1.0, tau_iso=15.0, tau_ramp=0.3, steps=120)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # isochores must thermalise: friction only
        modes = {(w, t): build(w, t).limit_cycle(np.eye(2) / 2)[0].mode()
                 for w, t in ((0.2, 0.64), (0.2, 0.2), (0.9, 0.1))}
    assert modes[(0.2, 0.64)] == "heater"
    assert modes[(0.2, 0.2)] == "accelerator"
    assert modes[(0.9, 0.1)] == "engine"
    # the quasi-static cycle at the same points is engine or refrigerator only
    for w, t in modes:
        lim = qt.ideal_otto(qt.qubit_hamiltonian(w) + 0.4 * qt.sigma_x,
                            qt.qubit_hamiltonian(1.0) + 0.4 * qt.sigma_x, t, 1.0)
        assert lim.mode in ("engine", "refrigerator")


def test_sweep_accepts_continuous_models():
    out = qt.sweep(lambda T_h: qt.models.absorption_refrigerator(T_h=T_h),
                   np.linspace(2.0, 8.0, 5), lambda r, m: r.current("cold"),
                   parameter="T_h", metric_name="J_cold")
    assert np.all(np.diff(out.metrics) > 0)           # hotter source, more cooling


def test_steady_state_mode_of_the_maser_is_engine():
    r = qt.models.three_level_maser().analyze()
    assert r.mode() == "engine"
