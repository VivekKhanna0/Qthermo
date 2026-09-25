"""Plain-data export and JSON round trip."""

import json

import numpy as np

import qthermo as qt


def test_steady_state_exports_currents_and_derived_quantities():
    r = qt.models.absorption_refrigerator().analyze()
    data = qt.export(r)
    json.dumps(data)                                   # JSON-safe
    assert data["type"] == "SteadyState"
    assert np.isclose(data["currents"]["cold"], r.current("cold"))
    assert np.isclose(data["entropy_production_rate"], r.entropy_production_rate)
    assert "rho" not in data                           # heavy fields left out
    assert "rho" in qt.export(r, include_states=True)


def test_every_result_type_is_json_safe():
    m = qt.models.three_level_maser()
    cycle = qt.otto_cycle(qt.qubit_hamiltonian(1.0), qt.qubit_hamiltonian(2.0), 0.5, 3.0,
                          [qt.sigma_x], tau_iso=15, tau_ramp=1, steps=60)
    result, _, _ = cycle.limit_cycle(np.eye(2) / 2)
    objects = [m.analyze(), qt.current_statistics(m, {"hot": "energy", "cold": "energy"}),
               qt.audit(m), result, qt.heat_flow_map(qt.models.spin_chain(3, master_equation="local").analyze()),
               qt.ideal_otto(qt.qubit_hamiltonian(1.0), qt.qubit_hamiltonian(2.0), 0.5, 3.0),
               qt.cycle_counting(cycle, {"cold_iso": "quanta"}).distribution(4)]
    for obj in objects:
        json.dumps(qt.export(obj))


def test_save_and_load_round_trip(tmp_path):
    stats = qt.current_statistics(qt.models.three_level_maser(), "hot")
    path = tmp_path / "maser.json"
    qt.save(stats, path, note="maser heat current")
    loaded = qt.load(path)
    assert loaded["qthermo_version"] == qt.__version__
    assert loaded["note"] == "maser heat current"
    assert np.isclose(loaded["result"]["tur_ratio"], stats.tur_ratio)
