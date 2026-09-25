"""Transient thermodynamics of continuous machines."""

import numpy as np

import qthermo as qt


def fridge(g=0.05, gamma=0.002):
    return qt.models.absorption_refrigerator(g=g, gamma=gamma, T_c=1.0, T_h=6.0, T_r=1.5)


def switched_on(m):
    return qt.product_thermal_state(m.local_H, [1.0, 6.0, 1.5])


def test_long_time_currents_approach_the_steady_state():
    m = fridge(gamma=0.01)
    run = qt.transient(m, switched_on(m), duration=1200.0, steps=600)
    steady = m.analyze()
    for name in ("cold", "hot", "room"):
        assert np.isclose(run.currents[name][-1], steady.current(name), rtol=1e-4)


def test_cumulative_heat_balances_the_energy_change():
    m = fridge(gamma=0.01)
    run = qt.transient(m.H, switched_on(m), duration=300.0, steps=1500, baths=m.baths)
    scale = max(abs(q[-1]) for q in run.heat.values())
    assert run.energy_balance_residual() < 1e-4 * scale


def test_coherent_fridge_cools_below_its_steady_state_transiently():
    """Single-shot cooling: Mitchison et al., NJP 17, 115013 (2015)."""
    m = fridge(g=0.05, gamma=0.002)
    run = qt.transient(m, switched_on(m), duration=4000.0, steps=800)
    T_min, _ = run.minimum_temperature(0)
    T_steady = qt.heat_flow_map(m.analyze()).virtual_temperature[0]
    assert T_min < T_steady - 0.05
    assert np.isclose(run.virtual_temperatures[0][-1], T_steady, rtol=1e-3)


def test_overdamped_fridge_barely_undershoots():
    m = fridge(g=0.05, gamma=0.05)
    run = qt.transient(m, switched_on(m), duration=200.0, steps=600)
    T_min, _ = run.minimum_temperature(0)
    T_steady = qt.heat_flow_map(m.analyze()).virtual_temperature[0]
    assert T_steady - T_min < 0.01
