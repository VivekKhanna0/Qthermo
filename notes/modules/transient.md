---
title: transient.py
---

`transient(model or H, rho0, duration, steps, baths=...)` -> `TransientResult`
(`currents`, `heat` cumulative, `energy`, `virtual_temperatures`,
`settling_time`, `minimum_temperature`, `energy_balance_residual`).
`product_thermal_state(local_H, temps)` = machine switched on at t=0.
Coarse `steps` under-resolves coherent oscillations; the energy balance
residual reports it. Tests: `tests/test_transient.py`.
