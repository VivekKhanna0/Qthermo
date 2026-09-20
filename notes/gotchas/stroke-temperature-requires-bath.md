---
title: stroke-temperature-requires-bath
---

`Stroke.__post_init__` (`cycle.py`) raises `QThermoError` if `temperature` is
set but `c_ops` is empty:

    stroke 'name' has a temperature but no collapse operators. A stroke
    with no bath exchanges no heat, so its entropy production is not
    defined by a bath temperature.

The reverse is legal: `c_ops` set with no `temperature` just means the
stroke won't report entropy production (silently — no warning), since
`Cycle.run` only calls `entropy_production` when `stroke.temperature is not None`.

Trap when building a new stroke type: passing collapse operators from
`channels.thermal_bath(...)` without also passing the matching `temperature`
argument to `Stroke` won't error — it'll just quietly skip entropy-production
reporting for that stroke.
