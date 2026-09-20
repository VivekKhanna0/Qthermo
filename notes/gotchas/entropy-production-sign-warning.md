---
title: entropy-production-sign-warning
---

`core.entropy_production` computes `sigma = dS - Q/T`. By Spohn's theorem
this should be non-negative for Markovian dynamics with a thermal fixed
point. If it comes out below `-tolerance` (default `1e-9`), the function
does NOT raise — it emits a `RuntimeWarning` and returns the (negative)
value anyway.

The warning text (from `core.py`) points at the actual usual cause: the
declared `temperature` argument is inconsistent with the collapse operators
actually used — e.g. labeling a zero-temperature `amplitude_damping` channel
with a nonzero `temperature`. `amplitude_damping` has no temperature
parameter at all (see modules/channels.md); if you build a stroke with it
and hand `Stroke` a nonzero `temperature` anyway, expect this warning.

Because it's a warning and not an exception, a negative sigma can silently
flow into `CycleResult.total_entropy_production` and downstream reports
unless you're watching for the warning.
