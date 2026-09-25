---
title: smaller modules
---

- `audit.py` — `audit(model | H, baths)` -> `AuditReport` of `Finding`s (error/warning/info/ok).
- `counting.py` — `cycle_counting(cycle, {stroke: "quanta"})`: exact P(n) per cycle via tilted propagators + DFT; long-run cumulants.
- `modes.py` — `classify(W, Q_h, Q_c)`, `mode_map`, `MODES`.
- `batteries.py` — `ergotropy_split`, `locked_ergotropy`, `asymptotic_ergotropy`, `multi_copy_ergotropy`, `charge`, `dicke_battery`, `collective_advantage`.
- `export.py` — `export`, `save`, `load` (JSON with version).
- `units.py` — `LabUnits(f0_GHz)`.
- `transient.py`, `response.py`, `floquet.py`, `geometry.py` — see their notes.
