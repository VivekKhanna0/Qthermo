"""Operation modes of two-reservoir thermal machines.

A machine exchanging heat ``Q_h`` with a hot bath and ``Q_c`` with a cold one,
and work ``W`` with an external agent (``W > 0``: work done on the machine),
runs in one of four modes allowed by the second law:

    engine        W < 0,  Q_h > 0,  Q_c < 0     heat -> work
    refrigerator  W > 0,  Q_c > 0,  Q_h < 0     work pumps heat cold -> hot
    accelerator   W > 0,  Q_h > 0,  Q_c < 0     work speeds up hot -> cold flow
    heater        W > 0,  Q_h < 0,  Q_c < 0     work dumped into both baths

Anything else is either idle (nothing exchanged) or forbidden by the second
law, and a forbidden label is a bug in the model, not a regime. Which of the
four a machine is in, across parameter space, is the first thing one draws for
a new machine; :func:`mode_map` and :func:`qthermo.plotting.plot_mode_map` do
that for any builder the sweep tools accept.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MODES", "classify", "mode_code", "mode_map"]

MODES = ("engine", "refrigerator", "accelerator", "heater", "idle", "forbidden")


def classify(work: float, heat_hot: float, heat_cold: float,
             tol: float = 1e-12) -> str:
    """Operation mode from the signs of work and the two heats.

    ``tol`` is relative to the largest of the three magnitudes; exchanges
    below it count as zero.
    """
    scale = max(abs(work), abs(heat_hot), abs(heat_cold))
    if scale == 0 or not np.isfinite(scale):
        return "idle"
    eps = tol * scale

    def sign(x):
        return 0 if abs(x) <= eps else (1 if x > 0 else -1)

    w, h, c = sign(work), sign(heat_hot), sign(heat_cold)
    if (w, h, c) == (-1, 1, -1):
        return "engine"
    if (w, h, c) == (1, -1, 1):
        return "refrigerator"
    if (w, h, c) == (1, 1, -1):
        return "accelerator"
    if w == 1 and h <= 0 and c <= 0:
        return "heater"
    if w == 0 and h == 1 and c == -1:
        return "accelerator"      # plain conduction hot -> cold (W = 0 boundary)
    if (w, h, c) == (0, 0, 0):
        return "idle"
    return "forbidden"


def mode_code(mode: str) -> int:
    return MODES.index(mode)


def mode_map(build, x_values, y_values, parameters=("x", "y"), **kwargs):
    """Grid of operation modes: ``build(x, y)`` returns anything with a
    ``.mode()`` or ``.mode`` (an ``OttoLimit``, a ``Model``, a ``Cycle``
    whose strokes are named ``hot_iso``/``cold_iso``).

    Returns a :class:`~qthermo.analysis.ScanResult` whose grid holds
    ``MODES`` indices; plot it with ``plot_mode_map``.
    """
    from .analysis import scan_2d

    def metric(result, machine):
        mode = getattr(result, "mode")
        mode = mode() if callable(mode) else mode
        return float(mode_code(mode))
    return scan_2d(build, x_values, y_values, metric, parameters=parameters,
                   metric_name="operation mode", **kwargs)
