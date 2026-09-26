"""Plain-data export of results, for saving, sharing and plotting elsewhere.

    data = qt.export(steady)           # nested dict of numbers and lists
    qt.save(steady, "fridge.json")     # the same, as JSON, with provenance

Every result object in the package exports its scalar and small-array fields.
Density matrices and state trajectories are left out unless
``include_states=True`` (they can be large), and complex arrays are stored as
``{"real": [...], "imag": [...]}``. The file records the qthermo version, so a
number in a paper can be traced to the code that produced it.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math

import numpy as np

__all__ = ["export", "save", "load"]

_HEAVY = {"states", "rho", "rho_initial", "rho_final", "rho_start", "H", "H0",
          "energy", "times", "stroke", "model", "baths", "cycle", "path",
          "eigenbasis", "eig_ops"}


def _plain(value, include_states, depth=0):
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else repr(v)
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            if np.allclose(value.imag, 0):
                return _plain(value.real, include_states, depth + 1)
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return [_plain(v, include_states, depth + 1) for v in value.tolist()] \
            if value.dtype == object else value.tolist()
    if isinstance(value, dict):
        return {_key(k): _plain(v, include_states, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, include_states, depth + 1) for v in value]
    if dataclasses.is_dataclass(value):
        return _object_dict(value, include_states, depth + 1)
    if hasattr(value, "__dict__"):
        return _object_dict(value, include_states, depth + 1)
    return repr(value)


def _key(k):
    if isinstance(k, tuple):
        return "-".join(str(x) for x in k)
    return str(k)


def _derived(obj) -> dict:
    """Useful properties that are not stored fields."""
    names = ["entropy_production_rate", "total_current", "work_rate", "power_output",
             "net_work", "total_entropy_production", "max_first_law_residual",
             "tur_ratio", "kur_ratio", "fano_factor", "relative_noise",
             "excess_heat", "entropy_production", "efficiency", "cop", "mode",
             "first_law_residual", "interaction_energy_change", "correlation_change",
             "length", "ok"]
    out = {}
    for name in names:
        if name in vars(type(obj)) and isinstance(vars(type(obj))[name], property):
            try:
                out[name] = getattr(obj, name)
            except Exception:  # noqa: BLE001 -- e.g. efficiency of a non-engine
                continue
    return out


def _object_dict(obj, include_states, depth):
    if dataclasses.is_dataclass(obj):
        fields = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    else:
        fields = dict(vars(obj))
    out = {"type": type(obj).__name__}
    for name, value in fields.items():
        if name.startswith("_"):
            continue
        if name in _HEAVY and not include_states:
            if name == "baths" and value is not None:
                out["baths"] = [{"name": b.name, "temperature": b.temperature,
                                 "kind": b.kind} for b in value]
            continue
        if callable(value) and not isinstance(value, np.ndarray):
            continue
        out[name] = _plain(value, include_states, depth)
    for name, value in _derived(obj).items():
        out.setdefault(name, _plain(value, include_states, depth))
    return out


def export(obj, include_states: bool = False):
    """Convert any qthermo result (or container of results) to plain data."""
    return _plain(obj, include_states)


def save(obj, path, include_states: bool = False, note: str | None = None) -> None:
    """Write ``export(obj)`` to a JSON file with version and timestamp."""
    from . import __version__
    payload = {
        "qthermo_version": __version__,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": note,
        "result": export(obj, include_states),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load(path) -> dict:
    """Read a file written by :func:`save` (as plain data)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
