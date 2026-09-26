"""Generate docs/api.md from the package's docstrings.

    python docs/build_api.py
"""

import inspect
import pathlib

import qthermo as qt

ROOT = pathlib.Path(__file__).resolve().parent
SECTIONS = [
    ("Machines and baths", ["models", "baths", "steady"]),
    ("Diagnostics", ["audit"]),
    ("Where the heat goes", ["network", "subsystems"]),
    ("Fluctuations and statistics", ["fluctuations", "counting", "stochastic"]),
    ("Stroke machines", ["cycle", "engines", "modes"]),
    ("Driven, transient and linear response", ["floquet", "transient", "response"]),
    ("Strong coupling", ["strong_coupling"]),
    ("Information and protocols", ["information", "geometry"]),
    ("Quantum batteries", ["batteries"]),
    ("Core quantities, channels, solver", ["core", "channels", "solver"]),
    ("Analysis and plotting", ["analysis", "plotting"]),
    ("Output, units, validation", ["export", "report", "units", "validation", "benchmarks"]),
]


def summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    paragraph = doc.split("\n\n")[0].replace("\n", " ").strip()
    return paragraph


def signature(obj) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""
    return sig if len(sig) < 110 else sig[:107] + "...)"


def main():
    import importlib
    lines = ["# API reference", "",
             "Generated from the docstrings by `python docs/build_api.py`. Every name",
             "below is importable as `qthermo.<name>` unless marked with its module.",
             "Definitions of the physics are in [physics.md](physics.md).", ""]
    top = set(qt.__all__)
    for title, modules in SECTIONS:
        lines += [f"## {title}", ""]
        for name in modules:
            module = importlib.import_module(f"qthermo.{name}")
            lines.append(f"### `qthermo.{name}`")
            lines.append("")
            intro = summary(module)
            if intro:
                lines += [intro, ""]
            for member in getattr(module, "__all__", []):
                obj = getattr(module, member)
                if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                    continue
                prefix = "" if member in top else f"{name}."
                kind = "class " if inspect.isclass(obj) else ""
                lines.append(f"- **`{kind}{prefix}{member}{signature(obj)}`**  ")
                lines.append(f"  {summary(obj)}")
            lines.append("")
    (ROOT / "api.md").write_text("\n".join(lines))
    print(f"wrote {ROOT / 'api.md'}")


if __name__ == "__main__":
    main()
