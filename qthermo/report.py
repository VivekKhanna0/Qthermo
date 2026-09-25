"""A self-contained HTML report of a continuous machine, for sharing.

    qt.report(model, "fridge.html")

One file, no dependencies to view: the audit findings, the per-bath balance
sheet, the heat-flow figure, the fluctuation ratios, and the machine's
parameters, with the qthermo version and date. The point is to send a
colleague a result together with the checks behind it.
"""

from __future__ import annotations

import base64
import datetime
import html
import io

import numpy as np

__all__ = ["report"]

_STYLE = """
:root { --fg:#1c2330; --muted:#5b6575; --line:#d9dee6; --bg:#ffffff; --panel:#f5f7fa;
        --ok:#1e7b4f; --warn:#a66300; --err:#b3261e; --accent:#1b4f72; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e9ee; --muted:#a3acb9; --line:#343b47; --bg:#14171c; --panel:#1c2128;
          --ok:#5cc28f; --warn:#e3a44a; --err:#f07a70; --accent:#7fb3e0; } }
body { font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--fg); background: var(--bg); margin: 0; }
main { max-width: 900px; margin: 0 auto; padding: 28px 16px 60px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 32px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.meta { color: var(--muted); font-size: 0.9rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }
td.num { text-align: right; font-family: ui-monospace, Menlo, Consolas, monospace; }
.sev { font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .03em; }
.error { color: var(--err); } .warning { color: var(--warn); } .ok { color: var(--ok); } .info { color: var(--muted); }
.detail { color: var(--muted); font-size: 0.9rem; }
img { max-width: 100%; height: auto; background: #fff; border-radius: 6px; border: 1px solid var(--line); }
.summary { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; }
pre { background: var(--panel); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; }
"""


def _figure_png(steady) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plotting import plot_machine
    except ImportError:
        return None
    fig = plot_machine(steady)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def report(model, path=None, title: str | None = None, fluctuations: bool = True) -> str:
    """Write (and return) a self-contained HTML report of a ``Model``."""
    from . import __version__
    from .audit import audit
    from .export import export
    from .fluctuations import current_statistics

    steady = model.analyze()
    findings = audit(model, fluctuations=False)
    title = title or (model.description or "qthermo report")
    esc = html.escape

    n_err = len(findings.by_severity("error"))
    n_warn = len(findings.by_severity("warning"))
    verdict = ("no problems found" if findings.ok else
               f"{n_err} error(s), {n_warn} warning(s): read the audit before using these numbers")

    rows = []
    for b in model.baths:
        J = steady.currents[b.name]
        T = b.temperature
        rows.append(f"<tr><td>{esc(b.name)}</td><td>{esc(b.kind)}</td>"
                    f"<td class='num'>{'' if T is None else f'{T:g}'}</td>"
                    f"<td class='num'>{J:+.5e}</td>"
                    f"<td class='num'>{'' if T is None else f'{-J / T:+.4e}'}</td></tr>")
    balance = "".join(rows)

    audit_rows = "".join(
        f"<tr><td class='sev {f.severity}'>{esc(f.severity)}</td><td>{esc(f.check)}</td>"
        f"<td>{esc(f.summary)}<div class='detail'>{esc(f.detail)}</div></td></tr>"
        for f in sorted(findings.findings,
                        key=lambda f: ["error", "warning", "info", "ok"].index(f.severity)))

    fluct_rows = ""
    if fluctuations and model.H.shape[0] <= 32:
        for b in model.baths:
            try:
                s = current_statistics(model, b.name)
            except Exception:  # noqa: BLE001 -- report what can be computed
                continue
            if abs(s.mean) < 1e-14:
                continue
            fluct_rows += (f"<tr><td>{esc(b.name)}</td><td class='num'>{s.mean:+.4e}</td>"
                           f"<td class='num'>{s.noise:.4e}</td><td class='num'>{s.fano_factor:.4f}</td>"
                           f"<td class='num'>{s.tur_ratio:.4f}</td><td class='num'>{s.kur_ratio:.3f}</td></tr>")

    scale = max((abs(J) for J in steady.currents.values()), default=0.0)
    work_note = ""
    if abs(steady.work_rate) > 1e-10 * max(scale, 1e-300):
        work_note = (f" &middot; work done on the system {steady.work_rate:+.4e} "
                     "(boundary work of the local model, or a drive)")
    image = _figure_png(steady)
    figure = (f"<img alt='heat-flow network, currents and correlations' "
              f"src='data:image/png;base64,{image}'>" if image else
              "<p class='detail'>(matplotlib not installed: no figure)</p>")

    import json
    data = esc(json.dumps(export(steady), indent=2))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><style>{_STYLE}</style></head>
<body><main>
<h1>{esc(title)}</h1>
<div class="meta">qthermo {esc(__version__)} &middot; {stamp} &middot; {model.H.shape[0]} levels &middot;
{esc(model.master_equation)} master equation</div>
<h2>Summary</h2>
<div class="summary">Audit: <b>{esc(verdict)}</b>.<br>
Entropy production rate {steady.entropy_production_rate:.4e}
&middot; sum of currents {steady.total_current:+.2e}{work_note}</div>
<h2>Audit</h2>
<table><tr><th>severity</th><th>check</th><th>finding</th></tr>{audit_rows}</table>
<h2>Heat currents</h2>
<table><tr><th>bath</th><th>kind</th><th>T</th><th>J (into system)</th><th>-J/T</th></tr>{balance}</table>
<h2>Where the heat goes</h2>
{figure}
{"<h2>Fluctuations</h2><table><tr><th>bath</th><th>mean J</th><th>noise D</th><th>Fano</th><th>TUR ratio (&ge;2 classical)</th><th>KUR ratio (&ge;1)</th></tr>" + fluct_rows + "</table>" if fluct_rows else ""}
<h2>Data</h2>
<pre>{data}</pre>
</main></body></html>"""
    if path is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(page)
    return page
