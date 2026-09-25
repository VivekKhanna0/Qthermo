"""Interactive explorer: drag a parameter, watch the heat flow change.

    qt.explorer(lambda g: qt.models.two_qubit_heat_valve(g=g),
                np.linspace(0, 1.2, 41), parameter="coupling g",
                path="valve.html")

Writes one self-contained HTML file (no network needed to view it). Every
frame is computed up front with the same solver as ``Model.analyze``; the page
only displays them: the heat-flow network with virtual temperatures, the
per-bath currents and entropy production, and the currents against the
parameter. When the model can be rebuilt under the other master equation, a
switch shows local and global side by side -- including where they disagree
on the direction of heat flow.
"""

from __future__ import annotations

import html
import json
import warnings

import numpy as np

from .validation import QThermoError

__all__ = ["explorer"]


def _frame(model):
    from .network import heat_flow_map
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        steady = model.analyze()
    fm = heat_flow_map(steady)
    blind = bool(fm.extras.get("secular_blind"))
    bath_flow = {}
    for name in steady.currents:
        bath_flow[name] = (fm.bath_current(name) if blind else
                           sum(J for (b, _), J in fm.bath_to_site.items() if b == name))
    bonds = []
    for key in sorted({k for (k, _) in fm.term_to_site}):
        if len(key) == 2:
            bonds.append([int(key[0]), int(key[1]), float(fm.bond_current(key))])
    T = [float(t) if np.isfinite(t) else None for t in fm.virtual_temperature]
    return {"currents": {k: float(v) for k, v in steady.currents.items()},
            "bathT": {b.name: b.temperature for b in model.baths},
            "bath_flow": {k: float(v) for k, v in bath_flow.items()},
            "sigma": float(steady.entropy_production_rate),
            "T": T, "bonds": bonds, "blind": blind}


def explorer(build, values, parameter: str = "parameter", path=None,
             title: str | None = None, compare: bool = True) -> str:
    """Write an interactive HTML explorer of ``build(value) -> Model``.

    Parameters
    ----------
    build : callable
        ``build(value)`` returns a :class:`~qthermo.models.Model`.
    values : sequence of float
        Parameter values (one precomputed frame each; 20-80 is typical).
    compare : bool
        Also compute every frame under the other master equation when the
        model supports it (``Model.rebuild``).
    """
    from . import __version__
    from .plotting import _layout

    values = [float(v) for v in values]
    if len(values) < 2:
        raise QThermoError("the explorer needs at least two parameter values")
    first = build(values[0])
    kinds = [first.master_equation]
    if compare and first.rebuild is not None:
        kinds = ["global", "local"]
    frames = {k: [] for k in kinds}
    for v in values:
        m = build(v)
        for kind in kinds:
            mk = m if m.master_equation == kind else m.rebuild(kind)
            frames[kind].append(_frame(mk))

    n = len(first.dims)
    bonds = sorted({tuple(b[:2]) for f in frames[kinds[0]] for b in f["bonds"]})
    anchor = first.baths[0].sites[0] if first.baths and first.baths[0].sites else 0
    positions, chain = _layout(n, bonds or [(i, i + 1) for i in range(n - 1)], anchor)
    centre = np.mean(list(positions.values()), axis=0)
    bath_pos = {}
    for k, b in enumerate(first.baths):
        sites = list(b.sites) or list(range(n))
        a = np.mean([positions[s] for s in sites], axis=0)
        if chain:
            off = (np.array([-1.6, 0.0]) if sites == [0] and n > 1 else
                   np.array([1.6, 0.0]) if sites == [n - 1] and n > 1 else
                   np.array([0.0, 1.5 if k % 2 == 0 else -1.5]))
        else:
            d = a - centre
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([0.0, 1.0])
            off = 1.6 * d
        bath_pos[b.name] = (a + off).tolist()

    data = {
        "parameter": parameter, "values": values, "kinds": kinds, "frames": frames,
        "sites": [{"name": nm, "x": float(positions[i][0]), "y": float(positions[i][1])}
                  for i, nm in enumerate(first.site_names or [f"q{i}" for i in range(n)])],
        "baths": [{"name": b.name, "T": b.temperature, "x": bath_pos[b.name][0],
                   "y": bath_pos[b.name][1], "sites": list(b.sites)} for b in first.baths],
        "version": __version__,
    }
    title = title or (first.description or "qthermo explorer")
    page = _TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    if path is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(page)
    return page


_TEMPLATE = r"""<title>__TITLE__</title>
<style>
:root { --ground:#F5F7FA; --panel:#FFFFFF; --ink:#16202B; --muted:#5E6B79; --line:#D8DEE6;
  --hot:#C2412D; --cold:#2F6FB0; --accent:#0E7C78; --ok:#1F7A4D; --bad:#B42318; --mid:#C9CFD6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --ground:#0F141A; --panel:#161D25; --ink:#E3E8EE; --muted:#9AA6B3; --line:#2A3440;
  --hot:#E0664F; --cold:#5E9BDB; --accent:#3FB5AE; --ok:#56C08A; --bad:#F07068; --mid:#3A4552; } }
:root[data-theme="dark"] { color-scheme: dark; --ground:#0F141A; --panel:#161D25; --ink:#E3E8EE;
  --muted:#9AA6B3; --line:#2A3440; --hot:#E0664F; --cold:#5E9BDB; --accent:#3FB5AE;
  --ok:#56C08A; --bad:#F07068; --mid:#3A4552; }
* { box-sizing: border-box; }
body { margin:0; background:var(--ground); color:var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; padding-block: 24px 48px; padding-inline: 16px;
  display:grid; gap:18px; }
header h1 { font-size: 1.35rem; margin:0; text-wrap: balance; }
header p { margin: 4px 0 0; color: var(--muted); font-size: .9rem; max-width: 70ch; }
.controls { display:flex; flex-wrap:wrap; gap:14px 24px; align-items:center;
  background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; }
.controls label { font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.slider { flex: 1 1 280px; display:grid; gap:4px; }
input[type=range] { width:100%; accent-color: var(--accent); }
.value { font: 600 1.05rem ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
.switch { display:flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.switch button { font: inherit; font-size:.88rem; padding:6px 14px; border:0; background:transparent;
  color:var(--ink); cursor:pointer; }
.switch button[aria-pressed=true] { background: var(--accent); color:#fff; }
.switch button:focus-visible, input:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.main { display:grid; grid-template-columns: minmax(0,1.6fr) minmax(0,1fr); gap:18px; }
@media (max-width: 760px) { .main { grid-template-columns: 1fr; } }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; min-width:0; }
.card h2 { font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:0 0 8px; font-weight:600; }
svg { width:100%; height:auto; display:block; }
table { width:100%; border-collapse:collapse; font-variant-numeric: tabular-nums; }
td, th { padding:6px 4px; border-bottom:1px solid var(--line); text-align:left; font-size:.9rem; }
td.num { text-align:right; font-family: ui-monospace, Menlo, Consolas, monospace; }
.chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.8rem; font-weight:600; margin-top:10px; }
.chip.ok { background: color-mix(in srgb, var(--ok) 16%, transparent); color: var(--ok); }
.chip.bad { background: color-mix(in srgb, var(--bad) 16%, transparent); color: var(--bad); }
.note { color:var(--muted); font-size:.82rem; margin-top:8px; }
.foot { color:var(--muted); font-size:.8rem; }
</style>
<div class="wrap">
  <header><h1 id="title">__TITLE__</h1>
    <p>Each frame is a steady state solved by qthermo. Circles are sites coloured by their virtual
    temperature, squares are baths, and arrows follow the heat current (width ∝ magnitude).</p></header>
  <div class="controls">
    <div class="slider"><label for="param" id="plabel">parameter</label>
      <input id="param" type="range" min="0" value="0" step="1">
      <span class="value" id="pvalue"></span></div>
    <div id="mewrap"><label>master equation</label><div class="switch" id="me"></div></div>
  </div>
  <div class="main">
    <div class="card"><h2>Heat flow</h2><svg id="net" role="img" aria-label="heat flow network"></svg>
      <div class="note" id="blind"></div></div>
    <div class="card"><h2>Bath currents (into system)</h2>
      <table><thead><tr><th>bath</th><th>T</th><th style="text-align:right">J</th></tr></thead><tbody id="rows"></tbody></table>
      <div id="sigma"></div><div class="note" id="flip"></div></div>
  </div>
  <div class="card"><h2>Currents against the parameter</h2><svg id="chart" role="img" aria-label="currents against parameter"></svg>
    <div class="note">Solid: the selected master equation. Dashed: the other one, when available.</div></div>
  <div class="foot">Generated by qthermo <span id="ver"></span>.</div>
</div>
<script>
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let kind = D.kinds[0], idx = 0;
const allT = [];
for (const k of D.kinds) for (const f of D.frames[k]) {
  f.T.forEach(t => t != null && t > 0 && allT.push(t));
  Object.values(f.bathT).forEach(t => t != null && allT.push(t)); }
const Tmin = Math.min(...allT), Tmax = Math.max(...allT);
let Jmax = 1e-300;
for (const k of D.kinds) for (const f of D.frames[k]) {
  Object.values(f.bath_flow).forEach(v => Jmax = Math.max(Jmax, Math.abs(v)));
  f.bonds.forEach(b => Jmax = Math.max(Jmax, Math.abs(b[2]))); }
function mix(a, b, t) { const p = s => [1,3,5].map(i => parseInt(s.slice(i, i+2), 16));
  const A = p(a), B = p(b); return "rgb(" + A.map((x, i) => Math.round(x + (B[i]-x)*t)).join(",") + ")"; }
function tempColour(T) { if (T == null || !(T > 0)) return css("--panel");
  const u = Math.log(T/Tmin) / Math.log(Tmax/Tmin || 2);
  return u < .5 ? mix(css("--cold"), css("--mid"), u*2) : mix(css("--mid"), css("--hot"), (u-.5)*2); }
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
const fmt = x => (x >= 0 ? "+" : "−") + Math.abs(x).toExponential(2);

function drawNet() {
  const svg = document.getElementById("net"); svg.innerHTML = "";
  const pts = D.sites.map(s => [s.x, -s.y]).concat(D.baths.map(b => [b.x, -b.y]));
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const pad = .7, x0 = Math.min(...xs)-pad, y0 = Math.min(...ys)-pad;
  const w = Math.max(...xs)-x0+pad, h = Math.max(...ys)-y0+pad;
  const hh = Math.max(h, w*0.3);
  svg.setAttribute("viewBox", `${x0} ${y0 - (hh-h)/2} ${w} ${hh}`);
  const defs = el("defs", {}, svg);
  const f = D.frames[kind][idx];
  const ink = css("--ink");
  function marker(id, colour) { const m = el("marker", {id, viewBox:"0 0 10 10", refX:"8", refY:"5",
    markerWidth:"4", markerHeight:"4", orient:"auto-start-reverse"}, defs);
    el("path", {d:"M0,0 L10,5 L0,10 z", fill: colour}, m); }
  marker("mh", css("--hot")); marker("mc", css("--cold")); marker("mi", ink);
  function arrow(a, b, J, colour, mid, label) {
    const dx = b[0]-a[0], dy = b[1]-a[1], L = Math.hypot(dx, dy) || 1, s = .33/L;
    let p = [a[0]+dx*s, a[1]+dy*s], q = [b[0]-dx*s, b[1]-dy*s];
    if (J < 0) [p, q] = [q, p];
    const width = .02 + .11*Math.abs(J)/Jmax;
    if (Math.abs(J) > 1e-14*Jmax) el("line", {x1:p[0], y1:p[1], x2:q[0], y2:q[1], stroke:colour,
      "stroke-width":width, "marker-end":`url(#${mid})`, "stroke-linecap":"round"}, svg);
    if (!label) return;
    const t = el("text", {x:(a[0]+b[0])/2 - dy/L*.22, y:(a[1]+b[1])/2 + dx/L*.22 - .06, "font-size":".13",
      "text-anchor":"middle", fill: css("--muted"), "font-family":"ui-monospace, Menlo, monospace"}, svg);
    t.textContent = label; }
  f.bonds.forEach(([i, j, J]) => arrow([D.sites[i].x, -D.sites[i].y], [D.sites[j].x, -D.sites[j].y], J, ink, "mi",
    Math.abs(J) > 1e-12*Jmax ? Math.abs(J).toExponential(2) : ""));
  D.baths.forEach(b => { const J = f.bath_flow[b.name];
    const sites = b.sites.length ? b.sites : D.sites.map((_, i) => i);
    const tx = sites.reduce((s, i) => s + D.sites[i].x, 0)/sites.length, ty = -sites.reduce((s, i) => s + D.sites[i].y, 0)/sites.length;
    const hot = J >= 0; arrow([b.x, -b.y], [tx, ty], J, hot ? css("--hot") : css("--cold"), hot ? "mh" : "mc", fmt(J)); });
  D.baths.forEach(b => { const T = f.bathT[b.name];
    el("rect", {x:b.x-.34, y:-b.y-.26, width:.68, height:.52, rx:.06, fill: tempColour(T), stroke: ink, "stroke-width":.02}, svg);
    const t = el("text", {x:b.x, y:-b.y-.03, "font-size":".14", "text-anchor":"middle", fill:"#fff", "font-weight":"700"}, svg);
    t.textContent = b.name;
    const u = el("text", {x:b.x, y:-b.y+.15, "font-size":".12", "text-anchor":"middle", fill:"#fff"}, svg);
    u.textContent = "T = " + (+T).toPrecision(3); });
  D.sites.forEach((s, i) => { el("circle", {cx:s.x, cy:-s.y, r:.3, fill: tempColour(f.T[i]), stroke: ink, "stroke-width":.025}, svg);
    const t1 = el("text", {x:s.x, y:-s.y-.02, "font-size":".16", "text-anchor":"middle", fill: ink, "font-weight":"700"}, svg); t1.textContent = s.name;
    const t2 = el("text", {x:s.x, y:-s.y+.15, "font-size":".12", "text-anchor":"middle", fill: ink}, svg);
    t2.textContent = f.T[i] == null ? "T*=∞" : (f.T[i] < 0 ? "T*<0" : "T*=" + f.T[i].toPrecision(3)); });
  document.getElementById("blind").textContent = f.blind ?
    "Global master equation: the steady state is diagonal in H, so bond currents vanish identically; heat enters the interaction energy." : "";
}

function drawTable() {
  const f = D.frames[kind][idx], rows = document.getElementById("rows"); rows.innerHTML = "";
  D.baths.forEach(b => { const tr = document.createElement("tr");
    tr.innerHTML = `<td>${b.name}</td><td class="num">${(+f.bathT[b.name]).toPrecision(3)}</td><td class="num">${fmt(f.currents[b.name])}</td>`; rows.appendChild(tr); });
  const ok = f.sigma >= -1e-12;
  document.getElementById("sigma").innerHTML = `<span class="chip ${ok ? "ok" : "bad"}">σ̇ = ${f.sigma.toExponential(2)} ${ok ? "≥ 0" : "< 0: second law violated"}</span>`;
  let note = "";
  if (D.kinds.length > 1) { const g = D.frames.global[idx], l = D.frames.local[idx];
    const flips = D.baths.filter(b => Math.sign(g.currents[b.name]) !== Math.sign(l.currents[b.name]) &&
      Math.abs(g.currents[b.name]) > 1e-12).map(b => b.name);
    if (flips.length) note = "Local and global disagree on the direction of heat flow for: " + flips.join(", ") + "."; }
  document.getElementById("flip").textContent = note;
}

function drawChart() {
  const svg = document.getElementById("chart"); svg.innerHTML = "";
  const W = 900, H = 270, m = {l:70, r:16, t:12, b:46};
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const xs = D.values, x0 = xs[0], x1 = xs[xs.length-1];
  let lo = 0, hi = 0;
  for (const k of D.kinds) for (const f of D.frames[k]) Object.values(f.currents).forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
  if (hi === lo) hi = lo + 1;
  const X = v => m.l + (v-x0)/(x1-x0 || 1)*(W-m.l-m.r), Y = v => m.t + (hi-v)/(hi-lo)*(H-m.t-m.b);
  const muted = css("--muted"), line = css("--line");
  for (let i = 0; i <= 4; i++) { const v = lo + (hi-lo)*i/4;
    el("line", {x1:m.l, x2:W-m.r, y1:Y(v), y2:Y(v), stroke: line, "stroke-width":1}, svg);
    const t = el("text", {x:m.l-8, y:Y(v)+4, "text-anchor":"end", "font-size":12, fill: muted, "font-family":"ui-monospace, Menlo, monospace"}, svg);
    t.textContent = v.toExponential(1); }
  el("line", {x1:m.l, x2:W-m.r, y1:Y(0), y2:Y(0), stroke: muted, "stroke-width":1}, svg);
  for (let i = 0; i <= 4; i++) { const v = x0 + (x1-x0)*i/4;
    const t = el("text", {x:X(v), y:H-m.b+18, "text-anchor":"middle", "font-size":12, fill: muted}, svg); t.textContent = v.toPrecision(3); }
  const tl = el("text", {x:(m.l+W-m.r)/2, y:H-6, "text-anchor":"middle", "font-size":12, fill: muted}, svg); tl.textContent = D.parameter;
  const ref = D.frames[kind][Math.floor(xs.length / 2)].bathT, temps = D.baths.map(b => ref[b.name]);
  const colours = D.baths.map(b => ref[b.name] === Math.max(...temps) ? css("--hot") :
    ref[b.name] === Math.min(...temps) ? css("--cold") : css("--accent"));
  for (const k of D.kinds) D.baths.forEach((b, j) => {
    const pts = D.frames[k].map((f, i) => `${X(xs[i])},${Y(f.currents[b.name])}`).join(" ");
    el("polyline", {points: pts, fill:"none", stroke: colours[j], "stroke-width": k === kind ? 2.4 : 1.4,
      "stroke-dasharray": k === kind ? "" : "5 4", opacity: k === kind ? 1 : .7}, svg);
    if (k === kind) { const t = el("text", {x:W-m.r-2, y:Y(D.frames[k][xs.length-1].currents[b.name])-6, "text-anchor":"end", "font-size":12, fill: colours[j]}, svg); t.textContent = b.name; } });
  el("line", {x1:X(xs[idx]), x2:X(xs[idx]), y1:m.t, y2:H-m.b, stroke: css("--accent"), "stroke-width":1.5}, svg);
}

function render() {
  document.getElementById("pvalue").textContent = D.parameter + " = " + D.values[idx].toPrecision(4);
  drawNet(); drawTable(); drawChart(); }

const slider = document.getElementById("param");
slider.max = D.values.length - 1; document.getElementById("plabel").textContent = D.parameter;
slider.addEventListener("input", () => { idx = +slider.value; render(); });
const sw = document.getElementById("me");
if (D.kinds.length < 2) document.getElementById("mewrap").hidden = true;
D.kinds.forEach(k => { const b = document.createElement("button"); b.type = "button"; b.textContent = k;
  b.setAttribute("aria-pressed", k === kind); b.addEventListener("click", () => { kind = k;
    sw.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x.textContent === kind)); render(); });
  sw.appendChild(b); });
document.getElementById("ver").textContent = D.version;
idx = Math.round((D.values.length - 1) * 0.6); slider.value = idx;
render();
if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", render);
</script>
"""
