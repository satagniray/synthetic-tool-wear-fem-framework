"""Run-level outputs: all figures, the progression GIF and an interactive HTML dashboard.

The dashboard is a single self-contained HTML file. Its data are embedded as
JSON and it loads Plotly.js from the official CDN, so no Python plotting
dependency beyond Matplotlib is required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.contour import ToolContour
from ..core.mesh import Mesh
from ..core.state import ProcessFieldState
from ..io import csv_io
from ..io.config import load_config
from ..io.results import RunDirectory, read_json, resolve_run_dir

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


@dataclass
class RunData:
    run: RunDirectory
    config: dict
    mesh: Mesh
    contour: ToolContour
    history: list[dict]
    snapshots: list[tuple[int, np.ndarray]]
    fields: dict[int, ProcessFieldState] = field(default_factory=dict)
    wear_rates: dict[int, np.ndarray] = field(default_factory=dict)
    stop_reason: str | None = None


def fields_from_table(table: dict) -> ProcessFieldState:
    return ProcessFieldState(
        node_index=np.asarray(table["node_id"]) - 1,
        surface_type=np.asarray(table["surface_type"]).astype("U8"),
        s_norm=table["s_norm"],
        d_mm=table["d_mm"],
        x_mm=table["x_mm"],
        y_mm=table["y_mm"],
        pressure_MPa=table["contact_pressure_MPa"],
        temperature_K=table["temperature_K"],
        sliding_velocity_m_s=table["sliding_velocity_m_s"],
    )


def load_run(run_dir: str | Path) -> RunData:
    run = RunDirectory(resolve_run_dir(run_dir))
    config = load_config(run.config_file)
    mesh = csv_io.read_mesh(run.nodes_initial_file, run.elements_file)
    contour = ToolContour(mesh)
    history = csv_io.read_rows(run.history_file)
    snapshots: list[tuple[int, np.ndarray]] = [(0, mesh.coords)]
    data = RunData(run, config, mesh, contour, history, snapshots)
    for cdir in run.cycle_dirs():
        c = int(cdir.name.split("_")[1])
        if (cdir / "nodes_after.csv").exists():
            snapshots.append((c, csv_io.read_node_coords(cdir / "nodes_after.csv")))
        if (cdir / "process_fields.csv").exists():
            data.fields[c] = fields_from_table(csv_io.read_table(cdir / "process_fields.csv"))
        if (cdir / "wear_rates.csv").exists():
            data.wear_rates[c] = csv_io.read_table(cdir / "wear_rates.csv")["wear_rate_mm_s"]
    if len(snapshots) == 1 and run.state_file.exists():  # CSV output disabled: final state only
        with np.load(run.state_file) as st:
            snapshots.append((int(st["cycle"]), np.array(st["coords"])))
    snapshots.sort(key=lambda s: s[0])
    if run.checkpoint_file.exists():
        data.stop_reason = read_json(run.checkpoint_file).get("stop_reason")
    return data


def make_run_outputs(run_dir: str | Path) -> dict[str, Path]:
    """Create figures 1-6, the GIF and the dashboard for a finished run."""
    from .mesh_plot import plot_displacement_heatmap, plot_initial_mesh
    from .wear_plot import (
        make_wear_gif,
        plot_process_fields,
        plot_wear_indicator,
        plot_wear_progression,
        plot_workflow_diagram,
        render_progression_frames,
    )

    data = load_run(run_dir)
    out = data.config["output"]
    zoom = float(out["zoom_window_mm"])
    figs = data.run.figures
    paths: dict[str, Path] = {}
    if out["save_png"]:
        paths["workflow"] = plot_workflow_diagram(figs / "fig1_workflow.png")
        paths["initial_mesh"] = plot_initial_mesh(data.mesh, data.contour, figs / "fig2_initial_mesh.png", zoom)
        if 1 in data.fields:
            paths["process_fields"] = plot_process_fields(
                data.fields[1], figs / "fig3_process_fields.png", wear_rate=data.wear_rates.get(1),
                title="Synthetic process fields and wear rate - cycle 1",
            )
        paths["wear_progression"] = plot_wear_progression(data.mesh, data.contour, data.snapshots, figs / "fig4_wear_progression.png", zoom)
        paths["wear_indicator"] = plot_wear_indicator(data.history, figs / "fig5_wear_indicator.png", data.config["wear"]["target_flank_wear_mm"])
        final_cycle, final_coords = data.snapshots[-1]
        final_mesh = data.mesh.with_coords(final_coords)
        paths["displacement_heatmap"] = plot_displacement_heatmap(
            final_mesh, data.contour, (final_coords - data.mesh.coords) * 1000.0, figs / "fig6_displacement_heatmap.png", zoom,
            reference_coords=data.mesh.coords, title=f"Cumulative node displacement after cycle {final_cycle}",
        )
        if len(data.snapshots) > 1:
            frames = render_progression_frames(data.mesh, data.contour, data.snapshots, data.history, zoom)
            paths["gif"] = make_wear_gif(frames, figs / "wear_progression.gif")
    if out["save_html"]:
        paths["dashboard"] = write_dashboard(data, figs / "dashboard.html")
    return paths


# ---------------------------------------------------------------------------
def _r(a, n=6) -> list:
    return np.round(np.asarray(a, dtype=float), n).tolist()


def dashboard_payload(data: RunData) -> dict:
    cycles = []
    for c, coords in data.snapshots:
        contour = data.contour.anchored_points(coords)
        outline = data.contour.boundary_polygon(coords)
        entry = {
            "cycle": c,
            "contour": {"x": _r(contour[:, 0]), "y": _r(contour[:, 1])},
            "outline": {"x": _r(np.append(outline[:, 0], outline[0, 0])), "y": _r(np.append(outline[:, 1], outline[0, 1]))},
            "fields": None,
        }
        f = data.fields.get(c)
        if f is not None:
            entry["fields"] = {
                "d_um": _r(f.d_mm * 1000, 3),
                "p": _r(f.pressure_MPa, 2),
                "T": _r(f.temperature_K, 2),
                "v": _r(f.sliding_velocity_m_s, 4),
                "rate_um_min": _r(np.asarray(data.wear_rates.get(c, np.zeros_like(f.d_mm))) * 6.0e4, 5),
            }
        cycles.append(entry)
    (x0, x1), (y0, y1) = ((-0.04, data.config["output"]["zoom_window_mm"]), (-0.03, 0.72 * data.config["output"]["zoom_window_mm"]))
    return {
        "name": data.config["project"]["name"],
        "stop_reason": data.stop_reason,
        "target_vb_um": data.config["wear"]["target_flank_wear_mm"] * 1000,
        "zoom": {"x": [x0, x1], "y": [y0, y1]},
        "history": data.history,
        "cycles": cycles,
    }


def write_dashboard(data: RunData, path: str | Path) -> Path:
    payload = json.dumps(dashboard_payload(data), separators=(",", ":"))
    html = DASHBOARD_TEMPLATE.replace("__PLOTLY__", PLOTLY_CDN).replace("__DATA__", payload)
    Path(path).write_text(html, encoding="utf-8")
    return Path(path)


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tool-Wear Dashboard</title>
<script src="__PLOTLY__"></script>
<style>
  :root { --ink:#1F2430; --muted:#6B7280; --line:#E5E7EB; --bg:#F6F7F9; --card:#FFFFFF;
          --rake:#2E86AB; --edge:#E8A33D; --flank:#C8553D; --navy:#1B3B6F; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }
  header { padding:20px 28px 8px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; }
  .badge { display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px; background:#FDECEA; color:#9B2C2C; margin-left:8px; vertical-align:middle; }
  .controls { display:flex; align-items:center; gap:14px; padding:10px 28px; flex-wrap:wrap; }
  .controls input[type=range] { flex:1; min-width:200px; accent-color: var(--navy); }
  button { border:1px solid var(--line); background:var(--card); padding:6px 14px; border-radius:8px; cursor:pointer; font-weight:600; color:var(--ink); }
  .kpis { display:grid; grid-template-columns: repeat(5, minmax(120px,1fr)); gap:12px; padding:4px 28px 10px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 14px; }
  .kpi .l { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .kpi .v { font-size:20px; font-weight:700; margin-top:2px; font-variant-numeric: tabular-nums; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; padding:6px 28px 28px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:6px; min-height:360px; }
  footer { color:var(--muted); font-size:11px; padding:0 28px 24px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .kpis { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<header>
  <h1>Synthetic Multi-Cycle Tool-Wear Simulation <span class="badge">synthetic data</span></h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="controls">
  <button id="play">&#9654; Play</button>
  <label for="cycle"><b>Cycle</b></label>
  <input id="cycle" type="range" min="0" value="0" step="1">
  <span id="cycleLabel" style="font-variant-numeric: tabular-nums; min-width:70px"></span>
</div>
<div class="kpis">
  <div class="kpi"><div class="l">equiv. cutting time</div><div class="v" id="k_t"></div></div>
  <div class="kpi"><div class="l">flank-wear indicator</div><div class="v" id="k_vb"></div></div>
  <div class="kpi"><div class="l">crater depth</div><div class="v" id="k_kt"></div></div>
  <div class="kpi"><div class="l">removed area</div><div class="v" id="k_a"></div></div>
  <div class="kpi"><div class="l">min. scaled Jacobian</div><div class="v" id="k_q"></div></div>
</div>
<div class="grid">
  <div class="card" id="geom"></div>
  <div class="card" id="fields"></div>
  <div class="card" id="rate"></div>
  <div class="card" id="wear"></div>
</div>
<footer>Portfolio demonstrator. All geometry, process fields and results are synthetic and not experimentally validated.</footer>
<script>
const D = __DATA__;
const C = D.cycles, H = D.history;
const cfg = {displaylogo:false, responsive:true, modeBarButtonsToRemove:['lasso2d','select2d']};
const base = {margin:{l:60,r:20,t:40,b:50}, font:{family:'-apple-system, Segoe UI, Roboto, Arial', size:12, color:'#1F2430'},
              paper_bgcolor:'#fff', plot_bgcolor:'#fff'};
const ax = (t, extra={}) => Object.assign({title:{text:t}, gridcolor:'#EEF0F3', zeroline:false, linecolor:'#C9CED6'}, extra);
const slider = document.getElementById('cycle');
slider.max = C.length - 1;
document.getElementById('subtitle').textContent =
  `${D.name} - ${C.length - 1} cycles - stop reason: ${D.stop_reason || 'n/a'} - VB target ${D.target_vb_um.toFixed(0)} um`;
const hist = c => H.find(h => +h.cycle === c) || H[H.length-1];
const fieldsCycle = i => { for (let k = i; k < C.length; k++) if (C[k].fields) return C[k]; for (let k = i; k >= 0; k--) if (C[k].fields) return C[k]; return null; };

function geom(i) {
  const c = C[i], ref = C[0];
  const traces = [
    {x:c.outline.x, y:c.outline.y, fill:'toself', fillcolor:'#DCE3EA', line:{color:'#4A4E69', width:1}, name:'tool', hoverinfo:'skip'},
    {x:ref.contour.x, y:ref.contour.y, mode:'lines', line:{color:'#1F2430', dash:'dash', width:1.4}, name:'unworn'},
  ];
  if (i > 0) traces.push({x:C[i-1].contour.x, y:C[i-1].contour.y, mode:'lines', line:{color:'#9CC3D5', width:1.4}, name:`cycle ${C[i-1].cycle}`});
  traces.push({x:c.contour.x, y:c.contour.y, mode:'lines+markers', marker:{size:4}, line:{color:'#1B3B6F', width:2.6}, name:`cycle ${c.cycle}`});
  Plotly.react('geom', traces, Object.assign({}, base, {title:{text:'Tool geometry near the cutting edge'},
    xaxis: ax('x [mm]', {range:D.zoom.x}), yaxis: ax('y [mm]', {range:D.zoom.y, scaleanchor:'x'}), legend:{x:0.99, xanchor:'right', y:0.99}}), cfg);
}
function fields(i) {
  const c = fieldsCycle(Math.max(i,1));
  if (!c) return;
  const f = c.fields;
  Plotly.react('fields', [
    {x:f.d_um, y:f.p, name:'pressure [MPa]', line:{color:'#2E86AB', width:2}, yaxis:'y'},
    {x:f.d_um, y:f.T, name:'temperature [K]', line:{color:'#C8553D', width:2}, yaxis:'y2'},
    {x:f.d_um, y:f.v, name:'sliding velocity [m/s]', line:{color:'#4A4E69', width:2}, yaxis:'y3'},
  ], Object.assign({}, base, {title:{text:`Synthetic process fields - cycle ${c.cycle}`}, showlegend:false, margin:{l:70,r:20,t:40,b:50},
    xaxis: ax('d [um]  (flank < 0 < rake)', {range:[-300, 400]}),
    yaxis: ax('pressure [MPa]', {domain:[0.68,1]}), yaxis2: ax('temp. [K]', {domain:[0.35,0.65]}), yaxis3: ax('v_s [m/s]', {domain:[0,0.32]})}), cfg);
}
function rate(i) {
  const c = fieldsCycle(Math.max(i,1));
  if (!c) return;
  const f = c.fields;
  Plotly.react('rate', [{x:f.d_um, y:f.rate_um_min, type:'bar', marker:{color:'#7A3B69'}, name:'wear rate'}],
    Object.assign({}, base, {title:{text:`Wear-rate distribution - cycle ${c.cycle}`}, bargap:0.05,
    xaxis: ax('d [um]  (flank < 0 < rake)', {range:[-300, 400]}), yaxis: ax('wear rate [um/min]')}), cfg);
}
function wear(i) {
  const cyc = H.map(h => +h.cycle), cur = C[i].cycle;
  Plotly.react('wear', [
    {x:cyc, y:H.map(h => 1000*h.vb_indicator_mm), name:'flank-wear indicator', mode:'lines+markers', line:{color:'#C8553D', width:2.4}},
    {x:cyc, y:H.map(h => +h.crater_depth_indicator_um), name:'crater depth', mode:'lines+markers', line:{color:'#2E86AB', width:2.4}},
    {x:cyc, y:H.map(h => +h.edge_recession_um), name:'edge recession', mode:'lines+markers', line:{color:'#E8A33D', width:2.4}},
  ], Object.assign({}, base, {title:{text:'Cumulative wear'}, xaxis: ax('cycle'), yaxis: ax('[um]'),
    legend:{x:0.01, y:0.99},
    shapes:[{type:'line', x0:cur, x1:cur, yref:'paper', y0:0, y1:1, line:{color:'#1B3B6F', dash:'dot'}},
            {type:'line', xref:'paper', x0:0, x1:1, y0:D.target_vb_um, y1:D.target_vb_um, line:{color:'#1F2430', dash:'dash', width:1}}]}), cfg);
}
function kpis(i) {
  const h = hist(C[i].cycle);
  document.getElementById('cycleLabel').textContent = `${C[i].cycle} / ${C[C.length-1].cycle}`;
  document.getElementById('k_t').textContent = (h.elapsed_time_s/60).toFixed(1) + ' min';
  document.getElementById('k_vb').textContent = (1000*h.vb_indicator_mm).toFixed(1) + ' um';
  document.getElementById('k_kt').textContent = (+h.crater_depth_indicator_um).toFixed(1) + ' um';
  document.getElementById('k_a').textContent = (+h.removed_area_um2).toFixed(0) + ' um²';
  document.getElementById('k_q').textContent = (+h.min_scaled_jacobian).toFixed(3);
}
function update() { const i = +slider.value; geom(i); fields(i); rate(i); wear(i); kpis(i); }
slider.addEventListener('input', update);
let timer = null;
document.getElementById('play').addEventListener('click', e => {
  if (timer) { clearInterval(timer); timer = null; e.target.innerHTML = '&#9654; Play'; return; }
  e.target.innerHTML = '&#10074;&#10074; Pause';
  if (+slider.value >= C.length - 1) slider.value = 0;
  timer = setInterval(() => { if (+slider.value >= C.length - 1) { clearInterval(timer); timer = null; e.target.innerHTML = '&#9654; Play'; return; }
    slider.value = +slider.value + 1; update(); }, 650);
});
slider.value = C.length - 1;
update();
</script>
</body>
</html>
"""
