/* FeatureScope dashboard - vanilla JS + Plotly (vendored, works offline on HPC nodes).
 *
 * Linked views: every panel draws the same stable random sample of objects
 * (server-side), and the selection is a bitmask over ALL objects returned by the
 * server, so highlighting is consistent across scatter, spatial, histogram, box
 * and heatmap views regardless of how many objects are loaded.
 */
"use strict";

// URLs are relative so the app works behind proxies (Open OnDemand /rnode/<host>/<port>/).
const api = async (path, opts = {}) => {
  const init = { headers: {}, ...opts };
  if (opts.body && !(opts.body instanceof FormData)) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const r = await fetch("api/" + path, init);
  const txt = await r.text();
  let data;
  try { data = txt ? JSON.parse(txt) : null; } catch { data = txt; }
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
};
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k.nodeType ? k : String(k));
  return e;
};
const store = {
  get(k, d) { try { const v = localStorage.getItem("fs:" + k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("fs:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
function toast(msg, error = false, ms = 4000) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (error ? " error" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), error ? 9000 : ms);
}
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message, true); console.error(e); } };

// ---------------------------------------------------------------- decoding
const decF32 = (b) => new Float32Array(Uint8Array.from(atob(b), (c) => c.charCodeAt(0)).buffer);
const decI32 = (b) => new Int32Array(Uint8Array.from(atob(b), (c) => c.charCodeAt(0)).buffer);
const decI16 = (b) => new Int16Array(Uint8Array.from(atob(b), (c) => c.charCodeAt(0)).buffer);
function decMask(b) {
  if (!b) return null;
  const bytes = Uint8Array.from(atob(b), (c) => c.charCodeAt(0));
  return { has: (i) => (bytes[i >> 3] >> (i & 7)) & 1 };
}

// ---------------------------------------------------------------- colour (reference palette, validated)
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const isDark = () => css("color-scheme") === "dark";
const CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
// Beyond 8 groups hue alone cannot stay distinguishable: further groups reuse the hues at
// a lighter/darker step, and every scatter gets direct centroid labels + a clickable legend.
const CAT_EXTRA = ["#86b6ef", "#f4a37f", "#7fd6b4", "#f6cf73", "#f3b6cd", "#59b359", "#a79cf0", "#f09a99",
  "#184f95", "#a3401a", "#0f6b49", "#8a5d00", "#a33964", "#004d00", "#3a2d8f", "#9b2423"];
const catColor = (i) => (i < 8 ? (isDark() ? CAT_DARK : CAT_LIGHT)[i] : CAT_EXTRA[(i - 8) % CAT_EXTRA.length]);
const SEQ = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]];
const SEQ_DARK = [[0, "#0d366b"], [0.25, "#1c5cab"], [0.5, "#3987e5"], [0.75, "#86b6ef"], [1, "#cde2fb"]];
const divScale = () => [[0, "#1c5cab"], [0.25, "#6da7ec"], [0.5, isDark() ? "#383835" : "#f0efec"], [0.75, "#ec8a83"], [1, "#b42b2b"]];

function baseLayout(extra = {}) {
  const ink = css("--text-secondary"), grid = css("--border");
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: css("--font") || "system-ui", size: 11, color: ink },
    margin: { l: 50, r: 12, t: 8, b: 40 }, showlegend: false, dragmode: "lasso",
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, automargin: true },
    yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, automargin: true },
    hoverlabel: { bgcolor: css("--surface-1"), bordercolor: grid, font: { color: css("--text-primary") } },
    ...extra,
  };
}
// Plotly.react on a panel closed/rebuilt while its data was loading throws; skip those.
const react = (p, traces, layout) => { if (!p.dead && p.plot.isConnected) return Plotly.react(p.plot, traces, layout, PLOT_CFG); };
const PLOT_CFG = { responsive: true, displaylogo: false, scrollZoom: true,
  modeBarButtonsToRemove: ["autoScale2d", "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"] };

// ---------------------------------------------------------------- state
const S = {
  state: null, ops: [], opsByName: {}, selMask: null, selCount: 0,
  panels: [], nextId: 1, openSteps: new Set(),
};

function columnOptions({ numeric = true, categorical = true, includeNone = false } = {}) {
  const c = S.state?.columns;
  if (!c) return [];
  const out = [];
  if (includeNone) out.push(["", "— none —"]);
  if (numeric) {
    for (const [k, n] of Object.entries(c.obsm)) {
      const dims = k === "spatial" ? 2 : Math.min(n, k === "X_pca" ? 10 : 3);
      for (let d = 0; d < dims; d++) out.push([`obsm:${k}:${d}`, `${k.replace(/^X_/, "")} ${d + 1}`, "Embeddings"]);
    }
    for (const f of c.features) out.push([`feat:${f}`, f, "Features (processed)"]);
    if (c.has_raw) for (const f of c.features) out.push([`raw:${f}`, f + " (raw)", "Features (raw)"]);
    for (const o of c.obs_numeric) out.push([`obs:${o}`, o, "Metadata / results (numeric)"]);
  }
  if (categorical) for (const o of c.obs_categorical) out.push([`obs:${o}`, o, "Groups (categorical)"]);
  return out;
}
function fillSelect(sel, options, value) {
  sel.innerHTML = "";
  let group = null, cur = sel;
  for (const [v, label, g] of options) {
    if (g && g !== group) { group = g; cur = h("optgroup", { label: g }); sel.append(cur); }
    else if (!g) cur = sel;
    cur.append(h("option", { value: v }, label));
  }
  if (value !== undefined && options.some((o) => o[0] === value)) sel.value = value;
}

// ================================================================= panels
const PANEL_TYPES = {
  scatter: { title: "Scatter", make: ScatterPanel },
  spatial: { title: "Spatial", make: (p) => ScatterPanel(p, true) },
  hist: { title: "Histogram", make: HistPanel },
  groups: { title: "Feature by group", make: GroupsPanel },
  heatmap: { title: "Group heatmap", make: HeatmapPanel },
  markers: { title: "Selection vs rest", make: MarkersPanel },
  result: { title: "Result", make: ResultPanel },
};

function addPanel(type, cfg = {}) {
  const id = S.nextId++;
  const el = h("div", { class: "panel" + (type === "heatmap" || type === "result" ? " wide" : "") });
  const p = { id, type, cfg: { ...cfg }, el };
  const head = h("div", { class: "phead" }, h("span", { class: "ptitle" }, PANEL_TYPES[type].title));
  p.head = head;
  p.plot = h("div", { class: "plot" });
  p.foot = h("div", { class: "pfoot" });
  el.append(head, p.plot, p.foot);
  if (type === "result") { $("#panels").prepend(el); setTimeout(() => el.scrollIntoView({ behavior: "smooth" }), 50); }
  else $("#panels").append(el);
  Object.assign(p, PANEL_TYPES[type].make(p));
  head.append(h("span", { class: "sp" }), h("button", { class: "ghost", title: "Close", onclick: () => removePanel(p) }, "✕"));
  S.panels.push(p);
  $("#empty").classList.add("hidden");
  p.refresh();
  saveLayout();
  return p;
}
function removePanel(p) {
  p.dead = true;
  Plotly.purge(p.plot);
  p.el.remove();
  S.panels = S.panels.filter((x) => x !== p);
  saveLayout();
}
function saveLayout() {
  if (!S.state?.project) return;
  store.set("layout:" + S.state.project.name, S.panels.filter((p) => p.type !== "result").map((p) => ({ type: p.type, cfg: p.cfg })));
}
function restoreLayout() {
  for (const p of [...S.panels]) removePanel(p);
  const saved = S.state?.project && store.get("layout:" + S.state.project.name, null);
  if (saved && saved.length) saved.forEach((p) => addPanel(p.type, p.cfg));
  else defaultLayout();
}
function defaultLayout() {
  const c = S.state.columns;
  const f = c.features;
  const emb = Object.keys(c.obsm).find((k) => k === "X_umap") || Object.keys(c.obsm).find((k) => k.startsWith("X_"));
  const cat = c.obs_categorical.find((x) => /leiden|louvain|kmeans|flowsom|cluster/.test(x)) || c.obs_categorical[0];
  if (emb) addPanel("scatter", { x: `obsm:${emb}:0`, y: `obsm:${emb}:1`, color: cat ? `obs:${cat}` : "" });
  addPanel("scatter", { x: `feat:${f[0]}`, y: `feat:${f[1] || f[0]}`, color: "" });
  if (c.obsm.spatial) addPanel("spatial", { color: cat ? `obs:${cat}` : "" });
  addPanel("hist", { x: `feat:${f[0]}` });
}
async function refreshPanels(onlySelection = false) {
  await Promise.all(S.panels.map((p) => (onlySelection && p.onSelection ? p.onSelection() : p.refresh()).catch((e) => console.error(e))));
}

function headSelect(p, key, options, label) {
  const sel = h("select", { title: label });
  fillSelect(sel, options, p.cfg[key]);
  if (p.cfg[key] === undefined || !options.some((o) => o[0] === p.cfg[key])) p.cfg[key] = sel.value;
  sel.addEventListener("change", () => { p.cfg[key] = sel.value; saveLayout(); p.refresh(); });
  p.head.append(sel);
  return sel;
}

// ---------------------------------------------------------------- scatter / spatial
function ScatterPanel(p, spatial = false) {
  if (spatial) { p.cfg.x = "obsm:spatial:0"; p.cfg.y = "obsm:spatial:1"; }
  else {
    headSelect(p, "x", columnOptions({ categorical: false }), "x axis");
    headSelect(p, "y", columnOptions({ categorical: false }), "y axis");
  }
  headSelect(p, "color", columnOptions({ includeNone: true }), "colour by");
  const gateSel = h("select", { title: "Show only objects within this gate (hierarchical gating)" });
  const gates = Object.keys(S.state.gates || {});
  fillSelect(gateSel, [["", "all objects"], ...gates.map((g) => [g, "within " + g])], p.cfg.within_gate || "");
  gateSel.addEventListener("change", () => { p.cfg.within_gate = gateSel.value; saveLayout(); p.refresh(); });
  p.head.append(gateSel);
  const bk = S.state.batch_key;
  if (spatial && bk && S.state.categories[bk]) {   // slides/images share coordinates: show one at a time
    const fs = h("select", { title: "Image / slide" });
    const vals = S.state.categories[bk];
    if (p.cfg.library === undefined) p.cfg.library = vals[0];
    fillSelect(fs, [["", "all " + bk + "s (overlaid)"], ...vals.map((v) => [v, v])], p.cfg.library);
    fs.addEventListener("change", () => { p.cfg.library = fs.value; saveLayout(); p.refresh(); });
    p.head.append(fs);
  }
  const legend = h("div", { class: "legend" });
  p.el.insertBefore(legend, p.foot);
  const saveGateBtn = h("button", { class: "ghost", disabled: true, title: "Save the last lasso/box region as a gate" }, "Save gate");
  saveGateBtn.addEventListener("click", guard(async () => {
    const name = prompt("Gate name", "");
    if (!name) return;
    const r = await api("gates", { method: "POST", body: { ...p.lastRegion, name, parent: p.cfg.within_gate || null } });
    S.state.gates = r.gates; S.state.gate_stats = r.gate_stats;
    renderGates(); toast(`gate “${name}” saved`); refreshPanels();
  }));
  p.foot.append(saveGateBtn);
  const info = h("span");
  p.foot.append(info);
  let data = null;

  async function load() {
    const body = { x: p.cfg.x, y: p.cfg.y, color: p.cfg.color || null, max_points: p.cfg.max_points || 150000,
      within_gate: p.cfg.within_gate || null,
      filter: spatial && p.cfg.library && S.state.batch_key ? { column: S.state.batch_key, value: p.cfg.library } : null };
    const r = await api("plot/scatter", { method: "POST", body });
    data = { idx: decI32(r.idx), x: decF32(r.x.values), y: decF32(r.y.values), color: r.color, n_total: r.n_total };
    if (r.color) {
      if (r.color.kind === "cat") r.color.codesArr = decI16(r.color.codes);
      else r.color.vals = decF32(r.color.values);
    }
    info.textContent = `${r.n_shown.toLocaleString()} of ${r.n_total.toLocaleString()} objects shown` +
      (r.n_shown < r.n_total ? " (display sample; selection & gates use all)" : "");
    draw();
  }

  function draw() {
    if (!data) return;
    const n = data.idx.length, c = data.color, sel = S.selMask;
    const colors = new Array(n);
    let cvals = null;
    if (c && c.kind === "cat") for (let i = 0; i < n; i++) colors[i] = c.codesArr[i] < 0 ? css("--dim") : catColor(c.codesArr[i]);
    else if (c) cvals = c.vals;
    const traces = [];
    const mk = (ids, opacity, size, override) => {
      const xs = new Float32Array(ids.length), ys = new Float32Array(ids.length);
      const col = override ? override : cvals ? new Float32Array(ids.length) : new Array(ids.length);
      const cd = new Int32Array(ids.length);
      ids.forEach((k, j) => { xs[j] = data.x[k]; ys[j] = data.y[k]; cd[j] = data.idx[k];
        if (!override) col[j] = cvals ? cvals[k] : (c ? colors[k] : css("--accent")); });
      const marker = { size, opacity, color: col };
      if (cvals && !override) Object.assign(marker, { colorscale: isDark() ? SEQ_DARK : SEQ, cmin: c.p1, cmax: c.p99,
        showscale: true, colorbar: { thickness: 10, len: 0.8, outlinewidth: 0, tickfont: { size: 10 } } });
      return { type: "scattergl", mode: "markers", x: xs, y: ys, customdata: cd, marker,
        hovertemplate: "object #%{customdata}<br>x %{x:.3g}<br>y %{y:.3g}<extra></extra>" };
    };
    const size = n > 50000 ? 2.5 : n > 10000 ? 3.5 : 5;
    const all = [...Array(n).keys()];
    if (sel) {
      const inS = [], outS = [];
      for (let i = 0; i < n; i++) (sel.has(data.idx[i]) ? inS : outS).push(i);
      traces.push(mk(outS, 0.35, size, css("--dim")));
      traces.push(mk(inS, 0.9, size + 1, c ? null : css("--sel")));
    } else traces.push(mk(all, n > 50000 ? 0.6 : 0.8, size));

    // gate outlines drawn on matching axes
    const shapes = [], annotations = [];
    for (const g of Object.values(S.state.gates || {})) {
      if (g.x !== p.cfg.x) continue;
      const line = { color: css("--text-primary"), width: 1.5, dash: "dot" };
      if (g.type === "polygon" && g.y === p.cfg.y)
        shapes.push({ type: "path", path: "M" + g.vertices.map((v) => v.join(",")).join("L") + "Z", line });
      else if (g.type === "rect" && g.y === p.cfg.y)
        shapes.push({ type: "rect", x0: g.xrange[0], x1: g.xrange[1], y0: g.yrange[0], y1: g.yrange[1], line });
      else continue;
      const vx = g.vertices ? g.vertices[0] : [g.xrange[0], g.yrange[1]];
      annotations.push({ x: vx[0], y: vx[1], text: g.name, showarrow: false, font: { size: 10, color: css("--text-primary") }, yshift: 8 });
    }
    // direct labels at category centroids (identity never relies on colour alone)
    legend.innerHTML = "";
    if (c && c.kind === "cat") {
      const k = c.categories.length, sx = new Float64Array(k), sy = new Float64Array(k), cnt = new Int32Array(k);
      for (let i = 0; i < n; i++) { const g = c.codesArr[i]; if (g >= 0) { sx[g] += data.x[i]; sy[g] += data.y[i]; cnt[g]++; } }
      const label = !spatial && !p.cfg.y.startsWith("feat:") && !p.cfg.y.startsWith("raw:") && k <= 40;
      c.categories.forEach((name, g) => {
        if (label && cnt[g]) annotations.push({ x: sx[g] / cnt[g], y: sy[g] / cnt[g], text: name, showarrow: false,
          font: { size: 11, color: css("--text-primary") }, bgcolor: css("--surface-1"), opacity: 0.85, borderpad: 1 });
        legend.append(h("span", { title: "click: select this group · shift-click: add",
          onclick: guard(async (ev) => applySelection(await api("select/category", { method: "POST",
            body: { column: p.cfg.color.replace(/^obs:/, ""), values: [name], mode: ev.shiftKey ? "add" : $("#selMode").value } }))) },
        h("i", { style: `background:${catColor(g)}` }), name));
      });
    }
    const lab = (s) => s.replace(/^(feat|obs|raw|obsm):/, "").replace(/^X_/, "").replace(/:(\d+)$/, (m, d) => " " + (+d + 1));
    const layout = baseLayout({ shapes, annotations, uirevision: p.cfg.x + p.cfg.y,
      xaxis: { ...baseLayout().xaxis, title: { text: lab(p.cfg.x), standoff: 4 } },
      yaxis: { ...baseLayout().yaxis, title: { text: lab(p.cfg.y), standoff: 4 } } });
    if (spatial) Object.assign(layout.yaxis, { scaleanchor: "x", autorange: "reversed" }); // image coordinates
    react(p, traces, layout, PLOT_CFG);
    if (!p.bound) {
      p.bound = true;
      p.plot.on("plotly_selected", guard(async (ev) => {
        if (!ev) return;
        let region;
        if (ev.lassoPoints) region = { type: "polygon", vertices: ev.lassoPoints.x.map((x, i) => [x, ev.lassoPoints.y[i]]) };
        else if (ev.range) region = { type: "rect", xrange: ev.range.x, yrange: ev.range.y };
        else return;
        region.x = p.cfg.x; region.y = p.cfg.y;
        p.lastRegion = region;
        saveGateBtn.disabled = false;
        const r = await api("select", { method: "POST", body: { ...region, mode: $("#selMode").value, within_gate: p.cfg.within_gate || null } });
        applySelection(r);
      }));
      p.plot.on("plotly_deselect", () => {});
    }
  }
  return { refresh: load, onSelection: async () => draw() };
}

// ---------------------------------------------------------------- histogram
function HistPanel(p) {
  headSelect(p, "x", columnOptions({ categorical: false }), "feature");
  const logBox = h("label", { class: "check" }, h("input", { type: "checkbox" }), "log y");
  logBox.firstChild.checked = !!p.cfg.logy;
  logBox.firstChild.addEventListener("change", (e) => { p.cfg.logy = e.target.checked; saveLayout(); p.refresh(); });
  p.head.append(logBox);
  const saveGateBtn = h("button", { class: "ghost", disabled: true }, "Save gate");
  saveGateBtn.addEventListener("click", guard(async () => {
    const name = prompt("Gate name (range gate)", "");
    if (!name) return;
    const r = await api("gates", { method: "POST", body: { ...p.lastRegion, name } });
    S.state.gates = r.gates; S.state.gate_stats = r.gate_stats; renderGates(); refreshPanels();
  }));
  p.foot.append(h("span", {}, "Box-select a range to select objects"), saveGateBtn);
  async function load() {
    const r = await api("plot/hist", { method: "POST", body: { x: p.cfg.x, bins: 80 } });
    const mids = r.edges.slice(0, -1).map((e, i) => (e + r.edges[i + 1]) / 2);
    const w = r.edges[1] - r.edges[0];
    const traces = [{ type: "bar", x: mids, y: r.all, width: w * 0.92, marker: { color: r.selected ? css("--dim") : css("--accent") },
      name: "all", hovertemplate: "%{x:.3g}: %{y} objects<extra>all</extra>" }];
    if (r.selected) traces.push({ type: "bar", x: mids, y: r.selected, width: w * 0.92, marker: { color: css("--sel") },
      name: "selected", hovertemplate: "%{x:.3g}: %{y} selected<extra></extra>" });
    const shapes = Object.values(S.state.gates || {}).filter((g) => g.type === "range" && g.x === p.cfg.x)
      .map((g) => ({ type: "rect", xref: "x", yref: "paper", x0: g.range[0], x1: g.range[1], y0: 0, y1: 1,
        line: { color: css("--text-primary"), dash: "dot", width: 1 }, fillcolor: "rgba(0,0,0,0)" }));
    react(p, traces, baseLayout({ barmode: "overlay", dragmode: "select", selectdirection: "h", shapes,
      showlegend: !!r.selected, legend: { orientation: "h", y: 1.08 },
      yaxis: { ...baseLayout().yaxis, type: p.cfg.logy ? "log" : "linear", title: { text: "objects" } },
      xaxis: { ...baseLayout().xaxis, title: { text: p.cfg.x.replace(/^\w+:/, "") } } }));
    if (!p.bound) {
      p.bound = true;
      p.plot.on("plotly_selected", guard(async (ev) => {
        if (!ev || !ev.range) return;
        p.lastRegion = { type: "range", x: p.cfg.x, range: ev.range.x };
        saveGateBtn.disabled = false;
        applySelection(await api("select", { method: "POST", body: { ...p.lastRegion, mode: $("#selMode").value } }));
      }));
    }
  }
  return { refresh: load };
}

// ---------------------------------------------------------------- feature by group (box statistics)
function GroupsPanel(p) {
  headSelect(p, "y", columnOptions({ categorical: false }), "feature");
  headSelect(p, "groupby", columnOptions({ numeric: false }), "group by");
  p.foot.append("Whiskers 5–95th percentile · click a box to select the group");
  async function load() {
    if (!p.cfg.groupby) { p.plot.textContent = "No categorical column yet (run clustering)"; return; }
    const r = await api("plot/groups", { method: "POST", body: { y: p.cfg.y, groupby: p.cfg.groupby } });
    const traces = r.map((g, i) => ({
      type: "box", name: g.group, x: [g.group], q1: [g.q[1]], median: [g.q[2]], q3: [g.q[3]],
      lowerfence: [g.q[0]], upperfence: [g.q[4]], mean: [g.mean], marker: { color: catColor(i) }, line: { width: 1.5 },
      fillcolor: catColor(i) + "55", hovertemplate: `${g.group}<br>n=${g.n.toLocaleString()}` +
        (g.n_sel !== undefined ? ` (${g.n_sel.toLocaleString()} selected)` : "") + "<extra></extra>",
    }));
    react(p, traces, baseLayout({ dragmode: false, yaxis: { ...baseLayout().yaxis, title: { text: p.cfg.y.replace(/^\w+:/, "") } } }));
    if (!p.bound) {
      p.bound = true;
      p.plot.on("plotly_click", guard(async (ev) => {
        const name = ev.points[0].data.name;
        applySelection(await api("select/category", { method: "POST", body: { column: p.cfg.groupby.replace(/^obs:/, ""), values: [name], mode: $("#selMode").value } }));
      }));
    }
  }
  return { refresh: load, onSelection: load };
}

// ---------------------------------------------------------------- group x feature heatmap
function HeatmapPanel(p) {
  headSelect(p, "groupby", columnOptions({ numeric: false }), "group by");
  const layerSel = h("select", { title: "values" });
  fillSelect(layerSel, [["", "current X"], ...(S.state.layers || []).map((l) => [l, l])], p.cfg.layer || "");
  layerSel.addEventListener("change", () => { p.cfg.layer = layerSel.value; saveLayout(); p.refresh(); });
  p.head.append(layerSel);
  p.foot.append("Group means, z-scored per feature (diverging: blue low · red high) · click a row to select the group");
  async function load() {
    if (!p.cfg.groupby) { p.plot.textContent = "No categorical column yet (run clustering)"; return; }
    const r = await api("plot/heatmap", { method: "POST", body: { groupby: p.cfg.groupby, layer: p.cfg.layer || null } });
    const rows = r.rows.map((g, i) => `${g} (${r.counts[i].toLocaleString()})`);
    react(p, [{ type: "heatmap", z: r.z, x: r.cols, y: rows, colorscale: divScale(), zmid: 0,
      xgap: 1, ygap: 1, colorbar: { thickness: 10, outlinewidth: 0, title: { text: "z" } },
      hovertemplate: "%{y}<br>%{x}: %{z:.2f}<extra></extra>" }],
    baseLayout({ dragmode: false, margin: { l: 110, r: 10, t: 8, b: 80 }, yaxis: { automargin: true }, xaxis: { tickangle: -45, automargin: true } }));
    if (!p.bound) {
      p.bound = true;
      p.plot.on("plotly_click", guard(async (ev) => {
        const name = ev.points[0].y.replace(/ \([\d,.\s]+\)$/, "");
        applySelection(await api("select/category", { method: "POST", body: { column: p.cfg.groupby.replace(/^obs:/, ""), values: [name], mode: $("#selMode").value } }));
      }));
    }
  }
  return { refresh: load };
}

// ---------------------------------------------------------------- selection vs rest
function MarkersPanel(p) {
  p.foot.append("Standardized mean difference of each feature, selection vs all other objects");
  async function load() {
    const r = await api("selection/stats");
    if (!r.length) { Plotly.purge(p.plot); p.plot.innerHTML = '<p class="hint" style="padding:12px">Make a selection in any view.</p>'; return; }
    p.plot.innerHTML = "";
    const top = r.slice(0, 25).reverse();
    react(p, [{ type: "bar", orientation: "h", y: top.map((d) => d.feature), x: top.map((d) => d.effect),
      marker: { color: top.map((d) => (d.effect > 0 ? "#d03b3b" : "#2a78d6")) },
      hovertemplate: "%{y}: d=%{x:.2f}<extra></extra>" }],
    baseLayout({ dragmode: false, margin: { l: 110, r: 10, t: 8, b: 36 }, xaxis: { ...baseLayout().xaxis, title: { text: "effect size (Cohen's d)" } } }));
  }
  return { refresh: load, onSelection: load };
}

// ---------------------------------------------------------------- analysis results (uns)
function ResultPanel(p) {
  p.head.querySelector(".ptitle").textContent = p.cfg.key;
  async function load() {
    const r = await api("results/" + encodeURIComponent(p.cfg.key));
    if (r.kind === "matrix") {
      const sym = p.cfg.key.startsWith("nhood");
      react(p, [{ type: "heatmap", z: r.z, x: r.cols, y: r.rows, xgap: 1, ygap: 1,
        colorscale: sym ? divScale() : (isDark() ? SEQ_DARK : SEQ), zmid: sym ? 0 : undefined,
        colorbar: { thickness: 10, outlinewidth: 0 }, hovertemplate: "%{y} · %{x}: %{z:.3g}<extra></extra>" }],
      baseLayout({ dragmode: false, margin: { l: 110, r: 10, t: 8, b: 80 }, xaxis: { tickangle: -45, automargin: true }, yaxis: { automargin: true } }));
      p.foot.textContent = sym ? "Neighborhood enrichment z-score: red = neighbors more often than chance, blue = avoid each other" : "";
    } else if (r.kind === "bar") {
      react(p, [{ type: "bar", x: r.x, y: r.y, marker: { color: css("--accent") } }], baseLayout({ dragmode: false }));
    } else if (r.kind === "curve") {
      react(p, [{ type: "scatter", mode: "lines", y: r.y, line: { color: css("--accent"), width: 2 } }],
        baseLayout({ dragmode: false, xaxis: { ...baseLayout().xaxis, title: { text: p.cfg.key === "pca" ? "component" : "epoch" } } }));
    } else if (r.kind === "table") {
      const wrap = h("div", { class: "tablewrap" });
      const t = h("table", { class: "tbl" }, h("tr", {}, r.columns.map((c) => h("th", {}, c))),
        r.rows.slice(0, 500).map((row) => h("tr", {}, row.map((v) => h("td", {}, typeof v === "number" ? (+v.toPrecision(4)).toString() : v ?? "")))));
      wrap.append(t);
      p.plot.replaceWith(wrap); p.plot = wrap;
    } else p.plot.textContent = JSON.stringify(r.value).slice(0, 2000);
  }
  return { refresh: load };
}

// ================================================================= selection
function applySelection(r) {
  S.selMask = decMask(r.mask);
  S.selCount = r.count;
  $("#selCount").textContent = `${r.count.toLocaleString()} selected`;
  $("#selCount").classList.toggle("on", r.count > 0);
  refreshPanels(true);
}

// ================================================================= sidebar: data
async function loadProjects() {
  const list = await api("projects");
  const el = $("#projectList");
  el.innerHTML = "";
  if (!list.length) el.append(h("p", { class: "hint" }, "No projects in the workspace yet."));
  for (const p of list) {
    const dir = p.root.split(/[\\/]/).pop();
    el.append(h("div", { class: "item", onclick: guard(async () => { setState(await api("projects/open", { method: "POST", body: { name: dir } })); restoreLayout(); }) },
      h("span", { class: "grow" }, p.name), h("small", {}, `${p.n_obs.toLocaleString()} × ${p.n_vars}`)));
  }
}

let pendingImport = null;
function renderImport(pre) {
  pendingImport = pre;
  const box = $("#importRoles");
  box.classList.remove("hidden");
  box.innerHTML = "";
  if (pre.anndata) {
    box.append(h("p", { class: "hint" }, "AnnData file: roles are taken from the file."), h("button", { class: "primary", onclick: createProject }, "Create project"));
    return;
  }
  const r = pre.roles;
  const opt = (v) => [["", "— none —"], ...pre.columns.map((c) => [c, c])];
  const sx = h("select"), sy = h("select"), sb = h("select"), si = h("select");
  fillSelect(sx, opt(), r.spatial?.[0] || ""); fillSelect(sy, opt(), r.spatial?.[1] || "");
  fillSelect(sb, opt(), r.batch || ""); fillSelect(si, opt(), r.id || "");
  const feats = h("div", { class: "feats" }, pre.columns.map((c) => h("label", {},
    h("input", { type: "checkbox", value: c, checked: r.features.includes(c) }), c)));
  box.append(h("div", { class: "roles" },
    h("b", {}, "Column roles"), h("p", { class: "hint" }, "Checked numeric columns become features (used for clustering/embedding). Everything else is kept as metadata."),
    feats,
    h("div", { class: "row" }, h("span", {}, "x"), sx, h("span", {}, "y"), sy),
    h("label", {}, "Batch / sample column", sb),
    h("label", {}, "Object ID column", si),
    h("button", { class: "primary", onclick: createProject }, "Create project")));
  pendingImport.read = () => ({
    features: $$("input:checked", feats).map((i) => i.value),
    spatial: sx.value && sy.value ? [sx.value, sy.value] : null,
    batch: sb.value || null, id_col: si.value || null,
  });
}
const createProject = guard(async () => {
  const name = $("#newName").value.trim() || pendingImport.path.split(/[\\/]/).pop().replace(/\.\w+$/, "");
  toast("Importing…");
  const body = { name, path: pendingImport.path, ...(pendingImport.read ? pendingImport.read() : {}) };
  setState(await api("projects/create", { method: "POST", body }));
  $("#importRoles").classList.add("hidden");
  restoreLayout();
  loadProjects();
  toast("Project created");
});

// ================================================================= sidebar: pipeline
const TEMPLATES = {
  imaging: [
    ["filter_objects", { column: null, drop_nan: true }],
    ["transform", { method: "arcsinh", cofactor: 150 }],
    ["clip", { low: 0.001, high: 0.999 }],
    ["batch_center", {}],
    ["normalize", { method: "zscore", max_value: 5 }],
    ["pca", { n_comps: 20 }],
    ["harmony", {}],
    ["neighbors", { use_rep: "X_harmony", n_neighbors: 15 }],
    ["umap", {}],
    ["leiden", { resolution: 0.8 }],
    ["markers", { groupby: "leiden" }],
  ],
  transcriptomics: [
    ["normalize", { method: "total", target_sum: 10000 }],
    ["transform", { method: "log1p" }],
    ["highly_variable", { n_top: 2000, flavor: "dispersion" }],
    ["save_layer", { name: "lognorm" }],
    ["normalize", { method: "zscore", max_value: 10 }],
    ["pca", { n_comps: 50 }],
    ["neighbors", { n_neighbors: 15 }],
    ["umap", {}],
    ["leiden", { resolution: 1.0 }],
    ["markers", { groupby: "leiden", layer: "lognorm" }],
  ],
  spatial: [
    ["transform", { method: "arcsinh", cofactor: 150 }],
    ["batch_center", {}],
    ["normalize", { method: "zscore", max_value: 5 }],
    ["pca", { n_comps: 15 }],
    ["harmony", {}],
    ["neighbors", { use_rep: "X_harmony" }],
    ["umap", {}],
    ["leiden", { resolution: 0.5 }],
    ["spatial_neighbors", { method: "knn", n_neighbors: 10 }],
    ["nhood_enrichment", { cluster_key: "leiden" }],
    ["cellular_neighborhoods", { cluster_key: "leiden", n_neighborhoods: 6 }],
    ["spatial_dgi", { use_rep: "X_harmony", epochs: 200 }],
    ["kmeans", { use_rep: "X_spatial_dgi", n_clusters: 6, key: "spatial_domain" }],
    ["spatial_autocorr", {}],
  ],
  deep: [
    ["transform", { method: "arcsinh", cofactor: 150 }],
    ["normalize", { method: "zscore", max_value: 5 }],
    ["vae", { latent_dim: 16, epochs: 30, condition_on_batch: true }],
    ["neighbors", { use_rep: "X_vae" }],
    ["umap", {}],
    ["leiden", { resolution: 0.8 }],
    ["markers", { groupby: "leiden" }],
  ],
};
const uid = () => Math.random().toString(16).slice(2, 10);

function renderPipeline() {
  const st = S.state;
  const pipe = st.pipeline;
  const planBy = Object.fromEntries((st.plan || []).map((p) => [p.id, p]));
  const repBy = Object.fromEntries((st.report || []).map((p) => [p.id, p]));
  const el = $("#steps");
  el.innerHTML = "";
  pipe.steps.forEach((step, i) => {
    const info = S.opsByName[step.op];
    const plan = planBy[step.id] || {};
    const rep = repBy[step.id] || {};
    const status = plan.status || "pending";
    const open = S.openSteps.has(step.id);
    const body = h("div", { class: "body" + (open ? "" : " hidden") });
    for (const prm of info.params) body.append(paramInput(step, prm));
    body.append(h("p", { class: "hint" }, info.doc));
    const tools = h("span", { class: "tools" },
      h("button", { class: "ghost", title: "Run up to here", onclick: (e) => { e.stopPropagation(); runPipeline({ upto: step.id }); } }, "▶"),
      h("button", { class: "ghost", title: "Move up", onclick: (e) => { e.stopPropagation(); move(i, -1); } }, "↑"),
      h("button", { class: "ghost", title: "Move down", onclick: (e) => { e.stopPropagation(); move(i, 1); } }, "↓"),
      h("button", { class: "ghost", title: step.enabled ? "Disable" : "Enable", onclick: (e) => { e.stopPropagation(); step.enabled = !step.enabled; pushPipeline(); } }, step.enabled ? "⏸" : "⏵"),
      h("button", { class: "ghost", title: "Remove", onclick: (e) => { e.stopPropagation(); pipe.steps.splice(i, 1); pushPipeline(); } }, "✕"));
    const logTxt = rep.log || plan.meta?.log || "";
    const secs = plan.meta?.seconds;
    const head = h("div", { class: "head", onclick: () => { open ? S.openSteps.delete(step.id) : S.openSteps.add(step.id); renderPipeline(); } },
      h("span", { class: "status-dot " + status, title: status }),
      h("span", { class: "title" }, `${i + 1}. ${info.label}`,
        h("span", { class: "log" }, status === "cached" ? `${logTxt}${secs ? ` · ${secs.toFixed(1)}s` : ""}` : status)),
      info.gpu ? h("span", { class: "badge", title: "GPU-accelerated when available" }, "GPU") : null,
      tools);
    el.append(h("div", { class: "step" + (step.enabled ? "" : " disabled") }, head, body));
  });
  if (!pipe.steps.length) el.append(h("p", { class: "hint" }, "Empty pipeline: add steps or start from a template."));
  fillSelect($("#pipelineSelect"), (st.project.pipelines || []).map((n) => [n, n]), pipe.name);
}
function paramInput(step, prm) {
  const v = step.params[prm.name] ?? prm.default;
  const set = (val) => { if (val === "" || val === null || (typeof val === "number" && isNaN(val))) delete step.params[prm.name]; else step.params[prm.name] = val; pushPipeline(false); };
  let input;
  if (prm.type === "bool") {
    input = h("input", { type: "checkbox", checked: !!v });
    input.addEventListener("change", () => set(input.checked));
    return h("label", { class: "check", title: prm.help }, input, prm.name);
  }
  if (prm.type === "choice") {
    input = h("select"); fillSelect(input, prm.choices.map((c) => [c, c]), v);
    input.addEventListener("change", () => set(input.value));
  } else if (prm.type === "column") {
    input = h("select"); fillSelect(input, columnOptions({ includeNone: true }), v || "");
    input.addEventListener("change", () => set(input.value || null));
  } else if (prm.type === "gate") {
    input = h("select"); fillSelect(input, [["", "— choose —"], ...Object.keys(S.state.gates || {}).map((g) => [g, g])], v || "");
    input.addEventListener("change", () => set(input.value || null));
  } else if (prm.type === "int" || prm.type === "float") {
    input = h("input", { type: "number", step: prm.type === "int" ? 1 : "any", value: v ?? "" });
    input.addEventListener("change", () => set(input.value === "" ? null : prm.type === "int" ? parseInt(input.value) : parseFloat(input.value)));
  } else if (prm.type === "columns" || prm.type === "features") {
    input = h("input", { value: Array.isArray(v) ? v.join(", ") : v ?? "", placeholder: prm.type === "features" ? "comma list or regex" : "obs:col1, obs:col2" });
    input.addEventListener("change", () => {
      const t = input.value.trim();
      set(!t ? null : t.includes(",") || prm.type === "columns" ? t.split(",").map((s) => s.trim()).filter(Boolean) : t);
    });
  } else {
    input = h("input", { value: v ?? "" });
    input.addEventListener("change", () => set(input.value.trim() || null));
  }
  return h("label", { title: prm.help }, `${prm.name}${prm.help ? " — " + prm.help : ""}`, input);
}
function move(i, d) {
  const s = S.state.pipeline.steps, j = i + d;
  if (j < 0 || j >= s.length) return;
  [s[i], s[j]] = [s[j], s[i]];
  pushPipeline();
}
const pushPipeline = guard(async (rerender = true) => {
  const r = await api("pipeline", { method: "PUT", body: S.state.pipeline });
  S.state.plan = r.plan;
  if (rerender) renderPipeline(); else renderPipelineStatusOnly();
});
function renderPipelineStatusOnly() {
  const planBy = Object.fromEntries((S.state.plan || []).map((p) => [p.id, p]));
  $$("#steps .step").forEach((el, i) => {
    const st = planBy[S.state.pipeline.steps[i]?.id]?.status || "pending";
    const dot = $(".status-dot", el); dot.className = "status-dot " + st; dot.title = st;
  });
}

let pollTimer = null;
const runPipeline = guard(async (opts = {}) => {
  await api("pipeline", { method: "PUT", body: S.state.pipeline });
  const { job } = await api("pipeline/run", { method: "POST", body: opts });
  $("#runStatus").classList.remove("hidden");
  $("#runBtn").disabled = true;
  $("#runCancel").onclick = () => api(`jobs/${job}/cancel`, { method: "POST" });
  clearInterval(pollTimer);
  pollTimer = setInterval(guard(async () => {
    const j = await api("jobs/" + job);
    $("#runBar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
    $("#runMsg").textContent = j.message || "";
    if (j.status !== "running") {
      clearInterval(pollTimer);
      $("#runStatus").classList.add("hidden");
      $("#runBtn").disabled = false;
      if (j.status === "error") { toast(j.error, true); console.error(j.trace); }
      else if (j.status === "done") toast(`Pipeline finished in ${j.elapsed.toFixed(1)}s`);
      setState(await api("state"));
      rebuildPanels();
      ensureEmbeddingPanel();
    }
  }), 600);
});
function ensureEmbeddingPanel() {
  const c = S.state.columns;
  const emb = ["X_umap", "X_tsne"].find((k) => c.obsm[k]);
  if (!emb || S.panels.some((p) => (p.cfg.x || "").startsWith(`obsm:${emb}`))) return;
  const cat = [...c.obs_categorical].reverse().find((x) => /leiden|louvain|kmeans|flowsom|gmm/.test(x));
  addPanel("scatter", { x: `obsm:${emb}:0`, y: `obsm:${emb}:1`, color: cat ? `obs:${cat}` : "" });
}
function rebuildPanels() {  // column lists may have changed (new embeddings / clusters)
  const cfgs = S.panels.map((p) => ({ type: p.type, cfg: p.cfg }));
  for (const p of [...S.panels]) removePanel(p);
  cfgs.forEach((c) => addPanel(c.type, c.cfg));
}

// ================================================================= sidebar: gates
function renderGates() {
  const el = $("#gateTree");
  el.innerHTML = "";
  const stats = S.state.gate_stats || [];
  if (!stats.length) el.append(h("p", { class: "hint" }, "No gates yet."));
  const depth = (name, d = 0) => { const g = S.state.gates[name]; return g && g.parent ? depth(g.parent, d + 1) : d; };
  for (const g of stats) {
    el.append(h("div", { class: "item", style: `padding-left:${8 + 14 * depth(g.name)}px`, title: g.error || `${g.x || ""} ${g.y || ""}`,
      onclick: guard(async () => applySelection(await api("select/gate", { method: "POST", body: { name: g.name } }))) },
    h("span", { class: "grow" }, g.name),
    g.error ? h("small", {}, "⚠ " + g.error.slice(0, 40)) : h("small", {}, `${g.count.toLocaleString()} · ${g.pct_parent.toFixed(1)}%${g.parent ? " of " + g.parent : ""}`),
    h("button", { class: "ghost", title: "Delete gate", onclick: guard(async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete gate ${g.name}?`)) return;
      const r = await api("gates/" + encodeURIComponent(g.name), { method: "DELETE" });
      S.state.gates = r.gates; S.state.gate_stats = r.gate_stats; renderGates(); refreshPanels();
    }) }, "✕")));
  }
  fillSelect($("#boolOf"), Object.keys(S.state.gates || {}).map((g) => [g, g]));
}

// ================================================================= sidebar: results
function renderResults() {
  const el = $("#resultList");
  el.innerHTML = "";
  const labels = { pca: "PCA variance explained", morans_i: "Spatial autocorrelation (Moran's I)" };
  const res = S.state.results || [];
  if (!res.length) el.append(h("p", { class: "hint" }, "Run markers, PCA, neighborhood or deep-learning steps to see results here."));
  for (const k of res) el.append(h("div", { class: "item", onclick: () => addPanel("result", { key: k }) },
    h("span", { class: "grow" }, labels[k] || k.replace(/_/g, " ")), h("small", {}, "open")));
}

// ================================================================= sidebar: compute
async function renderCompute() {
  if (!S.state?.project) return;
  const r = await api("compute/profiles");
  const rec = r.recommendation;
  $("#recommend").innerHTML = "";
  $("#recommend").append(h("b", {}, "Suggested: " + rec.where), h("div", { class: "hint" }, rec.why),
    h("div", { class: "hint" }, `Matrix ≈ ${rec.matrix_gb} GB · plan for ≈ ${rec.est_ram_gb} GB RAM` +
      (rec.heavy_steps.length ? ` · heavy pending steps: ${rec.heavy_steps.join(", ")}` : " · nothing heavy pending")));
  const remote = Object.entries(r.profiles).filter(([, v]) => v.type !== "local");
  fillSelect($("#profileSelect"), remote.map(([k, v]) => [k, `${k} (${v.type})`]));
  const showInfo = () => { const p = r.profiles[$("#profileSelect").value]; $("#profileInfo").textContent = p ? (p.description || JSON.stringify(p)) : "No remote profiles configured. Copy deploy/compute.example.yaml to ~/.featurescope/compute.yaml."; };
  $("#profileSelect").onchange = showInfo; showInfo();
  $("#submitRemote").disabled = !remote.length;
  renderJobs();
}
async function renderJobs() {
  const jobs = await api("compute/jobs");
  const el = $("#jobList");
  el.innerHTML = "";
  if (!jobs.length) el.append(h("p", { class: "hint" }, "No remote jobs yet."));
  for (const j of jobs) {
    el.append(h("div", { class: "item" },
      h("span", { class: "grow", title: (j.history || []).join("\n") }, `${j.pipeline} → ${j.profile}`, h("br"), h("small", {}, `${j.created} · ${j.status}${j.dry_run ? " (dry run)" : ""}`)),
      h("a", { href: `api/compute/jobs/${j.id}/file/commands.log`, target: "_blank", class: "button ghost", title: "commands run" }, "log"),
      h("button", { class: "ghost", title: "Poll status; fetch results when complete", onclick: guard(async () => {
        const r = await api(`compute/jobs/${j.id}/refresh`, { method: "POST" });
        toast(`${r.id}: ${r.status}`);
        if (r.status === "fetched") { setState(await api("state")); rebuildPanels(); }
        renderJobs();
      }) }, "↻")));
  }
}

// ================================================================= state wiring
function setState(st) {
  S.state = st;
  if (!st.project) return;
  $("#projectLabel").textContent = `${st.project.name} · ${st.n_obs.toLocaleString()} objects × ${st.n_vars} features`;
  $("#dataInfo").innerHTML = "";
  $("#dataInfo").append(h("h3", {}, "Current project"), h("div", { class: "card" },
    h("div", {}, h("b", {}, st.project.name)), h("div", { class: "hint" }, st.project.root),
    h("div", {}, `${st.n_obs.toLocaleString()} objects × ${st.n_vars} features`),
    h("div", { class: "hint" }, `batch: ${st.batch_key || "—"} · spatial: ${st.columns.obsm.spatial ? "yes" : "no"} · cache ${(st.project.cache_bytes / 1e6).toFixed(1)} MB in ${st.project.cache_entries} steps`)));
  renderPipeline(); renderGates(); renderResults();
  if (st.selection.count === 0) applySelection({ count: 0, mask: null });
}

async function init() {
  const ops = await api("ops");
  S.ops = ops;
  S.opsByName = Object.fromEntries(ops.map((o) => [o.name, o]));
  const cats = {};
  ops.forEach((o) => (cats[o.category] = cats[o.category] || []).push(o));
  const add = $("#addStep");
  for (const [cat, list] of Object.entries(cats)) {
    const g = h("optgroup", { label: cat });
    list.forEach((o) => g.append(h("option", { value: o.name }, o.label + (o.requires.length ? ` (needs ${o.requires.join(", ")})` : ""))));
    add.append(g);
  }
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
    $$(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
    if (b.dataset.tab === "compute") guard(renderCompute)();
  }));
  $$(".toolbar button").forEach((b) => b.addEventListener("click", () => {
    if (!S.state?.project) return toast("Open a project first", true);
    const c = S.state.columns, f = c.features;
    const cat = c.obs_categorical.find((x) => /leiden|louvain|kmeans|flowsom|cluster|neighborhood|domain/.test(x)) || c.obs_categorical[0];
    const defaults = { scatter: { x: `feat:${f[0]}`, y: `feat:${f[1] || f[0]}` }, spatial: { color: cat ? `obs:${cat}` : "" },
      hist: { x: `feat:${f[0]}` }, groups: { y: `feat:${f[0]}`, groupby: cat ? `obs:${cat}` : "" }, heatmap: { groupby: cat ? `obs:${cat}` : "" }, markers: {} };
    if (b.dataset.add === "spatial" && !c.obsm.spatial) return toast("This dataset has no spatial coordinates", true);
    addPanel(b.dataset.add, defaults[b.dataset.add]);
  }));
  $("#selClear").addEventListener("click", guard(async () => applySelection(await api("select/clear", { method: "POST" }))));
  $("#themeToggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
    store.set("theme", document.documentElement.dataset.theme);
    refreshPanels();
  });
  const theme = store.get("theme", null);
  if (theme) document.documentElement.dataset.theme = theme;

  // data tab
  $("#fileInput").addEventListener("change", guard(async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    toast(`Uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`, false, 60000);
    renderImport(await api("upload", { method: "POST", body: fd }));
    if (!$("#newName").value) $("#newName").value = f.name.replace(/\.[^.]+(\.gz)?$/, "");
    toast("Check the column roles, then Create project");
  }));
  $("#previewPath").addEventListener("click", guard(async () => renderImport(await api("preview", { method: "POST", body: { path: $("#serverPath").value } }))));
  $("#demoBtn").addEventListener("click", guard(async () => {
    toast("Generating synthetic dataset…", false, 30000);
    setState(await api("projects/create", { method: "POST", body: { name: $("#newName").value.trim() || "demo", demo: true, n: 30000, overwrite: true } }));
    S.state.pipeline = { name: "imaging", description: "", steps: TEMPLATES.spatial.map(([op, params]) => ({ op, params: { ...params }, id: uid(), enabled: true, note: "" })) };
    await pushPipeline();
    restoreLayout(); loadProjects();
    toast("Demo project ready. A spatial pipeline is loaded: press Run pipeline.");
  }));

  // pipeline tab
  $("#addStepBtn").addEventListener("click", () => {
    if (!S.state?.project) return;
    const step = { op: add.value, params: {}, id: uid(), enabled: true, note: "" };
    S.state.pipeline.steps.push(step); S.openSteps.add(step.id); pushPipeline();
  });
  $("#templateSelect").addEventListener("change", (e) => {
    const t = TEMPLATES[e.target.value];
    e.target.value = "";
    if (!t || !S.state?.project) return;
    if (S.state.pipeline.steps.length && !confirm("Replace the current steps with this template?")) return;
    S.state.pipeline.steps = t.map(([op, params]) => ({ op, params: { ...params }, id: uid(), enabled: true, note: "" }));
    pushPipeline();
  });
  $("#runBtn").addEventListener("click", () => runPipeline({}));
  $("#forceBtn").addEventListener("click", () => confirm("Recompute every step, ignoring cached results?") && runPipeline({ force: true }));
  $("#savePipeline").addEventListener("click", guard(async () => {
    const name = prompt("Save pipeline as", S.state.pipeline.name);
    if (!name) return;
    await api("pipeline", { method: "PUT", body: S.state.pipeline });
    const r = await api("pipelines/save", { method: "POST", body: { name } });
    S.state.pipeline.name = name; S.state.project.pipelines = r.pipelines; renderPipeline(); toast("Saved");
  }));
  $("#loadPipeline").addEventListener("click", guard(async () => {
    setState(await api("pipelines/load", { method: "POST", body: { name: $("#pipelineSelect").value } }));
    rebuildPanels();
    toast("Loaded: cached steps were restored without recomputation");
  }));

  // gates tab
  $("#boolAdd").addEventListener("click", guard(async () => {
    const of = $$("#boolOf option:checked").map((o) => o.value);
    const r = await api("gates", { method: "POST", body: { type: "boolean", name: $("#boolName").value, expr: $("#boolExpr").value, of, x: null } });
    S.state.gates = r.gates; S.state.gate_stats = r.gate_stats; renderGates();
  }));
  $("#annotBtn").addEventListener("click", guard(async () => {
    if (!S.selCount) return toast("Select objects first", true);
    setState(await api("selection/label", { method: "POST", body: { column: $("#annotCol").value, label: $("#annotLabel").value } }));
    rebuildPanels();
  }));
  // results / export
  $("#exportBtn").addEventListener("click", guard(async () => {
    const r = await api("export", { method: "POST", body: { name: $("#exportName").value } });
    $("#exportMsg").textContent = "Written: " + r.path;
  }));
  // compute
  $("#submitRemote").addEventListener("click", guard(async () => {
    const j = await api("compute/submit", { method: "POST", body: { profile: $("#profileSelect").value, dry_run: $("#dryRun").checked } });
    toast(`Job ${j.id}: ${j.status}${j.dry_run ? " (dry run: see log for the exact commands)" : ""}`);
    renderJobs();
  }));

  await loadProjects();
  const st = await api("state");
  if (st.project) { setState(st); restoreLayout(); }
}
init().catch((e) => toast(e.message, true));
