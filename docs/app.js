/* NetSentry — client-side simulation of the production pipeline.
   Mirrors the Python stack conceptually:
     - synthetic_kpi.py  -> rolling KPI generator with seasonality + fault episodes
     - features/anomaly  -> rolling robust z-score flagging
     - models/tft        -> multi-horizon latency forecast
     - graph/topology    -> common-cause ranking over a small topology
   No build step, no dependencies — runs entirely in the browser. */

(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- KPI generator -------------------------------------------------
  const WIN = 180;                 // points shown on canvas
  const SLA = 45;                  // latency SLA ceiling (ms)
  const kpis = { lat: [], jit: [], loss: [], thr: [] };
  let t = 0, faultUntil = -1, faultSev = 0;

  function step() {
    const day = 96; // points per "day" in this sped-up sim
    const season = 6 * Math.sin((2 * Math.PI * t) / day);
    let lat = 26 + season + randn() * 1.6;
    let thr = 320 - 60 * Math.sin((2 * Math.PI * t) / day + 2.5) + randn() * 12;
    // occasionally trigger a fault episode
    if (t > faultUntil && Math.random() < 0.012) {
      faultUntil = t + 18 + Math.floor(Math.random() * 22);
      faultSev = 0.7 + Math.random() * 0.8;
    }
    if (t <= faultUntil) {
      const ramp = faultSev;
      lat += 42 * ramp;
      thr *= 1 - 0.55 * ramp;
    }
    const jit = Math.max(0.3, 0.25 * lat + randn() * 1.0);
    const loss = Math.max(0, 0.05 + 0.0025 * Math.max(lat - 26, 0) + Math.abs(randn()) * 0.06);
    push(kpis.lat, lat); push(kpis.jit, jit); push(kpis.loss, loss); push(kpis.thr, thr);
    t++;
  }
  function push(a, v) { a.push(v); if (a.length > WIN) a.shift(); }
  function randn() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // ---- robust rolling z-score (mirrors features/anomaly.py) ----------
  function zscore(series, w = 48) {
    const n = series.length; if (n < 8) return new Array(n).fill(0);
    const out = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - w);
      const seg = series.slice(lo, i + 1).slice().sort((a, b) => a - b);
      const med = seg[Math.floor(seg.length / 2)];
      const mad = seg.map((x) => Math.abs(x - med)).sort((a, b) => a - b)[Math.floor(seg.length / 2)] || 1e-6;
      out[i] = (series[i] - med) / (1.4826 * mad);
    }
    return out;
  }

  // ---- canvas waveform ----------------------------------------------
  const cv = $("#wave"), ctx = cv.getContext("2d");
  function resize() { cv.width = cv.clientWidth * devicePixelRatio; cv.height = 260 * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); }
  window.addEventListener("resize", resize); resize();

  function line(series, color, lo, hi, alpha = 1) {
    const W = cv.clientWidth, H = 260, pad = 12;
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    series.forEach((v, i) => {
      const x = pad + (i / (WIN - 1)) * (W - pad * 2);
      const y = H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke(); ctx.globalAlpha = 1;
  }

  function draw(zlat) {
    const W = cv.clientWidth, H = 260;
    ctx.clearRect(0, 0, W, H);
    // grid
    ctx.strokeStyle = "#13203522"; ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) { const y = (H / 6) * i; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    line(kpis.thr, "#3ddc84", 0, 420, .55);
    line(kpis.loss, "#ff5c5c", 0, 6, .7);
    line(kpis.jit, "#a78bfa", 0, 40, .7);
    line(kpis.lat, "#39c0ed", 0, 110, 1);
    // anomaly markers on latency
    const pad = 12;
    kpis.lat.forEach((v, i) => {
      if (Math.abs(zlat[i]) > 3) {
        const x = pad + (i / (WIN - 1)) * (W - pad * 2);
        const y = H - pad - (v / 110) * (H - pad * 2);
        ctx.fillStyle = "#ff5c5c"; ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 7); ctx.fill();
        ctx.strokeStyle = "#ff5c5c55"; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
    });
  }

  // ---- forecast (mirrors a TFT multi-horizon head) -------------------
  const H_LABELS = ["15m", "1h", "3h", "6h"];
  function forecast() {
    const lat = kpis.lat; const n = lat.length;
    const recent = lat.slice(-12);
    const base = recent.reduce((a, b) => a + b, 0) / recent.length;
    const slope = (recent[recent.length - 1] - recent[0]) / recent.length;
    return [3, 12, 36, 72].map((h, i) => {
      const decay = [1, .9, .7, .5][i];
      return Math.max(8, base + slope * h * decay + randn() * 2);
    });
  }

  // ---- topology + common-cause (mirrors graph/topology.py) -----------
  const topo = $("#topo");
  const aggs = [
    { id: "agg-0", x: 90, y: 70, kids: [0, 1, 2] },
    { id: "agg-1", x: 90, y: 150, kids: [3, 4] },
  ];
  const core = { id: "core-0", x: 30, y: 110 };
  const access = [
    { x: 200, y: 40 }, { x: 250, y: 70 }, { x: 210, y: 100 },
    { x: 230, y: 150 }, { x: 270, y: 180 },
  ];
  function drawTopo(hotAgg) {
    let s = "";
    aggs.forEach((a) => {
      s += `<line class="edge" x1="${core.x}" y1="${core.y}" x2="${a.x}" y2="${a.y}"/>`;
      a.kids.forEach((k) => {
        const hot = a.id === hotAgg;
        s += `<line class="edge ${hot ? "hot" : ""}" x1="${a.x}" y1="${a.y}" x2="${access[k].x}" y2="${access[k].y}"/>`;
      });
    });
    s += node(core.x, core.y, "#39c0ed", 9, "core");
    aggs.forEach((a) => s += node(a.x, a.y, a.id === hotAgg ? "#ff5c5c" : "#a78bfa", 7, "agg"));
    access.forEach((p, i) => {
      const hot = aggs.find((a) => a.id === hotAgg)?.kids.includes(i);
      s += node(p.x, p.y, hot ? "#ff5c5c" : "#3ddc84", 4.5);
    });
    topo.innerHTML = s;
  }
  function node(x, y, c, r, label) {
    const t = label ? `<text x="${x}" y="${y - r - 4}" fill="#76859b" font-size="8" text-anchor="middle" font-family="JetBrains Mono">${label}</text>` : "";
    return `<circle class="node" cx="${x}" cy="${y}" r="${r}" fill="${c}" />${t}`;
  }

  // ---- pipeline stage lights ----------------------------------------
  let stageIdx = 0;
  function pulseStages() {
    document.querySelectorAll(".stage").forEach((el, i) => el.classList.toggle("on", i === stageIdx));
    stageIdx = (stageIdx + 1) % 4;
  }

  // ---- render loop ---------------------------------------------------
  function readout(zlat) {
    const last = (a) => a[a.length - 1] || 0;
    const lat = last(kpis.lat), jit = last(kpis.jit), loss = last(kpis.loss), thr = last(kpis.thr);
    const alert = Math.abs(zlat[zlat.length - 1]) > 3;
    $("#readout").innerHTML = `
      <div class="cell ${alert ? "alert" : ""}"><label>Latency</label><span>${lat.toFixed(1)} ms</span></div>
      <div class="cell"><label>Jitter</label><span>${jit.toFixed(1)} ms</span></div>
      <div class="cell"><label>Loss</label><span>${loss.toFixed(2)}%</span></div>
      <div class="cell"><label>Throughput</label><span>${thr.toFixed(0)}</span></div>`;
  }

  function renderForecast(f) {
    const breachAt = f.findIndex((v) => v > SLA);
    $("#horizons").innerHTML = f.map((v, i) => {
      const pct = Math.min(100, (v / 80) * 100);
      const breach = v > SLA;
      return `<li><span class="h-label">${H_LABELS[i]}</span>
        <span class="bar"><span class="fill ${breach ? "breach" : ""}" style="width:${pct}%"></span></span>
        <span class="h-val">${v.toFixed(1)}ms</span></li>`;
    }).join("");
    const v = $("#verdict");
    if (breachAt === -1) { v.className = "verdict"; v.textContent = "Stable — no SLA breach predicted within 6h."; }
    else if (breachAt >= 2) { v.className = "verdict warn"; v.textContent = `Watch — predicted SLA breach in ~${H_LABELS[breachAt]}.`; }
    else { v.className = "verdict crit"; v.textContent = `Outage risk — SLA breach predicted in ${H_LABELS[breachAt]}.`; }
  }

  function tick() {
    step();
    const zlat = zscore(kpis.lat);
    draw(zlat); readout(zlat);
    const inFault = t <= faultUntil;
    const hotAgg = inFault ? "agg-0" : null;
    drawTopo(hotAgg);
    $("#rootcause").innerHTML = hotAgg
      ? `root-cause → <b>${hotAgg}</b> · score 1.00 (3/3 dependents anomalous)`
      : `no common-cause cluster · all dependents nominal`;
    renderForecast(forecast());
  }

  function clock() {
    $("#clock").textContent = new Date().toLocaleTimeString("en-US", { hour12: false });
  }

  // GitHub Pages serves under /<repo>/ — repo link stays as-is.
  setInterval(clock, 1000); clock();
  setInterval(tick, reduce ? 1200 : 650);
  setInterval(pulseStages, reduce ? 1200 : 650);
  // warm up the buffers
  for (let i = 0; i < WIN; i++) step();
  tick();
})();
