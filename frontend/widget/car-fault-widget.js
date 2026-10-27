/*
 * <car-fault-diagnosis> — embeddable web component for the Car Fault Diagnosis module.
 *
 * Drop into any website / app webview:
 *   <script src="http://localhost:8005/widget.js"></script>
 *   <car-fault-diagnosis api-base="http://localhost:8005/"></car-fault-diagnosis>
 *
 * Attributes:
 *   api-base   Service root URL (defaults to where this script was loaded from)
 *   vehicle-id Vehicle identifier; alerts are confirmed per vehicle (default "demo-car")
 *   interval   Live-feed interval in ms (default 2000)
 *   theme      "light" | "dark" (default follows the OS)
 *
 * Events (bubble out of the shadow DOM, listen on the element or document):
 *   carfault:result  detail = diagnosis result
 *   carfault:alert   detail = diagnosis result (only when the alert is confirmed)
 */
(() => {
  const SCRIPT_SRC = document.currentScript && document.currentScript.src;
  const DEFAULT_BASE = SCRIPT_SRC
    ? SCRIPT_SRC.replace(/(ui\/widget\/car-fault-widget\.js|widget\.js)(\?.*)?$/, "")
    : location.origin + "/";

  const LEVEL_COLORS = { LOW: "var(--ok)", MEDIUM: "var(--warn)", HIGH: "var(--high)", CRITICAL: "var(--crit)" };

  const STYLE = `
  :host {
    --bg: #ffffff; --panel: #f6f7f9; --border: #e3e6ea; --text: #16191d; --muted: #646b75;
    --accent: #2f6fed; --ok: #1f9d55; --warn: #d4a106; --high: #e06c1b; --crit: #d62b3a;
    display: block; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--text); background: var(--bg); border-radius: 14px; container-type: inline-size;
  }
  :host([theme="dark"]) { --bg:#121519; --panel:#1b1f25; --border:#2a3038; --text:#e8ebef; --muted:#9aa3ad; --accent:#5b8cff; }
  @media (prefers-color-scheme: dark) {
    :host(:not([theme="light"])) { --bg:#121519; --panel:#1b1f25; --border:#2a3038; --text:#e8ebef; --muted:#9aa3ad; --accent:#5b8cff; }
  }
  * { box-sizing: border-box; }
  .wrap { padding: 16px; display: grid; gap: 14px; }
  .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .bar .title { font-weight: 650; font-size: 15px; margin-right: auto; display: flex; align-items: center; gap: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.on { background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 25%, transparent); }
  button, select, input[type=number] {
    font: inherit; font-size: 13px; color: var(--text); background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px;
  }
  button { cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .5; cursor: default; }
  .grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
  @container (min-width: 820px) { .grid { grid-template-columns: 1.1fr 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px; min-width: 0; }
  .card h3 { margin: 0 0 10px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; }
  .sensors { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
  .sensor { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px; }
  .sensor.flag { border-color: var(--crit); }
  .sensor label { display: block; font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sensor input { width: 100%; border: 0; background: transparent; padding: 2px 0; font-size: 15px; font-variant-numeric: tabular-nums; color: var(--text); }
  .sensor input:focus { outline: none; }
  .sensor .u { font-size: 11px; color: var(--muted); }
  .result-head { display: flex; gap: 14px; align-items: center; }
  .gauge { width: 120px; flex: none; }
  .fault { font-size: 20px; font-weight: 700; line-height: 1.2; }
  .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .chip { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; color: #fff; letter-spacing: .04em; }
  .probs { display: grid; gap: 5px; margin-top: 12px; }
  .prob { display: grid; grid-template-columns: 150px 1fr 48px; gap: 8px; align-items: center; font-size: 12px; }
  .prob .track { height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .prob .fill { height: 100%; background: var(--accent); border-radius: 4px; }
  .prob span:last-child { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
  ul { margin: 6px 0 0; padding-left: 18px; font-size: 13px; }
  li { margin: 3px 0; }
  .section { margin-top: 12px; }
  .section b { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .alerts { max-height: 220px; overflow: auto; font-size: 13px; }
  .alert-row { display: grid; grid-template-columns: 70px 1fr auto; gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--border); align-items: center; }
  .alert-row:last-child { border-bottom: 0; }
  .muted { color: var(--muted); font-size: 13px; }
  .banner { padding: 10px 12px; border-radius: 10px; color: #fff; font-weight: 600; font-size: 14px; }
  .err { color: var(--crit); font-size: 13px; }
  .banner.pending { background: var(--bg); color: var(--text); border: 1px dashed var(--warn); font-weight: 500; }
  .tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); background: var(--bg); }
  .tag.warn { border-color: var(--warn); }
  .shap { display: grid; gap: 4px; margin-top: 6px; }
  .shap-row { display: grid; grid-template-columns: minmax(0, 1fr) 120px; gap: 8px; align-items: center; font-size: 12px; }
  .shap-row .lbl { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .shap-bar { position: relative; height: 8px; background: var(--border); border-radius: 4px; }
  .shap-bar i { position: absolute; top: 0; height: 100%; border-radius: 4px; }
  .shap-bar::after { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px; background: var(--muted); }
  .batch td, .batch th { padding: 4px 8px; text-align: left; font-size: 12px; border-bottom: 1px solid var(--border); }
  .batch { border-collapse: collapse; width: 100%; }
  `;

  class CarFaultDiagnosis extends HTMLElement {
    constructor() {
      super();
      this.root = this.attachShadow({ mode: "open" });
      this.timer = null;
      this.simRun = 0;
      this.sensors = {};
      this.alerts = [];
    }

    get vehicleId() {
      const id = this.getAttribute("vehicle-id") || "demo-car";
      return this.simRun ? `${id}-sim${this.simRun}` : id;
    }

    get base() {
      const b = this.getAttribute("api-base") || DEFAULT_BASE;
      return new URL(b.endsWith("/") ? b : b + "/", location.href).href;
    }

    async api(path, opts) {
      const res = await fetch(this.base + path, opts);
      if (!res.ok) {
        let msg = res.statusText;
        try { msg = (await res.json()).detail || msg; } catch (_) {}
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
      return res.json();
    }

    async connectedCallback() {
      this.root.innerHTML = `<style>${STYLE}</style><div class="wrap"><div class="muted">Connecting to diagnosis service…</div></div>`;
      try {
        const [sensors, faults, scenarios] = await Promise.all([
          this.api("api/v1/sensors"), this.api("api/v1/faults"), this.api("api/v1/simulate/scenarios"),
        ]);
        this.sensors = sensors;
        this.faults = faults;
        this.scenarios = scenarios;
        this.render();
        const first = await this.api("api/v1/simulate?scenario=normal");
        this.fill(first.reading);
        this.diagnose();
      } catch (e) {
        this.root.querySelector(".wrap").innerHTML =
          `<div class="err">Could not reach the Car Fault Diagnosis API at ${this.base} — ${e.message}</div>`;
      }
    }

    disconnectedCallback() { this.stop(); }

    $(sel) { return this.root.querySelector(sel); }

    render() {
      const scenarios = Object.entries(this.scenarios)
        .map(([k, label]) => `<option value="${k}">${label}</option>`).join("");
      const tiles = Object.entries(this.sensors).map(([k, s]) => `
        <div class="sensor" data-k="${k}">
          <label title="${s.label}">${s.label}</label>
          <input type="number" step="any" name="${k}" value="${s.default}">
          <span class="u">${s.unit}</span>
        </div>`).join("");

      this.$(".wrap").innerHTML = `
        <div class="bar">
          <div class="title"><span class="dot" id="dot"></span>AI Car Fault Diagnosis</div>
          <select id="scenario" aria-label="Simulation scenario">
            <option value="random">Live feed: random</option>${scenarios}
          </select>
          <button id="live">Start live feed</button>
          <button id="diag" class="primary">Diagnose</button>
        </div>
        <div id="banner"></div>
        <div class="grid">
          <div class="card">
            <h3>Sensor feed (OBD-II)</h3>
            <div class="sensors">${tiles}</div>
            <div class="section bar">
              <label class="muted" for="csv">Batch CSV:</label>
              <input type="file" id="csv" accept=".csv">
            </div>
            <div id="batch"></div>
          </div>
          <div class="card" id="result"><div class="muted">No diagnosis yet.</div></div>
        </div>
        <div class="card"><h3>Alert history</h3><div class="alerts" id="alerts"><div class="muted">No alerts yet.</div></div></div>
      `;
      this.$("#diag").onclick = () => this.diagnose();
      // New demo scenario = a fresh simulated vehicle, so its alert history starts clean.
      // (Resetting a real vehicle's history needs the admin key, which a public page must not hold.)
      this.$("#scenario").onchange = () => { this.simRun += 1; };
      this.$("#live").onclick = () => (this.timer ? this.stop() : this.start());
      this.$("#csv").onchange = (e) => e.target.files[0] && this.uploadCsv(e.target.files[0]);
    }

    fill(reading) {
      for (const [k, v] of Object.entries(reading)) {
        const input = this.root.querySelector(`input[name="${k}"]`);
        if (input) input.value = v;
      }
    }

    reading() {
      const out = {};
      this.root.querySelectorAll(".sensor input").forEach((i) => (out[i.name] = parseFloat(i.value)));
      return out;
    }

    start() {
      const ms = parseInt(this.getAttribute("interval") || "2000", 10);
      const tick = async () => {
        try {
          const s = await this.api(`api/v1/simulate?scenario=${this.$("#scenario").value}`);
          this.fill(s.reading);
          await this.diagnose();
        } catch (e) { this.showError(e); this.stop(); }
      };
      tick();
      this.timer = setInterval(tick, ms);
      this.$("#live").textContent = "Stop live feed";
      this.$("#dot").classList.add("on");
    }

    stop() {
      clearInterval(this.timer);
      this.timer = null;
      if (this.$("#live")) {
        this.$("#live").textContent = "Start live feed";
        this.$("#dot").classList.remove("on");
      }
    }

    async diagnose() {
      const reading = this.reading();
      // Blank fields are sent as missing sensors; the service imputes them.
      for (const k of Object.keys(reading)) if (Number.isNaN(reading[k])) delete reading[k];
      reading.vehicle_id = this.vehicleId;
      try {
        const r = await this.api("api/v1/diagnose", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(reading),
        });
        this.showResult(r);
        this.dispatchEvent(new CustomEvent("carfault:result", { detail: r, bubbles: true, composed: true }));
        if (r.alert) {
          this.alerts.unshift(r);
          this.alerts = this.alerts.slice(0, 50);
          this.renderAlerts();
          this.dispatchEvent(new CustomEvent("carfault:alert", { detail: r, bubbles: true, composed: true }));
        }
      } catch (e) { this.showError(e); }
    }

    gauge(score, level) {
      const a = Math.PI * (1 - score / 100);
      const x = 60 + 48 * Math.cos(a), y = 60 - 48 * Math.sin(a);
      const large = 0;
      return `<svg class="gauge" viewBox="0 0 120 72" role="img" aria-label="Risk score ${score}">
        <path d="M12 60 A48 48 0 0 1 108 60" fill="none" stroke="var(--border)" stroke-width="10" stroke-linecap="round"/>
        ${score > 0.5 ? `<path d="M12 60 A48 48 0 ${large} 1 ${x.toFixed(2)} ${y.toFixed(2)}" fill="none" stroke="${LEVEL_COLORS[level]}" stroke-width="10" stroke-linecap="round"/>` : ""}
        <text x="60" y="56" text-anchor="middle" font-size="20" font-weight="700" fill="var(--text)">${Math.round(score)}</text>
        <text x="60" y="70" text-anchor="middle" font-size="8" fill="var(--muted)">RISK SCORE</text>
      </svg>`;
    }

    showResult(r) {
      const color = LEVEL_COLORS[r.risk_level];
      const probs = Object.entries(r.probabilities).slice(0, 5).map(([k, p]) => `
        <div class="prob"><span>${this.faults[k]?.label || k}</span>
          <div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
          <span>${(p * 100).toFixed(1)}%</span></div>`).join("");
      const expl = r.explanations.length
        ? `<ul>${r.explanations.map((e) => `<li>${e.message}</li>`).join("")}</ul>`
        : `<div class="muted">All diagnostic sensors within the healthy baseline.</div>`;

      // SHAP: bars grow right (supports this diagnosis) or left (argues against it).
      const shap = r.shap_contributions || [];
      const maxAbs = Math.max(1e-6, ...shap.map((s) => Math.abs(s.contribution)));
      const shapHtml = shap.map((s) => {
        const w = (Math.abs(s.contribution) / maxAbs) * 50;
        const pos = s.contribution >= 0;
        return `<div class="shap-row" title="SHAP ${s.contribution}">
          <span class="lbl">${s.label} <span class="muted">${s.value}${s.unit ? " " + s.unit : ""}</span></span>
          <span class="shap-bar"><i style="${pos ? "left:50%" : `left:${50 - w}%`};width:${w}%;background:${pos ? "var(--accent)" : "var(--crit)"}"></i></span>
        </div>`;
      }).join("");

      const tags = [];
      if (r.multiple_faults) tags.push(`<span class="tag warn">Multiple faults: ${r.detected_faults.map((d) => d.label).join(" + ")}</span>`);
      if (r.novelty.is_novel) tags.push(`<span class="tag warn">Pattern not seen in training</span>`);
      if (r.persistence && r.persistence.status !== "clear") tags.push(`<span class="tag">${r.persistence.reason}</span>`);

      this.$("#result").innerHTML = `
        <h3>Diagnosis</h3>
        <div class="result-head">
          ${this.gauge(r.risk_score, r.risk_level)}
          <div>
            <div class="fault">${r.fault_label}</div>
            <div class="sub">${r.fault === "unknown_anomaly"
              ? `closest known class: ${this.faults[r.model_prediction]?.label || r.model_prediction} (${(r.confidence * 100).toFixed(1)}%)`
              : `${r.system !== "—" ? r.system + " system · " : ""}calibrated confidence ${(r.confidence * 100).toFixed(1)}%
              ${r.uncertain ? " · <em>low confidence</em>" : ""}`}</div>
            <div style="margin-top:6px"><span class="chip" style="background:${color}">${r.risk_level}</span></div>
          </div>
        </div>
        ${tags.length ? `<div class="tags">${tags.join("")}</div>` : ""}
        <div class="sub" style="margin-top:10px">${r.description}</div>
        <div class="probs">${probs}</div>
        ${shapHtml ? `<div class="section"><b>Why the model decided (SHAP)</b><div class="shap">${shapHtml}</div></div>` : ""}
        <div class="section"><b>Abnormal readings</b>${expl}</div>
        <div class="section"><b>Recommended actions</b><ul>${r.recommended_actions.map((a) => `<li>${a}</li>`).join("")}</ul></div>
        ${r.warnings.length ? `<div class="section err">${r.warnings.join("<br>")}</div>` : ""}
        <div class="muted" style="margin-top:10px;font-size:11px">id ${r.diagnosis_id} · ${r.latency_ms} ms</div>`;

      if (r.alert) {
        this.$("#banner").innerHTML =
          `<div class="banner" style="background:${color}">⚠ ${r.fault_label} — ${r.risk_level} risk (${Math.round(r.risk_score)}/100)</div>`;
      } else if (r.raw_alert) {
        this.$("#banner").innerHTML =
          `<div class="banner pending">Possible ${r.fault_label} — ${r.persistence ? r.persistence.reason : "confirming"}</div>`;
      } else {
        this.$("#banner").innerHTML = "";
      }

      const flagged = new Set(r.explanations.map((e) => e.sensor));
      this.root.querySelectorAll(".sensor").forEach((el) => el.classList.toggle("flag", flagged.has(el.dataset.k)));
    }

    renderAlerts() {
      this.$("#alerts").innerHTML = this.alerts.map((a) => `
        <div class="alert-row">
          <span class="muted">${new Date(a.timestamp * 1000).toLocaleTimeString()}</span>
          <span>${a.fault_label} <span class="muted">· ${(a.confidence * 100).toFixed(0)}%</span></span>
          <span class="chip" style="background:${LEVEL_COLORS[a.risk_level]}">${a.risk_level}</span>
        </div>`).join("");
    }

    async uploadCsv(file) {
      const fd = new FormData();
      fd.append("file", file);
      this.$("#batch").innerHTML = `<div class="muted">Scoring ${file.name}…</div>`;
      try {
        const res = await this.api("api/v1/diagnose/csv", { method: "POST", body: fd });
        const s = res.summary;
        const rows = Object.entries(s.by_fault).sort((a, b) => b[1] - a[1])
          .map(([k, n]) => `<tr><td>${this.faults[k]?.label || k}</td><td>${n}</td></tr>`).join("");
        this.$("#batch").innerHTML = `
          <div class="section"><b>Batch result</b>
          <div class="muted">${s.total} readings · ${s.alerts} alerts</div>
          <table class="batch"><tr><th>Fault</th><th>Count</th></tr>${rows}</table></div>`;
      } catch (e) {
        this.$("#batch").innerHTML = `<div class="err">${e.message}</div>`;
      }
    }

    showError(e) {
      this.$("#banner").innerHTML = `<div class="err">${e.message}</div>`;
    }
  }

  if (!customElements.get("car-fault-diagnosis")) customElements.define("car-fault-diagnosis", CarFaultDiagnosis);
})();
