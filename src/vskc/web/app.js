"use strict";

/* ====================================================================== state */
const S = {
  meta: null,
  source: "heldout",
  mode: "recorded",
  lists: {},          // split -> {run_id, items}
  current: null,      // full scenario dict (held-out / dev) or null for custom
  results: null,
  hideAnswers: false,
};
const SYS = ["rule", "llm", "rag"];
const SYS_LABEL = { rule: "Rule-based baseline", llm: "Generic LLM", rag: "Vertical RAG + LLM" };
const SYS_SHORT = { rule: "R", llm: "L", rag: "K" };
const SLICE_LABEL = {
  all: "All scenarios",
  depends_on_supplier_fact: "Needs a supplier fact",
  no_supplier_fact: "No supplier fact needed",
  clear_label: "Clear label",
  judgment_label: "Judgement label",
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x == null ? "n/a" : (100 * x).toFixed(d) + "%");
const fmtP = (p) => (p < 0.001 ? "< 0.001" : p.toFixed(3));
const human = (s) => String(s || "").replace(/_/g, " ");

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { const j = await r.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

/* ====================================================================== boot */
document.addEventListener("DOMContentLoaded", async () => {
  bindChrome();
  try {
    S.meta = await api("/api/meta");
  } catch (e) {
    showError("Could not reach the API: " + e.message);
    return;
  }
  renderTopbar();
  renderActionDefs();
  fillSelects();
  setModeAvailability();
  await loadList("heldout");
  renderCustomExamples();
  await applyHash();
  window.addEventListener("hashchange", applyHash);
});

/* Deep links: #results, #about, #heldout/ho-022, #dev/gold-016, #custom, #heldout/ho-022/hide */
async function applyHash() {
  const parts = decodeURIComponent(location.hash.replace(/^#/, "")).split("/").filter(Boolean);
  if (!parts.length) return;
  if (parts[0] === "results" || parts[0] === "about") { switchView(parts[0]); return; }
  switchView("workbench");
  if (parts.includes("hide")) { $("#hide-answers").checked = true; S.hideAnswers = true; }
  if (["heldout", "dev", "custom"].includes(parts[0])) {
    if (S.source !== parts[0]) await switchSource(parts[0]);
    if (parts[1] && parts[1] !== "hide") await openScenario(parts[1]);
  }
}

function bindChrome() {
  $$(".tab").forEach((b) => b.addEventListener("click", () => switchView(b.dataset.view)));
  $$("#source-seg button").forEach((b) => b.addEventListener("click", () => switchSource(b.dataset.source)));
  $$("#mode-seg button").forEach((b) => b.addEventListener("click", () => { if (!b.disabled) setMode(b.dataset.mode); }));
  ["search", "filter-label", "filter-fact", "filter-clear", "filter-disagree"].forEach((id) =>
    $("#" + id).addEventListener("input", renderList));
  $("#hide-answers").addEventListener("change", (e) => { S.hideAnswers = e.target.checked; renderList(); if (S.current) renderScenario(S.current); rerunIfShown(); });
  $("#run-btn").addEventListener("click", run);
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
}

function switchView(view) {
  $$(".tab").forEach((t) => { const on = t.dataset.view === view; t.classList.toggle("is-active", on); t.setAttribute("aria-selected", on); });
  $$(".view").forEach((v) => v.classList.toggle("is-active", v.id === "view-" + view));
  if (view === "results" && !S.results) loadResults();
}

function renderTopbar() {
  const m = S.meta;
  $("#topbar-meta").innerHTML =
    `<span class="pill" title="Model under test">${esc(m.model_under_test || "no model set")}</span>` +
    `<span class="pill">${m.knowledge_count} knowledge items</span>` +
    `<span class="pill"><span class="led ${m.live_available ? "on" : ""}"></span>${m.live_available ? "Live model available" : "Live model off"}</span>`;
}

function renderActionDefs() {
  $("#action-defs").innerHTML = S.meta.actions.map((a) =>
    `<dt>${esc(a.value)}</dt><dd>${esc(a.when)} <span class="muted">${esc(a.boundary)}</span></dd>`).join("");
}

function fillSelects() {
  $("#filter-label").innerHTML = `<option value="">All gold labels</option>` +
    S.meta.actions.map((a) => `<option value="${a.value}">${esc(a.value)}</option>`).join("");
  $("#cu-stage").innerHTML = S.meta.stages.map((s) => `<option>${s}</option>`).join("");
}

/* ====================================================================== sources and list */
async function switchSource(src) {
  S.source = src;
  $$("#source-seg button").forEach((b) => b.classList.toggle("is-on", b.dataset.source === src));
  clearOutputs();
  if (src === "custom") {
    S.current = null;
    $("#filters").hidden = true;
    $("#list-head").hidden = true;
    $("#custom-hint").hidden = false;
    $("#scenario-list").innerHTML = "";
    $("#list-count").textContent = "";
    $("#scenario-card").hidden = true;
    $("#empty-state").hidden = true;
    $("#custom-card").hidden = false;
    $("#runbar").hidden = false;
    if (S.mode === "recorded") setMode(S.meta.live_available ? "live" : "mock");
    setModeAvailability();
    return;
  }
  $("#filters").hidden = false;
  $("#list-head").hidden = false;
  $("#custom-hint").hidden = true;
  $("#custom-card").hidden = true;
  if (S.mode !== "recorded") setMode("recorded");
  setModeAvailability();
  await loadList(src);
}

async function loadList(split) {
  if (!S.lists[split]) {
    try { S.lists[split] = await api("/api/scenarios?split=" + split); }
    catch (e) { showError(e.message); return; }
  }
  S.current = null;
  $("#scenario-card").hidden = true;
  $("#runbar").hidden = true;
  $("#empty-state").hidden = false;
  renderList();
}

function filtered() {
  const L = S.lists[S.source]; if (!L) return [];
  const q = $("#search").value.trim().toLowerCase();
  const lab = $("#filter-label").value;
  const fact = $("#filter-fact").checked, clear = $("#filter-clear").checked, dis = $("#filter-disagree").checked;
  return L.items.filter((it) => {
    if (q && !(it.id.toLowerCase().includes(q) || it.snippet.toLowerCase().includes(q))) return false;
    if (lab && it.label !== lab) return false;
    if (fact && !it.depends_on_supplier_fact) return false;
    if (clear && it.label_certainty !== "clear") return false;
    if (dis) { const v = Object.values(it.recorded || {}); if (!(v.length && v.some(Boolean) && v.some((x) => !x))) return false; }
    return true;
  });
}

function marksHtml(rec) {
  return `<span class="marks" aria-label="Recorded results">` + SYS.map((s) => {
    if (!(s in (rec || {}))) return `<span class="mark na" title="${SYS_LABEL[s]}: no record">${SYS_SHORT[s]}</span>`;
    const ok = rec[s];
    return `<span class="mark ${ok ? "ok" : "no"}" title="${SYS_LABEL[s]}: ${ok ? "correct" : "wrong"}">${ok ? "✓" : "✕"}</span>`;
  }).join("") + `</span>`;
}

function renderList() {
  if (S.source === "custom") return;
  const items = filtered();
  const L = S.lists[S.source];
  $("#list-count").textContent = `${items.length} of ${L ? L.items.length : 0} · recorded run ${L && L.run_id ? L.run_id : "none"}`;
  $(".legend-mini").innerHTML = "R rule · L generic LLM · K knowledge (RAG)";
  $("#scenario-list").innerHTML = items.map((it) => {
    const tags = [
      `<span class="tag">${human(it.sales_stage)}</span>`,
      `<span class="tag">${it.customer_size}</span>`,
      it.depends_on_supplier_fact ? `<span class="tag fact">supplier fact</span>` : "",
      it.label_certainty === "judgment" ? `<span class="tag judg">judgement</span>` : "",
    ].join("");
    const gold = S.hideAnswers ? "" : `<div class="li-snippet" style="-webkit-line-clamp:1"><span class="muted">gold</span> <code>${esc(it.label)}</code></div>`;
    const on = S.current && S.current.id === it.id ? " is-on" : "";
    return `<li class="${on}" data-id="${esc(it.id)}" tabindex="0">
      <div class="li-top"><span class="li-id">${esc(it.id)}</span>${marksHtml(it.recorded)}</div>
      <div class="li-snippet">${esc(it.snippet)}</div>${gold}
      <div class="li-tags">${tags}</div></li>`;
  }).join("") || `<li class="muted" style="cursor:default">No scenario matches these filters.</li>`;
  $$("#scenario-list li[data-id]").forEach((li) => {
    li.addEventListener("click", () => openScenario(li.dataset.id));
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") openScenario(li.dataset.id); });
  });
}

/* ====================================================================== scenario */
async function openScenario(id) {
  clearOutputs();
  try { S.current = await api(`/api/scenarios/${S.source}/${encodeURIComponent(id)}`); }
  catch (e) { showError(e.message); return; }
  $$("#scenario-list li").forEach((li) => li.classList.toggle("is-on", li.dataset.id === id));
  renderScenario(S.current);
  $("#empty-state").hidden = true;
  $("#scenario-card").hidden = false;
  $("#runbar").hidden = false;
  setModeAvailability();
  if (S.mode === "recorded") run();
}

function renderScenario(sc) {
  $("#sc-id").textContent = `${S.source === "heldout" ? "Held-out" : "Dev"} scenario · ${sc.id}`;
  $("#sc-title").textContent = `${cap(human(sc.sales_stage))} · ${sc.customer_size} customer`;
  $("#sc-facts").innerHTML = [
    `<span class="tag">stage: ${human(sc.sales_stage)}</span>`,
    `<span class="tag">size: ${sc.customer_size}</span>`,
    `<span class="tag">${sc.export_oriented ? "export oriented" : "domestic"}</span>`,
    `<span class="tag">${sc.has_incumbent ? "has incumbent supplier" : "no incumbent"}</span>`,
    sc.depends_on_supplier_fact ? `<span class="tag fact">decision needs a supplier fact</span>` : "",
    sc.ambiguous ? `<span class="tag judg">contains conflicting statements</span>` : "",
  ].join("");
  $("#sc-narrative").textContent = sc.narrative;
  listInto("#sc-pain", sc.pain_points);
  listInto("#sc-obj", sc.objections);
  const hide = S.hideAnswers ? " hidden-answer" : "";
  $("#sc-gold").innerHTML = `<div class="label-sm">Gold label</div><span class="action big${hide}">${esc(sc.label || "unlabelled")}</span>` +
    (sc.label_certainty ? `<div class="label-sm" style="margin-top:4px">${sc.label_certainty === "clear" ? "clear label" : "judgement label"}</div>` : "");
  let d = "";
  if (!S.hideAnswers) {
    if (sc.label_rationale) d += `<p><b>Why:</b> ${esc(sc.label_rationale)}</p>`;
    if (sc.alternative_label) d += `<p><b>Defensible alternative:</b> <code>${esc(sc.alternative_label)}</code></p>`;
    if (sc.ambiguity) d += `<p><b>Conflict in the record:</b> ${esc(sc.ambiguity)}</p>`;
  }
  $("#sc-gold-detail").innerHTML = d;
}

function listInto(sel, arr) {
  const el = $(sel);
  if (!arr || !arr.length) { el.className = "bullets empty-list"; el.innerHTML = "<li>None stated</li>"; return; }
  el.className = "bullets";
  el.innerHTML = arr.map((x) => `<li>${esc(x)}</li>`).join("");
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/* ====================================================================== mode and run */
function setMode(mode) {
  S.mode = mode;
  $$("#mode-seg button").forEach((b) => b.classList.toggle("is-on", b.dataset.mode === mode));
  const notes = {
    recorded: "Replays what each system answered in the evaluation run. No model call, no cost, exactly what was scored.",
    live: `Calls ${S.meta.model_under_test || "the model"} now. About two cents per scenario.`,
    mock: "Deterministic fake replies. Shows the interface without calling any model.",
  };
  $("#mode-note").textContent = notes[mode];
  $("#run-btn").textContent = mode === "recorded" ? "Show recorded answers" : "Get recommendations";
}

function setModeAvailability() {
  const rec = $('#mode-seg button[data-mode="recorded"]');
  const live = $('#mode-seg button[data-mode="live"]');
  rec.disabled = S.source === "custom";
  live.disabled = !S.meta.live_available;
  live.title = S.meta.live_available ? "Call the model now." : "No API key or model configured on the server.";
}

function rerunIfShown() { if ($("#outputs").children.length && S.mode === "recorded" && S.current) run(); }

function currentScenarioPayload() {
  if (S.source === "custom") {
    const lines = (v) => v.split(/\n|;/).map((x) => x.trim()).filter(Boolean);
    const narrative = $("#cu-narrative").value.trim();
    if (!narrative) throw new Error("Write a narrative first.");
    return {
      narrative, sales_stage: $("#cu-stage").value, customer_size: $("#cu-size").value,
      export_oriented: $("#cu-export").checked, has_incumbent: $("#cu-incumbent").checked,
      pain_points: lines($("#cu-pain").value), objections: lines($("#cu-obj").value),
    };
  }
  const sc = S.current;
  return {
    narrative: sc.narrative, sales_stage: sc.sales_stage, customer_size: sc.customer_size,
    export_oriented: sc.export_oriented, has_incumbent: sc.has_incumbent,
    pain_points: sc.pain_points, objections: sc.objections,
  };
}

async function run() {
  hideError();
  const btn = $("#run-btn");
  if (S.mode === "recorded") {
    const rr = S.current && S.current.recorded_run;
    if (!rr) { showError("There is no recorded run for this split yet."); return; }
    renderOutputs(rr.outputs, S.current.label, `Recorded run ${rr.run_id} · ${rr.model} · prompt ${rr.prompt_version}`);
    return;
  }
  let payload;
  try { payload = currentScenarioPayload(); } catch (e) { showError(e.message); return; }
  btn.disabled = true;
  renderSkeleton();
  try {
    const res = await api("/api/recommend", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: payload, mode: S.mode }),
    });
    const gold = S.source === "custom" ? null : S.current.label;
    renderOutputs(res.outputs, gold, `${S.mode === "live" ? "Live" : "Mock"} · ${res.model}`);
  } catch (e) {
    clearOutputs(); showError(e.message);
  } finally { btn.disabled = false; }
}

/* ====================================================================== outputs */
function clearOutputs() { $("#outputs").innerHTML = ""; hideError(); }

function renderSkeleton() {
  $("#outputs").innerHTML = SYS.map((s) => `<article class="card out loading">
    <div class="out-head"><span class="sys-key k-${s}"></span><h3>${SYS_LABEL[s]}</h3></div>
    <div class="out-body"><div class="skeleton" style="width:50%"></div><div class="skeleton"></div><div class="skeleton" style="width:80%"></div></div></article>`).join("");
}

function renderOutputs(outputs, gold, caption) {
  const tauConf = S.meta.gate ? S.meta.gate.tau_conf : null;
  const html = outputs.map((o) => {
    let verdict = "";
    if (gold && o.action) {
      verdict = o.action === gold
        ? `<span class="verdict ok">✓ Matches gold</span>`
        : `<span class="verdict no">✕ Gold is ${S.hideAnswers ? "hidden" : esc(gold)}</span>`;
      if (S.hideAnswers) verdict = "";
    }
    const action = o.action ? `<span class="action big">${esc(o.action)}</span>` : `<span class="action big" style="color:var(--bad-ink)">no recommendation</span>`;
    const conf = o.system === "rule" ? "" : (o.confidence == null ? "" : `
      <div class="meter" data-tip="Model's self-reported confidence. The tick marks the gate threshold ${tauConf != null ? tauConf.toFixed(2) : ""}.">
        <div class="meter-top"><span>Confidence</span><b>${o.confidence.toFixed(2)}</b></div>
        <div class="meter-track"><div class="meter-fill" style="width:${(o.confidence * 100).toFixed(1)}%"></div>
        ${tauConf != null ? `<div class="meter-tick" style="left:calc(${(tauConf * 100).toFixed(1)}% - 1px)"></div>` : ""}</div></div>`);
    let gate = "";
    if (o.gate) {
      if (o.gate.decision === "pass") gate = `<div class="gate pass"><span class="ic">✓</span><div><b>Passes the gate</b>Shown to the FDE as a recommendation.</div></div>`;
      else if (o.gate.decision === "human_review") gate = `<div class="gate review"><span class="ic">!</span><div><b>Human review required</b>${esc(o.gate.reason_text)}.</div></div>`;
      else gate = `<div class="gate none"><div>${esc(o.gate.reason_text)}</div></div>`;
    } else if (o.system === "rule") {
      gate = `<div class="gate none"><div>Deterministic rule on structured fields. No confidence gate.</div></div>`;
    }
    let ev = "";
    if (o.system === "rag") {
      const chips = o.evidence.map((e) => `<button class="ev${e.retrieved === false ? " bad" : ""}" data-kid="${esc(e.id)}"
          title="${e.retrieved === false ? "Not among the retrieved items" : esc((e.text || "").slice(0, 160))}">
          <span>${esc(e.id)}</span><span class="t">${esc(e.type || "")}</span></button>`).join("");
      ev = `<div><div class="label-sm">Evidence cited</div><div class="evidence">${chips || '<span class="muted">None cited</span>'}</div></div>`;
      if (o.retrieved && o.retrieved.length) {
        ev += `<details class="retrieved"><summary>Retrieved ${o.retrieved.length} items${o.top_score != null ? ` · top score ${o.top_score}` : ""}</summary><ol>` +
          o.retrieved.map((r) => `<li data-kid="${esc(r.id)}"><code>${esc(r.id)}</code> <span class="sc">${r.score != null ? r.score.toFixed(2) : ""}</span> ${esc((r.text || "").slice(0, 110))}…</li>`).join("") +
          `</ol></details>`;
      }
    }
    const err = o.error ? `<div class="gate review"><span class="ic">!</span><div><b>System error</b>${esc(o.error.slice(0, 220))}</div></div>` : "";
    const foot = o.system === "rule"
      ? `<span>No model call</span><span>cost $0</span>`
      : `<span>${o.input_tokens} in · ${o.output_tokens} out tokens</span><span>${o.latency_s.toFixed(1)} s</span>${o.cost_usd != null ? `<span>$${o.cost_usd.toFixed(4)}</span>` : ""}`;
    return `<article class="card out">
      <div class="out-head"><span class="sys-key k-${o.system}"></span><h3>${esc(o.system_label)}</h3>${verdict}</div>
      <div class="out-body">${action}${o.rationale ? `<p class="rationale">${esc(ruleText(o))}</p>` : ""}${conf}${gate}${err}${ev}</div>
      <div class="out-foot">${foot}</div></article>`;
  }).join("");
  $("#outputs").innerHTML = html + `<div class="muted" style="grid-column:1/-1;font-size:12px">${esc(caption)}</div>`;
  $$("#outputs [data-kid]").forEach((el) => el.addEventListener("click", () => openKnowledge(el.dataset.kid)));
  bindTips($("#outputs"));
}

function ruleText(o) {
  if (o.system !== "rule" || !/^rule:/.test(o.rationale || "")) return o.rationale;
  return "Matched rule: " + human(o.rationale.slice(5)) + ".";
}

/* ====================================================================== drawer */
async function openKnowledge(kid) {
  let k;
  try { k = await api("/api/knowledge/" + encodeURIComponent(kid)); }
  catch (e) { showError(e.message); return; }
  $("#drawer-kind").textContent = k.type === "product" ? "Product record" : "Playbook item";
  $("#drawer-title").textContent = k.id;
  $("#drawer-body").innerHTML = `<p>${esc(k.text)}</p><dl class="kv"><dt>Type</dt><dd>${esc(k.type)}</dd>
    <dt>Source</dt><dd>${esc(k.source)}</dd><dt>Version</dt><dd>${esc(k.version)}</dd></dl>`;
  $("#drawer").classList.add("is-open"); $("#drawer").setAttribute("aria-hidden", "false");
  $("#scrim").hidden = false;
}
function closeDrawer() { $("#drawer").classList.remove("is-open"); $("#drawer").setAttribute("aria-hidden", "true"); $("#scrim").hidden = true; }

/* ====================================================================== custom examples */
function renderCustomExamples() {
  const ex = [
    { name: "Exporter switching supplier", narrative: "An export-oriented industrial customer has an existing supplier, has experienced long lead times, and is evaluating alternative industrial sensor suppliers.", sales_stage: "qualifying", customer_size: "medium", export: true, inc: true, pain: "long lead times", obj: "" },
    { name: "Hot sensor head", narrative: "A furnace equipment builder needs temperature sensors whose head runs continuously at 300 degrees Celsius. The requirement is fixed by the final design. Budget is approved and they want to order this quarter.", sales_stage: "technical_review", customer_size: "large", export: false, inc: false, pain: "", obj: "" },
    { name: "Discount request", narrative: "All technical and commercial terms are agreed. The buyer requires a 12 percent price reduction to fit the component allowance and says volume or shipping changes will not help.", sales_stage: "negotiation", customer_size: "medium", export: true, inc: true, pain: "", obj: "requires a 12 percent price reduction" },
  ];
  $("#cu-examples").innerHTML = `<span class="label-sm" style="align-self:center">Examples:</span>` +
    ex.map((e, i) => `<button type="button" data-i="${i}">${esc(e.name)}</button>`).join("");
  $$("#cu-examples button").forEach((b) => b.addEventListener("click", () => {
    const e = ex[+b.dataset.i];
    $("#cu-narrative").value = e.narrative; $("#cu-stage").value = e.sales_stage; $("#cu-size").value = e.customer_size;
    $("#cu-export").checked = e.export; $("#cu-incumbent").checked = e.inc; $("#cu-pain").value = e.pain; $("#cu-obj").value = e.obj;
  }));
}

/* ====================================================================== errors and tooltip */
function showError(msg) { const b = $("#error-banner"); b.textContent = msg; b.hidden = false; }
function hideError() { $("#error-banner").hidden = true; }

function bindTips(root) {
  const tip = $("#tooltip");
  $$("[data-tip]", root).forEach((el) => {
    el.addEventListener("mousemove", (e) => { tip.innerHTML = el.dataset.tip; tip.hidden = false; placeTip(e); });
    el.addEventListener("mouseleave", () => { tip.hidden = true; });
    el.addEventListener("focus", () => { const r = el.getBoundingClientRect(); tip.innerHTML = el.dataset.tip; tip.hidden = false; placeTip({ clientX: r.right, clientY: r.top }); });
    el.addEventListener("blur", () => { tip.hidden = true; });
  });
}
function placeTip(e) {
  const tip = $("#tooltip"); const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}

/* ====================================================================== results */
async function loadResults() {
  try { S.results = await api("/api/results"); }
  catch (e) { $("#results-wrap").innerHTML = `<div class="notice">${esc(e.message)}</div>`; return; }
  renderResults();
}

function renderResults() {
  const R = S.results, H = R.heldout_raw, ST = R.heldout_stripped, D = R.dev_raw;
  if (!H) { $("#results-wrap").innerHTML = `<div class="notice">No held-out run on disk yet.</div>`; return; }
  const sysAcc = (run, s) => run && run.systems[s] ? run.systems[s] : null;
  const rag = H.systems.rag, llm = H.systems.llm, rule = H.systems.rule;
  const incomplete = !H.complete ? `<div class="notice">The held-out run is incomplete or contains service errors. The numbers below are provisional.</div>` : "";
  const retries = H.retries && H.retries.length ? `<div class="callout info">${H.retries.map((r) => `${r.rows} rows were re-run on ${r.at.slice(0, 10)} after a service error (${esc(r.errors.join("; "))}). Scenarios, knowledge, prompt, model and thresholds were unchanged.`).join(" ")}</div>` : "";

  const tile = (label, key, a, hero) => a ? `<div class="tile${hero ? " hero" : ""}">
      <div class="t-label">${key ? `<span class="sys-key k-${key}"></span>` : ""}${esc(label)}</div>
      <div class="t-value">${pct(a.accuracy)}</div>
      <div class="t-sub">${a.correct} of ${a.n} correct · 95% CI ${pct(a.ci_low, 0)} to ${pct(a.ci_high, 0)}</div></div>` : "";

  const paired = H.paired.filter((p) => (p.a === "rag" && p.b === "llm") || p.slice === "all");
  const pairRows = paired.map((p) => `<tr><td>${SYS_LABEL[p.a]} vs ${SYS_LABEL[p.b]}</td><td>${SLICE_LABEL[p.slice]}</td>
      <td class="num">${p.n}</td><td class="num">${p.only_a}</td><td class="num">${p.only_b}</td><td class="num">${fmtP(p.p_value)}</td>
      <td>${p.p_value < 0.05 ? `<span class="sig yes">✓ significant</span>` : `<span class="sig no">not significant</span>`}</td></tr>`).join("");

  const g = H.gate;
  const gateHtml = g ? `
    <div class="tiles">
      <div class="tile"><div class="t-label">Abstain rate</div><div class="t-value">${pct(g.abstain_rate)}</div><div class="t-sub">${g.abstained} of ${g.n} sent to human review</div></div>
      <div class="tile"><div class="t-label">Abstained cases that would be wrong</div><div class="t-value">${pct(g.abstained_would_be_wrong)}</div><div class="t-sub">${g.abstained_wrong} of ${g.abstained}</div></div>
      <div class="tile"><div class="t-label">Accuracy on answered cases</div><div class="t-value">${pct(g.answered_accuracy)}</div><div class="t-sub">${g.answered_correct} of ${g.answered}</div></div>
      <div class="tile"><div class="t-label">Accuracy with no gate</div><div class="t-value">${pct(rag.accuracy)}</div><div class="t-sub">for reference</div></div>
    </div>
    <div class="callout">Thresholds were fixed on the dev set before the held-out run (confidence ${(+g.tau_conf).toFixed(2)}, retrieval score ${(+g.tau_ret).toFixed(2)}).
      Dev scenarios are clean and the model was confident on almost all of them, so the thresholds came out high. On the messier
      held-out set they send most cases to review, and the escalated cases are only slightly more likely to be wrong than average.
      The thresholds did not transfer. They were not re-tuned on held-out.</div>
    <table class="tbl"><thead><tr><th>Reason for review</th><th class="num">Cases</th></tr></thead><tbody>
      ${Object.entries(g.reasons).map(([k, v]) => `<tr><td>${esc(S.meta.gate_reasons[k] || k)}</td><td class="num">${v}</td></tr>`).join("")}
    </tbody></table>` : `<div class="muted">No thresholds recorded for this run.</div>`;

  const variantRows = SYS.map((s) => `<tr><td><span class="sys-key k-${s}"></span>${SYS_LABEL[s]}</td>
      <td class="num">${sysAcc(D, s) ? pct(sysAcc(D, s).accuracy) : "n/a"}</td>
      <td class="num">${sysAcc(H, s) ? pct(sysAcc(H, s).accuracy) : "n/a"}</td>
      <td class="num">${sysAcc(ST, s) ? pct(sysAcc(ST, s).accuracy) : (ST ? "n/a" : "not run")}</td></tr>`).join("");
  const costRows = SYS.map((s) => { const c = H.cost[s]; if (!c) return ""; return `<tr><td><span class="sys-key k-${s}"></span>${SYS_LABEL[s]}</td>
      <td class="num">${c.cost_per_scenario == null ? "n/a" : "$" + c.cost_per_scenario.toFixed(4)}</td>
      <td class="num">${Math.round(c.mean_input_tokens)}</td><td class="num">${Math.round(c.mean_output_tokens)}</td>
      <td class="num">${c.mean_latency_s.toFixed(1)} s</td></tr>`; }).join("");

  $("#results-wrap").innerHTML = `
    <div class="results-head"><div><div class="eyebrow">Held-out evaluation · run ${esc(H.run_id)}</div>
      <h2>${H.n_scenarios} unseen scenarios, three systems, one model call each</h2></div>
      <div class="muted" style="font-size:12px">Model under test: ${esc(H.model_under_test)}</div></div>
    ${incomplete}${retries}
    <div class="tiles">
      ${tile("Vertical RAG + LLM", "rag", rag, true)}${tile("Generic LLM", "llm", llm)}${tile("Rule-based baseline", "rule", rule)}
      ${tile("Majority class (always " + H.majority.label + ")", null, H.majority)}
    </div>
    <div class="grid2">
      <div class="card chart-card" id="c-acc"></div>
      <div class="card chart-card" id="c-slice"></div>
    </div>
    <div class="card"><h2>Paired comparison</h2>
      <div class="chart-sub">Exact McNemar test on the same scenarios. Only the cases one system got right and the other got wrong count.</div>
      <table class="tbl"><thead><tr><th>Comparison</th><th>Subset</th><th class="num">n</th><th class="num">Only first right</th><th class="num">Only second right</th><th class="num">p</th><th></th></tr></thead>
      <tbody>${pairRows}</tbody></table></div>
    <div class="card"><h2>Confidence gate</h2><div class="chart-sub">Vertical RAG + LLM. A case goes to human review when the model is unsure or the knowledge base matches poorly.</div>${gateHtml}</div>
    <div class="grid2">
      <div class="card"><h2>Accuracy across sets</h2><div class="chart-sub">Stripped removes every sentence that names an action or gives advice.</div>
        <table class="tbl"><thead><tr><th>System</th><th class="num">Dev (40)</th><th class="num">Held-out</th><th class="num">Held-out stripped</th></tr></thead><tbody>${variantRows}</tbody></table>
        ${variantNote(R.raw_vs_stripped)}</div>
      <div class="card"><h2>Cost and latency</h2><div class="chart-sub">Held-out, measured from logged token usage.</div>
        <table class="tbl"><thead><tr><th>System</th><th class="num">Per scenario</th><th class="num">Tokens in</th><th class="num">Tokens out</th><th class="num">Latency</th></tr></thead><tbody>${costRows}</tbody></table></div>
    </div>`;
  drawAccuracyChart($("#c-acc"), H);
  drawSliceChart($("#c-slice"), H);
}

function variantNote(v) {
  if (!v) return "";
  const parts = Object.entries(v.systems).map(([s, x]) =>
    `${SYS_LABEL[s]} changed its answer on ${x.action_changed} of ${x.n} scenarios, ${x.action_changed_on_untouched} of them unchanged by stripping`);
  const onTouched = Object.values(v.systems).reduce((n, x) => n + x.correctness_flipped_on_touched, 0);
  return `<div class="callout info" style="margin-top:12px">Stripping changed ${v.touched_scenarios.length} scenarios
    (${esc(v.touched_scenarios.join(", "))}). ${parts.join("; ")}. ${onTouched === 0
      ? "Every change in accuracy between the two runs happened on scenarios the filter did not touch, so the difference measures run-to-run variation of the model, not leakage. The first held-out run is the reported result."
      : "Some changes happened on stripped scenarios; see the report tables."}</div>`;
}

/* ---------------------------------------------------------------- chart helpers */
function chartShell(el, title, sub, legendHtml, tableHtml) {
  el.innerHTML = `<h2>${esc(title)}</h2><div class="chart-sub">${esc(sub)}</div>
    <div class="chart-tools"><div class="legend">${legendHtml || ""}</div><button class="link-btn" type="button">Show table</button></div>
    <div class="chart-body"></div><div class="chart-table" hidden>${tableHtml}</div>`;
  const btn = $(".link-btn", el), body = $(".chart-body", el), tbl = $(".chart-table", el);
  btn.addEventListener("click", () => { const show = tbl.hidden; tbl.hidden = !show; body.hidden = show; btn.textContent = show ? "Show chart" : "Show table"; });
  return body;
}
const svgEl = (w, h) => `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img">`;
const colorVar = (s) => (s === "baseline" ? "var(--baseline)" : `var(--s-${s})`);

/* Horizontal bars with a 95% interval. One bar per system. */
function drawAccuracyChart(el, H) {
  const rows = [
    { key: "rag", label: SYS_LABEL.rag, a: H.systems.rag },
    { key: "llm", label: SYS_LABEL.llm, a: H.systems.llm },
    { key: "rule", label: SYS_LABEL.rule, a: H.systems.rule },
    { key: "baseline", label: "Majority class", a: H.majority },
  ].filter((r) => r.a);
  const table = `<table class="tbl"><thead><tr><th>System</th><th class="num">Accuracy</th><th class="num">Correct</th><th class="num">95% CI</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${pct(r.a.accuracy)}</td><td class="num">${r.a.correct}/${r.a.n}</td><td class="num">${pct(r.a.ci_low, 0)} to ${pct(r.a.ci_high, 0)}</td></tr>`).join("") + `</tbody></table>`;
  const body = chartShell(el, "Accuracy by system", "Exact match with the gold label. The thin line is the 95% interval.", "", table);
  const W = 560, left = 150, right = 56, top = 8, band = 44, bar = 20, H_ = top + band * rows.length + 28;
  const x = (v) => left + v * (W - left - right);
  let s = svgEl(W, H_);
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => { s += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H_ - 24}"/><text class="tick" x="${x(t)}" y="${H_ - 8}" text-anchor="middle">${t * 100}%</text>`; });
  rows.forEach((r, i) => {
    const y = top + i * band + (band - bar) / 2, w = Math.max(0, x(r.a.accuracy) - left), cy = y + bar / 2;
    s += `<text x="${left - 10}" y="${cy + 4}" text-anchor="end">${esc(r.label)}</text>`;
    s += `<path d="M${left},${y} h${Math.max(0, w - 4)} a4,4 0 0 1 4,4 v${bar - 8} a4,4 0 0 1 -4,4 h${-Math.max(0, w - 4)} z" fill="${colorVar(r.key)}"/>`;
    s += `<line class="ci" x1="${x(r.a.ci_low)}" x2="${x(r.a.ci_high)}" y1="${cy}" y2="${cy}"/><line class="ci" x1="${x(r.a.ci_low)}" x2="${x(r.a.ci_low)}" y1="${cy - 5}" y2="${cy + 5}"/><line class="ci" x1="${x(r.a.ci_high)}" x2="${x(r.a.ci_high)}" y1="${cy - 5}" y2="${cy + 5}"/>`;
    s += `<text class="val" x="${x(Math.max(r.a.accuracy, r.a.ci_high)) + 8}" y="${cy + 4}">${pct(r.a.accuracy)}</text>`;
    s += `<rect class="hit" x="0" y="${top + i * band}" width="${W}" height="${band}" tabindex="0" data-tip="<b>${esc(r.label)}</b><br>${pct(r.a.accuracy)} · ${r.a.correct} of ${r.a.n}<br>95% CI ${pct(r.a.ci_low)} to ${pct(r.a.ci_high)}"/>`;
  });
  s += `<line class="axis" x1="${left}" x2="${left}" y1="${top}" y2="${H_ - 24}"/></svg>`;
  body.innerHTML = s;
  bindTips(body);
}

/* Grouped horizontal bars: subset × system. */
function drawSliceChart(el, H) {
  const groups = ["all", "depends_on_supplier_fact", "no_supplier_fact", "clear_label", "judgment_label"].filter((g) => H.slices[g]);
  const legend = SYS.map((s) => `<span><i style="background:${colorVar(s)}"></i>${SYS_LABEL[s]}</span>`).join("");
  const table = `<table class="tbl"><thead><tr><th>Subset</th>${SYS.map((s) => `<th class="num">${SYS_LABEL[s]}</th>`).join("")}</tr></thead><tbody>` +
    groups.map((g) => `<tr><td>${SLICE_LABEL[g]} (${H.slices[g].rag ? H.slices[g].rag.n : ""})</td>${SYS.map((s) => { const a = H.slices[g][s]; return `<td class="num">${a ? `${pct(a.accuracy)} (${a.correct}/${a.n})` : "n/a"}</td>`; }).join("")}</tr>`).join("") + `</tbody></table>`;
  const body = chartShell(el, "Accuracy by subset", "Where the knowledge base helps: cases that need a supplier fact, and cases with a clear label.", legend, table);
  const W = 560, left = 168, right = 48, top = 6, bar = 11, gap = 2, groupGap = 16;
  const gh = SYS.length * bar + (SYS.length - 1) * gap;
  const H_ = top + groups.length * (gh + groupGap) + 22;
  const x = (v) => left + v * (W - left - right);
  let s = svgEl(W, H_);
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => { s += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${H_ - 22}"/><text class="tick" x="${x(t)}" y="${H_ - 6}" text-anchor="middle">${t * 100}%</text>`; });
  groups.forEach((g, gi) => {
    const y0 = top + gi * (gh + groupGap);
    const n = H.slices[g].rag ? H.slices[g].rag.n : "";
    s += `<text x="${left - 10}" y="${y0 + gh / 2 - 2}" text-anchor="end">${esc(SLICE_LABEL[g])}</text>`;
    s += `<text class="tick" x="${left - 10}" y="${y0 + gh / 2 + 12}" text-anchor="end">n = ${n}</text>`;
    SYS.forEach((sys, si) => {
      const a = H.slices[g][sys]; if (!a) return;
      const y = y0 + si * (bar + gap), w = Math.max(0, x(a.accuracy) - left);
      const r = Math.min(4, w);
      s += `<path d="M${left},${y} h${Math.max(0, w - r)} a${r},${r} 0 0 1 ${r},${r} v${bar - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${-Math.max(0, w - r)} z" fill="${colorVar(sys)}"/>`;
      if (sys === "rag" || sys === "llm") s += `<text class="val" x="${x(a.accuracy) + 6}" y="${y + bar - 1}" style="font-size:11px">${pct(a.accuracy, 0)}</text>`;
      s += `<rect class="hit" x="${left}" y="${y - 1}" width="${W - left}" height="${bar + 2}" tabindex="0" data-tip="<b>${esc(SLICE_LABEL[g])}</b><br>${SYS_LABEL[sys]}: ${pct(a.accuracy)} · ${a.correct} of ${a.n}"/>`;
    });
  });
  s += `<line class="axis" x1="${left}" x2="${left}" y1="${top}" y2="${H_ - 22}"/></svg>`;
  body.innerHTML = s;
  bindTips(body);
}
