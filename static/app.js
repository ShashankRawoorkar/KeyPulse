"use strict";

// ---------------------------------------------------------------- config
const PROMPTS = [
  "The quick brown fox jumps over the lazy dog while the early morning light spills across the kitchen table and the kettle starts to whistle.",
  "Every ordinary keystroke carries a tiny signal about how rested your brain is, how steady your hands are, and how much you are juggling right now.",
  "She packed her bag, checked the train times twice, and walked to the station through quiet streets that were still wet from last night's rain.",
  "A good habit is built from small moments repeated often, like a short walk after lunch or a glass of water before the very first coffee of the day.",
];
const PVT_MIN_WAIT = 2000, PVT_MAX_WAIT = 5000, PVT_TIMEOUT = 3000, PVT_LAPSE = 500;
const FEATURE_INFO = {
  pvt_lapses: "Sustained attention",
  pvt_median: "Alertness / processing speed",
  pvt_false_starts: "Impulse control",
  hold_cv: "Fine motor steadiness",
  hold_mean: "Motor execution",
  flight_sd: "Cognitive load",
  flight_mean: "Motor planning",
  pause_rate: "Retrieval / hesitation",
  backspace_rate: "Error monitoring",
  wpm: "Overall throughput",
};
const MIN_BASELINE = 2;

// ---------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// ---- on-device storage: profiles + check-in history live in this browser; the server only scores
const DB_KEY = "kp_db_v1";
const memDb = { users: [], nextId: 1 };
const db = {
  load() {
    try {
      const raw = localStorage.getItem(DB_KEY);
      return raw ? JSON.parse(raw) : JSON.parse(JSON.stringify(memDb));
    } catch { return memDb; }
  },
  save(d) {
    Object.assign(memDb, d);
    try { localStorage.setItem(DB_KEY, JSON.stringify(d)); } catch { /* private mode: memory only */ }
  },
  users() { return this.load().users; },
  user(id) { return this.load().users.find((u) => u.id === id); },
  createUser(name, sessions = [], isDemo = false) {
    const d = this.load();
    const id = d.nextId++;
    d.users.push({ id, name, is_demo: isDemo, sessions });
    this.save(d);
    return id;
  },
  addSession(id, s) {
    const d = this.load();
    d.users.find((u) => u.id === id).sessions.push(s);
    this.save(d);
  },
  replaceDemo(name, sessions) {
    const d = this.load();
    d.users = d.users.filter((u) => !u.is_demo);
    this.save(d);
    return this.createUser(name, sessions, true);
  },
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${res.status})`);
  return data;
}

function toast(msg, ms = 3000) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

function fmt(key, v) {
  if (v == null) return "–";
  if (key === "backspace_rate") return (v * 100).toFixed(1) + "%";
  if (key === "hold_cv") return v.toFixed(2);
  if (["pvt_lapses", "pvt_false_starts", "pause_rate"].includes(key)) return v.toFixed(1);
  return Math.round(v) + (["hold_mean", "flight_mean", "flight_sd", "pvt_median"].includes(key) ? " ms" : "");
}

function scoreColor(s) {
  if (s == null) return "#8A8577";
  if (s >= 80) return "#1F7A4D";
  if (s >= 60) return "#1D4ED8";
  if (s >= 40) return "#1D4ED8";
  return "#7F1D1D";
}

/** Local wall-clock time as a naive ISO string, e.g. 2026-11-15T19:42:05 */
function localIso() {
  const d = new Date();
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
}

function shortDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " " +
    d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

// ---------------------------------------------------------------- state
const state = { users: [], userId: null, session: null };

async function loadUsers(selectId) {
  state.users = db.users().map((u) => ({ id: u.id, name: u.name, is_demo: u.is_demo, n_sessions: u.sessions.length }));
  const sel = $("userSelect");
  sel.innerHTML = "";
  if (!state.users.length) {
    sel.innerHTML = `<option value="">No profiles yet</option>`;
    state.userId = null;
  }
  for (const u of state.users) {
    const o = document.createElement("option");
    o.value = u.id;
    o.textContent = `${u.name} (${u.n_sessions})`;
    sel.appendChild(o);
  }
  const want = selectId ?? Number(store.get("kp_user"));
  const found = state.users.find((u) => u.id === want) || state.users[0];
  if (found) {
    state.userId = found.id;
    sel.value = found.id;
    store.set("kp_user", found.id);
  }
  updateCalibNote();
}

function currentUser() { return state.users.find((u) => u.id === state.userId); }

function updateCalibNote() {
  const u = currentUser();
  const note = $("calibNote");
  if (!u) {
    note.hidden = false;
    note.innerHTML = "Create a profile with <b>+ New profile</b> to get started (or load the demo data).";
    return;
  }
  if (u.n_sessions < MIN_BASELINE) {
    note.hidden = false;
    note.innerHTML = `<b>Calibration run ${u.n_sessions + 1} of ${MIN_BASELINE}.</b> Your first ${MIN_BASELINE} check-ins set your personal baseline, so do them while you feel normal.`;
  } else {
    note.hidden = false;
    note.innerHTML = `Baseline ready (${u.n_sessions} check-ins). Try one while counting backwards from 300 in 7s, out loud, and watch your score react to the cognitive load.`;
  }
}


function showStep(name) {
  for (const s of ["setup", "type", "pvt", "result"]) {
    const el = $("step-" + s);
    const show = s === name;
    const wasHidden = el.hidden;
    el.hidden = !show;
    if (show && wasHidden) animateIn(el);
  }
}

/** Staggered entrance: each child rises in a beat after the previous one. */
function animateIn(el) {
  const groups = [el.children, ...el.querySelectorAll("#dashContent, .grid-2")].map((g) => g.children || g);
  for (const g of groups) [...g].forEach((c, i) => c.style.setProperty("--i", Math.min(i, 8)));
  el.classList.remove("entering");
  void el.offsetWidth; // restart the animation
  el.classList.add("entering");
  clearTimeout(el._enterT);
  // drop the class afterwards so hover transforms aren't held by the animation fill
  el._enterT = setTimeout(() => el.classList.remove("entering"), 1400);
}

/** Count a number up from 0 with an ease-out. */
function animateNumber(el, target, dur = 1100) {
  if (target == null || isNaN(target)) { el.textContent = target ?? "–"; return; }
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur);
    el.textContent = Math.round(target * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---------------------------------------------------------------- typing capture
const typing = { events: [], pending: new Map(), t0: null, prompt: "", holds: [], done: false };

function keyCategory(e) {
  if (e.key === "Backspace") return "b";
  if (e.key === " ") return "s";
  if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) return "c";
  return null; // modifiers, arrows, etc. are ignored
}

function startTyping() {
  Object.assign(typing, { events: [], pending: new Map(), t0: null, holds: [], done: false });
  typing.prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
  const box = $("promptBox");
  box.innerHTML = "";
  for (const ch of typing.prompt) {
    const s = document.createElement("span");
    s.textContent = ch;
    box.appendChild(s);
  }
  $("typeArea").value = "";
  $("liveStats").textContent = "";
  $("stepTotal").textContent = $("pvtToggle").checked ? "2" : "1";
  $("mobileWarn").hidden = !matchMedia("(pointer: coarse)").matches;
  showStep("type");
  renderPrompt();
  drawPulse();
  // focus once visible (a view transition may still be finishing)
  for (const ms of [50, 250, 500]) setTimeout(() => { if (document.activeElement !== $("typeArea")) $("typeArea").focus(); }, ms);
}

function renderPrompt() {
  const typed = $("typeArea").value;
  const spans = $("promptBox").children;
  for (let i = 0; i < spans.length; i++) {
    spans[i].className = i < typed.length ? (typed[i] === typing.prompt[i] ? "ok" : "err") : i === typed.length ? "cur" : "";
  }
}

function onKeyDown(e) {
  if (typing.done || e.repeat) return;
  const k = keyCategory(e);
  if (!k) return;
  const now = performance.now();
  if (typing.t0 === null) typing.t0 = now;
  typing.pending.set(e.code || e.key, { k, d: now - typing.t0 });
}

function onKeyUp(e) {
  const id = e.code || e.key;
  const p = typing.pending.get(id);
  if (!p || typing.done) return;
  typing.pending.delete(id);
  const u = performance.now() - typing.t0;
  typing.events.push({ k: p.k, d: +p.d.toFixed(2), u: +u.toFixed(2) });
  typing.holds.push(u - p.d);
  drawPulse();
  updateLiveStats();
}

function onInput() {
  renderPrompt();
  // small delay so the final key's keyup is captured
  if ($("typeArea").value.length >= typing.prompt.length) setTimeout(finishTyping, 250);
}

function updateLiveStats() {
  const h = typing.holds;
  if (h.length < 5) return;
  const mean = h.reduce((a, b) => a + b, 0) / h.length;
  const ev = typing.events;
  const mins = (ev[ev.length - 1].u - ev[0].d) / 60000;
  const chars = ev.filter((e) => e.k !== "b").length;
  $("liveStats").textContent = `avg hold ${Math.round(mean)} ms · ${Math.round(chars / 5 / mins)} wpm · ${ev.length} keys`;
}

function drawPulse() {
  const c = $("pulseCanvas");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = 70;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(21,20,15,.35)";
  ctx.beginPath(); ctx.moveTo(0, h - 6); ctx.lineTo(w, h - 6); ctx.stroke();
  const bars = typing.holds.slice(-Math.floor(w / 6));
  const cool = "#15140F", warm = "#1D4ED8"; // solid ink; long holds in blue
  bars.forEach((v, i) => {
    const bh = Math.min(h - 10, (v / 250) * (h - 10));
    const x = w - (bars.length - i) * 6;
    ctx.fillStyle = v > 180 ? warm : cool;
    ctx.globalAlpha = 0.3 + 0.7 * (i / bars.length);
    ctx.fillRect(x, h - 6 - bh, 4, bh);
    ctx.shadowBlur = 0;
  });
  ctx.globalAlpha = 1;
}

function finishTyping() {
  if (typing.done) return;
  typing.done = true;
  $("typeArea").blur();
  if ($("pvtToggle").checked) startPvt();
  else submitSession(null);
}

// ---------------------------------------------------------------- PVT (psychomotor vigilance task)
const pvt = { phase: "idle", rts: [], falseStarts: 0, timer: null, raf: null, stimT: 0, start: 0, duration: 60 };

function startPvt() {
  clearTimeout(pvt.timer);
  cancelAnimationFrame(pvt.raf);
  Object.assign(pvt, { phase: "idle", rts: [], falseStarts: 0, duration: Number($("pvtDuration").value) });
  setPvtBox("idle", "Click here to begin");
  $("pvtProgress").style.width = "0";
  $("pvtStats").textContent = "";
  showStep("pvt");
  $("pvtBox").focus();
}

function setPvtBox(cls, text) {
  $("pvtBox").className = "pvt " + cls;
  $("pvtText").textContent = text;
}

function pvtElapsed() { return (performance.now() - pvt.start) / 1000; }

function pvtSchedule() {
  if (pvtElapsed() >= pvt.duration) return pvtFinish();
  pvt.phase = "waiting";
  setPvtBox("waiting", "Wait for blue…");
  const wait = PVT_MIN_WAIT + Math.random() * (PVT_MAX_WAIT - PVT_MIN_WAIT);
  pvt.timer = setTimeout(pvtStimulus, wait);
}

function pvtStimulus() {
  pvt.phase = "stim";
  pvt.stimT = performance.now();
  setPvtBox("stim", "0");
  const tick = () => {
    if (pvt.phase !== "stim") return;
    const rt = performance.now() - pvt.stimT;
    $("pvtText").textContent = Math.round(rt);
    $("pvtProgress").style.width = Math.min(100, (pvtElapsed() / pvt.duration) * 100) + "%";
    if (rt >= PVT_TIMEOUT) return pvtRecord(PVT_TIMEOUT);
    pvt.raf = requestAnimationFrame(tick);
  };
  pvt.raf = requestAnimationFrame(tick);
}

function pvtRecord(rt) {
  cancelAnimationFrame(pvt.raf);
  pvt.phase = "feedback";
  if (rt < 100) pvt.falseStarts++;
  else pvt.rts.push(rt);
  setPvtBox(rt > PVT_LAPSE ? "early" : "waiting", rt >= PVT_TIMEOUT ? "Missed!" : `${Math.round(rt)} ms`);
  updatePvtStats();
  pvt.timer = setTimeout(pvtSchedule, 700);
}

function updatePvtStats() {
  const lapses = pvt.rts.filter((r) => r > PVT_LAPSE).length;
  $("pvtStats").textContent = `${pvt.rts.length} responses · ${lapses} lapses · ${pvt.falseStarts} false starts · ${Math.max(0, Math.ceil(pvt.duration - pvtElapsed()))}s left`;
}

function pvtRespond() {
  if (pvt.phase === "idle") {
    pvt.start = performance.now();
    return pvtSchedule();
  }
  if (pvt.phase === "waiting") {
    clearTimeout(pvt.timer);
    pvt.falseStarts++;
    pvt.phase = "feedback";
    setPvtBox("early", "Too soon!");
    updatePvtStats();
    pvt.timer = setTimeout(pvtSchedule, 900);
    return;
  }
  if (pvt.phase === "stim") pvtRecord(performance.now() - pvt.stimT);
}

function pvtFinish() {
  pvt.phase = "done";
  $("pvtProgress").style.width = "100%";
  setPvtBox("done", "Done. Crunching your numbers…");
  submitSession({ rts: pvt.rts.map((r) => +r.toFixed(1)), false_starts: pvt.falseStarts, duration_s: pvt.duration });
}

// ---------------------------------------------------------------- in-browser model
const ML = { study: null, pipes: null, ready: false, promise: null };

async function initModel() {
  ML.study = await fetch("data/study.json").then((r) => r.json());
  await new Promise((r) => setTimeout(r, 30)); // let the page paint first
  const t0 = performance.now();
  ML.pipes = {
    all: KPML.trainPipeline(ML.study, "mlp", ML.study.rows, { seed: 1 }),
    typing: KPML.trainPipeline(ML.study, "mlp_typing", ML.study.rows, { seed: 1 }),
  };
  ML.ready = true;
  const el = $("modelStatus");
  el.textContent = `model trained in-browser · ${ML.study.rows.length} sessions · ${Math.round(performance.now() - t0)} ms`;
  el.classList.add("ready");
}

const PVT_KEYS = ["pvt_lapses", "pvt_median", "pvt_false_starts"];

/** zMap: {feature: z} from the backend's personal-baseline comparison. */
function predict(zMap) {
  if (!ML.ready || !zMap || !Object.keys(zMap).length) return null;
  const hasPvt = PVT_KEYS.every((k) => zMap[k] != null);
  const pipe = hasPvt ? ML.pipes.all : ML.pipes.typing;
  const zArr = ML.study.features.map((k) => zMap[k] ?? 0);
  const p = pipe.predict({ z: zArr });
  return {
    ...p,
    readiness: Math.round(KPML.readiness(p.kss, p.tlx)),
    sleepPct: Math.round(Math.max(0, Math.min(1, (p.kss - 2) / 7)) * 100),
    pipe, typingOnly: !hasPvt,
    x: KPML.zInput(zArr, pipe.idx),
    keys: pipe.idx.map((i) => ML.study.features[i]),
  };
}

function labelFor(s) {
  if (s == null) return "Calibrating";
  return s >= 80 ? "Sharp" : s >= 60 ? "Okay" : s >= 40 ? "Running low" : "Depleted";
}

function verdict(p) {
  const s = p.sleepPct / 100, l = p.tlx / 100;
  if (s >= 0.5 && l >= 0.5) return ["Tired and overloaded",
    "Sleep pressure and mental workload are both high, the worst combination for mistakes. Stop and rest."];
  if (l >= 0.5 && l - s >= 0.15) return ["You're not tired, you're overloaded",
    "Your motor signals look like your normal, but your rhythm and pauses show heavy mental workload. Single-task for a while."];
  if (s >= 0.45 && s - l >= 0.15) return ["This is sleep pressure, not workload",
    "Reaction lapses and keypress steadiness point to fatigue. Coffee hides this; only sleep fixes it."];
  if (s < 0.35 && l < 0.4) return ["Clear-headed", "Low sleepiness and low load. A good window for deep work."];
  return ["Slightly off your best", "Mild signs of fatigue or load, nothing dramatic. Take breaks."];
}

const SLEEP_SIGNALS = new Set(["pvt_lapses", "pvt_median", "pvt_false_starts", "hold_cv", "hold_mean"]);
const LOAD_SIGNALS = new Set(["flight_sd", "flight_mean", "pause_rate", "backspace_rate", "wpm"]);

/** Keep the rule-based tips that agree with the model's verdict; the verdict replaces the generic summary tip. */
function tipsFor(r, p) {
  const tips = r?.tips ?? [], keys = r?.tip_keys ?? [];
  if (!p) return tips;
  const v = verdict(p)[0];
  return tips.filter((_, i) => {
    const k = keys[i];
    if (k === "summary") return false;
    if (v.startsWith("This is sleep pressure")) return !LOAD_SIGNALS.has(k);
    if (v.startsWith("You're not tired")) return !SLEEP_SIGNALS.has(k);
    if (v === "Clear-headed") return false;
    return true;
  });
}

function renderTips(el, r, p) {
  const tips = tipsFor(r, p);
  el.innerHTML = (tips.length ? tips : ["Nothing stands out against your baseline. Keep doing what you're doing."])
    .map((t) => `<li>${t}</li>`).join("");
}

const ML_FEATURE_LABELS = {
  pvt_lapses: "Attention lapses", pvt_median: "Reaction time", pvt_false_starts: "False starts",
  hold_cv: "Keypress steadiness", hold_mean: "Avg key hold", flight_sd: "Rhythm variability",
  flight_mean: "Gap between keys", pause_rate: "Hesitations", backspace_rate: "Corrections", wpm: "Typing speed",
};

// ---------------------------------------------------------------- submit + results
async function submitSession(pvtResult) {
  const sleep = $("sleepInput").value;
  try {
    const r = await api("/api/score/checkin", {
      method: "POST",
      body: {
        typing: typing.events, pvt: pvtResult, sleep_hours: sleep === "" ? null : Number(sleep),
        history: db.user(state.userId).sessions,
      },
    });
    r.session.created_at = localIso(); // the viewer's clock, not the server's (UTC on Vercel)
    db.addSession(state.userId, r.session);
    await loadUsers(state.userId);
    await ML.promise;
    renderResult(r);
  } catch (err) {
    toast(err.message, 5000);
    showStep("setup");
  }
}

function setGauge(s) {
  const arc = $("gaugeArc");
  const C = 2 * Math.PI * 52;
  arc.style.strokeDasharray = C;
  arc.style.strokeDashoffset = C;
  const color = scoreColor(s);
  arc.style.stroke = color;
  arc.closest(".gauge").style.setProperty("--gauge-glow", color + "88");
  requestAnimationFrame(() => requestAnimationFrame(() => {
    arc.style.strokeDashoffset = s == null ? C * 0.92 : C * (1 - s / 100);
  }));
  if (s == null) $("scoreNum").textContent = "…";
  else animateNumber($("scoreNum"), s, 1300);
  $("scoreNum").style.color = color;
  $("scoreLabel").textContent = labelFor(s);
}

function renderResult(r) {
  showStep("result");
  const zMap = Object.fromEntries(r.details.map((d) => [d.key, d.z]));
  const p = predict(zMap);
  // reaction test done, but no reaction-test baseline yet (calibration runs skipped it)
  if (p) p.noPvtBaseline = r.features.pvt_median != null && zMap.pvt_median == null;
  setGauge(p ? p.readiness : null);

  if (p) {
    const [title, text] = verdict(p);
    $("verdictTitle").textContent = title;
    $("verdictText").textContent = text;
    $("meters").hidden = false;
    $("kssVal").textContent = p.kss.toFixed(1) + " / 9";
    $("tlxVal").textContent = Math.round(p.tlx) + " / 100";
    $("kssBar").style.width = "0";
    $("tlxBar").style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $("kssBar").style.width = ((p.kss - 1) / 8) * 100 + "%";
      $("tlxBar").style.width = p.tlx + "%";
    }));
    renderShap(p);
  } else {
    $("verdictTitle").textContent = `Calibration run ${r.n_sessions} of ${MIN_BASELINE} saved`;
    $("verdictText").textContent = "KeyPulse learns your personal normal first. The model kicks in from your next check-in.";
    $("meters").hidden = true;
    $("shapSection").hidden = true;
  }

  const badge = $("anomalyBadge");
  if (r.anomaly) {
    badge.hidden = false;
    badge.className = "badge " + (r.anomaly.flagged ? "flag" : "ok");
    badge.textContent = r.anomaly.flagged
      ? `Isolation Forest: more unusual than any of your last ${r.anomaly.n_train} healthy sessions`
      : `Isolation Forest: within your normal range (${r.anomaly.n_train} sessions)`;
  } else badge.hidden = true;

  renderTips($("tipsList"), r, p);
  renderDetailTable(r);
}

function renderShap(p) {
  const e = KPML.explain(p.pipe, p.x, "readiness");
  const items = p.keys.map((k, i) => ({ k, v: e.phi[i] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const maxAbs = Math.max(5, ...items.map((i) => Math.abs(i.v)));
  const rows = items.map(({ k, v }) => {
    const w = (Math.abs(v) / maxAbs) * 50;
    const left = v < 0 ? 50 - w : 50;
    const color = v < -0.5 ? "var(--signal)" : v > 0.5 ? "var(--accent)" : "var(--muted)";
    return `<div class="shap-row">
      <span title="${FEATURE_INFO[k]}">${ML_FEATURE_LABELS[k] ?? k}</span>
      <div class="shap-track"><div class="shap-bar ${v < 0 ? "neg" : "pos"}" style="left:${left}%;width:${w}%"></div></div>
      <span class="shap-val" style="color:${color}">${v > 0 ? "+" : ""}${v.toFixed(1)}</span>
    </div>`;
  }).join("");
  $("shapChart").innerHTML = `<div class="shap-head"><span>← lowers readiness</span><span>you at your normal: <b>${e.base.toFixed(0)}</b> → now: <b>${e.full.toFixed(0)}</b></span><span>raises →</span></div>${rows}`;
  $("modelBadge").textContent = p.typingOnly
    ? `Model: typing-only neural net (7 signals → 24 → 2), used because ${p.noPvtBaseline
        ? "you don't have 2 reaction-test check-ins for a baseline yet"
        : "the reaction test was skipped"}. Trained in your browser on ${ML.study.rows.length} labelled check-ins.`
    : `Model: neural net (10 signals → 24 → 2), trained in your browser on ${ML.study.rows.length} labelled check-ins. See the Model Lab tab for how it was evaluated.`;
  $("shapSection").hidden = false;
}

function renderDetailTable(r) {
  const tbody = $("detailTable").querySelector("tbody");
  const pillClass = { worse: "worse", "slightly worse": "slightly", normal: "normal", better: "better" };
  const rows = (r.details.length ? r.details : Object.keys(r.features).map((key) => ({ key, value: r.features[key] })))
    .sort((a, b) => (b.bad_z ?? 0) - (a.bad_z ?? 0));
  tbody.innerHTML = rows.map((d) => `
    <tr>
      <td>${d.label ?? ML_FEATURE_LABELS[d.key] ?? d.key}</td>
      <td class="muted">${FEATURE_INFO[d.key] ?? ""}</td>
      <td class="num">${fmt(d.key, d.value)}</td>
      <td class="num muted">${d.baseline == null ? "–" : fmt(d.key, d.baseline)}</td>
      <td>${d.status ? `<span class="pill ${pillClass[d.status]}">${d.status}${Math.abs(d.z) >= 1 ? ` (${d.z > 0 ? "+" : ""}${d.z.toFixed(1)}σ)` : ""}</span>` : `<span class="pill normal">baseline</span>`}</td>
    </tr>`).join("");
}

// ---------------------------------------------------------------- dashboard
const charts = {};
let dashData = null;

async function renderDashboard() {
  if (!state.userId) {
    $("dashEmpty").hidden = false;
    $("dashContent").hidden = true;
    return;
  }
  await ML.promise;
  const u = db.user(state.userId);
  const h = u.sessions.length
    ? await api("/api/score/history", { method: "POST", body: { sessions: u.sessions } })
    : { sessions: [], features: {}, latest: null };
  dashData = { ...h, user: { id: u.id, name: u.name, is_demo: u.is_demo } };
  const sessions = dashData.sessions;
  $("dashEmpty").hidden = sessions.length > 0;
  $("dashContent").hidden = sessions.length === 0;
  if (!sessions.length) return;

  sessions.forEach((s) => (s.pred = predict(s.z)));
  $("demoBanner").hidden = !dashData.user.is_demo;
  const scored = sessions.filter((s) => s.pred);
  const last = scored[scored.length - 1];
  animateNumber($("statLatest"), last ? last.pred.readiness : null);
  $("statLatest").style.color = scoreColor(last?.pred.readiness);
  $("statLatestLabel").textContent = last ? labelFor(last.pred.readiness) : "Calibrating";
  const weekAgo = Date.now() - 7 * 864e5;
  const week = scored.filter((s) => new Date(s.created_at).getTime() >= weekAgo);
  animateNumber($("statAvg"), week.length ? Math.round(week.reduce((a, s) => a + s.pred.readiness, 0) / week.length) : null);
  animateNumber($("statCount"), sessions.length);
  const pairs = scored.filter((s) => s.sleep_hours != null);
  const r = pairs.length >= 5 ? KPML.pearson(pairs.map((s) => s.sleep_hours), pairs.map((s) => s.pred.readiness)) : null;
  $("statR").textContent = r == null ? "–" : (r >= 0 ? "+" : "") + r.toFixed(2);
  $("statRNote").textContent = r == null ? "log sleep on 5+ check-ins" : r > 0.4 ? "more sleep → sharper you" : r > 0.15 ? "weak link so far" : "no clear link yet";

  if (last) {
    $("dashVerdict").textContent = `${verdict(last.pred)[0]} · ${last.pred.readiness}`;
    $("dashVerdict").style.color = scoreColor(last.pred.readiness);
  } else {
    $("dashVerdict").textContent = "Calibrating…";
    $("dashVerdict").style.color = "";
  }
  renderTips($("dashTips"), dashData.latest, last && last === sessions[sessions.length - 1] ? last.pred : null);

  const labels = sessions.map((s) => shortDate(s.created_at));
  const readiness = sessions.map((s) => s.pred?.readiness ?? null);
  charts.readiness?.destroy();
  charts.readiness = new Chart($("readinessChart"), {
    data: {
      labels,
      datasets: [
        { type: "line", label: "Readiness (model)", data: readiness, yAxisID: "y",
          borderColor: "#15140F", backgroundColor: areaGrad("21,20,15"), fill: true, borderWidth: 2.5,
          tension: 0.35, spanGaps: true, pointBackgroundColor: readiness.map(scoreColor), pointBorderColor: "#FBFAF6",
          pointRadius: 4.5, pointHoverRadius: 7, order: 0 },
        { type: "bar", label: "Sleep (h)", data: sessions.map((s) => s.sleep_hours), yAxisID: "y1",
          backgroundColor: "rgba(29,78,216,.14)", hoverBackgroundColor: "rgba(29,78,216,.28)", borderRadius: 1, order: 1 },
      ],
    },
    options: chartOpts({
      y: { min: 0, max: 100, title: { display: true, text: "Readiness", color: "#6F6B5E" } },
      y1: { position: "right", min: 0, max: 12, grid: { display: false }, title: { display: true, text: "Sleep (h)", color: "#6F6B5E" } },
    }),
  });

  charts.cause?.destroy();
  charts.cause = new Chart($("causeChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Sleepiness", data: sessions.map((s) => s.pred?.sleepPct ?? null), borderColor: "#15140F",
          backgroundColor: areaGrad("21,20,15"), fill: true, tension: 0.35, spanGaps: true, pointRadius: 2, borderWidth: 2.5 },
        { label: "Cognitive load", data: sessions.map((s) => (s.pred ? Math.round(s.pred.tlx) : null)), borderColor: "#1D4ED8",
          backgroundColor: areaGrad("29,78,216"), fill: true, tension: 0.35, spanGaps: true, pointRadius: 2, borderWidth: 2.5 },
      ],
    },
    options: chartOpts({ y: { min: 0, max: 100 } }),
  });

  const sel = $("featureSelect");
  if (!sel.options.length) {
    for (const [k, v] of Object.entries(dashData.features)) {
      const o = document.createElement("option");
      o.value = k;
      o.textContent = `${v.label} (${FEATURE_INFO[k]})`;
      sel.appendChild(o);
    }
    sel.value = "pvt_lapses";
  }
  renderFeatureChart();
}

function renderFeatureChart() {
  if (!dashData) return;
  const key = $("featureSelect").value;
  const meta = dashData.features[key];
  const pts = dashData.sessions.map((s) => s.features[key] ?? null);
  const scale = key === "backspace_rate" ? 100 : 1;
  charts.feature?.destroy();
  charts.feature = new Chart($("featureChart"), {
    type: "line",
    data: {
      labels: dashData.sessions.map((s) => shortDate(s.created_at)),
      datasets: [{ label: meta.label, data: pts.map((v) => (v == null ? null : v * scale)), spanGaps: true,
        borderColor: "#15140F", backgroundColor: areaGrad("21,20,15"), fill: true, borderWidth: 2.5,
        pointBackgroundColor: "#15140F", tension: 0.35, pointRadius: 3, pointHoverRadius: 6 }],
    },
    options: chartOpts({
      y: { title: { display: true, text: `${meta.label} (${key === "backspace_rate" ? "%" : meta.unit})${meta.dir > 0 ? " ↑ worse" : " ↓ worse"}`, color: "#6F6B5E" } },
    }),
  });
}

// global Chart.js look
Chart.defaults.font.family = "'Schibsted Grotesk', 'Segoe UI', system-ui, sans-serif";
Chart.defaults.color = "#6F6B5E";
Chart.defaults.animation.duration = 900;
Chart.defaults.animation.easing = "easeOutQuart";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.padding = 16;
Object.assign(Chart.defaults.plugins.tooltip, {
  backgroundColor: "#FBFAF6", borderColor: "#15140F", borderWidth: 1,
  padding: 10, cornerRadius: 2, titleColor: "#15140F", bodyColor: "#3B3A33", usePointStyle: true,
});

/** Flat, light fill under a line in the given "r,g,b" colour (no gradients). */
function areaGrad(rgb) {
  return `rgba(${rgb},.07)`;
}

function chartOpts(scales, extra = {}) {
  const grid = { color: "rgba(21,20,15,.08)" };
  const ticks = { color: "#6F6B5E", maxRotation: 0, autoSkip: true, maxTicksLimit: 8 };
  for (const s of Object.values(scales)) {
    s.ticks = { ...ticks, ...(s.ticks || {}) };
    s.grid = s.grid ?? grid;
  }
  const { x: xExtra, ...rest } = extra;
  return {
    responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { color: "#15140F" } } },
    ...rest,
    scales: { x: { ticks, grid, ...(xExtra || {}) }, ...scales },
  };
}

// ---------------------------------------------------------------- model lab
const lab = { drawn: false, running: false };

async function renderLab() {
  await ML.promise;
  if (lab.drawn) return;
  lab.drawn = true;
  const st = ML.study, rows = st.rows;
  const lapIdx = st.features.indexOf("pvt_lapses");
  $("labN").textContent = st.n_participants;
  $("labRows").textContent = rows.length;
  $("labHours").textContent = Math.round(Math.max(...rows.map((r) => r.awake)));
  $("labDual").textContent = Math.round((rows.filter((r) => r.dual).length / rows.length) * 100) + "%";

  const bins = [];
  for (let t = 0; t < st.event_hours; t += 2) {
    const sel = rows.filter((r) => r.t >= t && r.t < t + 2);
    if (!sel.length) continue;
    const clock = Math.floor((10 + t + 1) % 24);
    bins.push({
      label: `${String(clock).padStart(2, "0")}:00`,
      kss: sel.reduce((a, r) => a + r.kss, 0) / sel.length,
      lapses: sel.reduce((a, r) => a + r.raw[lapIdx], 0) / sel.length,
    });
  }
  charts.study?.destroy();
  charts.study = new Chart($("studyChart"), {
    type: "line",
    data: {
      labels: bins.map((b) => b.label),
      datasets: [
        { label: "Mean sleepiness (KSS)", data: bins.map((b) => b.kss), borderColor: "#15140F", backgroundColor: "#15140F", tension: 0.35, yAxisID: "y" },
        { label: "Mean attention lapses / min", data: bins.map((b) => b.lapses), borderColor: "#1D4ED8", backgroundColor: "#1D4ED8", tension: 0.35, yAxisID: "y1", borderDash: [5, 4] },
      ],
    },
    options: chartOpts({
      y: { min: 1, max: 9, title: { display: true, text: "KSS (1–9)", color: "#6F6B5E" } },
      y1: { position: "right", min: 0, grid: { display: false }, title: { display: true, text: "Lapses / min", color: "#6F6B5E" } },
    }, { x: { title: { display: true, text: "Clock time during the hackathon (starts 10:00)", color: "#6F6B5E" } } }),
  });
}

async function runLab() {
  if (lab.running) return;
  lab.running = true;
  const btn = $("trainBtn");
  btn.disabled = true;
  const status = $("trainStatus");
  const st = ML.study;
  const pause = (ms = 20) => new Promise((r) => setTimeout(r, ms));

  // 1) live training curve for the deployed model
  const trL = [], vaL = [];
  charts.loss?.destroy();
  charts.loss = new Chart($("lossChart"), {
    type: "line",
    data: { labels: [], datasets: [
      { label: "Training loss", data: trL, borderColor: "#15140F", backgroundColor: "#15140F", pointRadius: 0, tension: 0.2 },
      { label: "Validation loss (unseen participants)", data: vaL, borderColor: "#1D4ED8", backgroundColor: "#1D4ED8", pointRadius: 0, tension: 0.2, borderDash: [5, 4] },
    ] },
    options: { ...chartOpts({ y: { min: 0, title: { display: true, text: "MSE", color: "#6F6B5E" } } }), animation: false },
  });
  status.textContent = "Training the deployed neural net (80% of participants train, 20% held out)…";
  await KPML.trainWithCurve(st, "mlp", {
    onEpoch: (ep, t, v) => {
      charts.loss.data.labels.push(ep + 1);
      trL.push(t);
      vaL.push(v);
      if (ep % 3 === 0 || ep === KPML.TRAIN_OPTS.epochs - 1) {
        charts.loss.update();
        status.textContent = `Training the deployed neural net: epoch ${ep + 1}/${KPML.TRAIN_OPTS.epochs} · train ${t.toFixed(3)} · validation ${v.toFixed(3)}`;
      }
    },
  });

  // 2) cross-validate every model
  const order = ["heuristic", "ridge", "mlp", "mlp_typing", "mlp_raw"];
  const tbody = $("resultsTable").querySelector("tbody");
  tbody.innerHTML = "";
  const results = {};
  const f = (v, d = 2) => (v == null ? "–" : v.toFixed(d));
  for (const key of order) {
    const spec = KPML.MODEL_SPECS[key];
    status.textContent = `Cross-validating: ${spec.label}…`;
    await pause();
    results[key] = await KPML.crossValidate(st, key, {
      onFold: (fold, k) => (status.textContent = `Cross-validating: ${spec.label} (fold ${fold}/${k})`),
    });
    const res = results[key];
    const star = res.proxy ? '<span class="muted small"> *</span>' : "";
    const tag = key === "mlp" ? '<span class="tag">deployed</span>'
      : key === "mlp_typing" ? '<span class="tag">deployed (no reaction test)</span>' : "";
    const row = document.createElement("tr");
    if (key === "mlp") row.className = "deployed";
    row.innerHTML = `<td>${spec.label}${tag}</td><td class="muted">${spec.inputs}</td>
      <td>${f(res.kss_r)}${star}</td><td>${f(res.tlx_r)}${star}</td><td>${f(res.rd_r)}</td><td>${f(res.rd_mae, 1)}</td>`;
    tbody.appendChild(row);
  }
  const foot = document.createElement("tr");
  foot.innerHTML = `<td colspan="6" class="small muted">* The heuristic outputs one score, so it can't tell tiredness and load apart. These are its correlations when used as a stand-in for each. r = Pearson correlation with the labels; MAE in readiness points.</td>`;
  tbody.appendChild(foot);

  const m = results.mlp, h = results.heuristic;
  $("labFindings").hidden = false;
  $("labFindings").innerHTML = `
    <div class="finding"><b>${h.tlx_r.toFixed(2)} → ${m.tlx_r.toFixed(2)}</b>Workload tracking, heuristic vs neural net. One score can't separate tired from overloaded; the model can.</div>
    <div class="finding"><b>${results.mlp_raw.tlx_r.toFixed(2)} → ${m.tlx_r.toFixed(2)}</b>Workload r without vs with the personal baseline. Everyone types differently, so normalising to <i>you</i> is what makes it work.</div>
    <div class="finding"><b>r = ${results.mlp_typing.rd_r.toFixed(2)}</b>Readiness from typing alone, no reaction test. Passive, all-day monitoring is feasible.</div>`;

  // 3) scatter of out-of-fold predictions
  const pts = st.rows.map((r, i) => ({ x: KPML.readiness(r.kss, r.tlx), y: KPML.readiness(m.oof[i].kss, m.oof[i].tlx), dual: r.dual }));
  charts.scatter?.destroy();
  charts.scatter = new Chart($("scatterChart"), {
    type: "scatter",
    data: { datasets: [
      { label: "Normal check-in", data: pts.filter((p) => !p.dual), backgroundColor: "rgba(21,20,15,.42)", pointRadius: 3 },
      { label: "Dual-task check-in", data: pts.filter((p) => p.dual), backgroundColor: "rgba(29,78,216,.75)", pointRadius: 3 },
      { label: "Perfect prediction", type: "line", data: [{ x: 0, y: 0 }, { x: 100, y: 100 }], borderColor: "rgba(21,20,15,.35)", borderDash: [4, 4], pointRadius: 0 },
    ] },
    options: chartOpts({
      y: { min: 0, max: 100, title: { display: true, text: "Predicted readiness", color: "#6F6B5E" } },
    }, {
      x: { type: "linear", min: 0, max: 100, title: { display: true, text: "Actual readiness (from labels)", color: "#6F6B5E" } },
      interaction: { mode: "nearest", intersect: true },
    }),
  });

  // 4) global importance
  status.textContent = "Computing Shapley importances…";
  await pause();
  const imp = KPML.globalImportance(ML.pipes.all, st.rows);
  const share = (arr) => { const s = arr.reduce((a, b) => a + b, 0); return arr.map((v) => (v / s) * 100); };
  charts.importance?.destroy();
  charts.importance = new Chart($("importanceChart"), {
    type: "bar",
    data: {
      labels: st.features.map((k) => ML_FEATURE_LABELS[k]),
      datasets: [
        { label: "Sleepiness", data: share(imp.kss), backgroundColor: "#15140F", borderRadius: 3 },
        { label: "Cognitive load", data: share(imp.tlx), backgroundColor: "#1D4ED8", borderRadius: 3 },
      ],
    },
    options: chartOpts({ y: { ticks: { autoSkip: false } } }, {
      indexAxis: "y",
      x: { title: { display: true, text: "% of prediction explained", color: "#6F6B5E" } },
    }),
  });

  status.textContent = "Done. Trained 21 models from scratch in your browser (1 with a live curve + 5 folds × 4 models).";
  btn.disabled = false;
  btn.textContent = "Retrain";
  lab.running = false;
}

// ---------------------------------------------------------------- navigation
let currentView = "checkin";
let viewSwapTimer = null;

function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  moveIndicator();
  const prev = $("view-" + currentView), next = $("view-" + name);
  currentView = name;
  clearTimeout(viewSwapTimer);
  const swap = () => {
    for (const v of ["checkin", "dashboard", "lab"]) {
      $("view-" + v).hidden = v !== name;
      $("view-" + v).classList.remove("leaving");
    }
    animateIn(next);
    if (name === "dashboard") renderDashboard();
    if (name === "lab") renderLab();
  };
  if (prev !== next && !prev.hidden) {
    prev.classList.add("leaving");
    viewSwapTimer = setTimeout(swap, 170);
  } else swap();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function moveIndicator() {
  const active = document.querySelector(".tab.active"), ind = $("tabIndicator");
  if (!active || !ind) return;
  ind.style.left = active.offsetLeft + "px";
  ind.style.width = active.offsetWidth + "px";
}

// ---------------------------------------------------------------- wiring
function wire() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
  $("userSelect").addEventListener("change", (e) => {
    state.userId = Number(e.target.value);
    store.set("kp_user", state.userId);
    updateCalibNote();
    if (!$("view-dashboard").hidden) renderDashboard();
  });
  $("newUserBtn").addEventListener("click", async () => {
    const name = (window.prompt("Profile name:") || "").trim();
    if (!name) return;
    const id = db.createUser(name.slice(0, 40));
    await loadUsers(id);
    showView("checkin");
    showStep("setup");
    toast(`Profile "${name}" created. Do ${MIN_BASELINE} calibration runs first.`);
  });
  $("seedBtn").addEventListener("click", async () => {
    const demo = await api("/api/demo/sessions?now=" + encodeURIComponent(localIso()));
    const id = db.replaceDemo(demo.name, demo.sessions);
    await loadUsers(id);
    showView("dashboard");
    toast("Loaded 3 weeks of synthetic demo history");
  });
  $("startBtn").addEventListener("click", () => {
    if (!state.userId) return toast("Create a profile first (+ New profile).");
    startTyping();
  });
  const ta = $("typeArea");
  ta.addEventListener("keydown", onKeyDown);
  ta.addEventListener("keyup", onKeyUp);
  ta.addEventListener("input", onInput);
  ta.addEventListener("paste", (e) => e.preventDefault());
  $("pvtBox").addEventListener("pointerdown", (e) => { e.preventDefault(); pvtRespond(); });
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !$("step-pvt").hidden && !e.repeat) { e.preventDefault(); pvtRespond(); }
  });
  $("againBtn").addEventListener("click", () => { showStep("setup"); updateCalibNote(); });
  $("toDashBtn").addEventListener("click", () => showView("dashboard"));
  $("featureSelect").addEventListener("change", renderFeatureChart);
  $("trainBtn").addEventListener("click", runLab);
  window.addEventListener("resize", () => { drawPulse(); moveIndicator(); });
}

wire();
moveIndicator();
document.fonts?.ready.then(moveIndicator);
animateIn($("view-checkin"));
animateIn($("step-setup"));
ML.promise = initModel().catch((e) => {
  $("modelStatus").textContent = "model failed to load";
  toast("Model failed to load: " + e.message, 6000);
});
loadUsers().catch((e) => toast("Can't reach the KeyPulse server: " + e.message, 6000));
