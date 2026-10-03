import { Population } from './neural.js';
import { KalmanDecoder } from './decoder.js';

const DT = 0.02;
const VMAX = 900;
const SIZE = 800;
const CAL_SECONDS = 40;
const RUN_SECONDS = 60;
const RASTER_BINS = 150;
const TRACE_BINS = 250;
const DRIFT_RATE = 0.06;
const ASSIST = 0.7;
const KEYS = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', ' ', '⌫', '.', '?'];
const KCOLS = 6;
const KROWS = 5;
const KTOP = 150;
const PHRASES = [
  'HELLO WORLD',
  'I CAN TYPE WITH MY MIND',
  'TELEPATHY FOR EVERYONE',
  'BRAIN COMPUTER INTERFACE',
  'THE QUICK BROWN FOX',
  'GOOD MORNING MARS',
  'READ AND WRITE NEURONS',
];

const $ = (id) => document.getElementById(id);
const task = $('task');
const ctx = task.getContext('2d');
const raster = $('raster');
const rctx = raster.getContext('2d');
const trace = $('trace');
const tctx = trace.getContext('2d');
const overlay = $('overlay');
const rasterImg = document.createElement('canvas');

const ui = {
  channels: $('channels'),
  noise: $('noise'),
  dropout: $('dropout'),
  gain: $('gain'),
  grid: $('grid'),
  drift: $('drift'),
  intent: $('showIntent'),
  mode: $('mode'),
  calib: $('calib'),
  refit: $('refit'),
  run: $('run'),
};

const state = {
  phase: 'idle',
  pop: null,
  dec: null,
  cursor: { x: SIZE / 2, y: SIZE / 2 },
  pointer: { x: SIZE / 2, y: SIZE / 2, inside: false },
  pressed: false,
  cal: null,
  data: null,
  t: 0,
  lastClickAt: -1,
  prevClick: false,
  grid: 10,
  target: 0,
  correct: 0,
  wrong: 0,
  runStart: 0,
  running: false,
  flash: null,
  calibratedAt: 0,
  quality: null,
  best: Number(localStorage.getItem('bci-best') || 0),
  lastResult: null,
  rasterBuf: null,
  rasterPos: 0,
  traceBuf: [],
  mode: 'grid',
  phrase: PHRASES[0],
  typed: '',
  charsDone: 0,
  dwell: 0,
  calKind: '',
};

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

function buildPopulation() {
  const n = Number(ui.channels.value);
  state.pop = new Population(n, 1000 + n);
  state.pop.setDropout(Number(ui.dropout.value) / 100);
  state.rasterBuf = new Uint8Array(n * RASTER_BINS);
  state.rasterPos = 0;
  rasterImg.width = RASTER_BINS;
  rasterImg.height = n;
  state.dec = null;
  state.quality = null;
  state.phase = 'idle';
  state.running = false;
}

function startCalibration() {
  buildPopulationIfNeeded();
  state.phase = 'calibrating';
  state.running = false;
  state.cursor = { x: SIZE / 2, y: SIZE / 2 };
  state.cal = calSegment(state.cursor);
  state.data = { Z: [], V: [], C: [] };
}

function buildPopulationIfNeeded() {
  if (!state.pop || state.pop.size !== Number(ui.channels.value)) buildPopulation();
}

function calSegment(from) {
  const to = { x: 60 + Math.random() * (SIZE - 120), y: 60 + Math.random() * (SIZE - 120) };
  const dist = Math.hypot(to.x - from.x, to.y - from.y);
  const peak = VMAX * (0.35 + 0.45 * Math.random());
  return { from: { ...from }, to, D: Math.max(0.35, (1.875 * dist) / peak), t: 0, dwell: 0.45 };
}

function calStep() {
  const c = state.cal;
  c.t += DT;
  if (c.t <= c.D) {
    const tau = c.t / c.D;
    const s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5;
    const ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / c.D;
    const dx = c.to.x - c.from.x;
    const dy = c.to.y - c.from.y;
    state.cursor = { x: c.from.x + dx * s, y: c.from.y + dy * s };
    return { vx: (dx * ds) / VMAX, vy: (dy * ds) / VMAX, click: 0 };
  }
  const held = c.t - c.D;
  state.cursor = { ...c.to };
  if (held >= c.dwell) state.cal = calSegment(c.to);
  return { vx: 0, vy: 0, click: held > 0.1 && held < 0.35 ? 1 : 0 };
}

function startRefit() {
  if (state.phase !== 'control') return;
  state.phase = 'refit';
  state.running = false;
  state.dwell = 0;
  state.dec.reset();
  state.data = { Z: [], V: [], C: [] };
  resetScore(false);
}

function goal() {
  if (state.mode === 'speller') {
    const r = keyRect(KEYS.indexOf(expectedKey() || ' '));
    return { x: r.x + r.w / 2, y: r.y + r.h / 2, hw: r.w * 0.4, hh: r.h * 0.4 };
  }
  const cell = SIZE / state.grid;
  return { x: (state.target % state.grid + 0.5) * cell, y: (Math.floor(state.target / state.grid) + 0.5) * cell, hw: cell * 0.4, hh: cell * 0.4 };
}

function refitIntent() {
  const g = goal();
  const dx = g.x - state.cursor.x;
  const dy = g.y - state.cursor.y;
  const dist = Math.hypot(dx, dy);
  state.dwell = Math.abs(dx) < g.hw && Math.abs(dy) < g.hh ? state.dwell + DT : 0;
  const speed = dist < 4 ? 0 : Math.min(VMAX, dist * 5) / VMAX;
  return { vx: dist ? (dx / dist) * speed : 0, vy: dist ? (dy / dist) * speed : 0, click: state.dwell > 0.15 ? 1 : 0 };
}

function finishCalibration() {
  const { Z, V, C } = state.data;
  state.calKind = state.phase === 'refit' ? 'ReFIT' : 'open-loop';
  const cut = Math.floor(Z.length * 0.8);
  const holdout = new KalmanDecoder().fit(Z.slice(0, cut), V.slice(0, cut), C.slice(0, cut));
  state.quality = holdout.evaluate(Z.slice(cut), V.slice(cut), C.slice(cut));
  state.dec = new KalmanDecoder().fit(Z, V, C);
  state.data = null;
  state.phase = 'control';
  state.cursor = { x: SIZE / 2, y: SIZE / 2 };
  state.calibratedAt = state.t;
  resetScore(false);
}

function pointerIntent() {
  const p = state.pointer;
  const click = state.pressed ? 1 : 0;
  if (state.phase !== 'control' || !p.inside) return { vx: 0, vy: 0, click };
  const dx = p.x - state.cursor.x;
  const dy = p.y - state.cursor.y;
  const dist = Math.hypot(dx, dy);
  if (dist < 4) return { vx: 0, vy: 0, click };
  const speed = Math.min(VMAX, dist * 5) / VMAX;
  return { vx: (dx / dist) * speed, vy: (dy / dist) * speed, click };
}

function resetScore(run) {
  state.correct = 0;
  state.wrong = 0;
  state.runStart = state.t;
  state.running = run;
  state.typed = '';
  state.charsDone = 0;
  state.phrase = PHRASES[Math.floor(Math.random() * PHRASES.length)];
  newTarget();
}

function expectedKey() {
  const { phrase, typed } = state;
  if (!phrase.startsWith(typed)) return '⌫';
  return typed.length < phrase.length ? phrase[typed.length] : null;
}

function keyAt(x, y) {
  if (y < KTOP) return -1;
  const col = clamp(Math.floor(x / (SIZE / KCOLS)), 0, KCOLS - 1);
  const row = clamp(Math.floor((y - KTOP) / ((SIZE - KTOP) / KROWS)), 0, KROWS - 1);
  return row * KCOLS + col;
}

function keyRect(idx) {
  const w = SIZE / KCOLS;
  const h = (SIZE - KTOP) / KROWS;
  return { x: (idx % KCOLS) * w, y: KTOP + Math.floor(idx / KCOLS) * h, w, h };
}

function prefixLen() {
  const { phrase, typed } = state;
  let i = 0;
  while (i < typed.length && typed[i] === phrase[i]) i += 1;
  return i;
}

function wpm() {
  const elapsed = Math.max(1, Math.min(state.t - state.runStart, state.running ? RUN_SECONDS : Infinity));
  return ((state.charsDone + prefixLen()) / 5) / (elapsed / 60);
}

function typeKey() {
  const idx = keyAt(state.cursor.x, state.cursor.y);
  if (idx < 0) return;
  const key = KEYS[idx];
  const ok = key === expectedKey();
  state.typed = key === '⌫' ? state.typed.slice(0, -1) : state.typed + key;
  if (ok) state.correct += 1;
  else state.wrong += 1;
  state.flash = { idx, ok, at: state.t };
  if (state.typed === state.phrase) {
    state.charsDone += state.phrase.length;
    state.typed = '';
    let next = state.phrase;
    while (next === state.phrase) next = PHRASES[Math.floor(Math.random() * PHRASES.length)];
    state.phrase = next;
  }
}

function newTarget() {
  const cells = state.grid * state.grid;
  let next = state.target;
  while (next === state.target) next = Math.floor(Math.random() * cells);
  state.target = next;
}

function select() {
  if (state.mode === 'speller') {
    typeKey();
    return;
  }
  const cell = SIZE / state.grid;
  const col = clamp(Math.floor(state.cursor.x / cell), 0, state.grid - 1);
  const row = clamp(Math.floor(state.cursor.y / cell), 0, state.grid - 1);
  const idx = row * state.grid + col;
  const ok = idx === state.target;
  if (ok) {
    state.correct += 1;
    newTarget();
  } else {
    state.wrong += 1;
  }
  state.flash = { idx, ok, at: state.t };
}

function bps() {
  const n = state.mode === 'speller' ? KEYS.length : state.grid * state.grid;
  const elapsed = Math.max(1, Math.min(state.t - state.runStart, state.running ? RUN_SECONDS : Infinity));
  return (Math.log2(n - 1) * Math.max(0, state.correct - state.wrong)) / elapsed;
}

function endRun() {
  const score = bps();
  state.lastResult = { bps: score, wpm: state.mode === 'speller' ? wpm() : null, correct: state.correct, wrong: state.wrong, grid: state.grid };
  if (score > state.best) {
    state.best = score;
    localStorage.setItem('bci-best', String(score));
  }
  resetScore(false);
}

function step() {
  state.t += DT;
  const pop = state.pop;
  if (ui.drift.checked) pop.drift(DT, DRIFT_RATE);
  const intent = state.phase === 'calibrating' ? calStep() : state.phase === 'refit' ? refitIntent() : pointerIntent();
  const z = pop.step(intent.vx, intent.vy, intent.click, DT, Number(ui.noise.value));
  pushRaster(z);

  if (state.phase === 'refit') {
    state.data.Z.push(Float32Array.from(z));
    state.data.V.push([intent.vx, intent.vy]);
    state.data.C.push(intent.click);
    const out = state.dec.step(z);
    pushTrace(intent, out);
    state.cursor.x = clamp(state.cursor.x + ((1 - ASSIST) * out.vx + ASSIST * intent.vx) * VMAX * DT, 0, SIZE);
    state.cursor.y = clamp(state.cursor.y + ((1 - ASSIST) * out.vy + ASSIST * intent.vy) * VMAX * DT, 0, SIZE);
    if (state.dwell > 0.3) {
      state.lastClickAt = state.t;
      select();
      state.dwell = 0;
    }
    if (state.data.Z.length >= CAL_SECONDS / DT) finishCalibration();
    return;
  }

  if (state.phase === 'calibrating') {
    state.data.Z.push(Float32Array.from(z));
    state.data.V.push([intent.vx, intent.vy]);
    state.data.C.push(intent.click);
    pushTrace(intent, null);
    if (state.data.Z.length >= CAL_SECONDS / DT) finishCalibration();
    return;
  }
  if (state.phase !== 'control') {
    pushTrace(intent, null);
    return;
  }

  const out = state.dec.step(z);
  const g = Number(ui.gain.value);
  state.cursor.x = clamp(state.cursor.x + out.vx * VMAX * g * DT, 0, SIZE);
  state.cursor.y = clamp(state.cursor.y + out.vy * VMAX * g * DT, 0, SIZE);
  pushTrace(intent, out);

  if (out.click && !state.prevClick && state.t - state.lastClickAt > 0.35) {
    state.lastClickAt = state.t;
    select();
  }
  state.prevClick = out.click;
  if (state.running && state.t - state.runStart >= RUN_SECONDS) endRun();
}

function pushRaster(z) {
  const n = state.pop.size;
  const base = state.rasterPos * n;
  for (let i = 0; i < n; i++) state.rasterBuf[base + i] = Math.min(255, z[i]);
  state.rasterPos = (state.rasterPos + 1) % RASTER_BINS;
}

function pushTrace(intent, out) {
  state.traceBuf.push({ ix: intent.vx, iy: intent.vy, dx: out ? out.vx : null, dy: out ? out.vy : null });
  if (state.traceBuf.length > TRACE_BINS) state.traceBuf.shift();
}

function fitCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  return dpr;
}

function drawTask() {
  const dpr = task.width / SIZE;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0b0f14';
  ctx.fillRect(0, 0, SIZE, SIZE);

  const closed = state.phase === 'control' || state.phase === 'refit';
  if (closed && state.mode === 'speller') drawKeyboard();
  if (closed && state.mode === 'grid') {
    const n = state.grid;
    const cell = SIZE / n;
    const tc = state.target % n;
    const tr = Math.floor(state.target / n);
    ctx.fillStyle = '#2f7cf6';
    ctx.fillRect(tc * cell + 1, tr * cell + 1, cell - 2, cell - 2);
    if (state.flash && state.t - state.flash.at < 0.25) {
      const fc = state.flash.idx % n;
      const fr = Math.floor(state.flash.idx / n);
      ctx.fillStyle = state.flash.ok ? 'rgba(52,211,153,.55)' : 'rgba(248,113,113,.55)';
      ctx.fillRect(fc * cell, fr * cell, cell, cell);
    }
    ctx.strokeStyle = 'rgba(255,255,255,.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < n; i++) {
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, SIZE);
      ctx.moveTo(0, i * cell);
      ctx.lineTo(SIZE, i * cell);
    }
    ctx.stroke();
  }

  if (state.phase === 'calibrating' || state.phase === 'refit') {
    if (state.phase === 'calibrating') {
      const c = state.cal;
      ctx.strokeStyle = '#2f7cf6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(c.to.x, c.to.y, 22, 0, Math.PI * 2);
      ctx.stroke();
    }
    const done = state.data.Z.length / (CAL_SECONDS / DT);
    ctx.fillStyle = 'rgba(255,255,255,.08)';
    ctx.fillRect(0, SIZE - 6, SIZE, 6);
    ctx.fillStyle = '#2f7cf6';
    ctx.fillRect(0, SIZE - 6, SIZE * done, 6);
  }

  if (state.phase === 'control' && ui.intent.checked && state.pointer.inside) {
    ctx.strokeStyle = 'rgba(255,255,255,.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(state.pointer.x, state.pointer.y, 10, 0, Math.PI * 2);
    ctx.stroke();
  }

  if (state.phase !== 'idle') {
    const clicking = closed && state.t - state.lastClickAt < 0.15;
    ctx.fillStyle = clicking ? '#34d399' : '#f5f7fa';
    ctx.beginPath();
    ctx.arc(state.cursor.x, state.cursor.y, clicking ? 9 : 7, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawKeyboard() {
  const want = expectedKey();
  const hover = keyAt(state.cursor.x, state.cursor.y);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  KEYS.forEach((key, i) => {
    const r = keyRect(i);
    let fill = 'rgba(255,255,255,.04)';
    if (key === want) fill = '#2f7cf6';
    else if (i === hover) fill = 'rgba(255,255,255,.12)';
    if (state.flash && state.flash.idx === i && state.t - state.flash.at < 0.25) {
      fill = state.flash.ok ? 'rgba(52,211,153,.7)' : 'rgba(248,113,113,.7)';
    }
    ctx.fillStyle = fill;
    ctx.fillRect(r.x + 3, r.y + 3, r.w - 6, r.h - 6);
    ctx.fillStyle = key === want ? '#fff' : 'rgba(245,247,250,.8)';
    ctx.font = '600 34px system-ui, sans-serif';
    ctx.fillText(key === ' ' ? '␣' : key, r.x + r.w / 2, r.y + r.h / 2);
  });

  const { phrase, typed } = state;
  const ok = prefixLen();
  ctx.textAlign = 'left';
  ctx.font = '500 26px ui-monospace, monospace';
  const charW = ctx.measureText('M').width;
  const x0 = Math.max(16, (SIZE - charW * phrase.length) / 2);
  ctx.fillStyle = 'rgba(245,247,250,.35)';
  ctx.fillText(phrase, x0, 46);
  ctx.font = '600 34px ui-monospace, monospace';
  const typedW = ctx.measureText('M').width;
  const tx0 = Math.max(16, (SIZE - typedW * Math.max(typed.length, phrase.length)) / 2);
  [...typed].forEach((ch, i) => {
    ctx.fillStyle = i < ok ? '#f5f7fa' : '#f87171';
    ctx.fillText(ch === ' ' ? '·' : ch, tx0 + i * typedW, 104);
  });
  if (Math.floor(state.t * 2) % 2 === 0) {
    ctx.fillStyle = '#2f7cf6';
    ctx.fillRect(tx0 + typed.length * typedW, 86, 3, 36);
  }
  ctx.fillStyle = 'rgba(255,255,255,.06)';
  ctx.fillRect(0, KTOP - 2, SIZE, 2);
}

function drawRaster() {
  const n = state.pop.size;
  const img = new ImageData(RASTER_BINS, n);
  const d = img.data;
  for (let x = 0; x < RASTER_BINS; x++) {
    const col = ((state.rasterPos + x) % RASTER_BINS) * n;
    for (let ch = 0; ch < n; ch++) {
      const v = state.rasterBuf[col + ch];
      const p = (ch * RASTER_BINS + x) * 4;
      const k = v ? Math.min(255, 110 + v * 70) : 0;
      d[p] = k * 0.75;
      d[p + 1] = k * 0.85;
      d[p + 2] = k;
      d[p + 3] = 255;
    }
  }
  rasterImg.getContext('2d').putImageData(img, 0, 0);
  rctx.imageSmoothingEnabled = false;
  rctx.drawImage(rasterImg, 0, 0, raster.width, raster.height);
}

function drawTrace() {
  const w = trace.width;
  const h = trace.height;
  tctx.fillStyle = '#0b0f14';
  tctx.fillRect(0, 0, w, h);
  const half = h / 2;
  const line = (key, color, top) => {
    tctx.strokeStyle = color;
    tctx.lineWidth = Math.max(1, w / 400);
    tctx.beginPath();
    let started = false;
    state.traceBuf.forEach((p, i) => {
      if (p[key] === null) { started = false; return; }
      const x = (i / (TRACE_BINS - 1)) * w;
      const y = top + half / 2 - clamp(p[key], -1.2, 1.2) * (half * 0.4);
      if (started) tctx.lineTo(x, y);
      else { tctx.moveTo(x, y); started = true; }
    });
    tctx.stroke();
  };
  tctx.strokeStyle = 'rgba(255,255,255,.08)';
  tctx.beginPath();
  tctx.moveTo(0, half);
  tctx.lineTo(w, half);
  tctx.stroke();
  line('ix', 'rgba(255,255,255,.45)', 0);
  line('dx', '#2f7cf6', 0);
  line('iy', 'rgba(255,255,255,.45)', half);
  line('dy', '#f59e0b', half);
}

function fmt(v, d = 2) {
  return Number.isFinite(v) ? v.toFixed(d) : '—';
}

function updatePanel() {
  const q = state.quality;
  $('m-bps').textContent = state.phase === 'control' ? fmt(bps()) : '—';
  $('m-wpm').textContent = state.phase === 'control' && state.mode === 'speller' ? fmt(wpm(), 1) : '—';
  $('m-hits').textContent = state.phase === 'control' ? `${state.correct} / ${state.wrong}` : '—';
  $('m-time').textContent = state.phase === 'control'
    ? (state.running ? `${Math.max(0, RUN_SECONDS - (state.t - state.runStart)).toFixed(0)} s left` : 'free play')
    : state.data ? `${state.phase === 'refit' ? 'ReFIT ' : ''}${(CAL_SECONDS - state.data.Z.length * DT).toFixed(0)} s` : '—';
  $('m-r2').textContent = q ? `${fmt(q.r2x)} / ${fmt(q.r2y)}` : '—';
  $('m-click').textContent = q ? `${fmt(q.click * 100, 0)}%` : '—';
  $('m-alive').textContent = state.pop ? `${state.pop.alive} / ${state.pop.size}` : '—';
  $('m-age').textContent = state.phase === 'control' ? `${(state.t - state.calibratedAt).toFixed(0)} s (${state.calKind})` : '—';
  $('m-best').textContent = state.best ? fmt(state.best) : '—';
  $('m-last').textContent = state.lastResult
    ? state.lastResult.wpm !== null
      ? `${fmt(state.lastResult.wpm, 1)} WPM, ${fmt(state.lastResult.bps)} bps (${state.lastResult.correct}/${state.lastResult.wrong} keys)`
      : `${fmt(state.lastResult.bps)} bps (${state.lastResult.correct}/${state.lastResult.wrong}, ${state.lastResult.grid}×${state.lastResult.grid})`
    : '—';
  ui.run.disabled = state.phase !== 'control';
  ui.run.textContent = state.running ? 'Stop run' : `${state.mode === 'speller' ? 'Type' : 'Webgrid'} ${RUN_SECONDS} s`;
  ui.calib.textContent = state.phase === 'idle' ? 'Calibrate' : 'Recalibrate';
  ui.refit.disabled = state.phase !== 'control';

  overlay.hidden = state.phase === 'refit' || (state.phase === 'control' && state.pointer.inside);
  if (state.phase === 'idle') {
    overlay.innerHTML = '<b>Step 1 — Calibrate.</b> The cursor moves on its own while the simulated motor cortex “attempts” each movement and click. The decoder learns from those 40 seconds.';
  } else if (state.phase === 'calibrating') {
    overlay.innerHTML = '<b>Calibrating…</b> Recording spikes during attempted movements.';
  } else if (state.mode === 'speller') {
    overlay.innerHTML = '<b>Step 2 — Type by thought.</b> Move your mouse over the keyboard: it is the <i>intention</i>. The white dot is driven only by decoded spikes. Hold the button (or Space) to attempt a click on the blue key and copy the phrase; ⌫ fixes mistakes.';
  } else {
    overlay.innerHTML = '<b>Step 2 — Control.</b> Move your mouse over the grid: it is the <i>intention</i>, not the cursor. The white dot is driven only by decoded spikes. Hold the button (or Space) to attempt a click on the blue cell.';
  }
}

let lastFrame = 0;
let acc = 0;
let panelAt = 0;
function frame(ts) {
  const dt = lastFrame ? Math.min(0.1, (ts - lastFrame) / 1000) : 0;
  lastFrame = ts;
  acc += dt;
  let steps = 0;
  while (acc >= DT && steps < 5) {
    step();
    acc -= DT;
    steps += 1;
  }
  if (steps === 5) acc = 0;
  drawTask();
  drawRaster();
  drawTrace();
  if (ts - panelAt > 100) {
    updatePanel();
    panelAt = ts;
  }
  requestAnimationFrame(frame);
}

function resize() {
  const box = task.parentElement.getBoundingClientRect();
  const side = Math.min(box.width, 800);
  task.style.width = `${side}px`;
  task.style.height = `${side}px`;
  fitCanvas(task, SIZE, SIZE);
  const rb = raster.getBoundingClientRect();
  fitCanvas(raster, rb.width, rb.height);
  const tb = trace.getBoundingClientRect();
  fitCanvas(trace, tb.width, tb.height);
}

function toLogical(e) {
  const r = task.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
}

task.addEventListener('pointermove', (e) => {
  Object.assign(state.pointer, toLogical(e), { inside: true });
});
task.addEventListener('pointerleave', () => { state.pointer.inside = false; });
task.addEventListener('pointerdown', (e) => {
  Object.assign(state.pointer, toLogical(e), { inside: true });
  state.pressed = true;
});
window.addEventListener('pointerup', () => { state.pressed = false; });
task.addEventListener('contextmenu', (e) => e.preventDefault());
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target === document.body) {
    e.preventDefault();
    state.pressed = true;
  }
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space') state.pressed = false;
});

function bindOutput(input, format = (v) => v) {
  const out = input.parentElement.querySelector('output');
  const sync = () => { if (out) out.textContent = format(input.value); };
  input.addEventListener('input', sync);
  sync();
}

bindOutput(ui.channels);
bindOutput(ui.noise, (v) => Number(v).toFixed(1));
bindOutput(ui.dropout, (v) => `${v}%`);
bindOutput(ui.gain, (v) => `${Number(v).toFixed(1)}×`);
bindOutput(ui.grid, (v) => `${v}×${v}`);

ui.channels.addEventListener('change', () => {
  buildPopulation();
  resize();
});
ui.dropout.addEventListener('input', () => state.pop.setDropout(Number(ui.dropout.value) / 100));
ui.grid.addEventListener('input', () => {
  state.grid = Number(ui.grid.value);
  if (state.phase === 'control') resetScore(false);
});
ui.calib.addEventListener('click', startCalibration);
ui.refit.addEventListener('click', startRefit);
ui.run.addEventListener('click', () => {
  if (state.phase !== 'control') return;
  if (state.running) resetScore(false);
  else resetScore(true);
});

ui.mode.addEventListener('change', () => {
  state.mode = ui.mode.value;
  if (state.phase === 'control') resetScore(false);
});

if (new URLSearchParams(location.search).get('mode') === 'speller') ui.mode.value = 'speller';
state.mode = ui.mode.value;
state.grid = Number(ui.grid.value);
buildPopulation();
window.addEventListener('resize', resize);
resize();
requestAnimationFrame(frame);
