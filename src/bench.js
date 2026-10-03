import { Population } from '../bci/neural.js';
import { KalmanDecoder } from '../bci/decoder.js';

const DT = 0.02;
const VMAX = 900;
const SIZE = 800;

function calibrate(pop, seconds, noise) {
  const Z = [];
  const V = [];
  const C = [];
  let pos = { x: SIZE / 2, y: SIZE / 2 };
  while (Z.length < seconds / DT) {
    const to = { x: 60 + Math.random() * (SIZE - 120), y: 60 + Math.random() * (SIZE - 120) };
    const dist = Math.hypot(to.x - pos.x, to.y - pos.y);
    const D = Math.max(0.35, (1.875 * dist) / (VMAX * (0.35 + 0.45 * Math.random())));
    for (let t = DT; t <= D; t += DT) {
      const tau = t / D;
      const ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / D;
      const v = [((to.x - pos.x) * ds) / VMAX, ((to.y - pos.y) * ds) / VMAX];
      Z.push(Float32Array.from(pop.step(v[0], v[1], 0, DT, noise)));
      V.push(v);
      C.push(0);
    }
    for (let t = DT; t <= 0.45; t += DT) {
      const click = t > 0.1 && t < 0.35 ? 1 : 0;
      Z.push(Float32Array.from(pop.step(0, 0, click, DT, noise)));
      V.push([0, 0]);
      C.push(click);
    }
    pos = to;
  }
  return { Z, V, C };
}

function webgrid(pop, dec, { grid, seconds, noise, drift, rec, assist = 0 }) {
  const cell = SIZE / grid;
  let cursor = { x: SIZE / 2, y: SIZE / 2 };
  let target = Math.floor(Math.random() * grid * grid);
  let correct = 0;
  let wrong = 0;
  let prev = false;
  let lastClick = -1;
  let dwell = 0;
  dec.reset();
  for (let t = 0; t < seconds; t += DT) {
    if (drift) pop.drift(DT, 0.06);
    const tx = (target % grid + 0.5) * cell;
    const ty = (Math.floor(target / grid) + 0.5) * cell;
    const dx = tx - cursor.x;
    const dy = ty - cursor.y;
    const dist = Math.hypot(dx, dy);
    const inside = Math.abs(dx) < cell * 0.4 && Math.abs(dy) < cell * 0.4;
    dwell = inside ? dwell + DT : 0;
    const speed = dist < 4 ? 0 : Math.min(VMAX, dist * 5) / VMAX;
    const vx = dist ? (dx / dist) * speed : 0;
    const vy = dist ? (dy / dist) * speed : 0;
    const press = dwell > 0.15 ? 1 : 0;
    const z = pop.step(vx, vy, press, DT, noise);
    if (rec) {
      rec.Z.push(Float32Array.from(z));
      rec.V.push([vx, vy]);
      rec.C.push(press);
    }
    const out = dec.step(z);
    const mx = (1 - assist) * out.vx + assist * vx;
    const my = (1 - assist) * out.vy + assist * vy;
    cursor = {
      x: Math.min(SIZE, Math.max(0, cursor.x + mx * VMAX * DT)),
      y: Math.min(SIZE, Math.max(0, cursor.y + my * VMAX * DT)),
    };
    const click = assist ? press && dwell > 0.3 : out.click;
    if (click && !prev && t - lastClick > 0.35) {
      lastClick = t;
      const idx = Math.min(grid - 1, Math.floor(cursor.y / cell)) * grid + Math.min(grid - 1, Math.floor(cursor.x / cell));
      if (idx === target) {
        correct += 1;
        let next = target;
        while (next === target) next = Math.floor(Math.random() * grid * grid);
        target = next;
      } else {
        wrong += 1;
      }
      dwell = 0;
    }
    prev = click;
  }
  return { correct, wrong, bps: (Math.log2(grid * grid - 1) * Math.max(0, correct - wrong)) / seconds };
}

const KEYS = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ', ' ', '⌫', '.', '?'];
const KCOLS = 6;
const KROWS = 5;
const KTOP = 150;
const PHRASES = ['HELLO WORLD', 'I CAN TYPE WITH MY MIND', 'TELEPATHY FOR EVERYONE', 'BRAIN COMPUTER INTERFACE', 'THE QUICK BROWN FOX'];

function speller(pop, dec, { seconds, noise }) {
  const kw = SIZE / KCOLS;
  const kh = (SIZE - KTOP) / KROWS;
  let cursor = { x: SIZE / 2, y: SIZE / 2 };
  let phrase = PHRASES[0];
  let typed = '';
  let done = 0;
  let keys = 0;
  let errors = 0;
  let prev = false;
  let lastClick = -1;
  let dwell = 0;
  let p = 0;
  dec.reset();
  for (let t = 0; t < seconds; t += DT) {
    const want = phrase.startsWith(typed) ? phrase[typed.length] : '⌫';
    const idx = KEYS.indexOf(want);
    const tx = (idx % KCOLS + 0.5) * kw;
    const ty = KTOP + (Math.floor(idx / KCOLS) + 0.5) * kh;
    const dx = tx - cursor.x;
    const dy = ty - cursor.y;
    const dist = Math.hypot(dx, dy);
    dwell = Math.abs(dx) < kw * 0.4 && Math.abs(dy) < kh * 0.4 ? dwell + DT : 0;
    const speed = dist < 4 ? 0 : Math.min(VMAX, dist * 5) / VMAX;
    const press = dwell > 0.15 && t - lastClick > 0.5 ? 1 : 0;
    const z = pop.step(dist ? (dx / dist) * speed : 0, dist ? (dy / dist) * speed : 0, press, DT, noise);
    const out = dec.step(z);
    cursor = {
      x: Math.min(SIZE, Math.max(0, cursor.x + out.vx * VMAX * DT)),
      y: Math.min(SIZE, Math.max(0, cursor.y + out.vy * VMAX * DT)),
    };
    if (out.click && !prev && t - lastClick > 0.35) {
      lastClick = t;
      dwell = 0;
      if (cursor.y >= KTOP) {
        const col = Math.min(KCOLS - 1, Math.floor(cursor.x / kw));
        const row = Math.min(KROWS - 1, Math.floor((cursor.y - KTOP) / kh));
        const key = KEYS[row * KCOLS + col];
        keys += 1;
        if (key !== want) errors += 1;
        typed = key === '⌫' ? typed.slice(0, -1) : typed + key;
        if (typed === phrase) {
          done += phrase.length;
          typed = '';
          phrase = PHRASES[++p % PHRASES.length];
        }
      }
    }
    prev = out.click;
  }
  let ok = 0;
  while (ok < typed.length && typed[ok] === phrase[ok]) ok += 1;
  return { wpm: ((done + ok) / 5) / (seconds / 60), keys, errors };
}

const cases = [
  { channels: 64, noise: 0.3, dropout: 0, drift: false },
  { channels: 256, noise: 0.3, dropout: 0, drift: false },
  { channels: 1024, noise: 0.3, dropout: 0, drift: false },
  { channels: 256, noise: 1.5, dropout: 0, drift: false },
  { channels: 256, noise: 0.3, dropout: 0.5, drift: false },
  { channels: 256, noise: 0.3, dropout: 0.5, drift: false, recal: true },
  { channels: 256, noise: 0.3, dropout: 0.5, drift: false, refit: true },
  { channels: 256, noise: 0.3, dropout: 0, drift: true },
];

console.log('channels noise dropout drift |  R2x   R2y  click |  10x10 bps (c/w)   35x35 bps (c/w) | speller WPM (keys/err)');
for (const c of cases) {
  const pop = new Population(c.channels, 1000 + c.channels);
  if (c.recal) pop.setDropout(c.dropout);
  const { Z, V, C } = calibrate(pop, 40, c.noise);
  const cut = Math.floor(Z.length * 0.8);
  const q = new KalmanDecoder().fit(Z.slice(0, cut), V.slice(0, cut), C.slice(0, cut))
    .evaluate(Z.slice(cut), V.slice(cut), C.slice(cut));
  let dec = new KalmanDecoder().fit(Z, V, C);
  pop.setDropout(c.dropout);
  if (c.refit) {
    const rec = { Z: [], V: [], C: [] };
    webgrid(pop, dec, { grid: 10, seconds: 40, noise: c.noise, drift: c.drift, rec, assist: 0.7 });
    dec = new KalmanDecoder().fit(rec.Z, rec.V, rec.C);
  }
  const g10 = webgrid(pop, dec, { grid: 10, seconds: 60, noise: c.noise, drift: c.drift });
  const g35 = webgrid(pop, dec, { grid: 35, seconds: 60, noise: c.noise, drift: c.drift });
  const sp = speller(pop, dec, { seconds: 60, noise: c.noise });
  console.log(
    `${String(c.channels).padStart(8)} ${c.noise.toFixed(1).padStart(5)} ${String(c.dropout * 100 + '%' + (c.recal ? '+recal' : c.refit ? '+ReFIT' : '')).padStart(7)} ${String(c.drift).padStart(5)} | ` +
    `${q.r2x.toFixed(2).padStart(5)} ${q.r2y.toFixed(2).padStart(5)} ${(q.click * 100).toFixed(0).padStart(5)}% | ` +
    `${g10.bps.toFixed(2).padStart(9)} (${g10.correct}/${g10.wrong})   ${g35.bps.toFixed(2).padStart(9)} (${g35.correct}/${g35.wrong}) | ` +
    `${sp.wpm.toFixed(1).padStart(6)} (${sp.keys}/${sp.errors})`
  );
}
