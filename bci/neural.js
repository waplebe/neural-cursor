export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(r) {
  let u = 0;
  while (!u) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

function poisson(lambda, r) {
  if (lambda <= 0) return 0;
  if (lambda > 30) return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * gauss(r)));
  const limit = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    p *= r();
  } while (p > limit);
  return k - 1;
}

/**
 * Cosine-tuned motor cortex population. Velocity is normalised to |v| <= 1.
 * A minority of units are tuned to click intent instead of direction.
 */
export class Population {
  constructor(channels, seed = 7) {
    this.r = rng(seed);
    this.units = [];
    for (let i = 0; i < channels; i++) {
      const clicky = this.r() < 0.18;
      this.units.push({
        angle: this.r() * Math.PI * 2,
        base: 4 + this.r() * 16,
        depth: clicky ? 2 + this.r() * 6 : 6 + this.r() * 24,
        click: clicky ? 15 + this.r() * 25 : this.r() * 3,
        alive: true,
      });
    }
    this.order = this.units.map((_, i) => i);
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(this.r() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
    this.counts = new Float32Array(channels);
  }

  get size() {
    return this.units.length;
  }

  get alive() {
    return this.units.reduce((n, u) => n + (u.alive ? 1 : 0), 0);
  }

  setDropout(fraction) {
    const dead = Math.round(this.units.length * fraction);
    this.order.forEach((idx, rank) => {
      this.units[idx].alive = rank >= dead;
    });
  }

  drift(dt, rate) {
    const s = Math.sqrt(dt);
    for (const u of this.units) {
      u.angle += gauss(this.r) * rate * s;
      u.base = Math.min(40, Math.max(1, u.base * (1 + gauss(this.r) * rate * 0.1 * s)));
    }
  }

  step(vx, vy, click, dt, noise) {
    const { units, counts, r } = this;
    for (let i = 0; i < units.length; i++) {
      const u = units[i];
      if (!u.alive) {
        counts[i] = 0;
        continue;
      }
      let rate = u.base + u.depth * (vx * Math.cos(u.angle) + vy * Math.sin(u.angle)) + u.click * click;
      if (noise > 0) rate += noise * u.base * 0.5 * gauss(r);
      counts[i] = poisson(Math.max(0, rate) * dt, r);
    }
    return counts;
  }
}
