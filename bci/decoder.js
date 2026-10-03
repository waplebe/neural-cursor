const inv2 = ([a, b, c, d]) => {
  const det = a * d - b * c || 1e-9;
  return [d / det, -b / det, -c / det, a / det];
};
const mul2 = (x, y) => [
  x[0] * y[0] + x[1] * y[2], x[0] * y[1] + x[1] * y[3],
  x[2] * y[0] + x[3] * y[2], x[2] * y[1] + x[3] * y[3],
];
const t2 = (m) => [m[0], m[2], m[1], m[3]];
const add2 = (x, y) => x.map((v, i) => v + y[i]);

const CLICK_ALPHA = 0.35;

/**
 * Velocity Kalman filter (Wu et al., 2003) with a diagonal observation
 * covariance, so every update is 2x2 regardless of channel count.
 * Click is decoded with LDA on the same binned counts.
 */
export class KalmanDecoder {
  fit(Z, V, C) {
    const T = Z.length;
    const N = Z[0].length;
    const lambda = 1e-3 * T;

    const mu = new Float64Array(N);
    for (const z of Z) for (let i = 0; i < N; i++) mu[i] += z[i];
    for (let i = 0; i < N; i++) mu[i] /= T;

    let vv = [lambda, 0, 0, lambda];
    for (const [x, y] of V) vv = add2(vv, [x * x, x * y, y * x, y * y]);
    const vvInv = inv2(vv);

    const H = new Float64Array(2 * N);
    const q = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let b0 = 0;
      let b1 = 0;
      for (let t = 0; t < T; t++) {
        const zc = Z[t][i] - mu[i];
        b0 += V[t][0] * zc;
        b1 += V[t][1] * zc;
      }
      const h0 = vvInv[0] * b0 + vvInv[1] * b1;
      const h1 = vvInv[2] * b0 + vvInv[3] * b1;
      H[2 * i] = h0;
      H[2 * i + 1] = h1;
      let res = 0;
      for (let t = 0; t < T; t++) {
        const e = Z[t][i] - mu[i] - h0 * V[t][0] - h1 * V[t][1];
        res += e * e;
      }
      q[i] = res / T + 1e-3;
    }

    let s01 = [0, 0, 0, 0];
    let s00 = [lambda, 0, 0, lambda];
    for (let t = 1; t < T; t++) {
      const [a, b] = V[t];
      const [c, d] = V[t - 1];
      s01 = add2(s01, [a * c, a * d, b * c, b * d]);
      s00 = add2(s00, [c * c, c * d, d * c, d * d]);
    }
    const A = mul2(s01, inv2(s00));
    let W = [1e-6, 0, 0, 1e-6];
    for (let t = 1; t < T; t++) {
      const [c, d] = V[t - 1];
      const e0 = V[t][0] - (A[0] * c + A[1] * d);
      const e1 = V[t][1] - (A[2] * c + A[3] * d);
      W = add2(W, [e0 * e0 / (T - 1), e0 * e1 / (T - 1), e1 * e0 / (T - 1), e1 * e1 / (T - 1)]);
    }

    let HtQiH = [0, 0, 0, 0];
    const G = new Float64Array(2 * N);
    for (let i = 0; i < N; i++) {
      const h0 = H[2 * i];
      const h1 = H[2 * i + 1];
      HtQiH = add2(HtQiH, [h0 * h0 / q[i], h0 * h1 / q[i], h1 * h0 / q[i], h1 * h1 / q[i]]);
      G[2 * i] = h0 / q[i];
      G[2 * i + 1] = h1 / q[i];
    }

    const m0 = new Float64Array(N);
    const m1 = new Float64Array(N);
    let n1 = 0;
    for (let t = 0; t < T; t++) {
      const target = C[t] ? m1 : m0;
      if (C[t]) n1 += 1;
      for (let i = 0; i < N; i++) target[i] += Z[t][i];
    }
    const n0 = Math.max(1, T - n1);
    n1 = Math.max(1, n1);
    for (let i = 0; i < N; i++) {
      m0[i] /= n0;
      m1[i] /= n1;
    }
    const varc = new Float64Array(N);
    for (let t = 0; t < T; t++) {
      const m = C[t] ? m1 : m0;
      for (let i = 0; i < N; i++) {
        const e = Z[t][i] - m[i];
        varc[i] += e * e;
      }
    }
    const w = new Float64Array(N);
    const center = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      w[i] = (m1[i] - m0[i]) / (varc[i] / T + 1e-2);
      center[i] = (m0[i] + m1[i]) / 2;
    }

    Object.assign(this, { N, mu, H, q, A, W, HtQiH, G, w, center, threshold: 0 });

    let ema = 0;
    let s1 = 0;
    let s0 = 0;
    for (let t = 0; t < T; t++) {
      ema += CLICK_ALPHA * (this.clickScore(Z[t]) - ema);
      if (C[t]) s1 += ema;
      else s0 += ema;
    }
    this.threshold = (s1 / n1 + s0 / n0) / 2;
    this.reset();
    return this;
  }

  clickScore(z) {
    let s = 0;
    for (let i = 0; i < this.N; i++) s += this.w[i] * (z[i] - this.center[i]);
    return s;
  }

  reset() {
    this.x = [0, 0];
    this.P = this.W.map((v) => v * 10);
    this.ema = 0;
  }

  step(z) {
    const { A, W, HtQiH, H, G, mu, N } = this;
    const xp = [A[0] * this.x[0] + A[1] * this.x[1], A[2] * this.x[0] + A[3] * this.x[1]];
    const Pp = add2(mul2(mul2(A, this.P), t2(A)), W);
    const P = inv2(add2(inv2(Pp), HtQiH));
    let y0 = 0;
    let y1 = 0;
    for (let i = 0; i < N; i++) {
      const e = z[i] - mu[i] - (H[2 * i] * xp[0] + H[2 * i + 1] * xp[1]);
      y0 += G[2 * i] * e;
      y1 += G[2 * i + 1] * e;
    }
    this.x = [xp[0] + P[0] * y0 + P[1] * y1, xp[1] + P[2] * y0 + P[3] * y1];
    this.P = P;
    this.ema += CLICK_ALPHA * (this.clickScore(z) - this.ema);
    return { vx: this.x[0], vy: this.x[1], click: this.ema > this.threshold, score: this.ema };
  }

  evaluate(Z, V, C) {
    this.reset();
    const out = Z.map((z) => this.step(z));
    const r2 = (k) => {
      const mean = V.reduce((s, v) => s + v[k], 0) / V.length;
      let res = 0;
      let tot = 0;
      V.forEach((v, t) => {
        const pred = k === 0 ? out[t].vx : out[t].vy;
        res += (v[k] - pred) ** 2;
        tot += (v[k] - mean) ** 2;
      });
      return 1 - res / (tot || 1);
    };
    let tp = 0;
    let p = 0;
    let tn = 0;
    let n = 0;
    out.forEach((o, t) => {
      if (C[t]) { p += 1; if (o.click) tp += 1; } else { n += 1; if (!o.click) tn += 1; }
    });
    this.reset();
    return { r2x: r2(0), r2y: r2(1), click: ((tp / (p || 1)) + (tn / (n || 1))) / 2 };
  }
}
