/* KeyPulse ML — dependency-free, runs in the browser (and in Node for testing).
 *
 *  - MLP (1 hidden layer, tanh, Adam, L2) with two outputs: sleepiness (KSS 1-9) and cognitive load (NASA-TLX 0-100)
 *  - Ridge regression baseline (closed form)
 *  - Grouped k-fold cross-validation (split by participant, so test people are never seen in training)
 *  - Exact Shapley values for per-prediction explanations (baseline = "you at your normal", z = 0)
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------ utils
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rand) {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function shuffle(arr, rand) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

  // ------------------------------------------------------------ targets
  // readiness combines both outputs: being sleepy costs up to 60 points, being overloaded up to 40
  function readiness(kss, tlx) {
    const s = clamp((kss - 2) / 7, 0, 1);
    const l = clamp(tlx / 100, 0, 1);
    return clamp(100 - 60 * s - 40 * l, 0, 100);
  }
  const encY = (kss, tlx) => [(kss - 5) / 4, (tlx - 50) / 50];
  const decY = (y) => ({ kss: clamp(y[0] * 4 + 5, 1, 9), tlx: clamp(y[1] * 50 + 50, 0, 100) });

  // ------------------------------------------------------------ metrics
  function pearson(a, b) {
    const ma = mean(a), mb = mean(b);
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < a.length; i++) {
      num += (a[i] - ma) * (b[i] - mb);
      da += (a[i] - ma) ** 2;
      db += (b[i] - mb) ** 2;
    }
    return da && db ? num / Math.sqrt(da * db) : 0;
  }
  const mae = (a, b) => mean(a.map((v, i) => Math.abs(v - b[i])));

  // ------------------------------------------------------------ MLP
  class MLP {
    constructor(nIn, nHidden, nOut, seed = 1) {
      const rand = mulberry32(seed);
      this.nIn = nIn; this.nH = nHidden; this.nOut = nOut;
      const s1 = Math.sqrt(1 / nIn), s2 = Math.sqrt(1 / nHidden);
      this.W1 = Float64Array.from({ length: nIn * nHidden }, () => gauss(rand) * s1);
      this.b1 = new Float64Array(nHidden);
      this.W2 = Float64Array.from({ length: nHidden * nOut }, () => gauss(rand) * s2);
      this.b2 = new Float64Array(nOut);
      this.rand = rand;
    }
    get params() { return [this.W1, this.b1, this.W2, this.b2]; }

    forward(x) {
      const h = new Float64Array(this.nH);
      for (let j = 0; j < this.nH; j++) {
        let s = this.b1[j];
        for (let i = 0; i < this.nIn; i++) s += x[i] * this.W1[i * this.nH + j];
        h[j] = Math.tanh(s);
      }
      const y = new Float64Array(this.nOut);
      for (let k = 0; k < this.nOut; k++) {
        let s = this.b2[k];
        for (let j = 0; j < this.nH; j++) s += h[j] * this.W2[j * this.nOut + k];
        y[k] = s;
      }
      return { h, y };
    }
    predict(x) { return this.forward(x).y; }

    loss(X, Y) {
      let l = 0;
      for (let n = 0; n < X.length; n++) {
        const y = this.predict(X[n]);
        for (let k = 0; k < this.nOut; k++) l += (y[k] - Y[n][k]) ** 2;
      }
      return l / (X.length * this.nOut);
    }

    /** Mini-batch Adam. onEpoch(epoch, trainLoss) is called after every epoch. */
    fit(X, Y, opts = {}) {
      for (const s of this._steps(X, Y, opts)) if (opts.onEpoch) opts.onEpoch(s.ep, s.loss);
      return this;
    }

    /** Same as fit, but yields to the browser every few epochs so a live loss curve can repaint. */
    async fitAsync(X, Y, opts = {}) {
      for (const s of this._steps(X, Y, opts)) {
        if (opts.onEpoch) opts.onEpoch(s.ep, s.loss);
        if (s.ep % 3 === 0) await new Promise((r) => setTimeout(r, 0));
      }
      return this;
    }

    *_steps(X, Y, { epochs = 150, lr = 0.01, batch = 32, l2 = 1e-4 } = {}) {
      const P = this.params;
      const m = P.map((p) => new Float64Array(p.length));
      const v = P.map((p) => new Float64Array(p.length));
      const g = P.map((p) => new Float64Array(p.length));
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      let step = 0;
      const idx = X.map((_, i) => i);
      for (let ep = 0; ep < epochs; ep++) {
        const order = shuffle(idx, this.rand);
        for (let start = 0; start < order.length; start += batch) {
          const bIdx = order.slice(start, start + batch);
          g.forEach((a) => a.fill(0));
          const [gW1, gb1, gW2, gb2] = g;
          for (const n of bIdx) {
            const x = X[n];
            const { h, y } = this.forward(x);
            const dy = new Float64Array(this.nOut);
            for (let k = 0; k < this.nOut; k++) dy[k] = (2 * (y[k] - Y[n][k])) / (bIdx.length * this.nOut);
            const dh = new Float64Array(this.nH);
            for (let j = 0; j < this.nH; j++) {
              let s = 0;
              for (let k = 0; k < this.nOut; k++) {
                gW2[j * this.nOut + k] += h[j] * dy[k];
                s += this.W2[j * this.nOut + k] * dy[k];
              }
              dh[j] = s * (1 - h[j] * h[j]);
            }
            for (let k = 0; k < this.nOut; k++) gb2[k] += dy[k];
            for (let j = 0; j < this.nH; j++) {
              gb1[j] += dh[j];
              for (let i = 0; i < this.nIn; i++) gW1[i * this.nH + j] += x[i] * dh[j];
            }
          }
          step++;
          for (let p = 0; p < P.length; p++) {
            const W = P[p], G = g[p], M = m[p], V = v[p];
            const decay = p % 2 === 0 ? l2 : 0; // weights only, not biases
            for (let i = 0; i < W.length; i++) {
              const gi = G[i] + decay * W[i];
              M[i] = b1 * M[i] + (1 - b1) * gi;
              V[i] = b2 * V[i] + (1 - b2) * gi * gi;
              const mh = M[i] / (1 - b1 ** step), vh = V[i] / (1 - b2 ** step);
              W[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
            }
          }
        }
        yield { ep, loss: this.loss(X, Y) };
      }
    }
  }

  // ------------------------------------------------------------ ridge regression
  function solve(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  class Ridge {
    constructor(lambda = 1.0) { this.lambda = lambda; }
    fit(X, Y) {
      const d = X[0].length + 1, nOut = Y[0].length;
      const Xa = X.map((x) => [1, ...x]);
      const A = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => {
        let s = 0;
        for (const x of Xa) s += x[i] * x[j];
        return s + (i === j && i > 0 ? this.lambda : 0);
      }));
      this.W = [];
      for (let k = 0; k < nOut; k++) {
        const b = Array.from({ length: d }, (_, i) => Xa.reduce((s, x, n) => s + x[i] * Y[n][k], 0));
        this.W.push(solve(A, b));
      }
      return this;
    }
    predict(x) {
      return this.W.map((w) => w[0] + x.reduce((s, v, i) => s + v * w[i + 1], 0));
    }
  }

  // ------------------------------------------------------------ feature pipelines
  const Z_CLIP = 6;
  /** Personal-baseline inputs: z-scores vs. your own normal (computed server-side), clipped and scaled. */
  function zInput(zArr, idx) { return idx.map((i) => clamp(zArr[i], -Z_CLIP, Z_CLIP) / 3); }

  /** Ablation: raw values, standardised against the *population* (no personal baseline). */
  function popScaler(rows, idx) {
    const mu = idx.map((i) => mean(rows.map((r) => r.raw[i])));
    const sd = idx.map((i, j) => Math.sqrt(mean(rows.map((r) => (r.raw[i] - mu[j]) ** 2))) || 1);
    return (r) => idx.map((i, j) => clamp((r.raw[i] - mu[j]) / sd[j], -Z_CLIP, Z_CLIP) / 3);
  }

  const MODEL_SPECS = {
    heuristic: { label: "v1 hand-tuned heuristic", inputs: "personal z, 10 signals" },
    ridge:     { label: "Ridge regression", inputs: "personal z, 10 signals", kind: "ridge", feats: "all", personal: true },
    mlp:       { label: "Neural net (10→24→2)", inputs: "personal z, 10 signals", kind: "mlp", feats: "all", personal: true, deployed: true },
    mlp_typing:{ label: "Neural net, typing only", inputs: "personal z, 7 typing signals", kind: "mlp", feats: "typing", personal: true, deployed: true },
    mlp_raw:   { label: "Neural net, no personal baseline", inputs: "population-scaled raw, 10 signals", kind: "mlp", feats: "all", personal: false },
  };
  const MLP_HIDDEN = 24;
  const TRAIN_OPTS = { epochs: 150, lr: 0.01, batch: 32, l2: 1e-4 };

  function featIndex(study, which) {
    return which === "typing" ? study.typing_features.map((k) => study.features.indexOf(k)) : study.features.map((_, i) => i);
  }

  /** Train one pipeline; returns { predict(row) -> {kss, tlx}, model, toInput }. */
  function trainPipeline(study, specKey, rows, { seed = 1, onEpoch = null, valRows = null } = {}) {
    const spec = MODEL_SPECS[specKey];
    const idx = featIndex(study, spec.feats);
    const toInput = spec.personal ? (r) => zInput(r.z, idx) : popScaler(rows, idx);
    const X = rows.map(toInput);
    const Y = rows.map((r) => encY(r.kss, r.tlx));
    let model;
    if (spec.kind === "ridge") model = new Ridge(1.0).fit(X, Y);
    else {
      model = new MLP(idx.length, MLP_HIDDEN, 2, seed);
      const Xv = valRows ? valRows.map(toInput) : null;
      const Yv = valRows ? valRows.map((r) => encY(r.kss, r.tlx)) : null;
      model.fit(X, Y, { ...TRAIN_OPTS, onEpoch: onEpoch && ((ep, l) => onEpoch(ep, l, Xv ? model.loss(Xv, Yv) : null)) });
    }
    return { spec, idx, model, toInput, predict: (r) => decY(model.predict(toInput(r))) };
  }

  function groupFolds(rows, k, seed) {
    const pids = shuffle([...new Set(rows.map((r) => r.pid))], mulberry32(seed));
    return Array.from({ length: k }, (_, f) => new Set(pids.filter((_, i) => i % k === f)));
  }

  function summarize(rows, preds) {
    const kssT = rows.map((r) => r.kss), tlxT = rows.map((r) => r.tlx);
    const kssP = preds.map((p) => p.kss), tlxP = preds.map((p) => p.tlx);
    const rdT = rows.map((r) => readiness(r.kss, r.tlx)), rdP = preds.map((p) => readiness(p.kss, p.tlx));
    return {
      kss_r: pearson(kssP, kssT), kss_mae: mae(kssP, kssT),
      tlx_r: pearson(tlxP, tlxT), tlx_mae: mae(tlxP, tlxT),
      rd_r: pearson(rdP, rdT), rd_mae: mae(rdP, rdT),
      oof: preds,
    };
  }

  /** Grouped k-fold CV. Yields control between folds so the UI can repaint. */
  async function crossValidate(study, specKey, { k = 5, seed = 11, onFold = null, onEpoch = null } = {}) {
    const rows = study.rows;
    if (specKey === "heuristic") {
      // one score can't separate the two causes; report how well it tracks each one as a proxy
      const ok = rows.filter((r) => r.heuristic != null);
      const h = ok.map((r) => -r.heuristic);
      return {
        kss_r: pearson(h, ok.map((r) => r.kss)),
        tlx_r: pearson(h, ok.map((r) => r.tlx)),
        rd_r: pearson(ok.map((r) => r.heuristic), ok.map((r) => readiness(r.kss, r.tlx))),
        proxy: true,
      };
    }
    const folds = groupFolds(rows, k, seed);
    const preds = new Array(rows.length);
    for (let f = 0; f < k; f++) {
      const test = [], train = [];
      rows.forEach((r, i) => (folds[f].has(r.pid) ? test : train).push(i));
      const trainRows = train.map((i) => rows[i]), testRows = test.map((i) => rows[i]);
      const pipe = trainPipeline(study, specKey, trainRows, {
        seed: 100 + f, valRows: f === 0 ? testRows : null, onEpoch: f === 0 ? onEpoch : null,
      });
      test.forEach((i) => (preds[i] = pipe.predict(rows[i])));
      if (onFold) onFold(f + 1, k);
      await new Promise((r) => setTimeout(r, 0));
    }
    return summarize(rows, preds);
  }

  /** Train a pipeline on 80% of participants with a live train/validation loss curve (for the Model Lab). */
  async function trainWithCurve(study, specKey, { onEpoch, seed = 3 } = {}) {
    const spec = MODEL_SPECS[specKey];
    const idx = featIndex(study, spec.feats);
    const valSet = groupFolds(study.rows, 5, seed)[0];
    const train = study.rows.filter((r) => !valSet.has(r.pid)), val = study.rows.filter((r) => valSet.has(r.pid));
    const toInput = (r) => zInput(r.z, idx);
    const X = train.map(toInput), Y = train.map((r) => encY(r.kss, r.tlx));
    const Xv = val.map(toInput), Yv = val.map((r) => encY(r.kss, r.tlx));
    const model = new MLP(idx.length, MLP_HIDDEN, 2, seed);
    await model.fitAsync(X, Y, { ...TRAIN_OPTS, onEpoch: (ep, l) => onEpoch(ep, l, model.loss(Xv, Yv)) });
    return { nTrain: train.length, nVal: val.length };
  }

  // ------------------------------------------------------------ explanations
  /** Exact Shapley values of f over the inputs in `x`, against baseline vector `base`. */
  function shapley(f, x, base) {
    const n = x.length, N = 1 << n;
    const val = new Float64Array(N);
    const buf = new Array(n);
    for (let mask = 0; mask < N; mask++) {
      for (let i = 0; i < n; i++) buf[i] = mask & (1 << i) ? x[i] : base[i];
      val[mask] = f(buf);
    }
    const fact = [1];
    for (let i = 1; i <= n; i++) fact[i] = fact[i - 1] * i;
    const popcount = (m) => { let c = 0; while (m) { c += m & 1; m >>= 1; } return c; };
    const phi = new Float64Array(n);
    for (let mask = 0; mask < N; mask++) {
      const s = popcount(mask);
      const w = (fact[s] * fact[n - s - 1]) / fact[n];
      for (let i = 0; i < n; i++) {
        if (mask & (1 << i)) continue;
        phi[i] += w * (val[mask | (1 << i)] - val[mask]);
      }
    }
    return { phi: Array.from(phi), base: val[0], full: val[N - 1] };
  }

  /** Explain a deployed pipeline's prediction for one z-vector. target: 'readiness' | 'kss' | 'tlx' */
  function explain(pipe, x, target = "readiness") {
    const f = (inp) => {
      const p = decY(pipe.model.predict(inp));
      return target === "kss" ? p.kss : target === "tlx" ? p.tlx : readiness(p.kss, p.tlx);
    };
    return shapley(f, x, new Array(x.length).fill(0));
  }

  /** Global importance: mean |Shapley| per feature for each output, over a sample of rows. */
  function globalImportance(pipe, rows, sampleN = 120, seed = 5) {
    const sample = shuffle(rows, mulberry32(seed)).slice(0, sampleN);
    const acc = { kss: new Array(pipe.idx.length).fill(0), tlx: new Array(pipe.idx.length).fill(0) };
    for (const r of sample) {
      const x = pipe.toInput(r);
      for (const t of ["kss", "tlx"]) explain(pipe, x, t).phi.forEach((v, i) => (acc[t][i] += Math.abs(v) / sample.length));
    }
    return acc;
  }

  const KPML = {
    MLP, Ridge, MODEL_SPECS, readiness, pearson, mae, mulberry32,
    trainPipeline, crossValidate, trainWithCurve, explain, globalImportance, zInput, featIndex, shapley,
    TRAIN_OPTS, MLP_HIDDEN,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = KPML;
  else root.KPML = KPML;
})(typeof window !== "undefined" ? window : globalThis);
