# Neural Cursor

**Live demo:** https://asterius.pro/bci/ · **Speller:** https://asterius.pro/bci/?mode=speller

An intracortical brain–computer interface loop you can break and fix in the browser. A simulated motor-cortex population spikes, a velocity Kalman filter decodes it, and the decoded cursor drives a Webgrid task and an on-screen keyboard. The mouse is only the *intention*: the cursor never sees it directly.

No build step, no dependencies, ~30 KB of plain JavaScript.

## What's inside

| File | What it does |
|---|---|
| `neural.js` | Cosine-tuned units (Georgopoulos 1986), ~18% click-tuned, Poisson spikes in 20 ms bins, rate noise, electrode dropout, random-walk drift of preferred directions |
| `decoder.js` | Velocity Kalman filter (Wu et al. 2003) with diagonal observation covariance, so every update is a 2×2 solve even at 1024 channels; LDA click decoder |
| `app.js` | Open-loop calibration on minimum-jerk reaches, assisted ReFIT recalibration, closed-loop Webgrid and Speller tasks, raster and velocity traces |
| `bench.js` | Headless scripted-user benchmark (Node) |

## Tasks and metrics

- **Webgrid:** bits per second = log₂(N − 1) · max(correct − wrong, 0) / t.
- **Speller (type by thought):** 30 keys (A–Z, space, ⌫, `.`, `?`), copy-type a phrase, fix mistakes with ⌫. Words per minute = (correct characters / 5) / minutes.

## Benchmark (scripted user, 60 s runs, 40 s calibration)

```
channels noise dropout       | R²x  R²y  click | 10x10 bps  35x35 bps | speller WPM
      64   0.3      0%       | 0.70 0.59   97% |    3.87       3.25   |   8.6
     256   0.3      0%       | 0.79 0.73   96% |    6.52       5.47   |  13.0
    1024   0.3      0%       | 0.77 0.92   96% |    7.07       9.92   |  16.0
     256   1.5      0%       | 0.70 0.70   95% |    6.19       6.33   |  13.6
     256   0.3     50%       |   —    —     —  |    0.00       0.00   |   0.0
     256   0.3     50%+recal | 0.82 0.82   96% |    4.97       4.62   |  11.4
     256   0.3     50%+ReFIT | 0.74 0.62   96% |    5.97       7.52   |  18.8
     256   0.3   drift       | 0.78 0.84   95% |    5.97       5.47   |  11.8
```

Losing half the electrodes kills a decoder fitted on the full array. A fresh 40 s open-loop calibration brings most of the performance back; **ReFIT** (closed-loop recalibration, Gilja et al. 2012) — 40 s of assisted task play, intended velocity assumed to point at the target — beats it, because the training data matches closed-loop use.

```
node src/bench.js
```

## Run locally

Any static server: `npx serve .` and open `/bci/`. Benchmark: `npm run bench`.

## Ideas / roadmap

- Language-model next-character prior for the speller (key enlargement / auto-complete)
- Load real recordings (e.g. NLB MC_Maze) instead of the simulated population

## Author

Vladislav Makarov — 12 years in software, also built [NeuroScan](https://asterius.pro/neuroscan/), an interactive 3D brain atlas. bookforwork@gmail.com

MIT License
