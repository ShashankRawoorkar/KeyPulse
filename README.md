# KeyPulse: tired, or just overloaded?

**Elevate the ordinary:** everyone types all day. *How* you type (how long you hold each key, the rhythm
between keys, how often you hesitate or correct yourself) shifts with sleep loss and mental workload.
KeyPulse turns a 30-second typing sample (plus an optional 60-second reaction test) into two things a
single "focus score" can't give you: **how sleepy you are** and **how overloaded you are**. It does this
with a neural network that trains and runs **entirely in your browser**.

> Wellness signal, not a diagnosis. Inspired by keystroke-dynamics research (e.g. neuroQWERTY on early
> motor changes in Parkinson's) and the psychomotor vigilance task (PVT), a standard sleep-research tool.

**Live:** https://keypulse-iota.vercel.app

## Run it

```bash
cd keypulse
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt     # macOS/Linux: .venv/bin/python
.venv/Scripts/python -m uvicorn app:app --port 8765
```

**Deploy:** `npx vercel deploy --prod` from this folder. Vercel serves `static/` from its CDN and runs
`api/index.py` (the FastAPI app) as a serverless function. The API is **stateless**: profiles and check-in
history live in the browser's localStorage and are sent along for scoring, so nothing is lost when a
serverless instance recycles. The old SQLite-backed `/api/users/...` endpoints are kept only for the local tests.

Open **http://localhost:8765** on a laptop or desktop keyboard. Everything works offline.

```bash
.venv/Scripts/python -m pytest -q        # backend: features, baselines, isolation forest, API
node ml/evaluate.js                      # the in-browser ML pipeline, run under Node: CV metrics + Shapley check
.venv/Scripts/python ml/generate_study.py  # regenerate the simulated training study
```

## The AI/ML

| Piece | Where | What |
|---|---|---|
| **Neural net** | `static/ml.js` | 10 → 24 (tanh) → 2 MLP, written from scratch (backprop, Adam, L2). Two outputs: sleepiness (KSS 1–9) and cognitive load (NASA-TLX 0–100). Trained in the browser on page load in about 1 s. |
| **Personal baseline** | `kp/scoring.py` | Every signal is z-scored against *your own* first check-ins before it reaches the model. |
| **Typing-only model** | `static/ml.js` | 7-signal variant used when the reaction test is skipped. This is the path to passive monitoring. |
| **Explanations** | `static/ml.js` | **Exact Shapley values** (all 2¹⁰ coalitions, baseline = you at your normal) shown on every result. |
| **Evaluation** | Model Lab tab | 5-fold CV **grouped by participant** (tested only on unseen people), with comparisons against the v1 heuristic, ridge regression, typing-only and no-personal-baseline variants, plus a live loss curve and global importance. |
| **Anomaly detection** | `kp/iforest.py` | Isolation Forest written from scratch in numpy, which flags sessions unlike anything in your healthy history. |
| **Training data** | `ml/generate_study.py` | **Simulated** "KeyPulse Hackathon Study": 40 participants, 539 labelled check-ins over 34 h. |

### Results (grouped 5-fold CV, unseen participants)

| Model | Sleepiness r | Load r | Readiness r | Readiness MAE |
|---|---|---|---|---|
| v1 hand-tuned heuristic | 0.79* | 0.51* | 0.87 | – |
| Ridge regression | 0.89 | 0.91 | 0.91 | 7.3 |
| **Neural net (deployed)** | **0.88** | **0.93** | **0.91** | **7.3** |
| Neural net, typing only | 0.82 | 0.93 | 0.88 | 8.4 |
| Neural net, no personal baseline | 0.84 | 0.65 | 0.78 | 10.9 |

\* The heuristic gives one score, so these are its correlations when used as a stand-in for each label.

**Findings to pitch:**
1. **One score can't separate tired from overloaded; the model can.** Load tracking goes from 0.51 to 0.93.
2. **Personal baselines are essential.** Without them, load r drops from 0.93 to 0.65, because everyone types differently.
3. **Typing alone gets readiness r = 0.88**, so passive all-day monitoring is feasible.
4. **The model learned the neuroscience by itself:** the Shapley importances show sleepiness driven by reaction time,
   lapses and keypress steadiness, while load is driven by typing speed, rhythm and pauses.
5. Being honest about it: ridge regression is about as good, because the signal is mostly linear. The neural net's advantage is
   modest, and we say so.

### The simulated study (be upfront about this)
Generated from published sleep science, not real participants:
- **Two-process model of sleep regulation** (Borbély): sleep pressure builds while awake (τ = 18.2 h) and clears during sleep (τ = 4.2 h),
  combined with a 24 h circadian rhythm. This produces the 4 am peak and the next-afternoon "second wind".
- **Trait-like vulnerability to sleep loss** (Van Dongen et al., 2004), individual typing styles, caffeine masking, naps.
- 30% of check-ins randomly assigned a **dual-task** (type while counting backwards by 7s) to vary cognitive load.
- Noisy self-report labels (KSS ±0.7, TLX ±8). The live app's baseline code is reused, so training and inference see the same features.

The pipeline is built so real labelled check-ins can replace the simulated study without code changes.

## Demo script (≈3 min)

1. **Hook (15s):** "You hit thousands of keys a day. Each one is a tiny motor-control experiment. Is the person typing tired, or overloaded?
   Those need opposite fixes: sleep vs. single-tasking."
2. **Dashboard (30s):** *Load demo data.* Alex slept 4 h last night, and KeyPulse says **"This is sleep pressure, not workload."**
   Show the *Tired or overloaded?* chart.
3. **Live (75s):** pick a judge → **+ New profile** → 2 quick calibration runs (untick the reaction test) → a 3rd run
   **while counting backwards from 300 by 7s out loud**. Result: **"You're not tired, you're overloaded"**, with the Shapley
   bars showing exactly which signals moved. *(In testing: normal run 81, dual-task run 52 with load 79/100.)*
4. **Model Lab (45s):** press **Train & evaluate**. A live loss curve, 21 models trained in about 8 s in the browser, evaluated only on
   unseen people. Point at the three finding cards and the importance chart.
5. **Close (15s):** on-device ML, timings only and no text; next step is a browser extension for passive monitoring plus real
   study data.

**Tip:** if you want the full 10-signal model in the live demo, keep the reaction test on during the calibration runs too.

## Known limits
- Training data is simulated; real-world accuracy will be lower.
- Two calibration runs is a deliberately short baseline for demos. SD floors keep small early variation from producing huge z-scores.
- scikit-learn wasn't used because Windows Application Control blocked scipy on the dev machine. Both ML components are from scratch.
