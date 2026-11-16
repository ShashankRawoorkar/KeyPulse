"""Personal-baseline scoring.

Readiness = 100 - 25 * (weighted mean of "bad-direction" z-scores), clamped to 0..100.
A session that's 2 SD worse than your normal on everything scores 50.
"""
from __future__ import annotations

from statistics import fmean, stdev

import numpy as np

from .iforest import IsolationForest

# dir: +1 = higher is worse, -1 = lower is worse
# floor: minimum SD so a very consistent baseline doesn't turn tiny wobbles into huge z-scores
FEATURES: dict[str, dict] = {
    "pvt_lapses":       dict(label="Attention lapses",      unit="/min", dir=+1, w=1.50, floor=0.5,   group="Alertness"),
    "pvt_median":       dict(label="Reaction time",         unit="ms",   dir=+1, w=1.25, floor=10.0,  group="Alertness"),
    "pvt_false_starts": dict(label="False starts",          unit="/min", dir=+1, w=0.50, floor=0.3,   group="Alertness"),
    "hold_cv":          dict(label="Keypress steadiness",   unit="CV",   dir=+1, w=1.00, floor=0.015, group="Motor"),
    "hold_mean":        dict(label="Avg key hold",          unit="ms",   dir=+1, w=0.50, floor=3.0,   group="Motor"),
    "flight_sd":        dict(label="Rhythm variability",    unit="ms",   dir=+1, w=1.00, floor=4.0,   group="Cognitive load"),
    "flight_mean":      dict(label="Gap between keys",      unit="ms",   dir=+1, w=0.50, floor=4.0,   group="Cognitive load"),
    "pause_rate":       dict(label="Hesitations",           unit="/100 keys", dir=+1, w=1.00, floor=0.6, group="Cognitive load"),
    "backspace_rate":   dict(label="Corrections",           unit="%",    dir=+1, w=0.75, floor=0.008, group="Error monitoring"),
    "wpm":              dict(label="Typing speed",          unit="wpm",  dir=-1, w=0.75, floor=2.0,   group="Throughput"),
}

BASELINE_WINDOW = 10      # most recent sessions used as "your normal"
MIN_BASELINE = 2          # sessions needed before a feature is scored
HEALTHY_CUTOFF = 60       # sessions below this don't pollute the baseline
REL_SD_FLOOR = 0.08       # SD is at least 8% of the mean
IF_MIN_TRAIN = 5
IF_WINDOW = 30
Z_CAP = 4.0


def _stats(values: list[float], floor: float) -> tuple[float, float]:
    mean = fmean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return mean, max(sd, floor, REL_SD_FLOOR * abs(mean))


def label_for(score: float | None) -> str:
    if score is None:
        return "Calibrating"
    if score >= 80:
        return "Sharp"
    if score >= 60:
        return "Okay"
    if score >= 40:
        return "Running low"
    return "Depleted"


def score_session(feats: dict, baseline: list[dict]) -> dict:
    details = []
    for key, meta in FEATURES.items():
        if feats.get(key) is None:
            continue
        vals = [b[key] for b in baseline if b.get(key) is not None]
        if len(vals) < MIN_BASELINE:
            continue
        mean, sd = _stats(vals, meta["floor"])
        z = (feats[key] - mean) / sd
        bad = min(max(0.0, meta["dir"] * z), Z_CAP)
        good = meta["dir"] * z <= -1.0
        status = "worse" if bad >= 2 else "slightly worse" if bad >= 1 else "better" if good else "normal"
        details.append(dict(key=key, label=meta["label"], unit=meta["unit"], group=meta["group"],
                            value=feats[key], baseline=mean, z=z, bad_z=bad, status=status))

    if not details:
        return dict(readiness=None, label=label_for(None), details=[], tips=[
            "Building your baseline — complete a couple more check-ins and KeyPulse will start "
            "comparing you against your own normal."])

    wsum = sum(FEATURES[d["key"]]["w"] for d in details)
    penalty = sum(FEATURES[d["key"]]["w"] * d["bad_z"] for d in details) / wsum
    readiness = round(max(0.0, min(100.0, 100.0 - 25.0 * penalty)))
    tips, tip_keys = make_tips(details, readiness, with_keys=True)
    return dict(readiness=readiness, label=label_for(readiness), details=details,
                tips=tips, tip_keys=tip_keys)


def _pct(d):
    return abs(d["value"] - d["baseline"]) / abs(d["baseline"]) * 100 if d["baseline"] else 0


def _fmt(key, v):
    if key == "backspace_rate":
        return f"{v * 100:.1f}%"
    if key == "hold_cv":
        return f"{v:.2f}"
    if key in ("pvt_lapses", "pvt_false_starts", "pause_rate"):
        return f"{v:.1f}"
    return f"{v:.0f}"


TIP_TEXT = {
    "pvt_lapses": "Attention lapses are at {v}/min vs your usual {b}/min — the classic signature of sleep "
                  "pressure. Skip the all-nighter tonight; a 20-minute nap beats a third coffee.",
    "pvt_median": "Your reactions are {p:.0f}% slower than your normal. Avoid anything where split-second "
                  "timing matters (including driving tired).",
    "pvt_false_starts": "More jumping-the-gun responses than usual — impulsive responding often rises when "
                        "you're over-caffeinated or trying to push through fatigue.",
    "hold_cv": "Your keypresses are less steady than usual. Fine motor control is one of the first things "
               "sleep loss erodes — take a real break, not a scroll break.",
    "hold_mean": "You're holding keys {p:.0f}% longer than usual — a sign of slowed motor execution.",
    "flight_sd": "Your typing rhythm is choppier than normal — typical of a mind juggling too much. "
                 "Try single-tasking for the next hour.",
    "flight_mean": "Gaps between keystrokes are {p:.0f}% longer than usual — retrieval and planning are "
                   "taking more effort right now.",
    "pause_rate": "You're hesitating mid-word much more than usual ({v} vs {b} per 100 keys). High "
                  "cognitive load — close some tabs, literally and mentally.",
    "backspace_rate": "You're correcting {v} of keystrokes vs your usual {b}. Double-check anything "
                      "important you send today.",
    "wpm": "Typing speed is down {p:.0f}% from your norm.",
}


def make_tips(details: list[dict], readiness: int, with_keys: bool = False):
    """Signal-specific tips. tip_keys names the signal behind each tip ('summary' for the overall one),
    so the frontend can drop tips that contradict the model's tired-vs-overloaded verdict."""
    flagged = sorted((d for d in details if d["bad_z"] >= 1.0), key=lambda d: -d["bad_z"])[:3]
    tips = [TIP_TEXT[d["key"]].format(v=_fmt(d["key"], d["value"]), b=_fmt(d["key"], d["baseline"]),
                                      p=_pct(d)) for d in flagged]
    keys = [d["key"] for d in flagged]
    if readiness >= 80:
        tips.insert(0, "You're at or above your normal. Good window for deep work or hard studying.")
        keys.insert(0, "summary")
    elif readiness < 40:
        tips.insert(0, "You're well off your baseline. Prioritise rest over output — today's errors "
                       "cost more than today's progress.")
        keys.insert(0, "summary")
    return (tips, keys) if with_keys else tips


def anomaly_check(feats: dict, train: list[dict]) -> dict | None:
    """Multivariate check: is this session more unusual than anything in your recent history?"""
    if len(train) < IF_MIN_TRAIN:
        return None
    keys = [k for k in FEATURES if feats.get(k) is not None and all(t.get(k) is not None for t in train)]
    if len(keys) < 3:
        return None
    X = np.array([[t[k] for k in keys] for t in train], dtype=float)
    mu = X.mean(axis=0)
    sd = np.array([max(X[:, i].std(), FEATURES[k]["floor"], REL_SD_FLOOR * abs(mu[i]))
                   for i, k in enumerate(keys)])
    Xs = (X - mu) / sd
    x = (np.array([feats[k] for k in keys]) - mu) / sd
    forest = IsolationForest(n_trees=150, seed=0).fit(Xs)
    s = float(forest.anomaly_score(x)[0])
    worst_train = float(forest.anomaly_score(Xs).max())
    return dict(score=s, threshold=worst_train, flagged=s > worst_train, n_train=len(train),
                n_features=len(keys))


def _eligible(prior: list[dict]) -> list[dict]:
    return [p for p in prior if p["readiness"] is None or p["readiness"] >= HEALTHY_CUTOFF]


def score_history(sessions: list[dict]) -> list[dict]:
    """sessions: chronological dicts with 'features'. Adds readiness/label to each (cheap pass)."""
    out = []
    for s in sessions:
        base = [p["features"] for p in _eligible(out)][-BASELINE_WINDOW:]
        r = score_session(s["features"], base)
        # z = personal-baseline z-scores; the in-browser model consumes these
        out.append({**s, "readiness": r["readiness"], "label": r["label"],
                    "z": {d["key"]: d["z"] for d in r["details"]}})
    return out


def score_latest(sessions: list[dict]) -> dict:
    """Full scoring (details, tips, anomaly) for the last session given its history."""
    hist = score_history(sessions[:-1])
    eligible = [p["features"] for p in _eligible(hist)]
    current = sessions[-1]["features"]
    result = score_session(current, eligible[-BASELINE_WINDOW:])
    result["anomaly"] = anomaly_check(current, eligible[-IF_WINDOW:])
    return result


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 5:
        return None
    a, b = np.array(xs, float), np.array(ys, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
