"""Simulate the "KeyPulse Hackathon Study": 40 participants checking in every ~2.5 h across a 36 h hackathon.

SYNTHETIC DATA. Every assumption is listed here and in the Model Lab page.

Ground-truth model
------------------
* Sleepiness follows Borbély's two-process model:
    - Process S (homeostatic sleep pressure) rises while awake (tau 18.2 h) and decays during sleep (tau 4.2 h)
    - Process C (circadian) is a 24 h rhythm with peak alertness ~18:00 and a trough ~06:00
  latent fatigue F = vulnerability_i * 2.2 * (S - 0.2) - 0.35 * C - caffeine
* Vulnerability to sleep loss is a stable individual trait (Van Dongen et al., 2004) -> lognormal per person
* About 30% of check-ins are randomly assigned a dual-task condition (typing while counting backwards by 7s),
  which raises cognitive load L. The PVT is done afterwards, without the dual task.
* Labels: Karolinska Sleepiness Scale (KSS 1-9) from F, NASA-TLX mental demand (0-100) from L (plus a little F)
* Features: each person has their own baseline typing style; every feature shifts with F and L at a
  feature-specific sensitivity (PVT lapses respond most to sleep loss; pauses/rhythm respond most to load),
  plus session noise.
* Each participant has 3 rested pre-event sessions -> personal baseline, z-scored with the same code the app uses.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kp.scoring import FEATURES, _stats, score_session  # noqa: E402

FEATS = list(FEATURES.keys())
N_PARTICIPANTS = 40
EVENT_HOURS = 34.0
START_CLOCK = 10.0     # hackathon starts 10:00
SEED = 7

# fractional change per unit of fatigue (a) and of load (b); session noise as fraction of baseline
SENS = {
    "pvt_median":     dict(a=0.14,  b=0.00, noise=0.035),
    "hold_cv":        dict(a=0.18,  b=0.10, noise=0.05),
    "hold_mean":      dict(a=0.06,  b=0.04, noise=0.03),
    "flight_sd":      dict(a=0.20,  b=1.00, noise=0.07),
    "flight_mean":    dict(a=0.12,  b=0.70, noise=0.05),
    "pause_rate":     dict(a=0.45,  b=2.20, noise=0.15),
    "backspace_rate": dict(a=0.35,  b=0.60, noise=0.15),
    "wpm":            dict(a=-0.10, b=-0.45, noise=0.035),
}


def person(rng):
    wpm = float(np.clip(rng.normal(58, 14), 25, 110))
    return {
        "base": {
            "hold_mean": rng.normal(105, 14),
            "hold_cv": float(np.clip(rng.normal(0.30, 0.04), 0.18, 0.45)),
            "flight_mean": float(np.clip(80 + (58 - wpm) * 1.2 + rng.normal(0, 12), 20, 200)),
            "flight_sd": float(np.clip(rng.normal(62, 12), 30, 110)),
            "pause_rate": float(np.clip(rng.normal(4.5, 1.5), 1.0, 10)),
            "backspace_rate": float(np.clip(rng.normal(0.045, 0.015), 0.01, 0.12)),
            "wpm": wpm,
            "pvt_median": rng.normal(285, 25),
            "pvt_lapses": float(np.clip(rng.normal(0.8, 0.5), 0, 3)),
            "pvt_false_starts": float(np.clip(rng.normal(0.3, 0.3), 0, 1.5)),
        },
        "vuln": float(rng.lognormal(0, 0.35)),
        "load_react": float(rng.lognormal(0, 0.25)),
        "wake_clock": rng.uniform(6.5, 9.0),
        "nap": (rng.uniform(15, 22), rng.uniform(1.5, 4.0)) if rng.random() < 0.4 else None,
        "caffeine": sorted(rng.uniform(0, EVENT_HOURS, size=rng.integers(0, 4)).tolist()),
    }


def circadian(clock):
    return math.cos(2 * math.pi * (clock - 18.0) / 24.0)


def simulate_S(p, t_end, dt=0.25):
    """Process S from wake-up to t_end (hours since event start). Returns dict t->(S, hours_awake)."""
    t = math.floor(-(START_CLOCK - p["wake_clock"]) / dt) * dt   # snap to the dt grid
    S, awake_since, out = 0.2, t, {}
    while t <= t_end + 1e-9:
        asleep = p["nap"] is not None and p["nap"][0] <= t < p["nap"][0] + p["nap"][1]
        if asleep:
            S *= math.exp(-dt / 4.2)
            awake_since = t + dt
        else:
            S = 1 - (1 - S) * math.exp(-dt / 18.2)
        out[round(t, 2)] = (S, t - awake_since, asleep)
        t += dt
    return out


def gen_features(rng, p, F, L):
    b, x = p["base"], {}
    for k, s in SENS.items():
        mult = 1 + s["a"] * F + s["b"] * L * p["load_react"]
        x[k] = b[k] * mult * (1 + rng.normal(0, s["noise"]))
    x["pvt_lapses"] = float(max(0, round(b["pvt_lapses"] + 2.2 * max(F, 0) ** 1.5 + rng.normal(0, 0.6))))
    x["pvt_false_starts"] = float(max(0, round(b["pvt_false_starts"] + 0.5 * max(F, 0) + rng.normal(0, 0.3))))
    x["wpm"] = max(8.0, x["wpm"])
    x["hold_cv"] = max(0.05, x["hold_cv"])
    x["pause_rate"] = max(0.0, x["pause_rate"])
    x["backspace_rate"] = max(0.0, x["backspace_rate"])
    return x


def main():
    rng = np.random.default_rng(SEED)
    rows, pmeta = [], []
    for pid in range(N_PARTICIPANTS):
        p = person(rng)
        S_at = simulate_S(p, EVENT_HOURS)
        baseline = [gen_features(rng, p, F=rng.normal(0.15, 0.08), L=0.1) for _ in range(3)]
        stats = {}
        for k in FEATS:
            stats[k] = _stats([bl[k] for bl in baseline], FEATURES[k]["floor"])

        t = rng.uniform(0.3, 1.0)
        while t <= EVENT_HOURS:
            key = round(round(t * 4) / 4, 2)
            S, awake, asleep = S_at[key]
            if asleep:
                t += 0.5
                continue
            clock = (START_CLOCK + t) % 24
            caf = sum(0.35 * math.exp(-(t - c) / 4.0) for c in p["caffeine"] if c <= t)
            F = p["vuln"] * 2.2 * (S - 0.2) - 0.35 * circadian(clock) - caf + rng.normal(0, 0.08)
            dual = bool(rng.random() < 0.3)
            L = float(np.clip(0.15 + 0.1 * rng.random() + (0.65 if dual else 0) + rng.normal(0, 0.05), 0, 1))
            kss = int(np.clip(round(2.6 + 3.2 * F + rng.normal(0, 0.7)), 1, 9))
            tlx = float(np.clip(100 * L + 8 * max(F, 0) + rng.normal(0, 8), 0, 100))
            x = gen_features(rng, p, F, L)
            z = [(x[k] - stats[k][0]) / stats[k][1] for k in FEATS]
            heur = score_session(x, baseline)["readiness"]
            rows.append({
                "pid": pid, "t": round(t, 2), "clock": round(clock, 2), "awake": round(awake, 2),
                "caffeine": any(0 <= t - c <= 3 for c in p["caffeine"]), "dual": dual,
                "kss": kss, "tlx": round(tlx, 1),
                "z": [round(v, 4) for v in z],
                "raw": [round(x[k], 4) for k in FEATS],
                "heuristic": heur,
            })
            t += max(1.5, rng.normal(2.5, 0.5))
        pmeta.append({"pid": pid, "vulnerability": round(p["vuln"], 3), "napped": p["nap"] is not None})

    out = {
        "name": "KeyPulse Hackathon Study (synthetic)",
        "synthetic": True,
        "seed": SEED,
        "features": FEATS,
        "typing_features": [k for k in FEATS if not k.startswith("pvt_")],
        "n_participants": N_PARTICIPANTS,
        "event_hours": EVENT_HOURS,
        "participants": pmeta,
        "rows": rows,
    }
    dest = ROOT / "static" / "data" / "study.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")))
    kss = np.array([r["kss"] for r in rows])
    print(f"wrote {dest} — {len(rows)} sessions, {N_PARTICIPANTS} participants, "
          f"KSS mean {kss.mean():.2f} (min {kss.min()}, max {kss.max()}), "
          f"dual-task {np.mean([r['dual'] for r in rows]):.0%}")
    for lo in range(0, 36, 4):
        sel = [r for r in rows if lo <= r["t"] < lo + 4]
        if sel:
            print(f"  t={lo:2d}-{lo + 4:2d}h clock~{(START_CLOCK + lo + 2) % 24:4.1f}  "
                  f"KSS {np.mean([r['kss'] for r in sel]):.1f}  lapses {np.mean([r['raw'][0] for r in sel]):.1f}")


if __name__ == "__main__":
    main()
