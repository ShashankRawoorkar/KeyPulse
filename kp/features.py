"""Turn raw timing events into features. Only timings + key *category* are ever received —
never which characters were typed.

Typing event: {"k": "c"|"s"|"b", "d": keydown_ms, "u": keyup_ms}
    c = character, s = space, b = backspace
PVT: list of reaction times (ms, lapses that timed out are recorded as the timeout value),
     number of false starts, and the test duration in seconds.
"""
from __future__ import annotations

from statistics import fmean, median, pstdev

PAUSE_MS = 500        # down-to-down gap that counts as a hesitation
LAPSE_MS = 500        # PVT lapse threshold (standard)
MIN_KEYS = 20
MIN_PVT_RESPONSES = 3


def typing_features(events: list[dict]) -> dict:
    ev = sorted(
        (e for e in events if e.get("k") in ("c", "s", "b") and e["u"] >= e["d"]),
        key=lambda e: e["d"],
    )
    if len(ev) < MIN_KEYS:
        raise ValueError(f"need at least {MIN_KEYS} keystrokes, got {len(ev)}")

    holds = [e["u"] - e["d"] for e in ev if 0 < e["u"] - e["d"] < 1000]
    # flight = key-up to next key-down; negative when keys overlap (rollover) — that's real
    flights = [ev[i + 1]["d"] - ev[i]["u"] for i in range(len(ev) - 1)]
    flights = [f for f in flights if -500 < f < 2000]
    dd_gaps = [ev[i + 1]["d"] - ev[i]["d"] for i in range(len(ev) - 1)]

    n = len(ev)
    produced = sum(1 for e in ev if e["k"] in ("c", "s"))
    minutes = max((ev[-1]["u"] - ev[0]["d"]) / 60000.0, 1e-6)
    hold_mean = fmean(holds)

    return {
        "hold_mean": hold_mean,
        "hold_cv": pstdev(holds) / hold_mean,
        "flight_mean": fmean(flights),
        "flight_sd": pstdev(flights),
        "pause_rate": 100.0 * sum(1 for g in dd_gaps if g > PAUSE_MS) / n,
        "backspace_rate": sum(1 for e in ev if e["k"] == "b") / n,
        "wpm": (produced / 5.0) / minutes,
    }


def pvt_features(rts: list[float], false_starts: int, duration_s: float) -> dict:
    valid = [r for r in rts if r >= 100]
    if len(valid) < MIN_PVT_RESPONSES:
        raise ValueError(f"need at least {MIN_PVT_RESPONSES} PVT responses")
    per_min = 60.0 / max(duration_s, 1.0)
    return {
        "pvt_median": median(valid),
        "pvt_lapses": sum(1 for r in valid if r > LAPSE_MS) * per_min,
        "pvt_false_starts": false_starts * per_min,
    }
