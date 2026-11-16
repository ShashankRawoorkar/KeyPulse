"""Synthetic 3-week history for a demo user, so trends and the anomaly model have data on stage.

Generated from a simple fatigue model: fatigue rises with sleep debt (and in the evening), and each
feature shifts with fatigue in the direction the sleep-deprivation literature reports. Clearly
labelled as synthetic in the UI.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from . import db

DEMO_NAME = "Alex (demo data)"
DAYS = 21
# specific bad nights so the story is visible: day index -> hours slept
# the last day is a short night, so the dashboard opens on a story ("Alex crammed last night")
BAD_NIGHTS = {5: 4.8, 12: 5.3, 13: 4.6, DAYS - 1: 4.0}


def _session(rng: random.Random, fatigue: float) -> dict:
    g = rng.gauss
    f = fatigue
    return {
        "hold_mean": 105 * (1 + 0.06 * f) + g(0, 3),
        "hold_cv": 0.28 * (1 + 0.25 * f) + g(0, 0.012),
        "flight_mean": 75 * (1 + 0.15 * f) + g(0, 4),
        "flight_sd": 60 * (1 + 0.30 * f) + g(0, 4),
        "pause_rate": max(0.0, 4.0 * (1 + 0.6 * f) + g(0, 0.5)),
        "backspace_rate": max(0.0, 0.04 * (1 + 0.5 * f) + g(0, 0.005)),
        "wpm": 62 * (1 - 0.12 * f) + g(0, 1.8),
        "pvt_median": 280 * (1 + 0.15 * f) + g(0, 9),
        "pvt_lapses": float(max(0, round(1.0 + 2.6 * f + g(0, 0.5)))),
        "pvt_false_starts": float(max(0, round(0.3 + 0.6 * f + g(0, 0.3)))),
    }


def demo_sessions(now: datetime | None = None) -> list[dict]:
    """The demo user's history as plain dicts (chronological), no database needed."""
    rng = random.Random(42)
    now = now or datetime.now()
    start = (now - timedelta(days=DAYS - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for day in range(DAYS):
        sleep = BAD_NIGHTS.get(day, min(8.6, max(6.6, rng.gauss(7.5, 0.4))))
        sleep = round(sleep * 2) / 2
        debt = max(0.0, 7.5 - sleep) / 3.0
        slots = [(9, 30, 0.0)] + ([(21, 15, 0.35)] if rng.random() < 0.55 else [])
        for hour, minute, evening in slots:
            ts = start + timedelta(days=day, hours=hour, minutes=minute + rng.randint(-20, 20))
            if ts > now:
                # always keep today's morning check-in (the short-night story), even before 9:30
                if day == DAYS - 1 and hour == slots[0][0]:
                    ts = now - timedelta(minutes=10)
                else:
                    continue
            out.append(dict(created_at=ts.isoformat(timespec="seconds"),
                            features=_session(rng, debt + evening), sleep_hours=sleep))
    return out


def seed_demo(con, now: datetime | None = None) -> int:
    for u in db.list_users(con):
        if u["is_demo"]:
            db.delete_user(con, u["id"])
    uid = db.create_user(con, DEMO_NAME, is_demo=True)
    for s in demo_sessions(now):
        db.add_session(con, uid, s["created_at"], s["features"], s["sleep_hours"])
    return uid
