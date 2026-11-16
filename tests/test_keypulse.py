import random

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kp import scoring
from kp.features import pvt_features, typing_features
from kp.iforest import IsolationForest


def make_typing(n=120, hold=100, gap=120, jitter=0, pause_every=0, backspace_every=0, seed=0):
    """Synthetic keystrokes: key i goes down at t, up at t+hold; next key down after `gap`."""
    rng = random.Random(seed)
    t, events = 0.0, []
    for i in range(n):
        h = hold + rng.uniform(-jitter, jitter)
        k = "b" if backspace_every and i % backspace_every == backspace_every - 1 else (
            "s" if i % 6 == 5 else "c")
        events.append({"k": k, "d": t, "u": t + h})
        t += h + gap + rng.uniform(-jitter, jitter)
        if pause_every and i % pause_every == 0:
            t += 900
    return events


def test_typing_features_basic_math():
    f = typing_features(make_typing(hold=100, gap=120))
    assert f["hold_mean"] == pytest.approx(100)
    assert f["hold_cv"] == pytest.approx(0)
    assert f["flight_mean"] == pytest.approx(120)
    assert f["pause_rate"] == 0
    assert f["backspace_rate"] == 0
    # 120 keys over (120*220 - 120) ms -> 24 "words" in ~0.438 min
    assert f["wpm"] == pytest.approx(24 / ((120 * 220 - 120) / 60000), rel=1e-6)


def test_typing_features_detects_pauses_and_corrections():
    f = typing_features(make_typing(pause_every=10, backspace_every=10))
    assert f["pause_rate"] == pytest.approx(100 * 11 / 120, rel=0.1)
    assert f["backspace_rate"] == pytest.approx(0.1)


def test_typing_features_rejects_tiny_samples_and_unknown_keys():
    with pytest.raises(ValueError):
        typing_features(make_typing(n=10))
    events = make_typing(n=30)
    for e in events:
        e["k"] = "x"
    with pytest.raises(ValueError):
        typing_features(events)


def test_pvt_features_normalise_to_per_minute():
    f = pvt_features([250, 300, 600, 700, 80], false_starts=1, duration_s=30)
    assert f["pvt_median"] == 450          # 80ms response is anticipatory, excluded
    assert f["pvt_lapses"] == 4            # 2 lapses in 30s
    assert f["pvt_false_starts"] == 2


def _baseline(n=6, seed=1):
    rng = random.Random(seed)
    return [typing_features(make_typing(jitter=10, seed=rng.randint(0, 999))) for _ in range(n)]


def test_score_calibrating_until_enough_baseline():
    feats = typing_features(make_typing(jitter=10))
    r = scoring.score_session(feats, [])
    assert r["readiness"] is None and r["label"] == "Calibrating"


def test_normal_session_scores_high_and_loaded_session_scores_low():
    base = _baseline()
    normal = typing_features(make_typing(jitter=10, seed=777))
    loaded = typing_features(make_typing(jitter=60, gap=250, pause_every=6, backspace_every=12, seed=5))
    rn = scoring.score_session(normal, base)
    rl = scoring.score_session(loaded, base)
    assert rn["readiness"] >= 80
    assert rl["readiness"] < 40
    assert rl["tips"], "a bad session should come with tips"
    assert {d["key"] for d in rl["details"] if d["status"] == "worse"} >= {"pause_rate", "wpm"}


def test_faster_than_baseline_is_not_penalised():
    base = _baseline()
    quicker = typing_features(make_typing(jitter=10, gap=80, seed=3))
    assert scoring.score_session(quicker, base)["readiness"] >= 90


def test_isolation_forest_scores_outlier_higher():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    forest = IsolationForest(n_trees=100, seed=0).fit(X)
    s_in = forest.anomaly_score(np.zeros((1, 4)))[0]
    s_out = forest.anomaly_score(np.full((1, 4), 6.0))[0]
    assert s_out > 0.6 > s_in


def test_seeded_demo_tells_the_sleep_story():
    from kp import db
    from kp.seed import seed_demo
    with db.connect() as con:
        db.init()
        uid = seed_demo(con)
        sessions = db.sessions_for(con, uid)
    scored = scoring.score_history(sessions)
    rated = [s for s in scored if s["readiness"] is not None]
    short = [s["readiness"] for s in rated if s["sleep_hours"] < 6]
    rested = [s["readiness"] for s in rated if s["sleep_hours"] >= 7]
    assert short and rested
    assert np.mean(rested) - np.mean(short) > 25
    r = scoring.pearson([s["sleep_hours"] for s in rated], [s["readiness"] for s in rated])
    assert r > 0.5


@pytest.fixture()
def client():
    from app import app
    return TestClient(app)


def test_api_end_to_end_judge_flow(client):
    uid = client.post("/api/users", json={"name": "Judge"}).json()["id"]

    # two calibration runs, then a third under cognitive load
    for seed in (1, 2):
        r = client.post(f"/api/users/{uid}/sessions",
                        json={"typing": make_typing(jitter=10, seed=seed)}).json()
    assert r["n_sessions"] == 2

    loaded = make_typing(jitter=60, gap=250, pause_every=6, backspace_every=12, seed=9)
    r = client.post(f"/api/users/{uid}/sessions", json={"typing": loaded}).json()
    assert r["readiness"] is not None and r["readiness"] < 50
    assert r["anomaly"] is None  # not enough history yet for the forest

    hist = client.get(f"/api/users/{uid}/history").json()
    # sessions 1-2 are calibration, session 3 is the first one scored
    assert [s["readiness"] is None for s in hist["sessions"]] == [True, True, False]


def test_api_validation(client):
    uid = client.post("/api/users", json={"name": "V"}).json()["id"]
    bad = [{"k": "q", "d": 0, "u": 10}]
    assert client.post(f"/api/users/{uid}/sessions", json={"typing": bad}).status_code == 422
    short = make_typing(n=5)
    assert client.post(f"/api/users/{uid}/sessions", json={"typing": short}).status_code == 422
    assert client.post("/api/users/99999/sessions", json={"typing": make_typing()}).status_code == 404


def test_api_demo_history_has_anomaly_model(client):
    uid = client.post("/api/demo/seed").json()["id"]
    hist = client.get(f"/api/users/{uid}/history").json()
    assert len(hist["sessions"]) > 25
    assert hist["latest"]["anomaly"] is not None
    assert hist["sleep_readiness_r"] > 0.5
    # re-seeding replaces rather than duplicates
    client.post("/api/demo/seed")
    assert sum(u["is_demo"] for u in client.get("/api/users").json()) == 1


def test_stateless_api_judge_flow(client):
    """The flow the deployed frontend uses: history lives in the browser, the server only scores."""
    history = []
    for seed in (1, 2):
        r = client.post("/api/score/checkin", json={"typing": make_typing(jitter=10, seed=seed), "history": history})
        assert r.status_code == 200
        history.append(r.json()["session"])
    assert r.json()["readiness"] is None  # still calibrating after 2

    loaded = make_typing(jitter=60, gap=250, pause_every=6, backspace_every=12, seed=9)
    r = client.post("/api/score/checkin", json={"typing": loaded, "history": history}).json()
    assert r["n_sessions"] == 3 and r["readiness"] < 50 and r["tip_keys"]
    history.append(r["session"])

    h = client.post("/api/score/history", json={"sessions": history}).json()
    assert [s["readiness"] is None for s in h["sessions"]] == [True, True, False]
    assert h["sessions"][-1]["z"] and h["latest"]["readiness"] == r["readiness"]


def test_stateless_demo_sessions(client):
    d = client.get("/api/demo/sessions").json()
    assert d["name"] and len(d["sessions"]) > 25
    h = client.post("/api/score/history", json={"sessions": d["sessions"]}).json()
    assert h["latest"]["anomaly"]["flagged"]


def test_demo_keeps_todays_short_night_even_early_morning(client):
    early = client.get("/api/demo/sessions", params={"now": "2026-11-15T06:00:00"}).json()["sessions"]
    assert early[-1]["sleep_hours"] == 4.0 and early[-1]["created_at"] < "2026-11-15T06:00:00"
    h = client.post("/api/score/history", json={"sessions": early}).json()
    assert h["latest"]["readiness"] < 65
    assert client.get("/api/demo/sessions", params={"now": "not-a-date"}).status_code == 422
