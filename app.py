"""KeyPulse API + static frontend.  Run:  .venv/Scripts/python -m uvicorn app:app --reload"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kp import db, scoring
from kp.features import pvt_features, typing_features
from kp.seed import DEMO_NAME, demo_sessions, seed_demo

app = FastAPI(title="KeyPulse")
db.init()


class KeyEvent(BaseModel):
    k: str = Field(pattern="^[csb]$")   # category only — never the character
    d: float
    u: float


class PvtResult(BaseModel):
    rts: list[float]
    false_starts: int = Field(ge=0)
    duration_s: float = Field(gt=0)


class SessionIn(BaseModel):
    typing: list[KeyEvent]
    pvt: PvtResult | None = None
    sleep_hours: float | None = Field(default=None, ge=0, le=16)


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)


def _require_user(con, user_id: int):
    user = db.get_user(con, user_id)
    if not user:
        raise HTTPException(404, "user not found")
    return user


@app.get("/api/users")
def users():
    with db.connect() as con:
        return db.list_users(con)


@app.post("/api/users")
def create_user(body: UserIn):
    with db.connect() as con:
        return {"id": db.create_user(con, body.name.strip())}


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int):
    with db.connect() as con:
        _require_user(con, user_id)
        db.delete_user(con, user_id)
    return {"ok": True}


@app.post("/api/demo/seed")
def seed():
    with db.connect() as con:
        return {"id": seed_demo(con)}


@app.post("/api/users/{user_id}/sessions")
def add_session(user_id: int, body: SessionIn):
    try:
        feats = typing_features([e.model_dump() for e in body.typing])
        if body.pvt:
            feats.update(pvt_features(body.pvt.rts, body.pvt.false_starts, body.pvt.duration_s))
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    with db.connect() as con:
        _require_user(con, user_id)
        sid = db.add_session(con, user_id, datetime.now().isoformat(timespec="seconds"),
                             feats, body.sleep_hours)
        sessions = db.sessions_for(con, user_id)
    result = scoring.score_latest(sessions)
    return {"session_id": sid, "features": feats, "n_sessions": len(sessions), **result}


@app.get("/api/users/{user_id}/history")
def history(user_id: int):
    with db.connect() as con:
        user = _require_user(con, user_id)
        sessions = db.sessions_for(con, user_id)
    scored = scoring.score_history(sessions)
    pairs = [(s["sleep_hours"], s["readiness"]) for s in scored
             if s["sleep_hours"] is not None and s["readiness"] is not None]
    latest = scoring.score_latest(sessions) if sessions else None
    return {
        "user": user,
        "sessions": scored,
        "features": {k: {"label": v["label"], "unit": v["unit"], "dir": v["dir"]}
                     for k, v in scoring.FEATURES.items()},
        "sleep_readiness_r": scoring.pearson([p[0] for p in pairs], [p[1] for p in pairs]),
        "latest": latest,
    }


# ---------------------------------------------------------------- stateless API (used by the frontend)
# Profiles and history live in the browser; the server only extracts features and scores.
# This is what makes serverless hosting (Vercel) work: no server-side state to lose.

class StoredSession(BaseModel):
    created_at: str
    features: dict[str, float]
    sleep_hours: float | None = None


class CheckinIn(SessionIn):
    history: list[StoredSession] = Field(default_factory=list, max_length=500)


class HistoryIn(BaseModel):
    sessions: list[StoredSession] = Field(max_length=500)


FEATURE_META = {k: {"label": v["label"], "unit": v["unit"], "dir": v["dir"]} for k, v in scoring.FEATURES.items()}


def _extract(body: SessionIn) -> dict:
    try:
        feats = typing_features([e.model_dump() for e in body.typing])
        if body.pvt:
            feats.update(pvt_features(body.pvt.rts, body.pvt.false_starts, body.pvt.duration_s))
        return feats
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/score/checkin")
def score_checkin(body: CheckinIn):
    feats = _extract(body)
    new = {"created_at": datetime.now().isoformat(timespec="seconds"), "features": feats,
           "sleep_hours": body.sleep_hours}
    sessions = [h.model_dump() for h in body.history] + [new]
    return {"session": new, "features": feats, "n_sessions": len(sessions), **scoring.score_latest(sessions)}


@app.post("/api/score/history")
def score_history_stateless(body: HistoryIn):
    sessions = [s.model_dump() for s in body.sessions]
    return {
        "sessions": scoring.score_history(sessions),
        "features": FEATURE_META,
        "latest": scoring.score_latest(sessions) if sessions else None,
    }


@app.get("/api/demo/sessions")
def demo(now: str | None = None):
    """`now` = the browser's local time (naive ISO), so the demo lines up with the viewer's clock, not the server's UTC."""
    try:
        anchor = datetime.fromisoformat(now).replace(tzinfo=None) if now else None
    except ValueError:
        raise HTTPException(422, "now must be an ISO datetime")
    return {"name": DEMO_NAME, "sessions": demo_sessions(anchor)}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():  # locally FastAPI serves the frontend; on Vercel the CDN does
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
