from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# on Vercel the code directory is read-only; /tmp is the only writable place
DB_PATH = os.environ.get("KEYPULSE_DB") or (
    "/tmp/keypulse.db" if os.environ.get("VERCEL") else str(Path(__file__).resolve().parent.parent / "keypulse.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  is_demo INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  features TEXT NOT NULL,
  sleep_hours REAL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, created_at);
"""


@contextmanager
def connect(path: str | None = None):
    con = sqlite3.connect(path or DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init(path: str | None = None):
    with connect(path) as con:
        con.executescript(SCHEMA)


def list_users(con):
    return [dict(r) for r in con.execute(
        "SELECT u.id, u.name, u.is_demo, COUNT(s.id) AS n_sessions FROM users u "
        "LEFT JOIN sessions s ON s.user_id = u.id GROUP BY u.id ORDER BY u.id")]


def get_user(con, user_id: int):
    r = con.execute("SELECT id, name, is_demo FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(r) if r else None


def create_user(con, name: str, is_demo: bool = False) -> int:
    return con.execute("INSERT INTO users (name, is_demo) VALUES (?, ?)", (name, int(is_demo))).lastrowid


def delete_user(con, user_id: int):
    con.execute("DELETE FROM users WHERE id = ?", (user_id,))


def add_session(con, user_id: int, created_at: str, features: dict, sleep_hours: float | None) -> int:
    return con.execute(
        "INSERT INTO sessions (user_id, created_at, features, sleep_hours) VALUES (?, ?, ?, ?)",
        (user_id, created_at, json.dumps(features), sleep_hours)).lastrowid


def sessions_for(con, user_id: int) -> list[dict]:
    rows = con.execute("SELECT id, created_at, features, sleep_hours FROM sessions "
                       "WHERE user_id = ? ORDER BY created_at, id", (user_id,))
    return [dict(id=r["id"], created_at=r["created_at"], features=json.loads(r["features"]),
                 sleep_hours=r["sleep_hours"]) for r in rows]
