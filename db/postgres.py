from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None


@dataclass(frozen=True)
class RunContext:
    conn: "psycopg.Connection"
    run_id: int


def get_db_url(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")


def open_connection(db_url: str) -> Optional["psycopg.Connection"]:
    if not db_url:
        return None
    if psycopg is None:
        print("psycopg is not installed; skipping Postgres logging.")
        return None
    conn = psycopg.connect(db_url)
    conn.autocommit = True
    return conn


def ensure_schema(conn: "psycopg.Connection") -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                run_name TEXT,
                notes TEXT,
                params_json TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_metrics (
                id SERIAL PRIMARY KEY,
                run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
                phase TEXT NOT NULL,
                episode INTEGER NOT NULL,
                score INTEGER,
                reward REAL,
                moves INTEGER,
                max_tile INTEGER,
                epsilon REAL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_metrics_run_episode
            ON training_metrics(run_id, episode);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_training_metrics_run_phase
            ON training_metrics(run_id, phase);
            """
        )


def create_run(
    conn: "psycopg.Connection",
    *,
    model_name: str,
    run_name: str = "",
    notes: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> int:
    params_json = json.dumps(params or {}, sort_keys=True)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO training_runs (model_name, run_name, notes, params_json)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (model_name, run_name or None, notes or None, params_json),
        )
        return int(cur.fetchone()[0])


def init_run(
    *,
    db_url: str | None,
    model_name: str,
    run_name: str = "",
    notes: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[RunContext]:
    if not db_url:
        return None
    conn = open_connection(db_url)
    if conn is None:
        return None
    ensure_schema(conn)
    run_id = create_run(
        conn,
        model_name=model_name,
        run_name=run_name,
        notes=notes,
        params=params,
    )
    return RunContext(conn=conn, run_id=run_id)


def log_metrics(
    ctx: Optional[RunContext],
    *,
    phase: str,
    episode: int,
    score: int,
    reward: float,
    moves: int,
    max_tile: int,
    epsilon: float | None,
) -> None:
    if ctx is None:
        return
    with ctx.conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO training_metrics (
                run_id, phase, episode, score, reward, moves, max_tile, epsilon
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                ctx.run_id,
                phase,
                episode,
                score,
                reward,
                moves,
                max_tile,
                epsilon,
            ),
        )


def close(ctx: Optional[RunContext]) -> None:
    if ctx is None:
        return
    ctx.conn.close()
