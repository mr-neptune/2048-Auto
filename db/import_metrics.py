from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from db import postgres as db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import CSV metrics into Postgres.")
    parser.add_argument("--db-url", type=str, default="", help="Postgres URL; defaults to POSTGRES_URL/DATABASE_URL")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to store with the run")
    parser.add_argument("--phase", type=str, default="train", help="Metric phase label (train/eval)")
    parser.add_argument("--run-name", type=str, default="", help="Run name stored in Postgres")
    parser.add_argument("--notes", type=str, default="", help="Optional run notes stored in Postgres")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to a CSV metrics file; repeat to import multiple files in order",
    )
    parser.add_argument(
        "--run-per-file",
        action="store_true",
        help="Create a separate run for each CSV file instead of one combined run",
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="Offset added to episode numbers (useful when combining runs)",
    )
    parser.add_argument(
        "--auto-offset",
        action="store_true",
        help="Increase episode offset by max episode in each file to keep episodes contiguous",
    )
    return parser.parse_args()


def iter_csv_rows(path: Path) -> Iterable[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def import_file(
    path: Path,
    *,
    db_ctx: db.RunContext,
    phase: str,
    episode_offset: int,
) -> int:
    max_episode = 0
    for row in iter_csv_rows(path):
        episode = int(row["episode"]) + episode_offset
        score = int(float(row["score"]))
        reward = float(row["reward"])
        moves = int(float(row["moves"]))
        max_tile = int(float(row["max_tile"]))
        epsilon = float(row.get("epsilon") or 0.0)
        db.log_metrics(
            db_ctx,
            phase=phase,
            episode=episode,
            score=score,
            reward=reward,
            moves=moves,
            max_tile=max_tile,
            epsilon=epsilon,
        )
        if episode > max_episode:
            max_episode = episode
    return max_episode


def main() -> None:
    args = parse_args()
    db_url = db.get_db_url(args.db_url)
    if not db_url:
        raise SystemExit("No database URL provided. Set POSTGRES_URL or pass --db-url.")

    episode_offset = args.episode_offset
    csv_paths = [Path(p) for p in args.csv]

    if args.run_per_file:
        for path in csv_paths:
            db_ctx = db.init_run(
                db_url=db_url,
                model_name=args.model_name,
                run_name=(args.run_name or path.stem),
                notes=args.notes,
                params={"source_csv": str(path), "phase": args.phase},
            )
            if db_ctx is None:
                raise SystemExit("Could not initialize Postgres connection.")
            max_episode = import_file(path, db_ctx=db_ctx, phase=args.phase, episode_offset=episode_offset)
            db.close(db_ctx)
            if args.auto_offset:
                episode_offset = max_episode
    else:
        db_ctx = db.init_run(
            db_url=db_url,
            model_name=args.model_name,
            run_name=args.run_name,
            notes=args.notes,
            params={"source_csv": [str(p) for p in csv_paths], "phase": args.phase},
        )
        if db_ctx is None:
            raise SystemExit("Could not initialize Postgres connection.")
        for path in csv_paths:
            max_episode = import_file(path, db_ctx=db_ctx, phase=args.phase, episode_offset=episode_offset)
            if args.auto_offset:
                episode_offset = max_episode
        db.close(db_ctx)


if __name__ == "__main__":
    main()
