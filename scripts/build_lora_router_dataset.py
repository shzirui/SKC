#!/usr/bin/env python3
"""Export router training data from MySQL rollout_run rows.

The exported examples are kept intentionally small and stable:
- instruction: model input
- app_name: label alias of run_id
- task_id: auxiliary task identifier

Outputs:
- <output_dir>/router_train.jsonl
- <output_dir>/router_val.jsonl
- <output_dir>/router_meta.json
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[1]
DART_ROLLOUTER_SRC = REPO_ROOT / "dart_rollouter" / "src"
if str(DART_ROLLOUTER_SRC) not in sys.path:
    sys.path.insert(0, str(DART_ROLLOUTER_SRC))

from services.mysql_rollout import MySQLRolloutORM, RolloutRun  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export router dataset from rollout_run table.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "router_data",
        help="Directory to write router_train.jsonl and router_val.jsonl.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=None,
        help="Optional run_ids to export. Defaults to all distinct run_ids in SQL.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Per-label validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    parser.add_argument(
        "--min-instruction-len",
        type=int,
        default=1,
        help="Skip rows whose instruction is shorter than this number of non-space chars.",
    )
    return parser.parse_args()


def get_db_config_from_env():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "username": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_DATABASE", ""),
        "port": int(os.getenv("DB_PORT", "3306")),
        "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    }


def discover_run_ids(orm):
    with orm.session_scope() as session:
        rows = session.execute(
            select(RolloutRun.run_id).distinct().order_by(RolloutRun.run_id)
        ).all()
    return [row[0] for row in rows if row and row[0]]


def fetch_rows(orm, run_ids):
    with orm.session_scope() as session:
        stmt = select(RolloutRun)
        if run_ids:
            stmt = stmt.where(RolloutRun.run_id.in_(list(run_ids)))
        stmt = stmt.order_by(RolloutRun.run_id, RolloutRun.task_id, RolloutRun.trajectory_id)
        rows = session.execute(stmt).scalars().all()
        return [row.to_dict() for row in rows]


def normalize_record(row, min_instruction_len):
    instruction = (row.get("instruction") or "").strip()
    if len(instruction) < min_instruction_len:
        return None

    run_id = str(row.get("run_id") or "").strip()
    task_id = str(row.get("task_id") or "").strip()
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    trace_id = str(row.get("trace_id") or "").strip()

    if not run_id or not task_id or not trajectory_id:
        return None

    return {
        "instruction": instruction,
        "app_name": run_id,
        "label": run_id,
        "run_id": run_id,
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "trace_id": trace_id,
        "reward": row.get("reward"),
        "used": row.get("used"),
        "model_version": row.get("model_version"),
        "num_chunks": row.get("num_chunks"),
    }


def split_by_label(rows, val_ratio, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    train = []
    val = []
    for label in sorted(grouped):
        items = grouped[label][:]
        rng.shuffle(items)
        if len(items) <= 1 or val_ratio <= 0:
            train.extend(items)
            continue
        n_val = int(round(len(items) * val_ratio))
        n_val = max(1, min(len(items) - 1, n_val))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path, rows):
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    orm = MySQLRolloutORM(config=get_db_config_from_env(), create_tables_if_missing=False)
    run_ids = list(args.run_ids) if args.run_ids else discover_run_ids(orm)
    if not run_ids:
        raise RuntimeError("No run_ids found in SQL.")

    raw_rows = fetch_rows(orm, run_ids)
    normalized = []
    seen = set()
    skipped = 0
    for row in raw_rows:
        item = normalize_record(row, args.min_instruction_len)
        if item is None:
            skipped += 1
            continue
        key = (item["run_id"], item["trajectory_id"])
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        normalized.append(item)

    train_rows, val_rows = split_by_label(normalized, args.val_ratio, args.seed)
    train_path = args.output_dir / "router_train.jsonl"
    val_path = args.output_dir / "router_val.jsonl"
    meta_path = args.output_dir / "router_meta.json"

    train_count = write_jsonl(train_path, train_rows)
    val_count = write_jsonl(val_path, val_rows)

    label_counts = {}
    for row in normalized:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    meta = {
        "run_ids": run_ids,
        "total_rows": len(normalized),
        "train_rows": train_count,
        "val_rows": val_count,
        "skipped_rows": skipped,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "label_counts": label_counts,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "train_path": str(train_path),
        "val_path": str(val_path),
        "meta_path": str(meta_path),
        "total_rows": len(normalized),
        "train_rows": train_count,
        "val_rows": val_count,
        "skipped_rows": skipped,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
