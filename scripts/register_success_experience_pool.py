#!/usr/bin/env python3
"""
Register successful trajectories from one rollout run as fallback experience
rows for another run_id, without changing training code.

The inserted rows use a sentinel model_version such as
experience://test_plan_o3. trainable_filter.py will not select them during
normal top-checkpoint sampling, but it can still find them as historical
positive examples for zero-success tasks under the same run_id.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_RUN_ID = "results/test_plan_o3"
DEFAULT_SOURCE_ROOT = "/path/to/dart-gui/dart_rollouter/results_remote/test_plan_o3"
DEFAULT_EXPERIENCE_MODEL_VERSION = "experience://test_plan_o3"


@dataclass
class MysqlConfig:
    mode: str
    container: str
    database: str
    user: str
    password: str
    host: str
    port: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy/symlink successful trajectories and insert them into rollout_run "
            "as fallback experience rows for a target run_id."
        )
    )
    parser.add_argument("--target-run-id", required=True, help="Target training run_id to receive fallback rows.")
    parser.add_argument(
        "--target-root-data-dir",
        required=True,
        help="root_data_dir used by the target training dataset config.",
    )
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--source-root-data-dir", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--experience-model-version", default=DEFAULT_EXPERIENCE_MODEL_VERSION)
    parser.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="Select source rows with reward > this value. Default: 0.0.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of source rows to register.")
    parser.add_argument("--copy-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument(
        "--require-files",
        nargs="*",
        default=["final_messages.json", "reward.txt"],
        help="Files that must exist under each source trajectory directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create paths and insert SQL rows. Without this flag, only dry-run.",
    )

    parser.add_argument("--mysql-mode", choices=("auto", "docker", "local"), default="auto")
    parser.add_argument("--mysql-container", default=os.environ.get("MYSQL_CONTAINER", "mysql-server"))
    parser.add_argument("--db-database", default=os.environ.get("DB_DATABASE", "dart"))
    parser.add_argument("--db-user", default=os.environ.get("DB_USER", "root"))
    parser.add_argument("--db-password", default=os.environ.get("DB_PASSWORD", "${DB_PASSWORD}"))
    parser.add_argument("--db-host", default=os.environ.get("DB_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("DB_PORT", "3306")))
    parser.add_argument("--chunk-size", type=int, default=200, help="SQL insert statements per transaction.")
    return parser.parse_args()


def resolve_mysql_mode(args: argparse.Namespace) -> str:
    if args.mysql_mode != "auto":
        return args.mysql_mode
    if shutil.which("docker"):
        probe = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", args.mysql_container],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            return "docker"
    return "local"


def mysql_cmd(conf: MysqlConfig, batch: bool) -> list[str]:
    if conf.mode == "docker":
        cmd = [
            "docker",
            "exec",
            "-i",
        ]
        if conf.password:
            cmd += ["-e", f"MYSQL_PWD={conf.password}"]
        cmd += [
            conf.container,
            "mysql",
            "-u",
            conf.user,
        ]
    else:
        cmd = [
            "mysql",
            "-h",
            conf.host,
            "-P",
            str(conf.port),
            "-u",
            conf.user,
        ]

    if batch:
        cmd += ["--batch", "--raw", "--skip-column-names"]
    cmd.append(conf.database)
    return cmd


def run_mysql(conf: MysqlConfig, sql: str, batch: bool = False) -> str:
    env = os.environ.copy()
    if conf.mode == "local" and conf.password:
        env["MYSQL_PWD"] = conf.password
    proc = subprocess.run(
        mysql_cmd(conf, batch=batch),
        input=sql,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "mysql command failed\n"
            f"mode={conf.mode}, database={conf.database}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    return proc.stdout


def sql_string(value: object) -> str:
    if value is None:
        return "NULL"
    text = str(value)
    text = text.replace("\\", "\\\\").replace("'", "''")
    return f"'{text}'"


def sql_float(value: object) -> str:
    if value is None:
        return "NULL"
    return repr(float(value))


def sql_int(value: object) -> str:
    if value is None:
        return "NULL"
    return str(int(value))


def process_reward_sql(value: object) -> str:
    if value is None or value == "":
        return "NULL"
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return sql_string(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))


def fetch_source_rows(conf: MysqlConfig, args: argparse.Namespace) -> list[dict]:
    limit_sql = f"\nLIMIT {int(args.limit)}" if args.limit and args.limit > 0 else ""
    query = f"""
SELECT JSON_OBJECT(
  'trajectory_id', trajectory_id,
  'task_id', task_id,
  'trace_id', trace_id,
  'split_dir', split_dir,
  'reward', reward,
  'process_reward', CAST(process_reward AS CHAR),
  'num_chunks', num_chunks,
  'instruction', instruction
)
FROM rollout_run
WHERE run_id = {sql_string(args.source_run_id)}
  AND reward > {sql_float(args.min_reward)}
ORDER BY task_id, create_at, id{limit_sql};
"""
    output = run_mysql(conf, query, batch=True)
    rows = []
    for line in output.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def count_target_experience_rows(conf: MysqlConfig, target_run_id: str, model_version: str) -> int:
    query = f"""
SELECT COUNT(*)
FROM rollout_run
WHERE run_id = {sql_string(target_run_id)}
  AND model_version = {sql_string(model_version)};
"""
    output = run_mysql(conf, query, batch=True).strip()
    return int(output or "0")


def required_files_exist(path: Path, filenames: Iterable[str]) -> bool:
    return all((path / filename).is_file() for filename in filenames)


def prepare_trajectory_path(
    row: dict,
    source_root: Path,
    target_root: Path,
    copy_mode: str,
    required_files: list[str],
    apply: bool,
) -> tuple[bool, str]:
    trajectory_id = row["trajectory_id"]
    source_path = source_root / trajectory_id
    target_path = target_root / trajectory_id

    if not source_path.is_dir():
        return False, "missing_source_dir"
    if not required_files_exist(source_path, required_files):
        return False, "missing_required_files"

    if target_path.exists() or target_path.is_symlink():
        try:
            if target_path.resolve() == source_path.resolve():
                return True, "already_linked"
        except OSError:
            return False, "broken_existing_path"
        if copy_mode == "copy" and target_path.is_dir() and required_files_exist(target_path, required_files):
            return True, "already_present"
        return False, "target_path_conflict"

    if apply:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode == "symlink":
            target_path.symlink_to(source_path.resolve(), target_is_directory=True)
        else:
            shutil.copytree(source_path, target_path)
    return True, "would_create" if not apply else "created"


def insert_statement(row: dict, args: argparse.Namespace, source_root: Path) -> str:
    split_dir = row.get("split_dir") or str(source_root)
    process_reward = process_reward_sql(row.get("process_reward"))
    return f"""
INSERT INTO rollout_run
  (run_id, trajectory_id, task_id, trace_id, split_dir, reward, process_reward,
   num_chunks, used, model_version, instruction, create_at)
VALUES
  ({sql_string(args.target_run_id)},
   {sql_string(row.get("trajectory_id"))},
   {sql_string(row.get("task_id"))},
   {sql_string(row.get("trace_id"))},
   {sql_string(split_dir)},
   {sql_float(row.get("reward"))},
   {process_reward},
   {sql_int(row.get("num_chunks"))},
   0,
   {sql_string(args.experience_model_version)},
   {sql_string(row.get("instruction"))},
   NOW())
ON DUPLICATE KEY UPDATE
  task_id = IF(rollout_run.model_version = VALUES(model_version), VALUES(task_id), rollout_run.task_id),
  trace_id = IF(rollout_run.model_version = VALUES(model_version), VALUES(trace_id), rollout_run.trace_id),
  split_dir = IF(rollout_run.model_version = VALUES(model_version), VALUES(split_dir), rollout_run.split_dir),
  reward = IF(rollout_run.model_version = VALUES(model_version), VALUES(reward), rollout_run.reward),
  process_reward = IF(rollout_run.model_version = VALUES(model_version), VALUES(process_reward), rollout_run.process_reward),
  num_chunks = IF(rollout_run.model_version = VALUES(model_version), VALUES(num_chunks), rollout_run.num_chunks),
  used = IF(rollout_run.model_version = VALUES(model_version), 0, rollout_run.used),
  instruction = IF(rollout_run.model_version = VALUES(model_version), VALUES(instruction), rollout_run.instruction),
  create_at = IF(rollout_run.model_version = VALUES(model_version), NOW(), rollout_run.create_at);
""".strip()


def chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    args = parse_args()
    mysql_mode = resolve_mysql_mode(args)
    conf = MysqlConfig(
        mode=mysql_mode,
        container=args.mysql_container,
        database=args.db_database,
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port,
    )

    source_root = Path(args.source_root_data_dir).expanduser().resolve()
    target_root = Path(args.target_root_data_dir).expanduser()

    rows = fetch_source_rows(conf, args)
    before_count = count_target_experience_rows(conf, args.target_run_id, args.experience_model_version)

    prepared_rows = []
    status_counts: dict[str, int] = {}
    for row in rows:
        ok, status = prepare_trajectory_path(
            row=row,
            source_root=source_root,
            target_root=target_root,
            copy_mode=args.copy_mode,
            required_files=args.require_files,
            apply=args.apply,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if ok:
            prepared_rows.append(row)

    if args.apply and prepared_rows:
        for batch in chunks(prepared_rows, max(1, args.chunk_size)):
            sql = "START TRANSACTION;\n"
            sql += "\n".join(insert_statement(row, args, source_root) for row in batch)
            sql += "\nCOMMIT;\n"
            run_mysql(conf, sql, batch=False)

    after_count = count_target_experience_rows(conf, args.target_run_id, args.experience_model_version)

    mode_label = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode_label}] mysql_mode={conf.mode} database={conf.database}")
    print(f"source_run_id={args.source_run_id}")
    print(f"target_run_id={args.target_run_id}")
    print(f"experience_model_version={args.experience_model_version}")
    print(f"source_success_rows={len(rows)}")
    print(f"prepared_rows={len(prepared_rows)}")
    print(f"target_experience_rows_before={before_count}")
    print(f"target_experience_rows_after={after_count}")
    print("path_status_counts=" + json.dumps(status_counts, ensure_ascii=False, sort_keys=True))
    if not args.apply:
        print("No changes were made. Re-run with --apply to create paths and insert SQL rows.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
