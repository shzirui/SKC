#!/usr/bin/env python3
"""
Prune VERL checkpoints while preserving Hugging Face exports.

For each experiment directory under the root, keep the newest N
`global_step_*` checkpoints intact. For older checkpoints, remove everything
inside the checkpoint directory except `actor/huggingface`.

Dry-run is the default. Pass --apply to actually delete files.
"""

import argparse
import os
import re
import shutil
from pathlib import Path


STEP_RE = re.compile(r"^global_step_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep latest full VERL checkpoints and preserve only actor/huggingface for older ones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/checkpoints/verl_osworld_grpo_ppu"),
        help="Checkpoint project root containing experiment directories.",
    )
    parser.add_argument("--keep", type=int, default=2, help="Number of newest checkpoints to keep intact per experiment.")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Only process one experiment directory name, e.g. DART-GUI-TRAIN_20260531_aey2zg45.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this, only print actions.")
    parser.add_argument("--skip-without-hf", action="store_true", help="Skip old checkpoints that do not have actor/huggingface.")
    return parser.parse_args()


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def disk_usage(path):
    total = 0
    if not path.exists():
        return 0
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        try:
            total += root_path.lstat().st_blocks * 512
        except OSError:
            pass
        for name in files:
            try:
                total += (root_path / name).lstat().st_blocks * 512
            except OSError:
                pass
    return total


def removable_entries(step_dir):
    hf_dir = step_dir / "actor" / "huggingface"
    entries = []
    for child in step_dir.iterdir():
        if child == step_dir / "actor":
            for actor_child in child.iterdir():
                if actor_child == hf_dir:
                    continue
                entries.append(actor_child)
        else:
            entries.append(child)
    return entries


def delete_path(path):
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except PermissionError as exc:
        print(f"    [ERROR] permission denied: {path} ({exc})")
        return False
    except OSError as exc:
        print(f"    [ERROR] failed to delete: {path} ({exc})")
        return False


def parse_step_dir(path):
    match = STEP_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def iter_experiment_dirs(root, experiment):
    if experiment:
        exp_dir = root / experiment
        if exp_dir.is_dir():
            yield exp_dir
        return
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def process_experiment(exp_dir, keep, apply, skip_without_hf):
    step_dirs = []
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        step = parse_step_dir(child)
        if step is not None:
            step_dirs.append((step, child))
    step_dirs.sort(key=lambda item: item[0])
    if len(step_dirs) <= keep:
        print(f"[SKIP] {exp_dir.name}: {len(step_dirs)} checkpoints <= keep={keep}")
        return 0

    old_steps = step_dirs[: -keep]
    kept_steps = step_dirs[-keep:]
    print(
        f"[EXPERIMENT] {exp_dir.name}: total={len(step_dirs)}, "
        f"old={len(old_steps)}, keep_full={[step for step, _ in kept_steps]}"
    )

    total_reclaimable = 0
    failed_deletes = 0
    for step, step_dir in old_steps:
        hf_dir = step_dir / "actor" / "huggingface"
        if skip_without_hf and not hf_dir.is_dir():
            print(f"  [SKIP] global_step_{step}: no actor/huggingface")
            continue

        entries = removable_entries(step_dir)
        reclaimable = sum(disk_usage(path) if path.is_dir() else path.lstat().st_blocks * 512 for path in entries if path.exists())
        total_reclaimable += reclaimable
        action = "DELETE" if apply else "DRY-RUN"
        print(f"  [{action}] global_step_{step}: remove {len(entries)} entries, reclaim ~{format_bytes(reclaimable)}")

        if apply:
            for path in entries:
                if path.exists() and not delete_path(path):
                    failed_deletes += 1

    print(f"[SUMMARY] {exp_dir.name}: reclaimable ~{format_bytes(total_reclaimable)}")
    if failed_deletes:
        print(f"[SUMMARY] {exp_dir.name}: failed deletes={failed_deletes}")
    return total_reclaimable


def main():
    args = parse_args()
    if args.keep < 0:
        raise ValueError("--keep must be non-negative.")
    if not args.root.is_dir():
        raise ValueError(f"Root directory does not exist: {args.root}")

    total = 0
    for exp_dir in iter_experiment_dirs(args.root, args.experiment):
        total += process_experiment(exp_dir, args.keep, args.apply, args.skip_without_hf)

    print(f"[TOTAL] reclaimable ~{format_bytes(total)}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()

