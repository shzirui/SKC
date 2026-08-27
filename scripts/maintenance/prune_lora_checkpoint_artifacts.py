#!/usr/bin/env python3
"""
Prune LoRA VERL checkpoint artifacts while preserving adapter exports.

For each experiment, this script always preserves actor/lora_adapter because it
is the artifact needed by rollout/vLLM together with the base model. It can
remove bulky actor/huggingface exports and old FSDP resume states.

Dry-run is the default. Pass --apply to actually delete files.
"""

import argparse
import os
import re
import shutil
from pathlib import Path


STEP_RE = re.compile(r"^global_step_(\d+)$")
RESUME_FILE_RE = re.compile(r"^(model|optim|extra_state)_world_size_\d+_rank_\d+\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preserve LoRA adapters and prune bulky HF exports / old resume states.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/checkpoints/verl_osworld_grpo_ppu"),
        help="Checkpoint project root containing experiment directories.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Only process one experiment directory name, e.g. DART-GUI-TRAIN_20260823_xxxxxxxx.",
    )
    parser.add_argument(
        "--keep-resume",
        type=int,
        default=1,
        help="Keep resume .pt files for the newest N global_step directories.",
    )
    parser.add_argument(
        "--keep-hf",
        type=int,
        default=0,
        help="Keep actor/huggingface for the newest N global_step directories. Use 0 after LoRA adapters are verified.",
    )
    parser.add_argument(
        "--require-lora-adapter",
        action="store_true",
        help="Skip a step if actor/lora_adapter is missing.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this, only print actions.")
    return parser.parse_args()


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def disk_usage(path):
    if not path.exists():
        return 0
    total = 0
    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_blocks * 512
        except OSError:
            return 0
    for root, _dirs, files in os.walk(path):
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


def removable_entries(step_dir, keep_hf_step, keep_resume_step):
    entries = []
    actor_dir = step_dir / "actor"
    hf_dir = actor_dir / "huggingface"
    lora_dir = actor_dir / "lora_adapter"

    if hf_dir.exists() and not keep_hf_step:
        entries.append(hf_dir)

    if actor_dir.is_dir() and not keep_resume_step:
        for child in actor_dir.iterdir():
            if child == hf_dir or child == lora_dir:
                continue
            if child.is_file() and RESUME_FILE_RE.match(child.name):
                entries.append(child)

    return entries


def process_experiment(exp_dir, keep_resume, keep_hf, apply, require_lora_adapter):
    step_dirs = []
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        step = parse_step_dir(child)
        if step is not None:
            step_dirs.append((step, child))
    step_dirs.sort(key=lambda item: item[0])
    if not step_dirs:
        print(f"[SKIP] {exp_dir.name}: no global_step_* directories")
        return 0

    keep_resume_steps = {step for step, _ in step_dirs[-keep_resume:]} if keep_resume > 0 else set()
    keep_hf_steps = {step for step, _ in step_dirs[-keep_hf:]} if keep_hf > 0 else set()
    print(
        f"[EXPERIMENT] {exp_dir.name}: total={len(step_dirs)}, "
        f"keep_resume={sorted(keep_resume_steps)}, keep_hf={sorted(keep_hf_steps)}"
    )

    total_reclaimable = 0
    failed_deletes = 0
    for step, step_dir in step_dirs:
        lora_dir = step_dir / "actor" / "lora_adapter"
        if require_lora_adapter and not lora_dir.is_dir():
            print(f"  [SKIP] global_step_{step}: no actor/lora_adapter")
            continue

        entries = removable_entries(
            step_dir,
            keep_hf_step=step in keep_hf_steps,
            keep_resume_step=step in keep_resume_steps,
        )
        if not entries:
            print(f"  [KEEP] global_step_{step}: nothing to remove")
            continue

        reclaimable = sum(disk_usage(path) for path in entries if path.exists())
        total_reclaimable += reclaimable
        action = "DELETE" if apply else "DRY-RUN"
        print(f"  [{action}] global_step_{step}: remove {len(entries)} entries, reclaim ~{format_bytes(reclaimable)}")
        for path in entries:
            print(f"    - {path.relative_to(exp_dir)}")

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
    if args.keep_resume < 0:
        raise ValueError("--keep-resume must be non-negative.")
    if args.keep_hf < 0:
        raise ValueError("--keep-hf must be non-negative.")
    if not args.root.is_dir():
        raise ValueError(f"Root directory does not exist: {args.root}")

    total = 0
    for exp_dir in iter_experiment_dirs(args.root, args.experiment):
        total += process_experiment(
            exp_dir=exp_dir,
            keep_resume=args.keep_resume,
            keep_hf=args.keep_hf,
            apply=args.apply,
            require_lora_adapter=args.require_lora_adapter,
        )

    print(f"[TOTAL] reclaimable ~{format_bytes(total)}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
