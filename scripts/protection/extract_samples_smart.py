#!/usr/bin/env python3
"""Extract successful task trajectories from evaluation results."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

DEFAULT_UUID_CATEGORY_JSON = ""


def load_uuid_category_map(json_path):
    """Load a UUID-to-category mapping from an evaluation JSON file."""
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {uid: cat for cat, uids in data.items() for uid in uids}


def load_task_ids(task_id_file: str) -> Set[str]:
    """Load task IDs from a JSON object whose values are ID lists."""
    task_ids = set()

    with open(task_id_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key, ids in data.items():
            if isinstance(ids, list):
                task_ids.update(ids)

    return task_ids


def find_successful_tasks_in_results(
    results_dir: str,
    target_task_ids: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """Find the highest-reward successful trace for each target task."""
    results_path = Path(results_dir)
    successful_tasks = {}

    if not results_path.exists():
        print("Warning: results directory not found", file=sys.stderr)
        return successful_tasks

    print("  Scanning configured results directory")

    for task_dir in results_path.iterdir():
        if not task_dir.is_dir():
            continue

        dir_name = task_dir.name
        parts = dir_name.split('_trace-')
        if len(parts) < 2:
            continue

        task_id = parts[0]
        trace_id = parts[1]

        if task_id not in target_task_ids:
            continue

        reward_file = task_dir / "reward.txt"
        if not reward_file.exists():
            continue

        try:
            with open(reward_file, 'r', encoding='utf-8') as f:
                reward = float(f.read().strip())
        except Exception:
            print("  Warning: Failed to read a reward file", file=sys.stderr)
            continue

        if reward <= 0:
            continue

        if task_id in successful_tasks:
            if reward > successful_tasks[task_id]['reward']:
                successful_tasks[task_id] = {
                    'task_dir': str(task_dir),
                    'reward': reward,
                    'trace_id': trace_id
                }
                print("    Found a higher-reward successful trace")
        else:
            successful_tasks[task_id] = {
                'task_dir': str(task_dir),
                'reward': reward,
                'trace_id': trace_id
            }
            print("    Found a successful trace")

    return successful_tasks


def extract_trajectory_from_final_messages(
    task_dir: str,
    task_id: str
) -> List[Dict[str, Any]]:
    """Extract assistant trajectory steps from a final-messages file."""
    final_messages_file = Path(task_dir) / "final_messages.json"

    if not final_messages_file.exists():
        print("  Warning: final_messages.json not found", file=sys.stderr)
        return []

    try:
        with open(final_messages_file, 'r', encoding='utf-8') as f:
            messages = json.load(f)
    except Exception:
        print("  Warning: Failed to load final_messages.json", file=sys.stderr)
        return []

    trajectory = []
    step_num = 0

    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system':
            continue

        if role == 'assistant':
            step_num += 1

            if isinstance(content, list):
                text_content = ''
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_content = item.get('text', '')
                        break
            else:
                text_content = str(content)

            screenshot = ''
            if i > 0 and messages[i-1].get('role') == 'user':
                user_content = messages[i-1].get('content', [])
                if isinstance(user_content, list):
                    for item in user_content:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            screenshot = item.get('image_url', '')
                            break

            trajectory.append({
                'step_num': step_num,
                'role': role,
                'content': text_content,
                'screenshot': screenshot,
                'message_index': i
            })

    return trajectory


def export_samples(
    successful_tasks: Dict[str, Dict[str, Any]],
    target_task_ids: Set[str],
    output_path: str,
    uuid_class_map: Dict[str, str] = None,
    max_samples: int = None
) -> Dict[str, int]:
    """Export successful trajectory steps as JSONL samples."""
    stats = {
        'total': len(target_task_ids),
        'found': len(successful_tasks),
        'not_found': len(target_task_ids) - len(successful_tasks),
        'steps_extracted': 0,
        'samples_exported': 0
    }

    samples = []
    not_found_ids = []

    for task_id in target_task_ids:
        if task_id not in successful_tasks:
            not_found_ids.append(task_id)
            continue

        task_info = successful_tasks[task_id]
        task_dir = task_info['task_dir']

        config_file = Path(task_dir) / "task_config.json"
        instruction = ''

        category = uuid_class_map.get(task_id) if uuid_class_map else None

        if category is None:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    category = config.get('snapshot', 'unknown')
                    instruction = config.get('instruction', '')
                except Exception:
                    print("  Warning: Failed to read task_config.json", file=sys.stderr)
                    category = 'unknown'
        else:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    instruction = config.get('instruction', '')
                except Exception:
                    print("  Warning: Failed to read task_config.json", file=sys.stderr)

        snapshot = category

        print("  Extracting a successful task")

        trajectory = extract_trajectory_from_final_messages(task_dir, task_id)
        stats['steps_extracted'] += len(trajectory)

        for step in trajectory:
            sample = {
                'sample_id': f"{task_id}_step_{step['step_num']}",
                'task_id': task_id,
                'snapshot': snapshot,
                'instruction': instruction,
                'step_num': step['step_num'],
                'content': step['content'],
                'screenshot': step['screenshot'],
                'reward': task_info['reward'],
                'trace_id': '',
                'source_dir': ''
            }

            samples.append(sample)
            stats['samples_exported'] += 1

            if max_samples and stats['samples_exported'] >= max_samples:
                break

        if max_samples and stats['samples_exported'] >= max_samples:
            break

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    if not_found_ids:
        print(f"\n  Not found or failed task_ids ({len(not_found_ids)}):", file=sys.stderr)

    return stats


def main():
    """Parse command-line arguments and run sample extraction."""
    parser = argparse.ArgumentParser(
        description='Extract successful experiment samples from evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--task_id_file', required=True,
                        help='Task ID JSON file')
    parser.add_argument('--results_dir', required=True,
                        help='Results directory path')
    parser.add_argument('--output_path', required=True,
                        help='Output jsonl file path')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to extract')
    parser.add_argument('--category_json', type=str, default=DEFAULT_UUID_CATEGORY_JSON,
                        help='Optional evaluation JSON in {category: [uuid, ...]} format')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    print("="*80)
    print("Sample Extraction")
    print("="*80)

    print("\n[1/4] Loading task IDs from the configured file")
    task_ids = load_task_ids(args.task_id_file)
    print(f"  Loaded {len(task_ids)} task IDs")

    uuid_class_map = load_uuid_category_map(args.category_json)
    if uuid_class_map:
        print(f"  UUID category map loaded: {len(uuid_class_map)} entries")
    else:
        print("  Warning: UUID category map not loaded, falling back to snapshot field")

    if args.verbose:
        print(f"  Verbose mode: {len(task_ids)} task IDs are configured")

    print(f"\n[2/4] Finding successful tasks in results directory")
    successful_tasks = find_successful_tasks_in_results(args.results_dir, task_ids)

    print(f"\n  Found {len(successful_tasks)} / {len(task_ids)} successful tasks")

    print(f"\n[3/4] Extracting samples...")
    stats = export_samples(
        successful_tasks=successful_tasks,
        target_task_ids=task_ids,
        output_path=args.output_path,
        uuid_class_map=uuid_class_map,
        max_samples=args.max_samples
    )

    print(f"\n[4/4] Summary")
    print("="*80)
    print(f"  Total target task IDs:    {stats['total']}")
    print(f"  Found successful:         {stats['found']}")
    print(f"  Not found:                {stats['not_found']}")
    print(f"  Total steps extracted:    {stats['steps_extracted']}")
    print(f"  Total samples exported:   {stats['samples_exported']}")
    print("\n  Output file: [configured]")

    if Path(args.output_path).exists():
        print(f"  Output size: {Path(args.output_path).stat().st_size / 1024:.2f} KB")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
