import asyncio
from typing import List, Optional, Union, Dict, Any
import json
import os
import hashlib
from pathlib import Path
from omegaconf import DictConfig
from dataclasses import dataclass, asdict
import copy
import logging
import random

# Import required modules
from src.core.prompts import COMPUTER_USE_PROMPT, COMPUTER_USE_PROMPT_WITH_CALL_USER
from src.utils.log_config import setup_logging
from src.services.mysql_rollout import MySQLRolloutORM, DB_CONFIG

# Setup unified logging system
setup_logging()
logger = logging.getLogger(__name__)

class TaskLoader:
    def __init__(self, task_cfg: DictConfig, storage_root):
        self.task_file = Path(task_cfg.task_file)
        self.osworld_root = Path(task_cfg.osworld_root)
        
        self._latest_sha: Optional[str] = None
        self.storage_root = storage_root
        self.resume = task_cfg.resume
        
        # Dynamic rollout_n related parameters
        self.success_rate_threshold = getattr(task_cfg, 'success_rate_threshold', 1.0)  # Success rate threshold
        self.min_rollout_n = getattr(task_cfg, 'min_rollout_n', 1)  # Minimum rollout_n
        self.run_id = getattr(task_cfg, 'run_id', None)  # Run ID for database queries
        self.db_enabled = getattr(task_cfg, 'db_enabled', False)  # Whether to enable database functionality
        
        # Initialize database connection
        self.mysql_orm = None
        if self.db_enabled and self.run_id:
            try:
                self.mysql_orm = MySQLRolloutORM(DB_CONFIG)
                logger.info("MySQL ORM initialized successfully")
            except Exception as e:
                logger.error(f"MySQL ORM initialization failed: {e}")
                self.mysql_orm = None
        
        # Task success rate mapping
        self.mapping_task_success: Dict[str, float] = {}
        # Task trajectory count mapping
        self.mapping_task_counts: Dict[str, float] = {}
        self.mapping_task_dynamic_n: Dict[str, float] = {}

        # Periodically update task success rate mapping
        self._update_mapping_task_success()

    def poll_for_tasks(self) -> List[Dict]:
        """Find new tasks json file
        Return list of TaskInfo dict if there is new json
        Else return []
        """
        self._maybe_refresh_dataset()
        
        # Update task success rate mapping
        self._update_mapping_task_success()
        
        tasks_list = [task.to_dict() for task in self._tasks]
        random.shuffle(tasks_list)

        return tasks_list 
    
    def _maybe_refresh_dataset(self):
        latest_json = self.task_file
        print("Current tasks file: ", str(latest_json))
        
        with open(latest_json) as f:
            data = json.load(f)
            
        raw_tasks = [
            {"task_type": task_type, "task_id": task_id}
            for task_type, task_ids in data.items()
            for task_id in task_ids
        ]
        
        if self.resume:
            # Filter completed or mismatched tasks
            filtered_tasks = []
            storage_root = Path(self.storage_root)

            for raw in raw_tasks:
                task_id = str(raw["task_id"])
                task_type_expected = raw["task_type"]

                # Find all subdirectories starting with task_id (allow multiple versions)
                candidate_dirs = [
                    d for d in storage_root.iterdir()
                    if d.is_dir() and d.name.startswith(task_id)
                ]

                # Default to assuming task is not finished
                task_finished = False

                for d in candidate_dirs:
                    cfg_path = d / "task_config.json"
                    if not cfg_path.exists():
                        print("Cannot find config file")
                        continue

                    try:
                        with cfg_path.open("r", encoding="utf-8") as cf:
                            cfg = json.load(cf)
                    except Exception:
                        print("Config corrupted, ignoring this directory")
                        continue

                    # 3.1 If task_type differs => not the same task, skip this directory
                    if cfg.get("raw", {}).get("task_type") != task_type_expected:
                        continue

                    # 3.2 If task_type matches, check reward.txt
                    if (d / "reward.txt").exists():
                        task_finished = True
                        break  # Found completion record, no need to check other directories
                        
                if not task_finished:
                    filtered_tasks.append(raw)
                    
            self._tasks = [build_task(raw, self.osworld_root) for raw in filtered_tasks]
            print(f"Total number of tasks: {len(raw_tasks)}, Remaining: {len(filtered_tasks)}")

        else:
            self._tasks = [build_task(raw, self.osworld_root) for raw in raw_tasks]
            print(f"Total number of tasks: {len(raw_tasks)}")

        return True
        
    def _find_latest_json(self) -> Optional[Path]:
        files = list(self.task_root.glob("*.json"))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None
    
    @staticmethod
    def _calc_sha1(fp: Path, chunk_size=2<<20) -> str:
        h = hashlib.sha1()
        with fp.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _update_mapping_task_success(self):
        """Update task success rate mapping"""
        if not self.mysql_orm or not self.run_id:
            logger.info("Database not enabled or run_id not configured, skipping update of task success rate mapping")
            return
        
        try:
            # Get latest model version success rate data
            avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts = self.mysql_orm.get_latest_success_for_each_task(self.run_id, self.mapping_task_dynamic_n)
            self.mapping_task_success = mapping_task_success
            self.mapping_task_counts = mapping_task_counts
            logger.info(f"Updated task success rate and trajectory count mapping, total {len(mapping_task_success)} tasks")
        except Exception as e:
            logger.error(f"Failed to update task success rate and trajectory count mapping: {e}")

    def get_dynamic_rollout_n(self, task_id: str, rollout_n) -> int:
        """
        Dynamically calculate rollout_n value based on task success rate
        - Tasks with success rate below threshold maintain maximum sampling count
        - Tasks with success rate above threshold adjust sampling count using inverse proportion function
        """
        # Get task success rate, default to 0
        success_rate = self.mapping_task_success.get(task_id, 0.0)
        traj_counts = self.mapping_task_counts.get(task_id, 0.0)
        
        # If success rate is below threshold, maintain maximum sampling count
        if success_rate < self.success_rate_threshold:
            dynamic_rollout_n = rollout_n
            logger.info(f"Task {task_id} trajectory count {traj_counts} success rate {success_rate:.2%} is below threshold {self.success_rate_threshold:.2%}, using default sampling count: {dynamic_rollout_n}")
        else:
            # To avoid division by zero and very small values, add a small constant
            dynamic_rollout_n = int(9 / success_rate - 7)
            dynamic_rollout_n = max(self.min_rollout_n, dynamic_rollout_n)
            logger.info(f"Task {task_id} trajectory count {traj_counts} success rate {success_rate:.2%} is above threshold {self.success_rate_threshold:.2%}, using dynamic sampling count: {dynamic_rollout_n} (calculated value: {int(9 / success_rate - 7)})")

        self.mapping_task_dynamic_n[task_id] = dynamic_rollout_n
        logger.info(f"Dynamic rollout_n for task {task_id}: {dynamic_rollout_n} success rate: {success_rate:.2%}")
        return traj_counts, success_rate, dynamic_rollout_n

    def sort_tasks_by_success_rate(self, tasks: List[Dict]) -> List[Dict]:
        """
        Sort tasks by success rate
        - Tasks not in mapping come first
        - Tasks in mapping are sorted by success rate ascending
        """
        def sort_key(task):
            task_id = task.get('task_config', {}).get('raw', {}).get('task_id', '')
            # If task is not in success rate mapping, put it first (return -1)
            if task_id not in self.mapping_task_success:
                return (-1, 0)  # (priority, success rate)
            # If task is in mapping, sort by success rate
            return (0, self.mapping_task_success[task_id])
        
        # Use stable sorting
        return sorted(tasks, key=sort_key)

@dataclass
class TaskInfo:
    messages: List
    instruction: str
    task_config: Dict

    def to_dict(self):
        return asdict(self)


def build_task(raw: Dict, osworld_root: Path, use_call_user: bool = False) -> TaskInfo:
    task_type = raw["task_type"]
    task_id = raw["task_id"]
    task_path = os.path.join(osworld_root, task_type, task_id + ".json")
    with open(task_path) as f:
        task_data = json.load(f)

    task_data["raw"] = {
        "task_type": task_type,
        "task_id": task_id
    }

    instruction = task_data["instruction"]

    if "human-ground-truth" in task_data and "single-action" in task_data["human-ground-truth"]:
        plan = task_data["human-ground-truth"]["single-action"]
        plan_text = "\n".join(plan)

        instruction = instruction.strip() + "\nHere is an instruction to help you complete the task: \n" + plan_text

    system_prompt = COMPUTER_USE_PROMPT if not use_call_user else COMPUTER_USE_PROMPT_WITH_CALL_USER
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": system_prompt.format(
                        instruction=instruction, 
                        language="English"
                )}
            ]
        }
    ]
    
    return TaskInfo(
        messages = messages,
        instruction = instruction,
        task_config = task_data
    )

def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name or path of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def process_text_sync(processor, messages):
    """Process text synchronously.
    
    Args:
        processor: HuggingFace processor
        messages: List of message dictionaries
        
    Returns:
        torch.Tensor: Input IDs tensor
    """
    formatted = ""
    for m in messages:
        content = m["content"]

        # If content is list (multimodal message)
        if isinstance(content, list):
            # Only take text part
            texts = [c["text"] for c in content if c.get("type") == "text"]
            content_str = "\n".join(texts)
        else:
            # Regular string
            content_str = content

        formatted += f"{content_str}<|im_end|>\n"
    model_inputs = processor(text=[formatted], images=None, return_tensors="pt")
    input_ids = model_inputs.pop("input_ids")
    return input_ids[0]

import torch
def process_system_prompt(instruction_root='evaluation_examples/examples_new'):
    """Process system prompts and generate token IDs.
    
    Args:
        instruction_root (str): Root directory containing instruction JSON files
    """
    # Assume hf_processor and COMPUTER_USE_PROMPT are already defined
    processor = hf_processor('/workspace/models/UI-TARS-1.5-7B', trust_remote_code=True, use_fast=True)
    system_prompt = COMPUTER_USE_PROMPT  
    
    # --- Modification 1: Define output file path ---
    output_dir = 'evaluation_examples/system_prompt_token_ids'
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'system_prompts.pt') # Define final output filename
    print(f"Output will be saved to '{output_file_path}'")

    # --- Modification 2: Initialize a dictionary to collect all data ---
    all_token_ids = {}

    # Traverse all subdirectories
    for subdir_name in sorted(os.listdir(instruction_root)):
        subdir_path = os.path.join(instruction_root, subdir_name)
        if not os.path.isdir(subdir_path):
            continue
            
        # Traverse files in directory
        for file_name in sorted(os.listdir(subdir_path)):
            if not file_name.endswith('.json'):
                continue

            file_path = os.path.join(subdir_path, file_name)
            task_id = file_name.split('.')[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                instruction = data.get('instruction')
                if not instruction:
                    print(f"Warning: 'instruction' key not found or empty in {file_path}. Skipping.")
                    continue

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": system_prompt.format(
                                    instruction=instruction, 
                                    language="English"
                            )}
                        ]
                    }
                ]
                
                input_ids = process_text_sync(processor, messages)
                
                # --- Modification 3: Store results in dictionary instead of saving individual files ---
                if input_ids is not None:
                    all_token_ids[task_id] = input_ids
                    print(f"Processed and collected token IDs for task: {task_id}")
                else:
                    print(f"Warning: process_text_sync returned None for task {task_id}. Skipping.")
                 
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {file_path}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {file_path}: {e}")

    # --- Modification 4: After loop completes, save entire dictionary to single file ---
    if all_token_ids:
        torch.save(all_token_ids, output_file_path)
        print(f"\nSuccessfully saved {len(all_token_ids)} token ID sets to {output_file_path}")
    else:
        print("\nNo token IDs were generated. Output file was not created.")


def main():
    process_system_prompt()

if __name__ == "__main__":
    main()