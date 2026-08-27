# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import logging
import os, sys
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import time
import hashlib
from itertools import groupby

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import PreTrainedTokenizer, ProcessorMixin

# === NEW: unified DB manager and trainable filter ===
# WORKSPACE_ROOT = "/workspace/computer-use/verl"
# if WORKSPACE_ROOT not in sys.path:
#     sys.path.insert(0, WORKSPACE_ROOT)
# os.chdir(WORKSPACE_ROOT)
from verl.utils.database.mysql import create_database_manager
from verl.utils.dataset.trainable_filter import filter_fn

from mm_agents.prompts import COMPUTER_USE_DOUBAO, COMPUTER_USE_DOUBAO_WITH_CALL_USER

logger = logging.getLogger(__name__)


def collate_fn(data_list: List[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def collate_async_fn(data_list: List[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    data_list = [d for sub_d in data_list for d in sub_d]
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    if "process_reward_tensors" in non_tensors:
        pr_array = non_tensors["process_reward_tensors"]
        print("\n" + "="*50)
        print("🎯 [实战探针] Process Reward (List模式) 成功接入 DataLoader！")
        print(f"🎯 Batch 中 PR 数组的形状: {pr_array.shape} (dtype: {pr_array.dtype})")
        
        if len(pr_array) > 0 and isinstance(pr_array[0], list):
            print(f"🎯 第一条轨迹的真实步数 (List 长度): {len(pr_array[0])}")
            print(f"🎯 第一条轨迹的 PR 奖励前 5 步预览: {pr_array[0][:5]}")
        print("="*50 + "\n")
    else:
        print("\n🚨 [实战探针] 警告：当前 Batch 没有抓到 process_reward_tensors 字段！\n")

    return {**tensors, **non_tensors}


class OSWorldAsyncDataset(IterableDataset):

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # ---- keep original basic configs ----
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.use_call_user = config.get("use_call_user", False)

        # dataloader workers（IterableDataset 里这个值只用于 DataLoader 外部配置）
        self.num_workers = config.get("num_workers", 2)

        # how many task_ids to wait for and to take each poll
        self.batch_size_min = int(config.get("train_batch_size_min", 4))
        self.batch_size_max = int(config.get("train_batch_size_max", 8))

        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False

        # 与 IterableDataset 配合：每个 __iter__（即每个 epoch）最多产出多少“批次”
        self.steps_per_epoch = int(config.get("steps_per_epoch", 0))  # 0/None 表示不限

        self.max_steps = config.get("max_steps", 0)
        if self.max_steps and not self.steps_per_epoch:
            self.steps_per_epoch = int(self.max_steps)

        self.osworld_root = config.get("osworld_root", "evaluation_examples/examples")
        self.run_id = config.get("run_id", None)
        self.rollout_n = int(config.get("rollout_n", 8))
        assert self.run_id is not None, "config.run_id is required"

        self.poll_interval_sec = float(config.get("poll_interval_sec", 30.0))  # 等待间隔
        self.use_replay = self._as_bool(config.get("use_replay", False))
        self.replay_group_size = int(config.get("replay_group_size", self.rollout_n))
        if self.replay_group_size <= 0:
            raise ValueError("replay_group_size must be positive.")
        self.replay_run_ids = self._normalize_replay_run_ids(config.get("replay_run_ids", []))

        # === NEW: Unified DB manager ===
        self.db_manager = create_database_manager()

        # root_data_dir for files
        self.root_data_dir = getattr(self.config, "root_data_dir", ".")
        self.produced_batches = 0
        self.wait_num = 0 
        self.max_wait_num = config.get("max_wait_num", 20000*30)//30
        
        # 使用静态数据时，按上一批的最后一个task定锚轮换
        self._last_used_task_id = None
        self._replay_run_cursor = 0

        self.use_pr = config.get("use_pr", "none")

    # 你原来的 messages 组装逻辑（若仍需要）
    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        return messages

    # 稳定哈希，用于多 worker 分片（不要用 Python 内置 hash，它有进程盐）
    @staticmethod
    def _stable_shard(task_id, num_workers, worker_id):
        if num_workers is None or num_workers <= 1:
            return True
        key = str(task_id).encode("utf-8")
        h = int(hashlib.md5(key).hexdigest(), 16)
        return (h % num_workers) == worker_id

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _normalize_replay_run_ids(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("[") and raw.endswith("]"):
                raw = raw[1:-1]
            parts = [p.strip().strip("'").strip('"') for p in raw.split(",")]
            return [p for p in parts if p]
        if isinstance(value, (list, tuple, set, ListConfig)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _group_rows_by_task(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["task_id"])].append(dict(row))
        for task_rows in grouped.values():
            task_rows.sort(key=lambda r: (r.get("create_at"), str(r.get("trajectory_id", ""))))
        return grouped

    def _sample_replay_datasets(self) -> List[Dict[str, Any]]:
        if not self.use_replay or not self.replay_run_ids:
            return []

        seed_cfg = self.config.get("random_state", None)
        if seed_cfg is None:
            rng = np.random.RandomState()
        else:
            seed = (int(seed_cfg) + self.produced_batches * 1009) % (2**32)
            rng = np.random.RandomState(seed)

        start_idx = self._replay_run_cursor % len(self.replay_run_ids)
        for attempt in range(len(self.replay_run_ids)):
            run_idx = (start_idx + attempt) % len(self.replay_run_ids)
            run_id = self.replay_run_ids[run_idx]
            try:
                rows = self.db_manager.get_rollouts_by_run_id(run_id=run_id)
            except Exception as e:
                logger.warning(f"Failed to load replay rows for run_id={run_id}: {e}")
                continue

            grouped = self._group_rows_by_task(rows)
            eligible = []
            for task_id, task_rows in grouped.items():
                if len(task_rows) < self.replay_group_size:
                    continue
                rewards = []
                for row in task_rows:
                    reward = row.get("reward")
                    if reward is None:
                        continue
                    try:
                        rewards.append(float(reward))
                    except Exception:
                        continue
                if not rewards:
                    continue
                has_positive = any(r > 0 for r in rewards)
                has_negative = any(r <= 0 for r in rewards)
                if has_positive and has_negative:
                    eligible.append((task_id, task_rows))

            if not eligible:
                logger.info(f"[replay] run_id={run_id} has no eligible task with >= {self.replay_group_size} rows and both positive/negative rewards.")
                continue

            task_idx = int(rng.randint(len(eligible)))
            task_id, task_rows = eligible[task_idx]

            pos_positions = []
            neg_positions = []
            for idx, row in enumerate(task_rows):
                reward = row.get("reward")
                if reward is None:
                    continue
                try:
                    reward_val = float(reward)
                except Exception:
                    continue
                if reward_val > 0:
                    pos_positions.append(idx)
                else:
                    neg_positions.append(idx)

            if not pos_positions or not neg_positions:
                continue

            if len(task_rows) == self.replay_group_size:
                selected = [dict(row) for row in task_rows]
            else:
                selected_positions = {int(rng.choice(pos_positions)), int(rng.choice(neg_positions))}
                remaining_positions = [idx for idx in range(len(task_rows)) if idx not in selected_positions]
                need = self.replay_group_size - len(selected_positions)
                if len(remaining_positions) < need:
                    logger.info(f"[replay] run_id={run_id}, task_id={task_id} has only {len(task_rows)} rows, need {self.replay_group_size}; skip.")
                    continue
                perm = rng.permutation(len(remaining_positions))
                for pos in perm[:need]:
                    selected_positions.add(remaining_positions[int(pos)])
                selected = [dict(task_rows[i]) for i in sorted(selected_positions)]

            if len(selected) < self.replay_group_size:
                continue
            if len(selected) > self.replay_group_size:
                selected = [selected[i] for i in rng.permutation(len(selected))[: self.replay_group_size]]
            else:
                selected = [selected[i] for i in rng.permutation(len(selected))]

            for row in selected:
                row["is_replay"] = True

            self._replay_run_cursor = (run_idx + 1) % len(self.replay_run_ids)
            logger.info(f"[replay] selected run_id={run_id}, task_id={task_id}, size={len(selected)}, replay_group_size={self.replay_group_size}")
            return selected

        return []

    def _sort_rotate_limit_main_datasets(self, datasets: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        _min_create_at = {}
        for row in datasets:
            tid = str(row["task_id"])
            ts = row["create_at"]
            if tid not in _min_create_at or ts < _min_create_at[tid]:
                _min_create_at[tid] = ts
        datasets.sort(key=lambda r: (_min_create_at[str(r["task_id"])], str(r["task_id"]), r["create_at"]))

        rotate_static = self.config.get("rotate_task_groups", False)
        print("rotate_static: ", rotate_static)

        if rotate_static:
            groups = [(tid, list(items)) for tid, items in groupby(datasets, key=lambda r: str(r["task_id"]))]
            offset = 0
            last_tid = self._last_used_task_id
            print("last_tid: ", last_tid)
            if last_tid:
                idx_map = {tid: i for i, (tid, _) in enumerate(groups)}
                if last_tid in idx_map:
                    offset = (idx_map[last_tid] + 1) % len(groups)

            rotated = groups[offset:] + groups[:offset] if offset else groups
            datasets = [row for _, items in rotated for row in items]

        if len(datasets) > limit:
            datasets = datasets[:limit]

        if rotate_static and datasets:
            try:
                self._last_used_task_id = str(datasets[-1]["task_id"])
            except Exception:
                self._last_used_task_id = None

        return datasets

    def _ensure_db_connected(self):
        if not self.db_manager.is_connected():
            self.db_manager.setup_database()

    def _close_dbs(self):
        try:
            self.db_manager.close_database()
        except Exception:
            pass

    def _row_to_sample(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        把 DB 的 row_dict 转成训练样本（单条样本：dict）
        也会将该 rollout 标记为 used（atomic in DB layer is preferred).
        Expected row_dict to contain: "trajectory_id"
        """
        trajectory_id = row_dict["trajectory_id"]
        run_id = row_dict["run_id"]
        #instruction = row_dict["instruction"]
        reward = row_dict["reward"]
        is_replay = bool(row_dict.get("is_replay", False))

        if not is_replay:
            try:
                self.db_manager.update_rollout_used(run_id=run_id, trajectory_id=trajectory_id)
            except Exception as e:
                logger.warning(f"update_rollout_used failed for {trajectory_id}: {e}")

        # data_dir = os.path.join(self.root_data_dir, trajectory_id)
        # with open(os.path.join(data_dir, "task_config.json")) as f:
        #     task_data = json.load(f)
        # with open(os.path.join(data_dir, "reward.txt")) as f:
        #     reward_tensor = float(f.read().strip())
        #     reward_tensor = torch.tensor([reward_tensor], dtype=torch.float32)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # instruction = task_data["instruction"]
        # system_prompt = COMPUTER_USE_DOUBAO if not self.use_call_user else COMPUTER_USE_DOUBAO_WITH_CALL_USER

        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text",
        #              "text": system_prompt.format(instruction=instruction, language="English")}
        #         ],
        #     },
        # ]

        dataset_path = os.path.join(self.root_data_dir, trajectory_id)
        if is_replay:
            dataset_path = os.path.join("dart_rollouter", "results", "success", run_id, trajectory_id)

        sample = {
            # "messages": messages,
            # "instruction": instruction,
            # "task_config": task_data,
            "dataset_ids": trajectory_id,
            "dataset_paths": dataset_path,
            "reward_tensors": reward_tensor,
            "reward_diff": float("nan"),
        }

        reward_agent_result_path = os.path.join(dataset_path, "reward_agent_result.json")
        if os.path.exists(reward_agent_result_path):
            try:
                with open(reward_agent_result_path, "r", encoding="utf-8") as f:
                    reward_agent_result = json.load(f)
                if "reward_diff" in reward_agent_result:
                    sample["reward_diff"] = float(reward_agent_result["reward_diff"])
            except Exception as e:
                logger.warning(f"failed to load reward_diff for {trajectory_id}: {e}")

        process_reward = row_dict.get("process_reward", None) 

        if process_reward and self.use_pr != "none":
            if isinstance(process_reward, str):
                try:
                    process_reward = json.loads(process_reward)
                except Exception as e:
                    logger.warning(f"JSON 解析 process_reward 失败: {e}")
                    process_reward = []
            
            sample["process_reward_tensors"] = process_reward

        return sample
    
    def __iter__(self):
        """
        每次迭代（Trainer 的一个 epoch）产出若干个“批次”（list[dict]）。
        配置：
          - steps_per_epoch > 0：精确产出这么多批次后结束本次迭代
          - steps_per_epoch == 0：持续产出（上层决定何时结束）
        """
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        self._ensure_db_connected()

        # 控制每个 epoch 产出多少批次
        while (self.steps_per_epoch == 0) or (self.produced_batches < self.steps_per_epoch):
            #try:
            # 1) 获取全部候选 rollouts（原始数据）
            data = self.db_manager.get_rollouts_by_run_id(run_id=self.run_id)
            print("len all data in SQL:", len(data))

            # 2) 获取最近的若干 checkpoint 对应的 model_versions（top_mvs）
            top_n = int(self.config.get("top_mvs_n", 2))
            top_mvs = self.db_manager.get_latest_n_checkpoint_paths(run_id=self.run_id, n=top_n)
            # print(data[0])
            print(top_mvs)

            # 3) 通过 filter_fn 选择本次要训练的 datasets（限制每 task 的数量）
            datasets: List[Dict[str, Any]] = filter_fn(
                data=data,
                per_task_limit=self.rollout_n,
                top_mvs=top_mvs,
                random_state=self.config.get("random_state", None),
                acc_max=self.config.get("acc_max", 1.0),
                use_pr=self.use_pr
            )
            # datasets = data
            print("len datasets filtered for this step:", len(datasets))
            
            # 若数量不足，继续轮询等待
            min_needed = self.batch_size_min * self.rollout_n
            if self.use_replay and self.replay_run_ids:
                replay_datasets = self._sample_replay_datasets()
                if not replay_datasets:
                    print(f"[worker {worker_id}] replay enabled but no eligible replay task found; sleep {self.poll_interval_sec}s and retry.")
                    time.sleep(self.poll_interval_sec)
                    self.wait_num += 1
                    continue

                replay_count = len(replay_datasets)
                main_needed = max(min_needed - replay_count, 0)
                if len(datasets) < main_needed:
                    print(f"[worker {worker_id}] datasets={len(datasets)} < main_needed={main_needed}; "
                                f"sleep {self.poll_interval_sec}s and retry.")
                    time.sleep(self.poll_interval_sec)
                    self.wait_num += 1
                    continue
                if self.wait_num >= self.max_wait_num:
                    raise RuntimeError(f"[worker {worker_id}] Waited {self.max_wait_num} times but still not enough data (datasets={len(datasets)} < main_needed={main_needed}). Giving up.")
                self.wait_num =0

                max_allowed = self.batch_size_max * self.rollout_n
                main_datasets = self._sort_rotate_limit_main_datasets(datasets, max_allowed)
                main_datasets = main_datasets[:main_needed]
                datasets = main_datasets + replay_datasets

                if len(datasets) < min_needed:
                    print(f"[worker {worker_id}] datasets+replay={len(datasets)} < min_needed={min_needed}; "
                                f"sleep {self.poll_interval_sec}s and retry.")
                    time.sleep(self.poll_interval_sec)
                    self.wait_num += 1
                    continue

                replay_count_by_run = {}
                replay_count_by_task = {}
                for row in replay_datasets:
                    rid = str(row.get("run_id", ""))
                    tid = str(row.get("task_id", ""))
                    replay_count_by_run[rid] = replay_count_by_run.get(rid, 0) + 1
                    replay_count_by_task[tid] = replay_count_by_task.get(tid, 0) + 1
                print(f"len data after filtered: {len(datasets)}, replay={replay_count}, replay_by_run={replay_count_by_run}, replay_by_task={replay_count_by_task}")
            else:
                if len(datasets) < min_needed:
                    print(f"[worker {worker_id}] datasets={len(datasets)} < min_needed={min_needed}; "
                                f"sleep {self.poll_interval_sec}s and retry.")
                    time.sleep(self.poll_interval_sec)
                    self.wait_num += 1
                    continue
                if self.wait_num >= self.max_wait_num:
                    raise RuntimeError(f"[worker {worker_id}] Waited {self.max_wait_num} times but still not enough data (datasets={len(datasets)} < min_needed={min_needed}). Giving up.")
                self.wait_num =0
                
                # 按照task id最早出现时间排序，优先选先出现的task id
                _min_create_at = {}
                for row in datasets:
                    tid = str(row["task_id"])
                    ts = row["create_at"]
                    if tid not in _min_create_at or ts < _min_create_at[tid]:
                        _min_create_at[tid] = ts
                # 先按task_id组的 min(create_at) 升序排列；再按 task_id排序
                datasets.sort(key=lambda r: (_min_create_at[str(r["task_id"])], str(r["task_id"]), r["create_at"]))

                # 使用静态数据时，按照task_id轮换使用，记录偏移量         
                rotate_static = self.config.get("rotate_task_groups", False)
                print("rotate_static: ",rotate_static)
                
                if rotate_static:
                    # 按task_id分组
                    groups = [(tid, list(items)) for tid, items in groupby(datasets, key=lambda r: str(r["task_id"]))]
                    offset = 0
                    last_tid = self._last_used_task_id
                    print("last_tid: ", last_tid)
                    if last_tid:
                        idx_map = {tid: i for i, (tid, _) in enumerate(groups)}
                        if last_tid in idx_map:
                            offset = (idx_map[last_tid] + 1) % len(groups)

                    rotated = groups[offset:] + groups[:offset] if offset else groups
                    datasets = [row for _, items in rotated for row in items]
                
                # 限制最大数量
                max_allowed = self.batch_size_max * self.rollout_n
                if len(datasets) > max_allowed:
                    datasets = datasets[:max_allowed]

                print("len data after filtered: ", len(datasets))
                
                # 记录“本批最后一个 task_id”，供下一批计算偏移
                if rotate_static and datasets:
                    try:
                        self._last_used_task_id = str(datasets[-1]["task_id"])
                    except Exception:
                        self._last_used_task_id = None
                
            # 保存采样数据，用于检查
            try:
                dt = time.strftime("%Y%m%d")
                ts = time.strftime("%Y%m%d-%H%M%S")
                dump_dir = f"debug_datasets/{self.run_id.split('/')[-1]}"
                os.makedirs(dump_dir, exist_ok=True)
                dump_base = f"datasets_step{self.produced_batches+1}_{ts}"

                # 保存完整样本列表
                with open(os.path.join(dump_dir, dump_base + ".json"), "w", encoding="utf-8") as f:
                    json.dump(datasets, f, ensure_ascii=False, indent=2, default=str)

                # 简要统计：按 task_id 分布
                from collections import Counter
                def _task_id_of(row: dict) -> str:
                    tid = row.get("task_id")
                    if tid is not None:
                        return str(tid)
                    traj = str(row.get("trajectory_id", ""))
                    return traj.split("/", 1)[0] if "/" in traj else traj

                counts = Counter(_task_id_of(r) for r in datasets)
                summary = {
                    "total": len(datasets),
                    "counts_by_task_id": {k: int(v) for k, v in counts.items()},
                }
                if self.use_replay and self.replay_run_ids:
                    summary["replay_total"] = int(sum(1 for r in datasets if r.get("is_replay")))
                    summary["counts_by_run_id"] = {
                        k: int(v) for k, v in Counter(str(r.get("run_id", "")) for r in datasets).items()
                    }
                with open(os.path.join(dump_dir, dump_base + ".summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                logger.info(f"[worker {worker_id}] Dumped datasets to {os.path.join(dump_dir, dump_base)}.*")
            except Exception as e:
                logger.warning(f"[worker {worker_id}] Failed to dump datasets for inspection: {e}")

            # 构造样本
            batch_samples: List[Dict[str, Any]] = []
            for row in datasets:
                try:
                    sample = self._row_to_sample(row)
                    #print(sample['dataset_ids'])
                    batch_samples.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to build sample for row={row.get('trajectory_id')}: {e}")

            if batch_samples:
                logger.info(f"[worker {worker_id}] step {self.produced_batches}, traj num: {len(batch_samples)}")
                self.produced_batches += 1
                yield batch_samples
            else:
                logger.info(f"[worker {worker_id}] No valid samples built; sleep {self.poll_interval_sec}s")
                time.sleep(self.poll_interval_sec)

            # except Exception as e:
            #     logger.error(f"[worker {worker_id}] Iteration error: {e}")
            #     time.sleep(self.poll_interval_sec)
            #     continue

        # 迭代完成
        self._close_dbs()

    def __len__(self):
        if self.steps_per_epoch and self.steps_per_epoch > 0:
            return self.steps_per_epoch
        raise TypeError("This IterableDataset has no static length; set config.steps_per_epoch to use len(dataset).")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]

            # 关闭数据库连接并移除数据库管理器，避免序列化问题
            if "db_manager" in state and state["db_manager"] is not None:
                try:
                    state["db_manager"].close_database()
                except Exception:
                    pass
                del state["db_manager"]

            return state

        return self.__dict__.copy()

def main(run_id, root_data_dir):

    import time
    from typing import Tuple

    import torch
    from omegaconf import OmegaConf
    
    def tensor_shape(x) -> Tuple[int, ...] | None:
        try:
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
        except Exception:
            pass
        return None
    
    # 组装最小配置（与数据集实现里的 config.get 字段对齐）
    cfg = OmegaConf.create({
        "run_id": run_id,
        "steps_per_epoch": 1,
        "train_batch_size_min": 4,
        "train_batch_size_max": 8,
        "rollout_n": 8,
        "top_mvs_n": 2,
        "poll_interval_sec": 30,
        "root_data_dir": root_data_dir,
        "rotate_task_groups": True,
        # 你也可以按需加：cache_dir / use_call_user / prompt_key / image_key / video_key 等
    })

    dataset = OSWorldAsyncDataset(
        data_files=[],
        tokenizer=None,   # 本测试路径中未用到 tokenizer
        config=cfg,
        processor=None,
    )

    min_needed = cfg.train_batch_size_min * cfg.rollout_n
    max_allowed = cfg.train_batch_size_max * cfg.rollout_n

    print(f"[config] run_id={cfg.run_id}")
    print(f"[config] steps={cfg.steps_per_epoch}, rollout_n={cfg.rollout_n}, "
          f"min_needed={min_needed}, max_allowed={max_allowed}")
    print(f"[config] top_mvs_n={cfg.top_mvs_n}, poll_interval_sec={cfg.poll_interval_sec}")
    print(f"[config] root_data_dir={cfg.root_data_dir}")

    t0 = time.time()
    pulled = 0
    for step_idx, batch in enumerate(dataset):
        pulled += 1
        print(f"\n=== Step {step_idx} ===")
        print(f"Batch size: {len(batch)}  (expected: {min_needed} ~ {max_allowed})")

        # 预览一条样本结构
        if batch:
            ex = batch[0]
            keys_preview = list(ex.keys())
            print(f"Example keys: {keys_preview}")

            # 打印 messages 的首条，便于核对模板是否正确
            first_msg = None
            try:
                first_msg = ex.get("messages", [None])[0]
            except Exception:
                pass
            print(f"First message: {first_msg}")

            # 打印 id 与奖励张量形状等关键字段
            print(
                "dataset_ids:", ex.get("dataset_ids"),
                "| reward_tensors shape:", tensor_shape(ex.get("reward_tensors")),
            )

        # 数据集已通过 steps_per_epoch 控制结束，这里只是双保险
        if pulled >= cfg.steps_per_epoch:
            break

    print(f"\nDone. Pulled {pulled} batch(es) in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    
    run_id = "results/test_for_train_pass8_gpu7_env69_20250821_2333"
    root_data_dir = "test_for_train_pass8_gpu7_env69_20250821_2333"
    main(run_id, root_data_dir)
