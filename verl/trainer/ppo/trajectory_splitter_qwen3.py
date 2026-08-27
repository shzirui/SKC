import copy
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from PIL import Image
from transformers import AutoProcessor

import verl.utils.torch_functional as verl_F
from verl import DataProto

try:
    from verl.models.transformers.qwen3_vl import get_rope_index
except Exception:
    from verl.models.transformers.qwen2_vl import get_rope_index


QWEN3_HISTORY_N = 100
QWEN3_IMAGE_MAX = 20
QWEN3_FOLD_SIZE = 10
QWEN3_COLLAPSE_TEXT = "This screenshot has been collapsed."
QWEN3_IMAGE_FACTOR = 32
QWEN3_MAX_PIXELS = 16 * 16 * 4 * 12800


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        print("stack", key)
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return ""


def _content_text(text: str) -> dict:
    return {"type": "text", "text": text}


def _content_image(path: str) -> dict:
    return {"type": "image", "image": path}


def _normalize_image_ref(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        url = value.get("url")
        if isinstance(url, str):
            return url
    return None


def _collect_image_refs(message: dict) -> list[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    refs = []
    for item in content:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ == "image":
            ref = _normalize_image_ref(item.get("image"))
        elif typ == "image_url":
            ref = _normalize_image_ref(item.get("image_url"))
        else:
            ref = None
        if ref:
            refs.append(ref)
    return refs


def _smart_resize_size(width: int, height: int, factor: int = QWEN3_IMAGE_FACTOR, max_pixels: int = QWEN3_MAX_PIXELS) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return width, height
    pixels = width * height
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        width = int(width * scale)
        height = int(height * scale)
    width = max(factor, round(width / factor) * factor)
    height = max(factor, round(height / factor) * factor)
    return width, height


class StepwiseTrajectorySplitter:
    """Qwen3/Qwen3VL stepwise splitter.

    This file is intended to replace trajectory_splitter.py by filename swap. It
    keeps the public class name and output fields used by RayOSWorldAsyncTrainer,
    but rebuilds each step with the official OSWorld Qwen3VL sliding history
    protocol and uses cached response token ids/logprobs from data_for_step_*.pt.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        root_dir: str,
        window_size: int = 5,
        stride_size: int = 1,
        max_prompt_length: int = 32048,
        max_response_length: int = 32000,
        truncation: str = "error",
        limit_images: int = 5,
        limit_messages: int = 35,
        use_vllm_logp: bool = False,
        use_token_ids_from_pt: bool = False,
        traj_filter: bool = False,
        history_n: int = QWEN3_HISTORY_N,
        image_max: int = QWEN3_IMAGE_MAX,
        fold_size: int = QWEN3_FOLD_SIZE,
        collapse_text: str = QWEN3_COLLAPSE_TEXT,
    ) -> None:
        self.processor = processor
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride_size = stride_size
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_response_length = max_response_length
        self.limit_images = limit_images
        self.limit_messages = limit_messages
        self.use_vllm_logp = use_vllm_logp
        self.use_token_ids_from_pt = use_token_ids_from_pt
        self.traj_filter = traj_filter
        self.history_n = history_n
        self.image_max = int(image_max)
        self.fold_size = int(fold_size)
        self.collapse_text = collapse_text

        if self.processor is None:
            raise ValueError("Qwen3 splitter requires a HuggingFace processor; hf_processor returned None.")

        self.tokenizer = self.processor.tokenizer
        self.system_token_ids = self._load_system_prompt_token_ids()

    def _load_system_prompt_token_ids(self):
        path = "evaluation_examples/system_prompt_token_ids/system_prompts.pt"
        if not os.path.exists(path):
            print(f"[Qwen3Splitter] {path} not found; prompt ids will be rebuilt from messages when pt prompt ids are empty.")
            return {}
        try:
            return torch.load(path)
        except Exception as exc:
            print(f"[Qwen3Splitter] failed to load {path}: {exc}")
            return {}

    def split(
        self,
        dataset_ids: list[str],
        reward_tensor: torch.Tensor | None = None,
        process_reward_tensor: torch.Tensor | None = None,
        dataset_paths: list[str] | None = None,
    ) -> DataProto:
        batch_output = []
        avg_len = self._positive_avg_len(dataset_ids, reward_tensor, dataset_paths) if self.traj_filter else 0

        for idx, dataset_id in enumerate(dataset_ids):
            reward = reward_tensor[idx].item() if reward_tensor is not None else 0.0
            process_rewards = process_reward_tensor[idx] if process_reward_tensor is not None else None
            dataset_path = dataset_paths[idx] if dataset_paths is not None else None
            batch_messages = self.split_dataset_id_from_pt(
                dataset_id,
                reward=reward,
                avg_len=avg_len,
                process_rewards=process_rewards,
                dataset_path=dataset_path,
            )
            batch_output += self.tokenize_from_pt(batch_messages, dataset_id, reward=reward, dataset_path=dataset_path)

        return collate_fn(batch_output)

    def split_parallel(
        self,
        dataset_ids: list[str],
        reward_tensor: torch.Tensor | None = None,
        process_reward_tensor: torch.Tensor | None = None,
        dataset_paths: list[str] | None = None,
        num_cpus: int = 4,
        parallel_size: int = 16,
    ) -> DataProto:
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus)

        avg_len = self._positive_avg_len(dataset_ids, reward_tensor, dataset_paths) if self.traj_filter else 0

        @ray.remote
        def process_single_dataset(splitter, dataset_id, reward, process_rewards, avg_len, dataset_path):
            batch_messages = splitter.split_dataset_id_from_pt(
                dataset_id,
                reward=reward,
                avg_len=avg_len,
                process_rewards=process_rewards,
                dataset_path=dataset_path,
            )
            return splitter.tokenize_from_pt(batch_messages, dataset_id, reward=reward, dataset_path=dataset_path)

        futures = []
        for idx, dataset_id in enumerate(dataset_ids):
            reward = reward_tensor[idx].item() if reward_tensor is not None else 0.0
            process_rewards = process_reward_tensor[idx] if process_reward_tensor is not None else None
            dataset_path = dataset_paths[idx] if dataset_paths is not None else None
            futures.append(process_single_dataset.remote(self, dataset_id, reward, process_rewards, avg_len, dataset_path))

        batch_output = []
        for result in ray.get(futures):
            batch_output += result
        return collate_fn(batch_output)

    def _positive_avg_len(self, dataset_ids, reward_tensor, dataset_paths):
        if reward_tensor is None:
            return 0
        positive_lengths = []
        for idx, dataset_id in enumerate(dataset_ids):
            if reward_tensor[idx].item() <= 0:
                continue
            dataset_dir = dataset_paths[idx] if dataset_paths is not None else os.path.join(self.root_dir, dataset_id)
            message_path = os.path.join(dataset_dir, "final_messages.json")
            try:
                with open(message_path, encoding="utf-8") as f:
                    dataset = json.load(f)
            except Exception:
                continue
            positive_lengths.append(sum(1 for msg in dataset if msg.get("role") == "assistant"))
        return int(sum(positive_lengths) / len(positive_lengths)) if positive_lengths else 0

    def split_dataset_id_from_pt(
        self,
        dataset_id: str,
        reward: float,
        avg_len: int,
        process_rewards: torch.Tensor = None,
        dataset_path: str | None = None,
    ) -> list[tuple]:
        dataset_dir = dataset_path if dataset_path is not None else os.path.join(self.root_dir, dataset_id)
        message_path = os.path.join(dataset_dir, "final_messages.json")
        with open(message_path, encoding="utf-8") as f:
            dataset = json.load(f)

        pt_data_files = sorted(Path(dataset_dir).glob("data_for_step_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
        pt_data_list = [torch.load(f) for f in pt_data_files]
        rollout_log_probs = [d.get("logp", torch.tensor([])).float() for d in pt_data_list]
        response_token_ids = [d.get("token_ids", torch.tensor([])).long() for d in pt_data_list]
        prompt_token_ids = [d.get("prompt_token_ids", torch.tensor([])).long() for d in pt_data_list]

        task_config = {"id": dataset_id.split("_")[0]}
        screenshots, responses, actions, system_prompt, instruction = self._extract_qwen3_trace(dataset, dataset_dir)
        n_steps = min(len(responses), len(response_token_ids), len(rollout_log_probs))

        if reward == 0 and avg_len > 0:
            start_step = avg_len
            if n_steps <= avg_len:
                return []
        else:
            start_step = 0

        return_rewards = self._build_return_rewards(reward, process_rewards)

        batch_data = []
        for step_idx in range(start_step, n_steps):
            messages = self._build_step_messages(
                system_prompt=system_prompt,
                instruction=instruction,
                screenshots=screenshots,
                responses=responses,
                actions=actions,
                step_idx=step_idx,
            )
            messages.append({"role": "assistant", "content": [_content_text(responses[step_idx])]})

            process_reward = None
            return_reward = None
            if process_rewards is not None and step_idx < len(process_rewards):
                process_reward = process_rewards[step_idx]
                if return_rewards is not None and step_idx < len(return_rewards):
                    return_reward = return_rewards[step_idx]

            batch_data.append(
                (
                    messages,
                    copy.deepcopy(task_config),
                    rollout_log_probs[step_idx],
                    prompt_token_ids[step_idx] if step_idx < len(prompt_token_ids) else torch.tensor([]),
                    response_token_ids[step_idx],
                    process_reward,
                    return_reward,
                    step_idx,
                )
            )
        return batch_data

    def split_dataset_id(self, dataset_id: str, dataset_path: str | None = None):
        raise RuntimeError("Qwen3 splitter requires use_vllm_logp=True and cached data_for_step_*.pt files.")

    def tokenize(self, batch_data: list, dataset_id: str, reward: float, dataset_path: str | None = None):
        raise RuntimeError("Qwen3 splitter requires use_vllm_logp=True and use_token_ids_from_pt=True.")

    def tokenize_from_pt(self, batch_data: list, dataset_id: str, reward: float, dataset_path: str | None = None):
        tokenized_batch_data = []
        for (messages, task_config, rollout_log_prob, prompt_ids, response_ids, process_reward, return_reward, step_idx) in batch_data:
            if response_ids is None or response_ids.numel() == 0:
                print(f"[Qwen3Splitter] empty response token ids for {dataset_id} step {step_idx}; skip.")
                continue

            input_ids, attention_mask, position_ids, multi_modal_data, model_inputs = self._get_inputs_from_pt(
                messages,
                dataset_id,
                prompt_ids,
                dataset_path=dataset_path,
            )
            response, response_attention_mask, response_position_ids, _, _ = self._get_responses_from_pt(
                messages,
                dataset_id,
                model_inputs,
                position_ids,
                response_ids,
                attention_mask,
                dataset_path=dataset_path,
            )

            position_ids = torch.cat([position_ids[0], response_position_ids[0]], dim=-1)
            attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            seq = torch.cat([input_ids, response], dim=-1)

            valid_response_length = response_attention_mask.sum()
            reward_tensor = torch.zeros_like(response[0], dtype=torch.float32)
            reward_tensor[valid_response_length - 1] = reward

            rollout_log_prob = self._align_rollout_logp(rollout_log_prob, valid_response_length, dataset_id, step_idx)

            row_dict = {
                "prompts": input_ids[0],
                "responses": response[0],
                "attention_mask": attention_mask[0],
                "input_ids": seq[0],
                "position_ids": position_ids,
                "reward_tensor": reward_tensor,
                "multi_modal_data": multi_modal_data,
                "multi_modal_inputs": dict(model_inputs),
                "raw_messages": messages,
                "dataset_ids": dataset_id,
                "uid": task_config["id"],
                "rollout_log_probs": rollout_log_prob[0],
                "step_idx": step_idx,
            }

            if process_reward is not None:
                process_reward_tensor = torch.zeros_like(response[0], dtype=torch.float32)
                process_reward_tensor[valid_response_length - 1] = reward if process_reward == -1 else reward + 0.5 * (process_reward - 0.5)
                row_dict["process_reward_tensor"] = process_reward_tensor

            if return_reward is not None and process_reward != -1:
                return_reward_tensor = torch.zeros_like(response[0], dtype=torch.float32)
                return_reward_tensor[valid_response_length - 1] = return_reward
                row_dict["return_reward_tensor"] = return_reward_tensor

            self.compute_mask(row_dict)
            tokenized_batch_data.append(row_dict)
        return tokenized_batch_data

    def _extract_qwen3_trace(self, dataset: list[dict], dataset_dir: str) -> tuple[list[str], list[str], list[str], str, str]:
        system_prompt = _as_text(dataset[0].get("content", "")) if dataset else ""
        screenshots = []
        responses = []
        instruction = ""

        for msg in dataset:
            if msg.get("role") == "user":
                screenshots.extend(_collect_image_refs(msg))
                text = _as_text(msg.get("content", ""))
                if text and not instruction:
                    instruction = self._extract_instruction(text)
            elif msg.get("role") == "assistant":
                text = _as_text(msg.get("content", ""))
                if text:
                    responses.append(text)

        actions = self._load_parsed_actions(dataset_dir, responses)
        if not system_prompt:
            system_prompt = self._build_default_system_prompt()
        if not instruction:
            instruction = self._extract_instruction(_as_text(dataset[1].get("content", ""))) if len(dataset) > 1 else ""

        return screenshots, responses, actions, system_prompt, instruction

    def _load_parsed_actions(self, dataset_dir: str, responses: list[str]) -> list[str]:
        path = os.path.join(dataset_dir, "parsed_actions.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                actions = [str(item.get("low_level_instruction", "")).strip() for item in data if isinstance(item, dict)]
                if actions:
                    return actions
            except Exception as exc:
                print(f"[Qwen3Splitter] failed to read parsed_actions.json from {dataset_dir}: {exc}")
        return [self._extract_action_line(resp) or self._compact_previous_action(resp) for resp in responses]

    def _extract_instruction(self, text: str) -> str:
        match = re.search(r"Instruction:\s*(.*?)(?:\n\s*Previous actions:|\Z)", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"## User Instruction\s*(.*?)(?:\n\s*##|\Z)", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _build_step_messages(self, system_prompt: str, instruction: str, screenshots: list[str], responses: list[str], actions: list[str], step_idx: int) -> list[dict]:
        total_steps = step_idx + 1
        if step_idx >= len(screenshots):
            raise IndexError(f"step {step_idx} requires screenshot index {step_idx}, only {len(screenshots)} screenshots found")

        folded_prefix_k = self._folded_prefix(total_steps)
        start_step = max(1, total_steps - self.history_n)
        previous_actions_str = self._previous_actions_text(actions, start_step)
        instruction_prompt = self._build_instruction_prompt(instruction, previous_actions_str)

        messages = [{"role": "system", "content": [_content_text(system_prompt)]}]
        for step_num in range(start_step, total_steps + 1):
            is_first_turn = step_num == start_step
            is_collapsed = step_num <= folded_prefix_k
            if is_collapsed:
                if is_first_turn:
                    user_content = [_content_text(instruction_prompt)]
                else:
                    user_content = self._wrap_tool_response([_content_text(self.collapse_text)])
            else:
                image_part = _content_image(screenshots[step_num - 1])
                if is_first_turn:
                    user_content = [image_part, _content_text(instruction_prompt)]
                else:
                    user_content = self._wrap_tool_response([image_part])
            messages.append({"role": "user", "content": user_content})

            if step_num <= total_steps - 1 and (step_num - 1) < len(responses):
                messages.append({"role": "assistant", "content": [_content_text(responses[step_num - 1])]})
        return messages

    def _folded_prefix(self, total_screenshots: int) -> int:
        folded_prefix_k = 0
        while (total_screenshots - folded_prefix_k) > self.image_max:
            folded_prefix_k += self.fold_size
        return min(folded_prefix_k, total_screenshots)

    def _previous_actions_text(self, actions: list[str], start_step: int) -> str:
        previous_actions = [
            f"Step {i + 1}: {actions[i]}"
            for i in range(0, min(start_step - 1, len(actions)))
        ]
        return "\n".join(previous_actions) if previous_actions else "None"

    def _wrap_tool_response(self, parts: list[dict]) -> list[dict]:
        return [_content_text("<tool_response>\n")] + parts + [_content_text("\n</tool_response>")]

    def _build_instruction_prompt(self, instruction: str, previous_actions: str) -> str:
        return (
            "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            "Previous actions:\n"
            f"{previous_actions}"
        )

    def _extract_action_line(self, response: str) -> str:
        for line in (response or "").split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("action:"):
                return stripped.split(":", 1)[-1].strip()
        return ""

    def _compact_previous_action(self, response: str) -> str:
        return self._extract_action_line(response) or re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL | re.IGNORECASE).strip()

    def _build_return_rewards(self, reward: float, process_rewards: torch.Tensor | None):
        if process_rewards is None:
            return None
        pr_baseline = 0.5
        or_scale = 10.0
        gamma = 0.99
        returns = [0.0] * len(process_rewards)
        future_return = 0.0
        for t in reversed(range(len(process_rewards))):
            or_val = reward if t == len(process_rewards) - 1 else 0.0
            r_t = (process_rewards[t] - pr_baseline) + (or_val * or_scale)
            future_return = r_t + gamma * future_return
            returns[t] = future_return
        return returns

    def _locate_context(self, messages: list[dict]):
        return len(messages) - 1

    def _get_inputs_from_pt(self, messages: list[dict], dataset_id: str, prompt_ids: torch.Tensor | None, dataset_path: str | None = None):
        context_len = self._locate_context(messages)
        images = self._load_images(messages[:context_len], dataset_id, dataset_path)
        multi_modal_data = {"image": images}

        raw_prompt = self.processor.apply_chat_template(messages[:context_len], add_generation_prompt=True, tokenize=False)
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

        if prompt_ids is not None and torch.is_tensor(prompt_ids) and prompt_ids.numel() > 0:
            model_inputs.pop("input_ids", None)
            model_inputs.pop("attention_mask", None)
            input_ids = prompt_ids.long().unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = [
            get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
        ]
        return input_ids, attention_mask, position_ids, multi_modal_data, model_inputs

    def _get_responses_from_pt(self, messages: list[dict], dataset_id: str, model_inputs: dict, position_ids, response_ids, attention_mask, dataset_path: str | None = None):
        input_ids = response_ids.long().unsqueeze(0)
        response = verl_F.pad_2d_list_to_length(input_ids, self.tokenizer.pad_token_id, max_length=self.max_response_length)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1)
        delta_position_id = delta_position_id.unsqueeze(0).expand(1, -1)
        if position_ids[0].dim() == 3:
            delta_position_id = delta_position_id.view(1, 1, -1).expand(1, 3, -1)
        response_position_ids = position_ids[0][..., -1:] + delta_position_id
        response_attention_mask = verl_F.get_response_mask(
            response_id=response,
            eos_token=self.tokenizer.eos_token_id,
            dtype=attention_mask.dtype,
        )
        return response, response_attention_mask, [response_position_ids], None, None

    def _align_rollout_logp(self, rollout_log_prob, valid_response_length, dataset_id: str, step_idx: int):
        valid_len = int(valid_response_length.item() if torch.is_tensor(valid_response_length) else valid_response_length)
        if rollout_log_prob is None:
            rollout_log_prob = torch.tensor([])
        rollout_log_prob = rollout_log_prob.float()
        if rollout_log_prob.shape[0] != valid_len:
            print(
                f"[Qwen3Splitter][ERROR] rollout_log_prob length {rollout_log_prob.shape[0]} "
                f"does not match valid_response_length {valid_len} for {dataset_id} step {step_idx}; using zeros."
            )
            rollout_log_prob = torch.zeros(valid_len, dtype=torch.float32)
        return verl_F.pad_2d_list_to_length(rollout_log_prob.unsqueeze(0), 0, max_length=self.max_response_length)

    def _load_images(self, messages: list[dict], dataset_id: str, dataset_path: str | None):
        image_refs = []
        for msg in messages:
            image_refs.extend(_collect_image_refs(msg))
        return [self._open_image(ref, dataset_id, dataset_path) for ref in image_refs]

    def _open_image(self, ref: str, dataset_id: str, dataset_path: str | None):
        if ref.startswith("data:image"):
            import base64
            from io import BytesIO
            raw = ref.split(",", 1)[1]
            image = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
        else:
            root = dataset_path if dataset_path is not None else os.path.join(self.root_dir, dataset_id)
            path = ref if os.path.isabs(ref) else os.path.join(root, ref)
            if not os.path.exists(path):
                dirname = os.path.dirname(path)
                basename = re.sub(r"(image_)0*([0-9]+)(\.\w+)$", r"\1\2\3", os.path.basename(path))
                alt = os.path.join(dirname, basename)
                if os.path.exists(alt):
                    path = alt
            image = Image.open(path).convert("RGB")

        width, height = image.size
        new_width, new_height = _smart_resize_size(width, height)
        if (new_width, new_height) != (width, height):
            image = image.resize((new_width, new_height))
        return image

    def _image_token_text(self, n_images: int) -> str:
        image_token = getattr(self.processor, "image_token", None) or getattr(self.tokenizer, "image_token", None) or "<|image_pad|>"
        return "\n".join([image_token] * n_images) if n_images else ""

    def compute_mask(self, row_dict: dict):
        input_ids = row_dict["input_ids"]
        response_mask = torch.zeros_like(input_ids)
        attention_mask = row_dict["attention_mask"]
        valid_response_length = attention_mask[self.max_prompt_length:].sum()
        response_mask[self.max_prompt_length:self.max_prompt_length + valid_response_length] = 1
        row_dict["response_mask"] = response_mask[self.max_prompt_length:]
        row_dict["loss_mask"] = response_mask

    def _build_default_system_prompt(self) -> str:
        tools_def = {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": self._build_description_prompt(),
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": self._internal_action_description(),
                            "enum": ["key", "key_down", "key_up", "left_mouse_down", "left_mouse_up", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "triple_click", "scroll", "hscroll", "screenshot", "wait", "terminate", "call_user"],
                        },
                        "keys": {"type": "array", "description": "Required only by `action=key`, `action=key_down`, or `action=key_up`."},
                        "text": {"type": "string", "description": "Required only by `action=type` and `action=call_user`."},
                        "coordinate": {"type": "array", "description": "(x, y) coordinates. Required only by `action=mouse_move` and `action=left_click_drag`, optional for `action=left_mouse_down` and `action=left_mouse_up`."},
                        "pixels": {"type": "number", "description": "Scroll amount. Required only by `action=scroll` or `action=hscroll`."},
                        "time": {"type": "number", "description": "Seconds to wait. Required only by `action=wait`."},
                        "status": {"type": "string", "description": "Task status for terminate.", "enum": ["success", "failure"]},
                    },
                },
            },
        }
        return (
            "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
            "# Tools\n\n"
            "You have access to the following functions:\n\n"
            "<tools>\n"
            + json.dumps(tools_def, ensure_ascii=False)
            + "\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "- Collapsed screenshots appear as text: This screenshot has been collapsed.\n"
            "</IMPORTANT>\n\n"
            "# Response format\n\n"
            "For normal UI interaction steps:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block.\n\n"
            "For terminal steps, you may either:\n"
            "- output a final natural-language response with no tool call, or\n"
            "- use a terminal tool call such as call_user or terminate.\n\n"
            "Rules:\n"
            "- For non-terminal UI steps, output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything after a tool call.\n"
            "- Use call_user when you need user information or confirmation.\n"
            "- Use terminate when you want to explicitly end the task with a success or failure status.\n"
            "- If the task is infeasible, say so explicitly in the response."
        )

    def _build_description_prompt(self) -> str:
        return "\n".join([
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            "* The screen's resolution is 1000x1000.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ])

    def _internal_action_description(self) -> str:
        return """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and hold a single key without releasing it.
* `key_up`: Release a previously held single key.
* `left_mouse_down`: Press and hold the left mouse button.
* `left_mouse_up`: Release the left mouse button.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `triple_click`: Triple-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `screenshot`: Capture a new screenshot of the current screen.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `call_user`: Ask user for information or confirmation.
""".strip()
