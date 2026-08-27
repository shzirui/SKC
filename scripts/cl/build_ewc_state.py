#!/usr/bin/env python3
"""
Build an EWC state file from a reference model and old-task trajectories.

The output is consumed by actor_rollout_ref.actor.use_ewc=True training:
    {
        "theta_star": {parameter_name: CPU tensor},
        "fisher": {parameter_name: CPU tensor},
        "metadata": {...},
    }

Fisher is estimated as the diagonal empirical Fisher:
    F_i ~= mean_batch((d loss / d theta_i)^2)
where loss is next-token negative log likelihood on assistant outputs from old-task trajectories.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", required=True, type=Path, help="HF model/checkpoint directory used as theta_star.")
    parser.add_argument("--trajectory-root", type=Path, default=None, help="Root directory containing trajectory subdirectories.")
    parser.add_argument("--trajectory-list", type=Path, default=None, help="txt/json/jsonl file listing trajectory directories or ids.")
    parser.add_argument("--output-state", required=True, type=Path, help="Output ewc_state.pt path.")
    parser.add_argument("--max-samples", type=int, default=128, help="Maximum trajectories to use for Fisher estimation.")
    parser.add_argument("--max-length", type=int, default=32768, help="Maximum token length for NLL estimation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--trainable-only", action="store_true", help="Only save params that require grad after loading the model.")
    parser.add_argument("--assistant-only", action="store_true", default=True, help="Compute Fisher only on assistant message spans.")
    parser.add_argument("--param-name-contains", nargs="*", default=None, help="Only compute/save EWC state for parameters whose names contain any of these substrings.")
    return parser.parse_args()


def _dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_model(model_path: Path, torch_dtype, trust_remote_code: bool):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    architectures = getattr(config, "architectures", []) or []
    model_cls = AutoModelForCausalLM
    if architectures and "ForConditionalGeneration" in architectures[0]:
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except Exception:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq
    model = model_cls.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
    return model


def _read_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_trajectory_dirs(root: Path | None, list_path: Path | None) -> Iterable[Path]:
    if list_path is None:
        if root is None:
            raise ValueError("Either --trajectory-root or --trajectory-list is required")
        for path in sorted(root.iterdir()):
            if path.is_dir() and (path / "final_messages.json").exists():
                yield path
        return

    if list_path.suffix in {".json", ".jsonl"}:
        rows = _read_json_or_jsonl(list_path)
        if isinstance(rows, dict):
            rows = rows.get("trajectories", rows.get("data", []))
        for row in rows:
            if isinstance(row, str):
                value = row
            else:
                value = row.get("path") or row.get("trajectory_path") or row.get("trajectory_id") or row.get("dataset_id")
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                if root is None:
                    raise ValueError(f"Relative trajectory entry needs --trajectory-root: {value}")
                path = root / value
            if (path / "final_messages.json").exists():
                yield path
        return

    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value)
            if not path.is_absolute():
                if root is None:
                    raise ValueError(f"Relative trajectory entry needs --trajectory-root: {value}")
                path = root / value
            if (path / "final_messages.json").exists():
                yield path


def _resolve_image_path(trajectory_dir: Path, image_value: str) -> Path:
    if image_value.endswith(".png") or image_value.endswith(".jpg") or image_value.endswith(".jpeg"):
        return trajectory_dir / image_value
    return trajectory_dir / f"{image_value}.png"


def _load_messages_and_images(trajectory_dir: Path):
    with open(trajectory_dir / "final_messages.json", "r", encoding="utf-8") as f:
        messages = json.load(f)

    image_paths = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                value = item.get("image")
                if value:
                    image_paths.append(_resolve_image_path(trajectory_dir, str(value)))

    images = []
    for path in image_paths:
        if path.exists():
            images.append(Image.open(path).convert("RGB"))
    return messages, images


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    parts.append("<image>")
                elif item.get("type") == "video":
                    parts.append("<video>")
        return "".join(parts)
    return str(content)


def _encode_with_processor(processor, tokenizer, messages, images, device: str, max_length: int, add_generation_prompt: bool = False):
    if processor is not None and hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
        kwargs = {"text": [text], "return_tensors": "pt", "padding": False, "truncation": True, "max_length": max_length}
        if images:
            kwargs["images"] = images
        inputs = processor(**kwargs)
    else:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def _build_assistant_labels(processor, tokenizer, messages, images, device: str, max_length: int):
    inputs = _encode_with_processor(processor, tokenizer, messages, images, device, max_length, add_generation_prompt=False)
    input_ids = inputs["input_ids"]
    labels = torch.full_like(input_ids, -100)
    attention_mask = inputs.get("attention_mask")

    assistant_spans = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        assistant_text = _message_content_to_text(msg.get("content", ""))
        if not assistant_text.strip():
            continue

        prefix_messages = messages[:idx]
        prefix_inputs = _encode_with_processor(processor, tokenizer, prefix_messages, images, device, max_length, add_generation_prompt=True)
        prefix_len = int(prefix_inputs["input_ids"].shape[-1])

        content_ids = tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"][0].to(device)
        content_len = int(content_ids.shape[-1])
        if content_len == 0:
            continue

        start = min(prefix_len, int(input_ids.shape[-1]))
        end = min(prefix_len + content_len, int(input_ids.shape[-1]))
        if end <= start:
            continue
        labels[0, start:end] = input_ids[0, start:end]
        assistant_spans.append((start, end))

    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    return inputs, labels, assistant_spans


def main():
    args = parse_args()
    torch_dtype = _dtype(args.torch_dtype)
    model = _load_model(args.model_path, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code)
    model.to(args.device)
    model.train()

    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    except Exception:
        processor = None
    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    param_filters = args.param_name_contains or []

    def _param_selected(name: str) -> bool:
        return not param_filters or any(token in name for token in param_filters)

    if param_filters:
        for name, param in model.named_parameters():
            param.requires_grad_(_param_selected(name))

    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if _param_selected(name) and (param.requires_grad or not args.trainable_only)
    ]
    if not named_params:
        raise ValueError(f"No parameters matched --param-name-contains={param_filters}")
    print(f"EWC parameter tensors: {len(named_params)}; filters={param_filters or 'ALL'}")
    theta_star = {name: param.detach().cpu().float().clone() for name, param in named_params}
    fisher = {name: torch.zeros_like(theta_star[name]) for name, _ in named_params}

    used = 0
    skipped = 0
    for trajectory_dir in _iter_trajectory_dirs(args.trajectory_root, args.trajectory_list):
        if used >= args.max_samples:
            break
        try:
            model.zero_grad(set_to_none=True)
            messages, images = _load_messages_and_images(trajectory_dir)
            inputs, labels, assistant_spans = _build_assistant_labels(processor, tokenizer, messages, images, args.device, args.max_length)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss: {loss}")
            loss.backward()
            for name, param in named_params:
                if param.grad is not None:
                    fisher[name].add_(param.grad.detach().cpu().float().pow(2))
            used += 1
            if used % 10 == 0:
                print(f"processed {used} trajectories")
        except Exception as exc:
            skipped += 1
            print(f"[skip] {trajectory_dir}: {exc}")

    if used == 0:
        raise RuntimeError("No trajectories were processed; cannot build EWC state")

    for name in fisher:
        fisher[name].div_(used)

    state = {
        "theta_star": theta_star,
        "fisher": fisher,
        "metadata": {
            "format": "diagonal_empirical_fisher_v1",
            "model_path": str(args.model_path),
            "trajectory_root": str(args.trajectory_root) if args.trajectory_root else None,
            "trajectory_list": str(args.trajectory_list) if args.trajectory_list else None,
            "num_samples": used,
            "num_skipped": skipped,
            "max_length": args.max_length,
            "trainable_only": args.trainable_only,
            "assistant_only": args.assistant_only,
            "param_name_contains": args.param_name_contains,
        },
    }
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output_state)
    print(f"Saved EWC state to {args.output_state}; samples={used}, skipped={skipped}, params={len(theta_star)}")


if __name__ == "__main__":
    main()
