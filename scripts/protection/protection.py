#!/usr/bin/env python3
"""Identify neurons with the largest mean absolute MLP activations."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import transformers
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ActivationRecorder:
    """Manage hooks that record MLP activations."""

    def __init__(self):
        self.activations = {}
        self.hooks = []
        self.layer_names = []

    def register_hooks(self, model: torch.nn.Module):
        """Register hooks that capture the input of every down projection."""
        target_modules = []
        for name, module in model.named_modules():
            if "language_model.layers." in name and ".mlp.down_proj" in name:
                try:
                    layer_num = int(name.split("language_model.layers.")[1].split(".")[0])
                    target_modules.append((layer_num, name, module))
                except (IndexError, ValueError):
                    continue

        target_modules.sort(key=lambda x: x[0])

        for layer_num, name, module in target_modules:
            self.layer_names.append(name)
            hook = module.register_forward_hook(self._make_hook_fn(name))
            self.hooks.append(hook)

        print(f"\nRegistered hooks on {len(self.layer_names)} modules:")
        print()

        return self.layer_names

    def _make_hook_fn(self, name: str):
        """Create a hook that captures a down-projection input."""
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                self.activations[name] = input[0].detach().cpu()
            else:
                self.activations[name] = input.detach().cpu()
        return hook_fn

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

    def clear_activations(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_names = []


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    dtype: str = "auto",
    load_in_8bit: bool = False,
    cpu_offload: bool = False,
):
    """Load a model and its tokenizer or processor."""
    print("Loading model: [configured]")
    print(f"Device: {device}, dtype: {dtype}")
    if load_in_8bit:
        print("Enabling 8-bit quantization")
    if cpu_offload:
        print("Enabling CPU offload")

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype.lower(), "auto")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = config.model_type if hasattr(config, 'model_type') else None
    print("Detected model type: [configured]")

    load_kwargs = {"trust_remote_code": True}

    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif cpu_offload:
        load_kwargs["device_map"] = "auto"
        load_kwargs["offload_folder"] = "offload"
    else:
        if not load_in_8bit:
            load_kwargs["torch_dtype"] = torch_dtype
        if device == "cpu":
            pass
        else:
            load_kwargs["device_map"] = "auto"
            print("Using device_map='auto' for automatic layer placement")

    try:
        architecture_names = getattr(config, "architectures", None) or []
        model_class = (
            getattr(transformers, architecture_names[0], AutoModelForCausalLM)
            if architecture_names
            else AutoModelForCausalLM
        )
        print("Loading the configured model architecture")
        model = model_class.from_pretrained(model_path, **load_kwargs)
    except Exception:
        print("Model loading failed")
        raise

    if device == "cpu" and "device_map" not in load_kwargs:
        model = model.to("cpu")

    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("Using AutoProcessor")
        tokenizer_or_processor = processor
    except Exception:
        print("Processor loading failed; trying a tokenizer")
        tokenizer_or_processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Using AutoTokenizer")

    print("Loaded model class: [configured]")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    return model, tokenizer_or_processor


def load_jsonl_samples(data_path: str, input_field: str = "text", max_samples: Optional[int] = None) -> List[Dict]:
    """Load samples from a JSONL file."""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and len(samples) >= max_samples:
                break
            sample = json.loads(line.strip())
            samples.append(sample)

    print(f"\nLoaded {len(samples)} samples from the configured dataset")
    print(f"Input field: {input_field}")

    if samples:
        first_sample = samples[0]
        if input_field in first_sample:
            print("Configured input field is available")
        else:
            print(f"Warning: field '{input_field}' is missing")

    return samples


def process_sample(
    model: torch.nn.Module,
    tokenizer_or_processor: Any,
    text: str,
    recorder: ActivationRecorder,
    device: str,
    max_length: int = 2048,
) -> Dict[str, np.ndarray]:
    """Return token-averaged absolute activations for one sample."""
    recorder.clear_activations()

    try:
        if hasattr(tokenizer_or_processor, 'tokenizer'):
            inputs = tokenizer_or_processor(
                text=[text],
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
        else:
            inputs = tokenizer_or_processor(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
    except Exception:
        print("Tokenization failed")
        inputs = tokenizer_or_processor(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        try:
            outputs = model(**inputs, output_hidden_states=False)
        except Exception:
            print("Forward pass failed")
            return {}

    activations = recorder.get_activations()
    token_averaged = {}

    for layer_name, activation in activations.items():
        avg_activation = activation.abs().mean(dim=1).squeeze(0)
        token_averaged[layer_name] = avg_activation.numpy()

    return token_averaged


def compute_dataset_mean_activations(
    model: torch.nn.Module,
    tokenizer_or_processor: Any,
    samples: List[Dict],
    recorder: ActivationRecorder,
    input_field: str,
    device: str,
    max_length: int = 2048,
) -> Dict[str, np.ndarray]:
    """Compute dataset-level mean absolute activations for every layer."""
    layer_names = recorder.layer_names
    activation_sums = {name: None for name in layer_names}
    activation_counts = {name: 0 for name in layer_names}

    print(f"\nProcessing {len(samples)} samples...")
    for sample in tqdm(samples, desc="Processing samples"):
        if input_field not in sample:
            print(f"Warning: sample has no '{input_field}' field; skipping")
            continue

        text = sample[input_field]
        if not isinstance(text, str) or not text.strip():
            print(f"Warning: sample field '{input_field}' is empty; skipping")
            continue

        try:
            token_averaged = process_sample(
                model, tokenizer_or_processor, text, recorder, device, max_length
            )
        except Exception:
            print("Sample processing failed")
            continue

        for layer_name, activation in token_averaged.items():
            if activation_sums[layer_name] is None:
                activation_sums[layer_name] = activation.copy()
            else:
                activation_sums[layer_name] += activation
            activation_counts[layer_name] += 1

    layer_mean_activations = {}
    for layer_name in layer_names:
        if activation_counts[layer_name] > 0:
            layer_mean_activations[layer_name] = (
                activation_sums[layer_name] / activation_counts[layer_name]
            )
        else:
            layer_mean_activations[layer_name] = None

    return layer_mean_activations


def identify_topk_neurons(mean_activation: np.ndarray, topk_ratio: float = 0.02):
    """Identify the neurons with the largest mean absolute activations."""
    top_k = max(1, int(len(mean_activation) * topk_ratio))
    act_tensor = torch.tensor(mean_activation)
    top_indices = torch.topk(act_tensor, k=min(top_k, len(act_tensor))).indices

    return {
        "topk_count": len(top_indices),
        "intermediate_dim": len(mean_activation),
        "indices": top_indices.tolist(),
        "values": act_tensor[top_indices].tolist(),
    }


def convert_to_json_serializable(obj):
    """Recursively convert an object to JSON-serializable values."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj


def analyze_and_save(
    layer_mean_activations: Dict[str, np.ndarray],
    model_path: str,
    num_samples: int,
    output_dir: str,
    topk_ratio: float = 0.02,
):
    """Analyze target-layer Top-K neurons and save the results."""
    os.makedirs(output_dir, exist_ok=True)

    target_layer_activations = {
        name: act
        for name, act in layer_mean_activations.items()
        if f".layers.{TARGET_LAYER}." in name
    }
    if not target_layer_activations:
        available = ", ".join(sorted(layer_mean_activations.keys()))
        raise ValueError(f"Layer {TARGET_LAYER} activations not found. Available layers: {available}")

    results = {
        "model": "",
        "num_samples": num_samples,
        "topk_ratio": topk_ratio,
        "target_layer": TARGET_LAYER,
        "num_layers": len(target_layer_activations),
        "layer_topk": {},
        "layer_stats": {},
    }

    print(f"\nLayer {TARGET_LAYER} Top-K neuron analysis:")
    print("-" * 80)

    protected_neurons = []
    for layer_name in sorted(target_layer_activations.keys()):
        mean_act = target_layer_activations[layer_name]
        if mean_act is None:
            print(f"Skipping {layer_name}: no data")
            continue

        topk = identify_topk_neurons(mean_act, topk_ratio)
        results["layer_topk"][layer_name] = topk

        stats = {
            "mean": float(mean_act.mean()),
            "std": float(mean_act.std()),
            "max": float(mean_act.max()),
            "min": float(mean_act.min()),
        }
        results["layer_stats"][layer_name] = stats

        print(f"\n{layer_name}:")
        print(f"  Dimension: {topk['intermediate_dim']}")
        print(f"  Top-K count: {topk['topk_count']} ({topk_ratio:.1%})")
        print(f"  Mean activation: {stats['mean']:.6f}; maximum: {stats['max']:.6f}")
        print("  Top 5 neurons:")
        for i in range(min(5, len(topk['indices']))):
            print(f"    Neuron {topk['indices'][i]:6d}: {topk['values'][i]:.6f}")

        protected_neurons = [
            {"index": int(index), "mean_abs_activation": float(value)}
            for index, value in zip(topk["indices"], topk["values"])
        ]

    summary = convert_to_json_serializable(results)
    json_path = os.path.join(output_dir, "protection_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {Path(json_path).name}")

    flat_path = os.path.join(output_dir, "topk_neurons.json")
    flat = {
        layer_name: data["indices"]
        for layer_name, data in results["layer_topk"].items()
    }
    with open(flat_path, 'w') as f:
        json.dump(flat, f, indent=2)
    print(f"Saved Top-K neuron indices: {Path(flat_path).name}")

    protected_path = os.path.join(output_dir, "protected_neurons.json")
    protected_payload = {
        "model": "",
        "layer": TARGET_LAYER,
        "source": "down_proj input mean absolute activation",
        "order": "mean_abs_activation_desc",
        "neurons": protected_neurons,
    }
    with open(protected_path, 'w') as f:
        json.dump(protected_payload, f, indent=2)
    print(f"Saved protected neurons for layer {TARGET_LAYER}: {Path(protected_path).name}")

    _plot_mean_activations(target_layer_activations, output_dir)

    _generate_report(results, output_dir)

    return results


def _plot_mean_activations(layer_mean_activations: Dict[str, np.ndarray], output_dir: str):
    """Plot mean absolute activation by layer."""
    layers = sorted(layer_mean_activations.keys())
    means = []
    for l in layers:
        act = layer_mean_activations[l]
        means.append(float(act.mean()) if act is not None else 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(means, marker='o', linewidth=2)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean Abs Activation', fontsize=12)
    ax.set_title('Mean Absolute Activation (down_proj input) per Layer', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "mean_abs_activation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {Path(plot_path).name}")
    plt.close()


def _generate_report(results: dict, output_dir: str):
    """Write a text report for the activation analysis."""
    report_path = os.path.join(output_dir, "report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MLP Top-K Neuron Analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {results['model']}\n")
        f.write(f"Samples: {results['num_samples']}\n")
        f.write(f"Top-K ratio: {results['topk_ratio']:.2%}\n")
        f.write(f"Analyzed layers: {results['num_layers']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Global statistics\n")
        f.write("-" * 80 + "\n\n")

        all_means = [s["mean"] for s in results["layer_stats"].values()]
        all_maxs  = [s["max"]  for s in results["layer_stats"].values()]
        f.write(f"Mean absolute activation: {np.mean(all_means):.6f} ± {np.std(all_means):.6f}\n")
        f.write(f"Mean maximum activation: {np.mean(all_maxs):.6f} ± {np.std(all_maxs):.6f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Top-K neurons by layer\n")
        f.write("-" * 80 + "\n\n")

        for layer_name in sorted(results["layer_topk"].keys()):
            topk = results["layer_topk"][layer_name]
            stats = results["layer_stats"][layer_name]
            f.write(f"{layer_name}:\n")
            f.write(f"  Dimension: {topk['intermediate_dim']}, Top-K count: {topk['topk_count']}\n")
            f.write(f"  Mean activation: {stats['mean']:.6f}, maximum: {stats['max']:.6f}\n")
            f.write("  Top 10 neurons (index: abs_mean_activation):\n")
            for i in range(min(10, len(topk['indices']))):
                f.write(f"    Neuron {topk['indices'][i]:6d}: {topk['values'][i]:.6f}\n")
            f.write("\n")

    print(f"Saved report: {Path(report_path).name}")


def main():
    """Parse command-line arguments and run neuron protection analysis."""
    parser = argparse.ArgumentParser(
        description="Extract Top-K neurons by absolute activation from down-projection inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model path")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Dataset path (.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Output subdirectory name")
    parser.add_argument("--input_field", type=str, default="content",
                        choices=["content", "text", "response"],
                        help="Sample field used as model input")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples")
    parser.add_argument("--topk_ratio", type=float, default=0.02,
                        help="Top-K neuron ratio")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Execution device")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "fp16", "float32", "fp32", "bfloat16", "bf16"],
                        help="Model data type")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load the model with 8-bit quantization")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload model layers to the CPU")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")

    args = parser.parse_args()

    def resolve_path(path_str: str) -> str:
        if not os.path.isabs(path_str):
            return os.path.join(str(PROJECT_ROOT), path_str)
        return path_str

    args.model     = resolve_path(args.model)
    args.data_path = resolve_path(args.data_path)
    args.output_dir = resolve_path(args.output_dir)

    print("=" * 80)
    print("MLP Top-K Neuron Extraction")
    print("=" * 80)
    print("Model: [configured]")
    print("Dataset: [configured]")
    print("Output name: [configured]")
    print(f"Top-K ratio: {args.topk_ratio}")
    print("=" * 80)

    full_output_dir = os.path.join(args.output_dir, args.output_name)
    os.makedirs(full_output_dir, exist_ok=True)

    samples = load_jsonl_samples(args.data_path, args.input_field, args.max_samples)
    if not samples:
        print("Error: no samples were loaded")
        return

    print("\n" + "=" * 80)
    print("Loading model")
    print("=" * 80)
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.device, args.dtype, args.load_in_8bit, args.cpu_offload
    )

    recorder = ActivationRecorder()
    recorder.register_hooks(model)

    layer_mean_activations = compute_dataset_mean_activations(
        model, tokenizer, samples, recorder, args.input_field, args.device, args.max_length
    )

    recorder.remove_hooks()
    del model
    del tokenizer
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Analyzing Top-K neurons")
    print("=" * 80)
    analyze_and_save(
        layer_mean_activations,
        args.model,
        len(samples),
        full_output_dir,
        args.topk_ratio,
    )

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print("\nOutput files:")
    print("  - protection_summary.json")
    print("  - topk_neurons.json")
    print("  - mean_abs_activation.png")
    print("  - report.txt")


if __name__ == "__main__":
    main()
