#!/usr/bin/env python3
"""
Inspect module names relevant to gradient surgery.

Example:
    python scripts/inspect_gradient_surgery_modules.py \
        --model-path /path/to/model \
        --layer 20 \
        --trust-remote-code
"""

import argparse


def current_layer_match(module_name, layer_idx):
    pattern = f"layers.{layer_idx}"
    return module_name == pattern or module_name.endswith(f".{pattern}") or f"{pattern}." in module_name


def inspect_module_names(model, layer_idx=20, limit=300):
    names = [name for name, _ in model.named_modules()]
    layer_token = f"layers.{layer_idx}"

    print(f"Total modules: {len(names)}")
    print()

    current_rule_matches = [name for name in names if current_layer_match(name, layer_idx)]
    print(f"Current dp_actor layer-match rule hits: {len(current_rule_matches)}")
    for name in current_rule_matches[:limit]:
        print(f"  {name}")
    if len(current_rule_matches) > limit:
        print(f"  ... {len(current_rule_matches) - limit} more")
    print()

    layer_candidates = [name for name in names if layer_token in name]
    print(f"Names containing '{layer_token}': {len(layer_candidates)}")
    for name in layer_candidates[:limit]:
        print(f"  {name}")
    if len(layer_candidates) > limit:
        print(f"  ... {len(layer_candidates) - limit} more")
    print()

    proj_candidates = [
        name
        for name in names
        if layer_token in name and any(name.endswith(f"mlp.{proj}") for proj in ("down_proj", "gate_proj", "up_proj"))
    ]
    print(f"Layer {layer_idx} MLP projection module candidates: {len(proj_candidates)}")
    for name in proj_candidates:
        print(f"  {name}")
    print()

    down_hook_candidates = [name for name in proj_candidates if name.endswith("mlp.down_proj")]
    print(f"Down-proj hook candidates: {len(down_hook_candidates)}")
    for name in down_hook_candidates:
        print(f"  {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a model and print module names relevant to gradient surgery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model path.")
    parser.add_argument("--layer", type=int, default=20, help="Layer index to inspect.")
    parser.add_argument("--limit", type=int, default=300, help="Maximum names to print per section.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    parser.add_argument(
        "--model-class",
        choices=("auto", "causal-lm", "seq2seq-lm", "vision2seq"),
        default="auto",
        help="AutoModel class to use.",
    )
    return parser.parse_args()


def load_model(args):
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForVision2Seq

    kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": True,
    }
    if args.model_class == "causal-lm":
        cls = AutoModelForCausalLM
    elif args.model_class == "seq2seq-lm":
        cls = AutoModelForSeq2SeqLM
    elif args.model_class == "vision2seq":
        cls = AutoModelForVision2Seq
    else:
        cls = AutoModel
    return cls.from_pretrained(args.model_path, **kwargs)


def main():
    args = parse_args()
    model = load_model(args)
    inspect_module_names(model, layer_idx=args.layer, limit=args.limit)


if __name__ == "__main__":
    main()
