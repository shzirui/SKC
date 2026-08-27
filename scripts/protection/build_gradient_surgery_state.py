#!/usr/bin/env python3
"""
Build a gradient-surgery state file from protection artifacts.

Supported input directory layouts:
    Old protection layout:
        protected_neurons.json or layer20_protection_matrices_meta.json
        layer20_protection_matrices.pt

    dSVD layout:
        layer20_dsvd_meta.json
        layer20_C_union.pt
        layer20_down_proj_Hhist.pt
        layer20_gate_proj_Hhist.pt
        layer20_up_proj_Hhist.pt
        layer20_singular_values.pt

Output format:
    {
        "protected_indices": list[int],
        "directions": {
            "down_proj": Tensor(hidden_size, num_dirs, num_protected) or Tensor(hidden_size, num_protected),
            "gate_proj": Tensor(hidden_size, num_dirs, num_protected) or Tensor(hidden_size, num_protected),
            "up_proj": Tensor(hidden_size, num_dirs, num_protected) or Tensor(hidden_size, num_protected),
        },
    }
"""

import argparse
import json
from pathlib import Path

torch = None


PROJ_KEYS = ("down_proj", "gate_proj", "up_proj")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert protection or dSVD files into gradient_surgery_state.pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/data/merge/gimp_protection"),
        help="Directory containing protection metadata or dSVD artifacts.",
    )
    parser.add_argument("--neurons-json", type=Path, default=None, help="Override protection metadata JSON path.")
    parser.add_argument("--matrices-pt", type=Path, default=None, help="Override layer20_protection_matrices.pt path.")
    parser.add_argument("--output-state", type=Path, default=None, help="Output .pt path.")
    parser.add_argument("--hidden-size", type=int, default=3584, help="Expected direction vector size.")
    return parser.parse_args()


def load_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "neuron_indices" in data:
        indices = [int(idx) for idx in data["neuron_indices"]]
        if not indices:
            raise ValueError(f"Expected non-empty 'neuron_indices' list in {path}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate protected indices found in {path}")
        return indices, [], data

    neurons = data.get("neurons")
    if not isinstance(neurons, list) or not neurons:
        raise ValueError(f"Expected non-empty 'neurons' or 'neuron_indices' list in {path}")

    indices = []
    activations = []
    for item in neurons:
        if "index" not in item:
            raise ValueError(f"Neuron entry missing 'index' in {path}: {item}")
        indices.append(int(item["index"]))
        if "mean_abs_activation" in item:
            activations.append(float(item["mean_abs_activation"]))

    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate protected indices found in {path}")
    return indices, activations, data


def validate_directions(matrices, num_protected, hidden_size, source):
    if not isinstance(matrices, dict):
        raise ValueError(f"Expected {source} to contain a dict with {PROJ_KEYS}.")

    directions = {}
    for key in PROJ_KEYS:
        if key not in matrices:
            raise ValueError(f"Missing '{key}' in {source}")
        tensor = matrices[key]
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor)
        if tensor.dim() != 2:
            raise ValueError(f"Expected {key} to be 2D, got shape {tuple(tensor.shape)}")

        if tensor.shape == (hidden_size, num_protected):
            directions[key] = tensor.float().contiguous()
        elif tensor.shape == (num_protected, hidden_size):
            directions[key] = tensor.t().float().contiguous()
        else:
            raise ValueError(
                f"Expected {key} shape ({hidden_size}, {num_protected}) or "
                f"({num_protected}, {hidden_size}), got {tuple(tensor.shape)}"
            )
    return directions


def _tensor_to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _load_optional_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_dsvd_direction(tensor, num_protected, hidden_size, source):
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    if tensor.dim() != 3:
        raise ValueError(f"Expected {source} to be 3D, got shape {tuple(tensor.shape)}")

    if tensor.shape[0] == hidden_size and tensor.shape[2] == num_protected:
        return tensor.float().contiguous()
    if tensor.shape[2] == hidden_size and tensor.shape[0] == num_protected:
        return tensor.permute(2, 1, 0).float().contiguous()

    raise ValueError(
        f"Expected {source} shape ({hidden_size}, num_dirs, {num_protected}) or "
        f"({num_protected}, num_dirs, {hidden_size}), got {tuple(tensor.shape)}"
    )


def _build_dsvd_state(args, output_state):
    union_path = args.input_dir / "layer20_C_union.pt"
    meta_path = args.input_dir / "layer20_dsvd_meta.json"
    singular_values_path = args.input_dir / "layer20_singular_values.pt"

    indices_tensor = torch.load(union_path, map_location="cpu")
    if not torch.is_tensor(indices_tensor):
        indices_tensor = torch.tensor(indices_tensor)
    indices = [int(idx) for idx in indices_tensor.flatten().tolist()]
    if not indices:
        raise ValueError(f"Expected non-empty protected index tensor in {union_path}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate protected indices found in {union_path}")

    directions = {}
    for key in PROJ_KEYS:
        hist_path = args.input_dir / f"layer20_{key}_Hhist.pt"
        if not hist_path.exists():
            raise FileNotFoundError(f"Missing dSVD direction file: {hist_path}")
        directions[key] = _validate_dsvd_direction(
            torch.load(hist_path, map_location="cpu"),
            len(indices),
            args.hidden_size,
            hist_path,
        )

    dsvd_meta = _load_optional_json(meta_path)
    metadata = {
        "format": "dsvd_subspace",
        "source_dsvd_meta": str(meta_path) if meta_path.exists() else None,
        "source_union_indices": str(union_path),
        "source_singular_values": str(singular_values_path) if singular_values_path.exists() else None,
        "layer": dsvd_meta.get("layer"),
        "task_names": dsvd_meta.get("task_order"),
        "num_protected": len(indices),
        "num_directions": int(next(iter(directions.values())).shape[1]),
    }

    if singular_values_path.exists():
        sv_state = torch.load(singular_values_path, map_location="cpu")
        if isinstance(sv_state, dict):
            for key in ("task_names", "membership", "valid_update_counts", "ranks", "singular_values"):
                if key in sv_state:
                    metadata[key] = _tensor_to_list(sv_state[key])

    state = {
        "protected_indices": indices,
        "directions": directions,
        "metadata": metadata,
    }

    output_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_state)

    print(
        f"Saved {output_state}: format=dsvd_subspace, protected={len(indices)}, "
        f"direction_shape={tuple(next(iter(directions.values())).shape)}"
    )


def _build_legacy_state(args, output_state):
    if args.neurons_json is not None:
        neurons_json = args.neurons_json
    else:
        protected_neurons_json = args.input_dir / "protected_neurons.json"
        matrices_meta_json = args.input_dir / "layer20_protection_matrices_meta.json"
        neurons_json = protected_neurons_json if protected_neurons_json.exists() else matrices_meta_json
    matrices_pt = args.matrices_pt or args.input_dir / "layer20_protection_matrices.pt"

    indices, activations, neurons_meta = load_indices(neurons_json)
    matrices = torch.load(matrices_pt, map_location="cpu")
    directions = validate_directions(matrices, len(indices), args.hidden_size, matrices_pt)

    state = {
        "protected_indices": indices,
        "directions": directions,
        "metadata": {
            "format": "legacy_single_direction",
            "source_neurons_json": str(neurons_json),
            "source_matrices_pt": str(matrices_pt),
            "layer": neurons_meta.get("layer"),
            "source": neurons_meta.get("source"),
            "order": neurons_meta.get("order"),
            "num_protected": len(indices),
            "num_directions": 1,
        },
    }
    if activations:
        state["metadata"]["mean_abs_activation"] = activations

    output_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_state)

    print(
        f"Saved {output_state}: format=legacy_single_direction, protected={len(indices)}, "
        f"direction_shape={tuple(next(iter(directions.values())).shape)}"
    )


def main():
    global torch
    args = parse_args()
    import torch as torch_module

    torch = torch_module

    output_state = args.output_state or args.input_dir / "gradient_surgery_state.pt"
    dsvd_files = [
        args.input_dir / "layer20_C_union.pt",
        args.input_dir / "layer20_down_proj_Hhist.pt",
        args.input_dir / "layer20_gate_proj_Hhist.pt",
        args.input_dir / "layer20_up_proj_Hhist.pt",
    ]
    if args.matrices_pt is None and all(path.exists() for path in dsvd_files):
        _build_dsvd_state(args, output_state)
    else:
        _build_legacy_state(args, output_state)


if __name__ == "__main__":
    main()
