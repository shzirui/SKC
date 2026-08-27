#!/usr/bin/env python3
"""
Merge gradient-surgery memory after a training stage.

The output file is the single state consumed by
`gradient_surgery_state_path` in `verl/workers/actor/dp_actor.py`:

    {
        "protected_indices": list[int],
        "directions": {
            "down_proj": Tensor,  # shape: (hidden_size, len(protected_indices))
            "gate_proj": Tensor,
            "up_proj": Tensor,
        },
    }

Current-stage input can either be:
1. A full activation/direction dump:
       {
           "x_curr": Tensor[intermediate_size],
           "directions": {
               "down_proj": Tensor[hidden_size, intermediate_size],
               "gate_proj": Tensor[hidden_size, intermediate_size],
               "up_proj": Tensor[hidden_size, intermediate_size],
           },
       }
   The script selects top-k/top-ratio channels from x_curr.
2. A preselected state:
       {
           "protected_indices": [...],
           "directions": {
               "down_proj": Tensor[hidden_size, n],
               "gate_proj": Tensor[hidden_size, n],
               "up_proj": Tensor[hidden_size, n],
           },
       }
   The script uses those indices directly.
"""

import argparse
from pathlib import Path

torch = None


INDEX_KEYS = ("protected_indices", "indices", "C_hist", "C_curr")
ACTIVATION_KEYS = ("x_curr", "activation_mean", "activation_scores", "activations")
PROJ_KEYS = ("down_proj", "gate_proj", "up_proj")


def _get_first_key(state, keys):
    for key in keys:
        if key in state:
            return key, state[key]
    return None, None


def _as_index_list(value, source):
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten().tolist()
    return [int(idx) for idx in value]


def _normalize_columns(tensor):
    tensor = tensor.float().contiguous()
    return tensor / tensor.norm(dim=0, keepdim=True).clamp_min(1e-12)


def _select_direction_columns(direction, indices, hidden_size, intermediate_size, source):
    if not torch.is_tensor(direction):
        direction = torch.tensor(direction)
    if direction.dim() != 2:
        raise ValueError(f"Expected direction tensor in {source} to be 2D, got {tuple(direction.shape)}")

    index_tensor = torch.tensor(indices, dtype=torch.long)
    max_idx = int(index_tensor.max().item()) if indices else -1

    if direction.shape == (hidden_size, intermediate_size):
        selected = direction[:, index_tensor]
    elif direction.shape == (intermediate_size, hidden_size):
        selected = direction[index_tensor, :].t()
    elif direction.shape == (hidden_size, len(indices)):
        selected = direction
    elif direction.shape == (len(indices), hidden_size):
        selected = direction.t()
    elif direction.shape[0] == hidden_size and direction.shape[1] > max_idx:
        selected = direction[:, index_tensor]
    elif direction.shape[1] == hidden_size and direction.shape[0] > max_idx:
        selected = direction[index_tensor, :].t()
    else:
        raise ValueError(
            f"Cannot align direction shape {tuple(direction.shape)} from {source} "
            f"with hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
            f"and {len(indices)} indices."
        )

    return _normalize_columns(selected)


def _get_directions_dict(state, source):
    directions = state.get("directions")
    if not isinstance(directions, dict):
        raise ValueError(f"{source} must contain a 'directions' dict with {PROJ_KEYS}.")
    for proj_name in PROJ_KEYS:
        if proj_name not in directions:
            raise ValueError(f"Missing directions['{proj_name}'] in {source}.")
    return directions


def _load_selected_state(path, hidden_size, intermediate_size):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected {path} to contain a dict.")

    index_key, indices = _get_first_key(state, INDEX_KEYS)
    if index_key is None:
        raise ValueError(f"No protected/current indices found in {path}. Expected one of {INDEX_KEYS}.")

    indices = _as_index_list(indices, path)
    direction_state = _get_directions_dict(state, path)
    directions = {
        proj_name: _select_direction_columns(direction_state[proj_name], indices, hidden_size, intermediate_size, f"{path}:{proj_name}")
        for proj_name in PROJ_KEYS
    }
    return indices, directions


def _load_current_state(path, hidden_size, intermediate_size, top_k, top_ratio):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected {path} to contain a dict.")

    activation_key, activation = _get_first_key(state, ACTIVATION_KEYS)
    index_key, indices = _get_first_key(state, INDEX_KEYS)
    direction_state = _get_directions_dict(state, path)

    if activation_key is not None:
        if not torch.is_tensor(activation):
            activation = torch.tensor(activation)
        activation = activation.float().flatten()
        if activation.numel() != intermediate_size:
            raise ValueError(
                f"Expected activation vector in {path} to have {intermediate_size} elements, "
                f"got {activation.numel()}."
            )

        if top_k is None:
            if top_ratio is None:
                raise ValueError("Either --top-k or --top-ratio is required when current state contains activations.")
            top_k = int(intermediate_size * top_ratio)
        top_k = max(1, min(intermediate_size, int(top_k)))
        indices = torch.topk(activation, k=top_k).indices.detach().cpu().tolist()
    elif index_key is not None:
        indices = _as_index_list(indices, path)
    else:
        raise ValueError(
            f"{path} must contain either activations {ACTIVATION_KEYS} or preselected indices {INDEX_KEYS}."
        )

    directions = {
        proj_name: _select_direction_columns(direction_state[proj_name], indices, hidden_size, intermediate_size, f"{path}:{proj_name}")
        for proj_name in PROJ_KEYS
    }
    return indices, directions


def _merge_states(prev_indices, prev_directions, curr_indices, curr_directions, overwrite_existing):
    merged_indices = []
    merged_columns = {proj_name: [] for proj_name in PROJ_KEYS}
    index_to_pos = {}

    def add_or_update(idx, directions):
        if idx in index_to_pos:
            if overwrite_existing:
                for proj_name in PROJ_KEYS:
                    merged_columns[proj_name][index_to_pos[idx]] = directions[proj_name]
            return
        index_to_pos[idx] = len(merged_indices)
        merged_indices.append(idx)
        for proj_name in PROJ_KEYS:
            merged_columns[proj_name].append(directions[proj_name])

    if prev_indices is not None:
        for pos, idx in enumerate(prev_indices):
            add_or_update(idx, {proj_name: prev_directions[proj_name][:, pos] for proj_name in PROJ_KEYS})

    for pos, idx in enumerate(curr_indices):
        add_or_update(idx, {proj_name: curr_directions[proj_name][:, pos] for proj_name in PROJ_KEYS})

    merged_directions = {
        proj_name: _normalize_columns(torch.stack(merged_columns[proj_name], dim=1))
        for proj_name in PROJ_KEYS
    }
    return merged_indices, merged_directions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge gradient surgery protected indices and direction vectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--current-state", required=True, type=Path, help="Current-stage activation/direction or selected state file.")
    parser.add_argument("--output-state", required=True, type=Path, help="Output state path for the next training stage.")
    parser.add_argument("--prev-state", default=None, type=Path, help="Previous gradient_surgery_state_path file.")
    parser.add_argument("--top-ratio", default=None, type=float, help="Top activation ratio used when current-state has x_curr.")
    parser.add_argument("--top-k", default=None, type=int, help="Top activation count used when current-state has x_curr.")
    parser.add_argument("--hidden-size", default=3584, type=int, help="Model hidden size.")
    parser.add_argument("--intermediate-size", default=18944, type=int, help="MLP intermediate size.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Use current direction for indices that already exist in prev-state. By default old directions are kept.",
    )
    return parser.parse_args()


def main():
    global torch
    args = parse_args()
    import torch as torch_module
    torch = torch_module

    if args.top_ratio is not None and args.top_ratio < 0:
        raise ValueError("--top-ratio must be non-negative.")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    prev_indices = None
    prev_directions = None
    if args.prev_state is not None:
        prev_indices, prev_directions = _load_selected_state(args.prev_state, args.hidden_size, args.intermediate_size)

    curr_indices, curr_directions = _load_current_state(
        args.current_state,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.top_ratio,
    )
    merged_indices, merged_directions = _merge_states(
        prev_indices,
        prev_directions,
        curr_indices,
        curr_directions,
        args.overwrite_existing,
    )

    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protected_indices": merged_indices,
            "directions": {proj_name: merged_directions[proj_name].cpu() for proj_name in PROJ_KEYS},
        },
        args.output_state,
    )

    prev_count = 0 if prev_indices is None else len(prev_indices)
    print(
        f"Saved {args.output_state}: prev={prev_count}, current={len(curr_indices)}, "
        f"merged={len(merged_indices)}"
    )


if __name__ == "__main__":
    main()
