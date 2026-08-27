#!/usr/bin/env python3
"""Build per-neuron DSVD history directions for MLP projections."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

DEFAULT_TASKS: Tuple[Tuple[str, str], ...] = ()
KNOWN_ROUND_NAMES: Dict[int, Tuple[str, ...]] = {}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def resolve_path(path_str: str, root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = root / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-neuron SVD history direction tensors for "
            "gate_proj/up_proj/down_proj using historical task checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=str, default="", help="repository/workspace root")
    parser.add_argument("--model-root", type=str, default="", help="checkpoint root")
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=None,
        metavar="ROUND_DIR",
        help="explicit model directories for round0..roundN; must be one more than task count",
    )
    parser.add_argument(
        "--extra-task",
        action="append",
        default=[],
        metavar="NAME=PROTECTION_FOLDER",
        help="append a task specification",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="NAME=PROTECTION_FOLDER",
        help=(
            "use an explicit task list; repeat this option in history order"
        ),
    )
    parser.add_argument(
        "--protection-root",
        type=str,
        default="",
        help="directory containing task protection folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="directory for Hhist outputs",
    )
    parser.add_argument("--layer", type=int, help="MLP layer index")
    parser.add_argument("--num-directions", type=int, default=5, help="fixed number of SVD directions to save")
    parser.add_argument("--eps", type=float, default=1e-8, help="normalization and singular-value threshold")
    parser.add_argument(
        "--save-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="dtype used for saved Hhist tensors",
    )
    parser.add_argument(
        "--write-gradient-surgery-state",
        action="store_true",
        help="also write gradient_surgery_state.pt with protected_indices and all projection directions",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=None,
        help="directory for cached layer-neuron projection matrices",
    )
    parser.add_argument(
        "--cache-first-n-models",
        type=int,
        default=0,
        help="cache/read the first N model dirs from extracted layer-neuron matrices; later models read safetensors",
    )
    parser.add_argument(
        "--cache-model-indices",
        nargs="*",
        default=[],
        metavar="MODEL_INDEX",
        help=(
            "additional model indices to cache/read from extracted layer-neuron matrices; "
            "supports integers or comma-separated lists, for example 6 7 8 or 6,7,8"
        ),
    )
    parser.add_argument(
        "--refresh-model-cache",
        action="store_true",
        help="rebuild model cache files even when matching cache files already exist",
    )
    return parser.parse_args()


def save_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_protected_neurons(path: Path, layer: int) -> List[int]:
    data = load_json(path)

    if isinstance(data, dict) and "neurons" in data:
        neurons = data["neurons"]
        if not isinstance(neurons, list):
            raise ValueError("Protected-neuron data is not a list")
        return [int(item["index"] if isinstance(item, dict) else item) for item in neurons]

    if isinstance(data, dict):
        preferred = f"model.layers.{layer}.mlp.down_proj"
        language_model = f"model.language_model.layers.{layer}.mlp.down_proj"
        for key in (preferred, language_model):
            if key in data:
                return [int(index) for index in data[key]]
        matches = [value for key, value in data.items() if f".layers.{layer}." in key]
        if len(matches) == 1:
            return [int(index) for index in matches[0]]

    if isinstance(data, list):
        return [int(item["index"] if isinstance(item, dict) else item) for item in data]

    raise ValueError("Cannot parse protected neuron indices")


def parse_task_specs(task_specs: Sequence[str]) -> List[Tuple[str, str]]:
    parsed = []
    for spec in task_specs:
        if "=" in spec:
            name, folder = spec.split("=", 1)
        elif ":" in spec:
            name, folder = spec.split(":", 1)
        else:
            raise ValueError("Invalid task spec; expected NAME=PROTECTION_FOLDER")
        name = name.strip()
        folder = folder.strip()
        if not name or not folder:
            raise ValueError("Invalid task spec; name and folder must be non-empty")
        parsed.append((name, folder))
    return parsed


def parse_model_indices(raw_indices: Sequence[str]) -> List[int]:
    parsed = []
    for raw in raw_indices:
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            index = int(item)
            if index < 0:
                raise ValueError(f"Invalid model index {index}; indices must be non-negative")
            parsed.append(index)
    return sorted(set(parsed))


def resolve_round_dirs(
    model_root: Path,
    explicit_dirs: Optional[Sequence[str]],
    root: Path,
    expected_count: int,
    allow_missing_indices: Optional[set] = None,
) -> List[Path]:
    allow_missing_indices = allow_missing_indices or set()
    if explicit_dirs:
        if len(explicit_dirs) != expected_count:
            raise ValueError(f"--model-dirs expects {expected_count} paths, got {len(explicit_dirs)}")
        round_dirs = [resolve_path(item, root) for item in explicit_dirs]
    else:
        round_dirs = []
        for round_id in range(expected_count):
            resolved = None
            for name in KNOWN_ROUND_NAMES.get(round_id, (f"round{round_id}",)):
                candidate = model_root / name
                if candidate.exists():
                    resolved = candidate
                    break
            if resolved is None and round_id > 0:
                matches = sorted(path for path in model_root.glob(f"round{round_id}*") if path.is_dir())
                if matches:
                    resolved = matches[0]
            if resolved is None:
                raise FileNotFoundError(f"Cannot resolve checkpoint directory for round{round_id}")
            round_dirs.append(resolved)

    missing = [path for idx, path in enumerate(round_dirs) if not path.is_dir() and idx not in allow_missing_indices]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} checkpoint directorie(s)")
    return round_dirs


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    data = load_json(index_path)
    return data["weight_map"]


def find_tensor_key(weight_map: Dict[str, str], layer: int, projection: str) -> str:
    exact = f"model.layers.{layer}.mlp.{projection}.weight"
    if exact in weight_map:
        return exact

    suffix = f"layers.{layer}.mlp.{projection}.weight"
    matches = [key for key in weight_map if key.endswith(suffix)]
    if not matches:
        contains = [key for key in weight_map if suffix in key]
        matches = contains
    if len(matches) != 1:
        raise KeyError(f"Expected one tensor key for layer={layer}, projection={projection}; got {matches}")
    return matches[0]


def load_tensor(model_dir: Path, weight_map: Dict[str, str], tensor_key: str) -> torch.Tensor:
    shard_path = model_dir / weight_map[tensor_key]
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_key)


def extract_neuron_matrix(weight: torch.Tensor, projection: str, union_indices: torch.Tensor) -> torch.Tensor:
    if projection == "down_proj":
        if int(union_indices.max()) >= weight.shape[1]:
            raise IndexError(f"max neuron index {int(union_indices.max())} exceeds {projection} shape {tuple(weight.shape)}")
        return weight.index_select(1, union_indices).float().contiguous()

    if int(union_indices.max()) >= weight.shape[0]:
        raise IndexError(f"max neuron index {int(union_indices.max())} exceeds {projection} shape {tuple(weight.shape)}")
    return weight.index_select(0, union_indices).transpose(0, 1).float().contiguous()


def union_digest(union_indices: Sequence[int]) -> str:
    encoded = ",".join(str(index) for index in union_indices).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def safe_model_name(model_dir: Path) -> str:
    return ""


def model_cache_path(
    cache_dir: Path,
    model_idx: int,
    model_dir: Path,
    layer: int,
    union_indices: Sequence[int],
) -> Path:
    digest = union_digest(union_indices)
    name = f"model{model_idx:02d}_{safe_model_name(model_dir)}_layer{layer}_n{len(union_indices)}_{digest}.pt"
    return cache_dir / name


def validate_cache_bundle(cache_path: Path, bundle: Dict, union_index_tensor: torch.Tensor) -> None:
    if "neuron_indices" not in bundle or "projections" not in bundle:
        raise ValueError("Invalid cache bundle: missing neuron_indices/projections")
    cached_indices = bundle["neuron_indices"].to(dtype=torch.long, device="cpu")
    if not torch.equal(cached_indices, union_index_tensor.cpu()):
        raise ValueError("Cached neuron_indices do not match current C_union")
    missing = [projection for projection in PROJECTIONS if projection not in bundle["projections"]]
    if missing:
        raise ValueError(f"Cache bundle is missing projections: {missing}")


def ensure_model_cache(
    cache_dir: Path,
    model_idx: int,
    model_dir: Path,
    weight_map: Optional[Dict[str, str]],
    tensor_keys_by_projection: Optional[Dict[str, List[Optional[str]]]],
    union_index_tensor: torch.Tensor,
    union_indices: Sequence[int],
    layer: int,
    refresh: bool,
    zero_fill_allowed_positions: Optional[torch.Tensor] = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_cache_path(cache_dir, model_idx, model_dir, layer, union_indices)
    if cache_path.exists() and not refresh:
        bundle = torch.load(cache_path, map_location="cpu")
        validate_cache_bundle(cache_path, bundle, union_index_tensor)
        print(f"Using cached model matrix bundle: {cache_path.name}")
        return cache_path

    if not refresh:
        superset = find_best_superset_cache(cache_dir, model_idx, model_dir, layer, union_index_tensor)
        if superset is not None:
            superset_path, superset_bundle, positions = superset
            return slice_superset_cache(
                superset_cache_path=superset_path,
                superset_bundle=superset_bundle,
                positions=positions,
                target_cache_path=cache_path,
                model_idx=model_idx,
                model_dir=model_dir,
                layer=layer,
                union_index_tensor=union_index_tensor,
            )

    if not model_dir.is_dir() or weight_map is None or tensor_keys_by_projection is None:
        partial = find_best_partial_cache(cache_dir, model_idx, model_dir, layer, union_index_tensor)
        if partial is not None and zero_fill_allowed_positions is not None:
            partial_path, partial_bundle, current_positions, cached_positions, missing_positions = partial
            if all(bool(zero_fill_allowed_positions[pos].item()) for pos in missing_positions):
                return expand_partial_cache_with_zero_fill(
                    partial_cache_path=partial_path,
                    partial_bundle=partial_bundle,
                    current_positions=current_positions,
                    cached_positions=cached_positions,
                    missing_positions=missing_positions,
                    target_cache_path=cache_path,
                    model_idx=model_idx,
                    model_dir=model_dir,
                    layer=layer,
                    union_index_tensor=union_index_tensor,
                )

        subset = find_best_subset_cache(cache_dir, model_idx, model_dir, layer, union_index_tensor)
        if subset is not None and zero_fill_allowed_positions is not None:
            subset_path, subset_bundle, positions, missing_positions = subset
            if all(bool(zero_fill_allowed_positions[pos].item()) for pos in missing_positions):
                return expand_subset_cache_with_zero_fill(
                    subset_cache_path=subset_path,
                    subset_bundle=subset_bundle,
                    positions=positions,
                    missing_positions=missing_positions,
                    target_cache_path=cache_path,
                    model_idx=model_idx,
                    model_dir=model_dir,
                    layer=layer,
                    union_index_tensor=union_index_tensor,
                )

        if not model_dir.is_dir():
            raise FileNotFoundError("A configured checkpoint is required to build the missing cache")
        raise ValueError("A weight map is required to build the missing cache")
    if weight_map is None or tensor_keys_by_projection is None:
        raise ValueError("A weight map is required to build the missing cache")

    projections = {}
    tensor_keys = {}
    for projection in PROJECTIONS:
        tensor_key = tensor_keys_by_projection[projection][model_idx]
        if tensor_key is None:
            raise ValueError(f"A tensor key for {projection} is required to build the missing cache")
        weight = load_tensor(model_dir, weight_map, tensor_key)
        projections[projection] = extract_neuron_matrix(weight, projection, union_index_tensor)
        tensor_keys[projection] = tensor_key
        del weight

    bundle = {
        "model_index": model_idx,
        "model_dir": "",
        "layer": layer,
        "neuron_indices": union_index_tensor.cpu(),
        "projections": projections,
        "tensor_keys": tensor_keys,
        "format": "layer_neuron_projection_cache",
    }
    torch.save(bundle, cache_path)
    print(f"Saved cached model matrix bundle: {cache_path.name}")
    return cache_path


def load_cached_model_matrices(cache_path: Path, union_index_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    bundle = torch.load(cache_path, map_location="cpu")
    validate_cache_bundle(cache_path, bundle, union_index_tensor)
    return {projection: bundle["projections"][projection].float().contiguous() for projection in PROJECTIONS}


def current_positions_for_cached_indices(cached_indices: torch.Tensor, union_index_tensor: torch.Tensor) -> Optional[List[int]]:
    current_pos = {int(index): pos for pos, index in enumerate(union_index_tensor.tolist())}
    positions = []
    for index in cached_indices.tolist():
        pos = current_pos.get(int(index))
        if pos is None:
            return None
        positions.append(pos)
    return positions


def cached_positions_for_current_indices(cached_indices: torch.Tensor, union_index_tensor: torch.Tensor) -> Optional[List[int]]:
    cached_pos = {int(index): pos for pos, index in enumerate(cached_indices.tolist())}
    positions = []
    for index in union_index_tensor.tolist():
        pos = cached_pos.get(int(index))
        if pos is None:
            return None
        positions.append(pos)
    return positions


def find_best_superset_cache(
    cache_dir: Path,
    model_idx: int,
    model_dir: Path,
    layer: int,
    union_index_tensor: torch.Tensor,
) -> Optional[Tuple[Path, Dict, List[int]]]:
    prefix = f"model{model_idx:02d}_{safe_model_name(model_dir)}_layer{layer}_n"
    best = None
    for candidate in sorted(cache_dir.glob(f"{prefix}*.pt")):
        bundle = torch.load(candidate, map_location="cpu")
        if "neuron_indices" not in bundle or "projections" not in bundle:
            continue
        missing = [projection for projection in PROJECTIONS if projection not in bundle["projections"]]
        if missing:
            continue
        cached_indices = bundle["neuron_indices"].to(dtype=torch.long, device="cpu")
        positions = cached_positions_for_current_indices(cached_indices, union_index_tensor)
        if positions is None:
            continue
        if len(cached_indices) <= len(union_index_tensor):
            continue
        score = len(cached_indices)
        if best is None or score < best[0]:
            best = (score, candidate, bundle, positions)
    if best is None:
        return None
    _, candidate, bundle, positions = best
    return candidate, bundle, positions


def find_best_partial_cache(
    cache_dir: Path,
    model_idx: int,
    model_dir: Path,
    layer: int,
    union_index_tensor: torch.Tensor,
) -> Optional[Tuple[Path, Dict, List[int], List[int], List[int]]]:
    prefix = f"model{model_idx:02d}_{safe_model_name(model_dir)}_layer{layer}_n"
    current_indices = [int(index) for index in union_index_tensor.tolist()]
    best = None
    for candidate in sorted(cache_dir.glob(f"{prefix}*.pt")):
        bundle = torch.load(candidate, map_location="cpu")
        if "neuron_indices" not in bundle or "projections" not in bundle:
            continue
        missing = [projection for projection in PROJECTIONS if projection not in bundle["projections"]]
        if missing:
            continue
        cached_indices = bundle["neuron_indices"].to(dtype=torch.long, device="cpu")
        cached_pos = {int(index): pos for pos, index in enumerate(cached_indices.tolist())}
        current_positions = []
        cached_positions = []
        missing_positions = []
        for current_pos, index in enumerate(current_indices):
            cached_position = cached_pos.get(index)
            if cached_position is None:
                missing_positions.append(current_pos)
            else:
                current_positions.append(current_pos)
                cached_positions.append(cached_position)
        if not current_positions or not missing_positions:
            continue
        score = len(current_positions)
        if best is None or score > best[0]:
            best = (score, candidate, bundle, current_positions, cached_positions, missing_positions)
    if best is None:
        return None
    _, candidate, bundle, current_positions, cached_positions, missing_positions = best
    return candidate, bundle, current_positions, cached_positions, missing_positions


def expand_partial_cache_with_zero_fill(
    partial_cache_path: Path,
    partial_bundle: Dict,
    current_positions: Sequence[int],
    cached_positions: Sequence[int],
    missing_positions: Sequence[int],
    target_cache_path: Path,
    model_idx: int,
    model_dir: Path,
    layer: int,
    union_index_tensor: torch.Tensor,
) -> Path:
    current_position_tensor = torch.tensor(list(current_positions), dtype=torch.long)
    cached_position_tensor = torch.tensor(list(cached_positions), dtype=torch.long)
    projections = {}
    for projection in PROJECTIONS:
        old_matrix = partial_bundle["projections"][projection].float().contiguous()
        expanded = torch.zeros(
            (old_matrix.shape[0], len(union_index_tensor)),
            dtype=old_matrix.dtype,
            device=old_matrix.device,
        )
        expanded[:, current_position_tensor] = old_matrix.index_select(1, cached_position_tensor)
        projections[projection] = expanded.cpu()

    bundle = {
        "model_index": model_idx,
        "model_dir": "",
        "layer": layer,
        "neuron_indices": union_index_tensor.cpu(),
        "projections": projections,
        "tensor_keys": partial_bundle.get("tensor_keys", {}),
        "format": "layer_neuron_projection_cache",
        "source_partial_cache": "",
        "zero_filled_indices": union_index_tensor[list(missing_positions)].cpu(),
        "zero_fill_reason": "expanded partial cache for neurons not referenced by tasks touching this model",
    }
    torch.save(bundle, target_cache_path)
    print(f"Expanded partial cached model matrix bundle: {target_cache_path.name}")
    print(f"  source partial cache: {partial_cache_path.name}")
    print(f"  zero-filled columns: {len(missing_positions)}")
    return target_cache_path


def slice_superset_cache(
    superset_cache_path: Path,
    superset_bundle: Dict,
    positions: Sequence[int],
    target_cache_path: Path,
    model_idx: int,
    model_dir: Path,
    layer: int,
    union_index_tensor: torch.Tensor,
) -> Path:
    position_tensor = torch.tensor(list(positions), dtype=torch.long)
    projections = {}
    for projection in PROJECTIONS:
        old_matrix = superset_bundle["projections"][projection].float().contiguous()
        projections[projection] = old_matrix.index_select(1, position_tensor).cpu()

    bundle = {
        "model_index": model_idx,
        "model_dir": "",
        "layer": layer,
        "neuron_indices": union_index_tensor.cpu(),
        "projections": projections,
        "tensor_keys": superset_bundle.get("tensor_keys", {}),
        "format": "layer_neuron_projection_cache",
        "source_superset_cache": "",
        "slice_reason": "sliced exact current C_union from a larger cached layer-neuron projection bundle",
    }
    torch.save(bundle, target_cache_path)
    print(f"Sliced cached model matrix bundle from larger cache: {target_cache_path.name}")
    print(f"  source superset cache: {superset_cache_path.name}")
    return target_cache_path


def find_best_subset_cache(
    cache_dir: Path,
    model_idx: int,
    model_dir: Path,
    layer: int,
    union_index_tensor: torch.Tensor,
) -> Optional[Tuple[Path, Dict, List[int], List[int]]]:
    prefix = f"model{model_idx:02d}_{safe_model_name(model_dir)}_layer{layer}_n"
    best = None
    for candidate in sorted(cache_dir.glob(f"{prefix}*.pt")):
        bundle = torch.load(candidate, map_location="cpu")
        if "neuron_indices" not in bundle or "projections" not in bundle:
            continue
        missing = [projection for projection in PROJECTIONS if projection not in bundle["projections"]]
        if missing:
            continue
        cached_indices = bundle["neuron_indices"].to(dtype=torch.long, device="cpu")
        positions = current_positions_for_cached_indices(cached_indices, union_index_tensor)
        if positions is None:
            continue
        if len(positions) >= len(union_index_tensor):
            continue
        missing_positions = sorted(set(range(len(union_index_tensor))) - set(positions))
        score = len(positions)
        if best is None or score > best[0]:
            best = (score, candidate, bundle, positions, missing_positions)
    if best is None:
        return None
    _, candidate, bundle, positions, missing_positions = best
    return candidate, bundle, positions, missing_positions


def expand_subset_cache_with_zero_fill(
    subset_cache_path: Path,
    subset_bundle: Dict,
    positions: Sequence[int],
    missing_positions: Sequence[int],
    target_cache_path: Path,
    model_idx: int,
    model_dir: Path,
    layer: int,
    union_index_tensor: torch.Tensor,
) -> Path:
    projections = {}
    for projection in PROJECTIONS:
        old_matrix = subset_bundle["projections"][projection].float().contiguous()
        expanded = torch.zeros(
            (old_matrix.shape[0], len(union_index_tensor)),
            dtype=old_matrix.dtype,
            device=old_matrix.device,
        )
        expanded[:, torch.tensor(list(positions), dtype=torch.long)] = old_matrix
        projections[projection] = expanded.cpu()

    bundle = {
        "model_index": model_idx,
        "model_dir": "",
        "layer": layer,
        "neuron_indices": union_index_tensor.cpu(),
        "projections": projections,
        "tensor_keys": subset_bundle.get("tensor_keys", {}),
        "format": "layer_neuron_projection_cache",
        "source_subset_cache": "",
        "zero_filled_indices": union_index_tensor[list(missing_positions)].cpu(),
        "zero_fill_reason": "expanded subset cache for neurons not referenced by tasks touching this model",
    }
    torch.save(bundle, target_cache_path)
    print(f"Expanded cached model matrix bundle: {target_cache_path.name}")
    print(f"  source subset cache: {subset_cache_path.name}")
    print(f"  zero-filled columns: {len(missing_positions)}")
    return target_cache_path


def zero_fill_allowed_positions_for_model(model_idx: int, task_count: int, membership: torch.Tensor) -> torch.Tensor:
    adjacent_tasks = []
    if model_idx > 0:
        adjacent_tasks.append(model_idx - 1)
    if model_idx < task_count:
        adjacent_tasks.append(model_idx)
    if not adjacent_tasks:
        return torch.ones((membership.shape[1],), dtype=torch.bool)
    return ~membership[adjacent_tasks].any(dim=0)


def normalize_columns(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(matrix, dim=0, keepdim=True)
    return matrix / (norms + eps)


def build_task_membership(task_sets: Sequence[set], union_indices: Sequence[int]) -> torch.Tensor:
    rows = []
    for task_set in task_sets:
        rows.append([index in task_set for index in union_indices])
    return torch.tensor(rows, dtype=torch.bool)


def projection_deltas(
    projection: str,
    round_dirs: Sequence[Path],
    weight_maps: Sequence[Optional[Dict[str, str]]],
    tensor_keys: Sequence[Optional[str]],
    union_index_tensor: torch.Tensor,
    eps: float,
    cached_model_matrices: Optional[Sequence[Optional[Dict[str, torch.Tensor]]]] = None,
) -> List[torch.Tensor]:
    def load_neuron_matrix(model_idx: int) -> torch.Tensor:
        if cached_model_matrices and cached_model_matrices[model_idx] is not None:
            return cached_model_matrices[model_idx][projection].float().contiguous()
        if weight_maps[model_idx] is None or tensor_keys[model_idx] is None:
            raise FileNotFoundError(
                f"model index {model_idx} has no cache for {projection} and no readable checkpoint metadata"
            )
        weight = load_tensor(round_dirs[model_idx], weight_maps[model_idx], tensor_keys[model_idx])
        matrix = extract_neuron_matrix(weight, projection, union_index_tensor)
        del weight
        return matrix

    deltas = []
    for task_idx in range(len(round_dirs) - 1):
        prev_matrix = load_neuron_matrix(task_idx)
        curr_matrix = load_neuron_matrix(task_idx + 1)
        if prev_matrix.shape != curr_matrix.shape:
            raise ValueError(
                f"{projection} round{task_idx}->round{task_idx + 1} shape mismatch: "
                f"{tuple(prev_matrix.shape)} vs {tuple(curr_matrix.shape)}"
            )
        deltas.append(normalize_columns(curr_matrix - prev_matrix, eps))
        del prev_matrix, curr_matrix
    return deltas


def compute_projection_hhist(
    deltas: Sequence[torch.Tensor],
    membership: torch.Tensor,
    eps: float,
    num_directions: int,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_model, protected_size = deltas[0].shape
    hhist = torch.zeros((d_model, num_directions, protected_size), dtype=torch.float32)
    singular_values = torch.zeros((protected_size, num_directions), dtype=torch.float32)
    ranks = torch.zeros((protected_size,), dtype=torch.int64)
    full_ranks = torch.zeros((protected_size,), dtype=torch.int64)

    for neuron_pos in range(protected_size):
        valid_tasks = torch.nonzero(membership[:, neuron_pos], as_tuple=False).flatten().tolist()
        if not valid_tasks:
            continue

        history = torch.stack([deltas[task_idx][:, neuron_pos] for task_idx in valid_tasks], dim=1)
        if history.numel() == 0:
            continue

        u, svals, _ = torch.linalg.svd(history, full_matrices=False)
        full_rank = int((svals > eps).sum().item())
        rank = min(full_rank, num_directions)
        singular_count = min(svals.numel(), num_directions)
        singular_values[neuron_pos, :singular_count] = svals[:singular_count]
        ranks[neuron_pos] = rank
        full_ranks[neuron_pos] = full_rank
        if rank == 0:
            continue

        valid_u = u[:, :rank]
        ref = history.sum(dim=1)
        if torch.linalg.vector_norm(ref).item() >= eps:
            dots = valid_u.transpose(0, 1).matmul(ref)
            signs = torch.where(dots < 0, -1.0, 1.0).to(valid_u.dtype)
            valid_u = valid_u * signs.unsqueeze(0)
        hhist[:, :rank, neuron_pos] = valid_u

    return hhist.to(output_dtype), singular_values, ranks, full_ranks


def stats_for_tensor(values: torch.Tensor) -> Dict[str, float]:
    values = values.float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Build and save history directions from configured checkpoints."""
    args = parse_args()
    if args.num_directions <= 0:
        raise ValueError("--num-directions must be positive")
    if args.cache_first_n_models < 0:
        raise ValueError("--cache-first-n-models must be non-negative")
    extra_cache_indices = parse_model_indices(args.cache_model_indices)
    cache_model_indices = set(range(args.cache_first_n_models)) | set(extra_cache_indices)

    project_root = resolve_path(args.project_root or ".", Path.cwd())
    model_root = resolve_path(args.model_root, project_root)
    if not args.protection_root:
        raise ValueError("--protection-root must be configured")
    if not args.output_dir:
        raise ValueError("--output-dir must be configured")
    protection_root = resolve_path(args.protection_root, project_root)
    output_dir = resolve_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task:
        if args.extra_task:
            raise ValueError("--task provides the full task list; do not combine it with --extra-task")
        task_specs = parse_task_specs(args.task)
    else:
        task_specs = list(DEFAULT_TASKS) + parse_task_specs(args.extra_task)
    if not task_specs:
        raise ValueError("At least one --task or --extra-task must be configured")
    public_task_names = [f"task_{index}" for index in range(len(task_specs))]
    round_dirs = resolve_round_dirs(
        model_root=model_root,
        explicit_dirs=args.model_dirs,
        root=project_root,
        expected_count=len(task_specs) + 1,
        allow_missing_indices=cache_model_indices if not args.refresh_model_cache else set(),
    )
    if args.cache_first_n_models > len(round_dirs):
        raise ValueError(
            f"--cache-first-n-models={args.cache_first_n_models} exceeds model count {len(round_dirs)}"
        )
    invalid_cache_indices = sorted(index for index in cache_model_indices if index >= len(round_dirs))
    if invalid_cache_indices:
        raise ValueError(f"cache model index out of range for {len(round_dirs)} model(s): {invalid_cache_indices}")

    task_indices: Dict[str, List[int]] = {}
    task_sets = []
    for task_name, folder_name in task_specs:
        neurons_path = protection_root / folder_name / "protected_neurons.json"
        indices = load_protected_neurons(neurons_path, args.layer)
        if not indices:
            raise ValueError("A configured task has an empty protection set")
        deduped = sorted(set(indices))
        task_indices[task_name] = deduped
        task_sets.append(set(deduped))

    union_indices = sorted(set().union(*task_sets))
    union_index_tensor = torch.tensor(union_indices, dtype=torch.long)
    membership = build_task_membership(task_sets, union_indices)
    valid_update_counts = membership.sum(dim=0).to(torch.int64)

    model_cache_dir = resolve_path(args.model_cache_dir, project_root) if args.model_cache_dir else None
    if cache_model_indices and model_cache_dir is None:
        model_cache_dir = output_dir / "model_layer_cache"

    weight_maps: List[Optional[Dict[str, str]]] = []
    for model_idx, model_dir in enumerate(round_dirs):
        cache_path = (
            model_cache_path(model_cache_dir, model_idx, model_dir, args.layer, union_indices)
            if model_cache_dir is not None and model_idx in cache_model_indices
            else None
        )
        if (
            model_idx in cache_model_indices
            and not args.refresh_model_cache
            and cache_path is not None
            and cache_path.exists()
        ):
            weight_maps.append(None)
        elif not (model_dir / "model.safetensors.index.json").exists():
            weight_maps.append(None)
        else:
            weight_maps.append(load_weight_map(model_dir))
    tensor_keys_by_projection = {
        projection: [
            find_tensor_key(weight_map, args.layer, projection) if weight_map is not None else None
            for weight_map in weight_maps
        ]
        for projection in PROJECTIONS
    }

    cache_paths: List[Optional[Path]] = [None] * len(round_dirs)
    cached_model_matrices: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(round_dirs)
    if cache_model_indices:
        print("Model matrix cache directory: [configured]")
        print(f"Reading model indices {sorted(cache_model_indices)} from extracted cache bundles.")
        for model_idx in sorted(cache_model_indices):
            cache_path = ensure_model_cache(
                cache_dir=model_cache_dir,
                model_idx=model_idx,
                model_dir=round_dirs[model_idx],
                weight_map=weight_maps[model_idx],
                tensor_keys_by_projection=tensor_keys_by_projection,
                union_index_tensor=union_index_tensor,
                union_indices=union_indices,
                layer=args.layer,
                refresh=args.refresh_model_cache,
                zero_fill_allowed_positions=zero_fill_allowed_positions_for_model(
                    model_idx=model_idx,
                    task_count=len(task_specs),
                    membership=membership,
                ),
            )
            cache_paths[model_idx] = cache_path
            cached_model_matrices[model_idx] = load_cached_model_matrices(cache_path, union_index_tensor)

    output_dtype = save_dtype(args.save_dtype)
    singular_bundle = {
        "neuron_indices": union_index_tensor,
        "task_names": public_task_names,
        "membership": membership,
        "valid_update_counts": valid_update_counts,
        "eps": args.eps,
        "layer": args.layer,
        "num_directions": args.num_directions,
        "singular_values": {},
        "ranks": {},
        "full_ranks": {},
    }

    metadata = {
        "layer": args.layer,
        "eps": args.eps,
        "num_tasks": len(task_specs),
        "num_directions": args.num_directions,
        "save_dtype": args.save_dtype,
        "task_order": public_task_names,
        "task_protection_sizes": {
            public_task_names[index]: len(task_indices[task_name])
            for index, (task_name, _) in enumerate(task_specs)
        },
        "union_size": len(union_indices),
        "union_min": min(union_indices),
        "union_max": max(union_indices),
        "round_dirs": [""] * len(round_dirs),
        "protection_root": "",
        "output_dir": "",
        "tensor_keys": tensor_keys_by_projection,
        "model_cache_dir": "" if model_cache_dir else None,
        "cache_first_n_models": args.cache_first_n_models,
        "cache_model_indices": sorted(cache_model_indices),
        "cache_paths": ["" if path else None for path in cache_paths],
        "valid_update_count_histogram": {
            str(count): int((valid_update_counts == count).sum().item())
            for count in range(1, len(task_specs) + 1)
        },
        "outputs": {},
    }

    print("Task protection sizes:")
    for index, (task_name, _) in enumerate(task_specs):
        print(f"  {public_task_names[index]}: {len(task_indices[task_name])}")
    print(f"C_union size: {len(union_indices)}")
    print(f"C_union range: {min(union_indices)}..{max(union_indices)}")
    print(f"Saved directions per neuron: {args.num_directions}")
    print(f"Valid update count histogram: {metadata['valid_update_count_histogram']}")

    directions_bundle = {}
    for projection in PROJECTIONS:
        print(f"\nBuilding {projection} ...")
        deltas = projection_deltas(
            projection=projection,
            round_dirs=round_dirs,
            weight_maps=weight_maps,
            tensor_keys=tensor_keys_by_projection[projection],
            union_index_tensor=union_index_tensor,
            eps=args.eps,
            cached_model_matrices=cached_model_matrices,
        )
        hhist, singular_values, ranks, full_ranks = compute_projection_hhist(
            deltas=deltas,
            membership=membership,
            eps=args.eps,
            num_directions=args.num_directions,
            output_dtype=output_dtype,
        )

        hhist_path = output_dir / f"layer{args.layer}_{projection}_Hhist.pt"
        torch.save(hhist, hhist_path)
        directions_bundle[projection] = hhist.cpu()
        singular_bundle["singular_values"][projection] = singular_values
        singular_bundle["ranks"][projection] = ranks
        singular_bundle["full_ranks"][projection] = full_ranks

        nonzero_rank = int((ranks > 0).sum().item())
        rank_histogram = {str(rank): int((ranks == rank).sum().item()) for rank in range(0, args.num_directions + 1)}
        full_rank_histogram = {
            str(rank): int((full_ranks == rank).sum().item())
            for rank in range(0, len(task_specs) + 1)
        }
        metadata["outputs"][projection] = {
            "path": hhist_path.name,
            "shape": list(hhist.shape),
            "dtype": str(hhist.dtype),
            "rank_histogram": rank_histogram,
            "full_rank_histogram": full_rank_histogram,
            "nonzero_rank_neurons": nonzero_rank,
            "singular_value_stats": stats_for_tensor(singular_values),
        }
        print(f"  saved {hhist_path.name}")
        print(f"  shape: {tuple(hhist.shape)}, nonzero-rank neurons: {nonzero_rank}/{len(union_indices)}")
        print(f"  rank histogram: {rank_histogram}")
        print(f"  full rank histogram before truncation: {full_rank_histogram}")
        del deltas, hhist

    singular_path = output_dir / f"layer{args.layer}_singular_values.pt"
    torch.save(singular_bundle, singular_path)
    metadata["singular_values_path"] = singular_path.name

    union_path = output_dir / f"layer{args.layer}_C_union.pt"
    torch.save(union_index_tensor, union_path)
    metadata["union_indices_path"] = union_path.name

    meta_path = output_dir / f"layer{args.layer}_dsvd_meta.json"
    write_json(meta_path, metadata)

    if args.write_gradient_surgery_state:
        state_path = output_dir / "gradient_surgery_state.pt"
        state = {
            "protected_indices": union_indices,
            "directions": directions_bundle,
            "metadata": {
                "format": "dsvd_subspace",
                "source_dsvd_meta": meta_path.name,
                "source_union_indices": union_path.name,
                "source_singular_values": singular_path.name,
                "layer": args.layer,
                "task_names": public_task_names,
                "num_protected": len(union_indices),
                "num_tasks": len(task_specs),
                "num_directions": args.num_directions,
                "model_cache_dir": "" if model_cache_dir else None,
                "cache_first_n_models": args.cache_first_n_models,
                "cache_model_indices": sorted(cache_model_indices),
                "cache_paths": ["" if path else None for path in cache_paths],
                "membership": membership,
                "valid_update_counts": valid_update_counts,
                "ranks": singular_bundle["ranks"],
                "full_ranks": singular_bundle["full_ranks"],
                "singular_values": singular_bundle["singular_values"],
            },
        }
        torch.save(state, state_path)
        metadata["gradient_surgery_state_path"] = state_path.name
        write_json(meta_path, metadata)
    else:
        state_path = None

    print(f"\nSaved singular values: {singular_path.name}")
    print(f"Saved C_union: {union_path.name}")
    print(f"Saved metadata: {meta_path.name}")
    if state_path:
        print(f"Saved gradient surgery state: {state_path.name}")


if __name__ == "__main__":
    main()
