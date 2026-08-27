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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_ewc = self.config.get("use_ewc", False)
        self.ewc_coef = float(self.config.get("ewc_coef", 0.0))
        self.ewc_state = self.config.get("ewc_state", None)
        self._ewc_missing_shard_info_warned = set()

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        self.protected_indices = []
        self.protected_index_to_pos = {}
        self.gradient_surgery_directions = {}
        self.gradient_surgery_metadata = {}
        self.gradient_surgery_project_indices = []
        self.gradient_surgery_activation_sum = None
        self.gradient_surgery_activation_count = 0
        self.gradient_surgery_collect_activations = False
        self.gradient_surgery_hook_handle = None
        if self.config.get("gradient_surgery", False):
            self._load_gradient_surgery_state()
            self._register_gradient_surgery_activation_hook()

    def _get_ewc_shard_info_maps(self):
        shard_info_by_name = {}
        shard_info_by_param_id = {}
        seen_handles = set()

        def _add_handle(handle):
            if handle is None or id(handle) in seen_handles:
                return
            seen_handles.add(id(handle))
            flat_param = getattr(handle, "flat_param", None)
            if flat_param is None:
                return
            params = getattr(flat_param, "_params", None)
            param_infos = getattr(flat_param, "_param_infos", None)
            shard_infos = getattr(flat_param, "_shard_param_infos", None)
            if not param_infos or not shard_infos:
                return
            if params:
                for param, shard_info in zip(params, shard_infos):
                    shard_info_by_param_id[id(param)] = shard_info
            for param_info, shard_info in zip(param_infos, shard_infos):
                module_name = getattr(param_info, "module_name", "") or ""
                param_name = getattr(param_info, "param_name", "")
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                clean_name = full_name.replace("_fsdp_wrapped_module.", "")
                names = {clean_name}
                if clean_name.startswith("model.layers."):
                    names.add(clean_name.replace("model.layers.", "model.language_model.layers.", 1))
                if clean_name.startswith("model.language_model.layers."):
                    names.add(clean_name.replace("model.language_model.layers.", "model.layers.", 1))
                for candidate_name in names:
                    shard_info_by_name[candidate_name] = shard_info

        for module in self.actor_module.modules():
            _add_handle(getattr(module, "_handle", None))
            for handle in getattr(module, "_all_handles", []) or []:
                _add_handle(handle)
            state = getattr(module, "_fsdp_state", None)
            if state is not None:
                _add_handle(getattr(state, "_handle", None))
                for handle in getattr(state, "_all_handles", []) or []:
                    _add_handle(handle)
                for handle in (getattr(state, "_fully_sharded_module_to_handle", {}) or {}).values():
                    _add_handle(handle)
        return shard_info_by_name, shard_info_by_param_id

    def _compute_ewc_penalty(self):
        if not self.use_ewc or self.ewc_state is None or self.ewc_coef <= 0:
            return None

        theta_star = self.ewc_state.get("theta_star", None)
        fisher = self.ewc_state.get("fisher", None)
        if theta_star is None or fisher is None:
            raise ValueError("EWC state must contain 'theta_star' and 'fisher'")
        if not isinstance(theta_star, dict) or not isinstance(fisher, dict):
            raise ValueError("EWC state 'theta_star' and 'fisher' must be parameter-name dictionaries")

        penalty = None
        matched = 0
        found_state_key = 0
        shard_info_by_name, shard_info_by_param_id = self._get_ewc_shard_info_maps()
        for name, current_param in self.actor_module.named_parameters():
            clean_name = name.replace("_fsdp_wrapped_module.", "")
            if clean_name not in theta_star or clean_name not in fisher:
                continue
            found_state_key += 1
            if current_param.numel() == 0:
                continue
            ref_param = theta_star[clean_name].to(device=current_param.device, dtype=torch.float32)
            fisher_param = fisher[clean_name].to(device=current_param.device, dtype=torch.float32)
            current_float = current_param.float()
            if ref_param.shape != current_param.shape or fisher_param.shape != current_param.shape:
                if ref_param.numel() == current_float.numel() and fisher_param.numel() == current_float.numel():
                    ref_param = ref_param.reshape_as(current_float)
                    fisher_param = fisher_param.reshape_as(current_float)
                else:
                    shard_info = shard_info_by_name.get(clean_name) or shard_info_by_param_id.get(id(current_param))
                    start = getattr(shard_info, "intra_param_start_idx", None) if shard_info is not None else None
                    end = getattr(shard_info, "intra_param_end_idx", None) if shard_info is not None else None
                    if (
                        start is not None
                        and end is not None
                        and end >= start
                        and (end - start + 1) == current_float.numel()
                        and ref_param.numel() >= end + 1
                        and fisher_param.numel() >= end + 1
                    ):
                        ref_param = ref_param.reshape(-1)[start : end + 1].reshape_as(current_float)
                        fisher_param = fisher_param.reshape(-1)[start : end + 1].reshape_as(current_float)
                    else:
                        if current_float.numel() < ref_param.numel() and current_float.numel() < fisher_param.numel():
                            warn_key = (clean_name, tuple(current_param.shape))
                            if warn_key not in self._ewc_missing_shard_info_warned:
                                self._ewc_missing_shard_info_warned.add(warn_key)
                                if torch.distributed.get_rank() == 0:
                                    print(
                                        f"[WARN] Skip EWC partial shard without FSDP shard metadata for {clean_name}: "
                                        f"current={tuple(current_param.shape)} theta_star={tuple(ref_param.shape)} fisher={tuple(fisher_param.shape)}"
                                    )
                            continue
                        raise ValueError(
                            f"EWC state shape mismatch for {clean_name}: "
                            f"current={tuple(current_param.shape)} theta_star={tuple(ref_param.shape)} fisher={tuple(fisher_param.shape)} "
                            f"shard_range={(start, end)}"
                        )
            term = (fisher_param * (current_float - ref_param).pow(2)).sum()
            penalty = term if penalty is None else penalty + term
            matched += 1

        if matched == 0:
            if found_state_key == 0:
                raise ValueError("No model parameters matched EWC state keys")
            return None
        return 0.5 * penalty

    def _load_gradient_surgery_state(self):
        state_path = self.config.get("gradient_surgery_state_path", None)
        if not state_path:
            raise ValueError("gradient_surgery_state_path is required when gradient_surgery is enabled.")

        state = torch.load(state_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"Expected gradient surgery state at {state_path} to be a dict.")
        for key in ("protected_indices", "indices", "C_hist"):
            if key in state:
                self.protected_indices = [int(idx) for idx in state[key]]
                break
        else:
            raise ValueError(f"No protected_indices found in {state_path}")

        if not self.protected_indices:
            raise ValueError(f"No protected_indices found in {state_path}")
        self.protected_index_to_pos = {idx: pos for pos, idx in enumerate(self.protected_indices)}
        self.gradient_surgery_metadata = state.get("metadata", {}) if isinstance(state.get("metadata", {}), dict) else {}

        directions = state.get("directions")
        if not isinstance(directions, dict):
            raise ValueError(f"{state_path} must contain a 'directions' dict.")
        for proj_name in ("down_proj", "gate_proj", "up_proj"):
            if proj_name not in directions:
                raise ValueError(f"No directions['{proj_name}'] found in {state_path}")
            self.gradient_surgery_directions[proj_name] = self._prepare_gradient_surgery_direction(
                directions[proj_name],
                state_path,
                proj_name,
            )

    def _prepare_gradient_surgery_direction(self, direction, source, proj_name):
        if not torch.is_tensor(direction):
            direction = torch.tensor(direction)

        hidden_size = int(self.config.get("gradient_surgery_hidden_size", 3584))
        protected_tensor = torch.tensor(self.protected_indices, dtype=torch.long)
        max_protected_idx = int(protected_tensor.max().item())

        if direction.dim() == 2:
            if direction.shape[0] == hidden_size and direction.shape[1] > max_protected_idx:
                u_protected = direction[:, protected_tensor]
            elif direction.shape[1] == hidden_size and direction.shape[0] > max_protected_idx:
                u_protected = direction[protected_tensor, :].t()
            elif direction.shape[0] == hidden_size and direction.shape[1] == len(self.protected_indices):
                u_protected = direction
            elif direction.shape[1] == hidden_size and direction.shape[0] == len(self.protected_indices):
                u_protected = direction.t()
            else:
                raise ValueError(
                    f"Cannot align {proj_name} direction shape {tuple(direction.shape)} with hidden_size={hidden_size} "
                    f"and {len(self.protected_indices)} protected indices from {source}."
                )
            u_protected = u_protected.unsqueeze(1)
        elif direction.dim() == 3:
            if direction.shape[0] == hidden_size and direction.shape[2] > max_protected_idx:
                u_protected = direction[:, :, protected_tensor]
            elif direction.shape[0] == hidden_size and direction.shape[2] == len(self.protected_indices):
                u_protected = direction
            elif direction.shape[2] == hidden_size and direction.shape[0] > max_protected_idx:
                u_protected = direction[protected_tensor, :, :].permute(2, 1, 0)
            elif direction.shape[2] == hidden_size and direction.shape[0] == len(self.protected_indices):
                u_protected = direction.permute(2, 1, 0)
            else:
                raise ValueError(
                    f"Cannot align {proj_name} direction shape {tuple(direction.shape)} with hidden_size={hidden_size} "
                    f"and {len(self.protected_indices)} protected indices from {source}."
                )
        else:
            raise ValueError(
                f"Expected {proj_name} direction to be 2D or 3D, got shape {tuple(direction.shape)} from {source}"
            )

        u_protected = u_protected.float().contiguous()
        u_protected = u_protected / u_protected.norm(dim=0, keepdim=True).clamp_min(1e-12)
        return u_protected

    def _get_gradient_surgery_decision(self):
        project_indices = self.gradient_surgery_project_indices
        if project_indices is None:
            project_indices = []
        if torch.is_tensor(project_indices):
            project_indices = project_indices.detach().cpu().tolist()
        project_set = {int(idx) for idx in project_indices}
        protected_set = set(self.protected_indices)
        unknown_indices = project_set - protected_set
        if unknown_indices:
            raise ValueError(f"Project indices must be a subset of protected_indices, got {sorted(unknown_indices)[:10]}")

        project_indices = [idx for idx in self.protected_indices if idx in project_set]
        truncate_indices = [idx for idx in self.protected_indices if idx not in project_set]
        if project_indices and set(self.gradient_surgery_directions.keys()) != {"down_proj", "gate_proj", "up_proj"}:
            raise ValueError("gradient_surgery_state_path must contain down/gate/up projection directions.")
        return project_indices, truncate_indices

    def _gradient_surgery_module_matches_layer(self, module_name, layer_idx):
        pattern = f"layers.{layer_idx}"
        return module_name == pattern or module_name.endswith(f".{pattern}") or f"{pattern}." in module_name

    def _find_gradient_surgery_down_proj(self):
        layer_idx = int(self.config.get("gradient_surgery_layer", 20))
        for module_name, module in self.actor_module.named_modules():
            if self._gradient_surgery_module_matches_layer(module_name, layer_idx) and module_name.endswith("mlp.down_proj"):
                return module_name, module
        raise RuntimeError(f"Could not find layers.{layer_idx}.mlp.down_proj for activation collection.")

    def _register_gradient_surgery_activation_hook(self):
        target_name, target_module = self._find_gradient_surgery_down_proj()

        def _activation_hook(module, inputs, output):
            del module, output
            if not self.gradient_surgery_collect_activations:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                return

            activation = inputs[0].detach()
            intermediate_size = int(self.config.get("gradient_surgery_intermediate_size", 18944))
            if activation.shape[-1] != intermediate_size:
                raise RuntimeError(
                    f"Expected activation last dim {intermediate_size} for {target_name}, "
                    f"got shape {tuple(activation.shape)}."
                )

            reduce_dims = tuple(range(activation.dim() - 1))
            activation_sum = activation.float().abs().sum(dim=reduce_dims)
            activation_count = activation.numel() // activation.shape[-1]

            if self.gradient_surgery_activation_sum is None:
                self.gradient_surgery_activation_sum = torch.zeros_like(activation_sum)
            elif self.gradient_surgery_activation_sum.device != activation_sum.device:
                self.gradient_surgery_activation_sum = self.gradient_surgery_activation_sum.to(activation_sum.device)

            self.gradient_surgery_activation_sum += activation_sum
            self.gradient_surgery_activation_count += activation_count

        self.gradient_surgery_hook_handle = target_module.register_forward_hook(_activation_hook)

    def _reset_gradient_surgery_activation_stats(self):
        self.gradient_surgery_activation_sum = None
        self.gradient_surgery_activation_count = 0
        self.gradient_surgery_project_indices = []
        self.gradient_surgery_collect_activations = False

    def _update_gradient_surgery_decision_from_activation(self):
        self.gradient_surgery_collect_activations = False
        if self.gradient_surgery_activation_sum is None or self.gradient_surgery_activation_count == 0:
            self.gradient_surgery_project_indices = []
            return

        activation_sum = self.gradient_surgery_activation_sum
        activation_count = torch.tensor(float(self.gradient_surgery_activation_count), device=activation_sum.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(activation_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(activation_count, op=torch.distributed.ReduceOp.SUM)

        activation_mean = activation_sum / activation_count.clamp_min(1.0)
        intermediate_size = int(self.config.get("gradient_surgery_intermediate_size", 18944))
        top_ratio = self.config.get("gradient_surgery_realtime_top_ratio", None)
        if top_ratio is None:
            top_k = len(self.protected_indices)
        else:
            top_ratio = float(top_ratio)
            if top_ratio <= 0:
                self.gradient_surgery_project_indices = []
                return
            top_k = int(intermediate_size * top_ratio)

        top_k = max(1, min(intermediate_size, top_k))
        current_top_indices = torch.topk(activation_mean, k=top_k).indices.detach().cpu().tolist()
        current_top_set = set(int(idx) for idx in current_top_indices)
        self.gradient_surgery_project_indices = [idx for idx in self.protected_indices if idx in current_top_set]

    def _get_project_basis(self, proj_name, project_indices, device, dtype):
        positions = [self.protected_index_to_pos[idx] for idx in project_indices]
        positions = torch.tensor(positions, dtype=torch.long)
        return self.gradient_surgery_directions[proj_name][:, :, positions].to(device=device, dtype=dtype)

    def _project_vectors_to_basis_orthogonal_complement(self, vectors, basis):
        eps = float(self.config.get("gradient_surgery_direction_eps", 1e-12))
        if vectors.dim() != 2 or basis.dim() != 3:
            raise ValueError(
                f"Expected vectors to be 2D and basis to be 3D, got {tuple(vectors.shape)} and {tuple(basis.shape)}"
            )
        if basis.shape[0] != vectors.shape[0] or basis.shape[2] != vectors.shape[1]:
            raise ValueError(f"Cannot align vectors {tuple(vectors.shape)} with basis {tuple(basis.shape)}")

        projected = vectors.clone()
        for col_idx in range(vectors.shape[1]):
            u = basis[:, :, col_idx].float()
            direction_norms = u.norm(dim=0)
            u = u[:, direction_norms > eps]
            if u.numel() == 0:
                continue

            if u.shape[1] == 1:
                q = u / u.norm(dim=0, keepdim=True).clamp_min(eps)
            else:
                q, r = torch.linalg.qr(u, mode="reduced")
                independent = torch.abs(torch.diagonal(r, 0)) > eps
                q = q[:, independent]
                if q.numel() == 0:
                    continue

            g = vectors[:, col_idx].float()
            projected[:, col_idx] = (g - q @ (q.transpose(0, 1) @ g)).to(dtype=vectors.dtype)
        return projected

    def _apply_adaptive_gradient_surgery(self):
        for n, p in self.actor_module.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"NaN grad at {n}")

        layer_idx = int(self.config.get("gradient_surgery_layer", 20))
        layer_fsdp_module = None
        layer_module_name = None
        for module_name, module in self.actor_module.named_modules():
            if self._gradient_surgery_module_matches_layer(module_name, layer_idx) and isinstance(module, FSDP):
                layer_fsdp_module = module
                layer_module_name = module_name
                break
        if layer_fsdp_module is None:
            raise RuntimeError(f"Could not find an FSDP-wrapped module for layers.{layer_idx}.")

        project_indices, truncate_indices = self._get_gradient_surgery_decision()
        hidden_size = int(self.config.get("gradient_surgery_hidden_size", 3584))
        intermediate_size = int(self.config.get("gradient_surgery_intermediate_size", 18944))

        with FSDP.summon_full_params(layer_fsdp_module, writeback=True, recurse=True, with_grads=True):
            for n, p in layer_fsdp_module.named_parameters():
                full_name = f"{layer_module_name}.{n}"
                if p.grad is None:
                    continue

                if torch.isnan(p.grad).any():
                    print(f"NaN grad at {full_name}")

                if "mlp.down_proj.weight" in n:
                    expected_shape = (hidden_size, intermediate_size)
                    if p.grad.numel() != expected_shape[0] * expected_shape[1]:
                        raise RuntimeError(
                            f"Expected full grad for {full_name} to have {expected_shape[0] * expected_shape[1]} elements "
                            f"for shape {expected_shape}, but got grad shape {tuple(p.grad.shape)} "
                            f"with {p.grad.numel()} elements and param shape {tuple(p.shape)}."
                        )
                    g_2d = p.grad.data.view(*expected_shape)

                    if truncate_indices:
                        truncate_tensor = torch.tensor(truncate_indices, dtype=torch.long, device=g_2d.device)
                        g_2d[:, truncate_tensor] = 0.0

                    if project_indices:
                        project_tensor = torch.tensor(project_indices, dtype=torch.long, device=g_2d.device)
                        u = self._get_project_basis("down_proj", project_indices, device=g_2d.device, dtype=g_2d.dtype)
                        original = g_2d[:, project_tensor]
                        projected = self._project_vectors_to_basis_orthogonal_complement(original, u)
                        g_2d[:, project_tensor] = projected
                elif "mlp.gate_proj.weight" in n:
                    expected_shape = (intermediate_size, hidden_size)
                    if p.grad.numel() != expected_shape[0] * expected_shape[1]:
                        raise RuntimeError(
                            f"Expected full grad for {full_name} to have {expected_shape[0] * expected_shape[1]} elements "
                            f"for shape {expected_shape}, but got grad shape {tuple(p.grad.shape)} "
                            f"with {p.grad.numel()} elements and param shape {tuple(p.shape)}."
                        )
                    g_2d = p.grad.data.view(*expected_shape)

                    if truncate_indices:
                        truncate_tensor = torch.tensor(truncate_indices, dtype=torch.long, device=g_2d.device)
                        g_2d[truncate_tensor, :] = 0.0

                    if project_indices:
                        project_tensor = torch.tensor(project_indices, dtype=torch.long, device=g_2d.device)
                        u = self._get_project_basis("gate_proj", project_indices, device=g_2d.device, dtype=g_2d.dtype)
                        original = g_2d[project_tensor, :]
                        projected = self._project_vectors_to_basis_orthogonal_complement(original.t(), u).t()
                        g_2d[project_tensor, :] = projected
                elif "mlp.up_proj.weight" in n:
                    expected_shape = (intermediate_size, hidden_size)
                    if p.grad.numel() != expected_shape[0] * expected_shape[1]:
                        raise RuntimeError(
                            f"Expected full grad for {full_name} to have {expected_shape[0] * expected_shape[1]} elements "
                            f"for shape {expected_shape}, but got grad shape {tuple(p.grad.shape)} "
                            f"with {p.grad.numel()} elements and param shape {tuple(p.shape)}."
                        )
                    g_2d = p.grad.data.view(*expected_shape)

                    if truncate_indices:
                        truncate_tensor = torch.tensor(truncate_indices, dtype=torch.long, device=g_2d.device)
                        g_2d[truncate_tensor, :] = 0.0

                    if project_indices:
                        project_tensor = torch.tensor(project_indices, dtype=torch.long, device=g_2d.device)
                        u = self._get_project_basis("up_proj", project_indices, device=g_2d.device, dtype=g_2d.dtype)
                        original = g_2d[project_tensor, :]
                        projected = self._project_vectors_to_basis_orthogonal_complement(original.t(), u).t()
                        g_2d[project_tensor, :] = projected

        num_dirs = next(iter(self.gradient_surgery_directions.values())).shape[1]
        print(
            f"Applied gradient surgery on layer {layer_idx}: truncate={len(truncate_indices)}, "
            f"project={len(project_indices)}, num_dirs={num_dirs}"
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                # Special handling for MiniCPM-o model: pixel_values, image_bound, and tgt_sizes
                # need different concatenation strategies compared to other multimodal inputs
                if (key == "pixel_values" and isinstance(micro_batch["multi_modal_inputs"][0]["pixel_values"], list)) or key == "image_bound" or key == "tgt_sizes":
                    # For MiniCPM-o: keep as list structure instead of concatenating tensors
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

            if "multi_modal_inputs" in micro_batch:
                # MiniCPM-o specific processing for image bounds and pixel values
                if "image_bound" in multi_modal_inputs:
                    # Adjust image bounds based on left padding and cumulative sequence lengths
                    # This is necessary for MiniCPM-o's vision-language alignment
                    left_padding_length = torch.argmax(attention_mask, dim=1)
                    image_bounds = []
                    for i in range(len(multi_modal_inputs["image_bound"])):
                        image_bound = multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
                        image_bounds.append(image_bound)
                    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
                    # Flatten pixel values list for MiniCPM-o processing
                    pixel_values = []
                    for i in range(len(multi_modal_inputs["pixel_values"])):
                        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])
                    multi_modal_inputs["pixel_values"] = [pixel_values]
                # Handle target sizes for MiniCPM-o vision processing
                if "tgt_sizes" in multi_modal_inputs:
                    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False

                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            print("_forward_micro_batch Final", entropy.shape if entropy is not None else None, log_probs.shape if log_probs is not None else None)
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if "loss_mask" in data.batch.keys():
            select_keys.append("loss_mask")
        if "padding_mask" in data.batch.keys():
            select_keys.append("padding_mask")
        if "attention_mask_eff" in data.batch.keys():
            select_keys.append("attention_mask_eff")
        if multi_turn:
            select_keys.append("loss_mask")
        # if "entropy_mask" in data.batch.keys():
        #     select_keys.append("entropy_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.use_vllm_logp:
            select_keys.append("rollout_log_probs")
            print("Using vllm log probs for ppo update")
            if self.config.use_sft_loss:
                print("Also using sft loss")
                select_keys.append("token_level_scores")

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            print(
                "Split batch size", 
                num_mini_batches,  
                data.batch.batch_size[0], 
                self.config.ppo_mini_batch_size
            )
            print(f"data.batch.batch_size[0] = {data.batch.batch_size}, mini bz = {self.config.ppo_mini_batch_size}, chunks = {num_mini_batches}")
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(batch=batch_tensordict_for_rearrange, max_token_len=max_token_len)

                        for current_original_indices, text_mb_td in zip(textual_indices, rearranged_text_micro_batches_tds):
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if self.config.get("gradient_surgery", False):
                    self._reset_gradient_surgery_activation_stats()

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [{kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    elif  "attention_mask_eff" in data.keys():
                        response_mask = data["attention_mask_eff"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    # if "entropy_mask" in data.keys():
                    #     response_mask = response_mask * data["entropy_mask"]

                    old_log_prob = data["old_log_probs"]
                    
                    if self.config.use_vllm_logp:
                        vllm_log_prob = data["rollout_log_probs"]
                    else: 
                        vllm_log_prob = None
                        sft_loss = None
                        reward = None
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    if self.config.get("gradient_surgery", False):
                        self.gradient_surgery_collect_activations = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    if self.config.get("gradient_surgery", False):
                        self.gradient_surgery_collect_activations = False
                    if vllm_log_prob is not None:
                        if self.config.use_sft_loss:
                            print("Using sft loss, aggregating sft loss")
                            sft_loss = -agg_loss(loss_mat=log_prob, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            reward = data["token_level_scores"][:,0]
                        else:
                            sft_loss = None
                            reward = None

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower,pg_loss_no_old, oldlogp_vllm_ratio = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            vllm_log_prob=vllm_log_prob,
                            reward=reward,
                            sft_loss=sft_loss,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower,pg_loss_no_old, oldlogp_vllm_ratio = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            vllm_log_prob=vllm_log_prob,
                            reward=reward,
                            sft_loss=sft_loss,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.use_ewc and self.ewc_state is not None and self.ewc_coef > 0:
                        ewc_penalty = self._compute_ewc_penalty()
                        if ewc_penalty is not None:
                            policy_loss = policy_loss + ewc_penalty * self.ewc_coef
                            metrics["actor/ewc_penalty"] = ewc_penalty.detach().item()
                            metrics["actor/ewc_coef"] = self.ewc_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                        
                    def _check_finite(name, t):
                        if torch.is_tensor(t):
                            if not torch.isfinite(t).all():
                                print(f"[NaN/Inf] {name}")

                    valid = (response_mask > 0.5)

                    _check_finite("log_prob", log_prob)
                    _check_finite("old_log_prob", old_log_prob)
                    if self.config.use_kl_loss:
                        _check_finite("ref_log_prob", data["ref_log_prob"])
                    _check_finite("advantages", advantages)
                    _check_finite("response_mask", response_mask)

                    # 尤其看“被掩掉的位置”是否已经是坏的
                    bad_ratio = torch.exp((log_prob - old_log_prob))
                    _check_finite("ratio", bad_ratio)
                    _check_finite("ratio_on_mask0", bad_ratio[~valid])
                    
                    loss.backward()

                    data = {
                        "actor/advantages_mean": advantages.mean().detach().item(),
                        "actor/advantages_std": advantages.std().detach().item(),
                        "actor/log_prob_mean": log_prob.mean().detach().item(),
                        "actor/old_log_prob_mean": old_log_prob.mean().detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_loss_no_old": pg_loss_no_old.detach().item(),
                        "actor/loss": loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    if vllm_log_prob is not None and oldlogp_vllm_ratio is not None:
                        data["actor/vllm_log_prob_mean"] = vllm_log_prob.mean().detach().item()
                        data["actor/oldlogp_vllm_ratio"] = oldlogp_vllm_ratio.detach().tolist()
                        
                    if sft_loss is not None:
                        data["actor/sft_loss"] = sft_loss.detach().item()

                    append_to_dict(metrics, data)
                    
                # for n,p in self.actor_module.named_parameters():
                #     if p.grad is not None and torch.isnan(p.grad).any():
                #         print("NaN grad at", n)

                if self.config.get("gradient_surgery", False):
                    self._update_gradient_surgery_decision_from_activation()
                    self._apply_adaptive_gradient_surgery()
                else:
                    for n,p in self.actor_module.named_parameters():
                        if p.grad is not None and torch.isnan(p.grad).any():
                            print("NaN grad at", n)

                grad_norm = self._optimizer_step()
                print("call optimizer step, update actor")
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
                with open("loss_dump/minibs32_adv_mean.txt", 'a', encoding='utf-8') as f:
                    line_to_write = " ".join(map(str, metrics["actor/advantages_mean"])) + "\n"
                    f.write(line_to_write)
        self.actor_optimizer.zero_grad()

        return metrics
