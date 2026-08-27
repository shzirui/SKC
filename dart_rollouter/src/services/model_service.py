"""
Model Service Pool - vLLM Service Management
"""

import asyncio
import os
import argparse
from typing import List, Dict, Any, Optional, Union
import subprocess
import aiohttp
import time
from contextlib import asynccontextmanager
import pynvml
import logging
import logging.handlers
import json
import base64
import torch

from fastapi import HTTPException, status
from pydantic import BaseModel, Field
from safetensors import safe_open

import socket
from datetime import datetime
import re

# Global flag: Whether NVML is available
NVML_AVAILABLE = False
try:
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except pynvml.NVMLError as e:
    logging.warning(f"NVML initialization failed, GPU monitoring functionality will be disabled: {e}")

def set_logger(log_file: str = "logs/model_service.log", log_level: int = logging.INFO):
    """
    Setup logger to output logs to both file and console
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/model_service_{timestamp}.log"
    # Create logger
    logger = logging.getLogger('model_service')
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Create file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=100)  # 50MB per file, keep 5 backups
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.info(f"Logger initialized with log file: {log_file}")
    return logger


# Create global logger instance
logger = set_logger()


# -------------------------------------------------------------------------
# Step 1: Define data models and core configuration
# -------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """
    Model service configuration class
    """
    ckpt_path: str = Field(..., description="Model checkpoint path, e.g. /data/models/llama-2-7b-chat-hf")
    base_port: int = Field(8000, description="Base port number used by vLLM service")
    replicas: int = Field(1, description="Expected number of service replicas")
    vllm_params: Dict[str, Any] = Field(default_factory=dict, description="Additional vLLM startup parameters")
    save_local: bool = Field(False, description="Whether to save locally")
    save_path: str = Field(default="./", description="Save path")
    enable_lora: bool = Field(False, description="Whether to start vLLM with a LoRA adapter")
    lora_adapter_path: Optional[str] = Field(None, description="LoRA adapter path for initial startup")
    max_lora_rank: int = Field(32, description="Maximum LoRA rank for vLLM")
    
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class GenerationRequest(BaseModel):
    messages: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None

class TokenizeRequest(BaseModel):
    prompt: str
    parameters: Optional[Dict] = None

class SaveRequest(BaseModel):
    messages: List[Dict]
    reward: float
    task_id: str
    trace_id: str
    parsed_actions: Optional[List[Dict[str, Any]]] = None
    
class ReloadRequest(BaseModel):
    new_ckpt_path: str = Field(..., description="New model checkpoint path")
    batch_size: int = Field(1, description="Rolling update batch size", gt=0)
    lora_adapter_path: Optional[str] = Field(None, description="LoRA adapter path to serve with the base checkpoint")


# -------------------------------------------------------------------------
# Step 2: Simplify and refactor ModelServicePool class
# -------------------------------------------------------------------------

class ServiceInstance:
    """
    Use a data class to uniformly manage the state of each service instance.
    """
    def __init__(
        self,
        port: int,
        gpu_id: int,
        process: subprocess.Popen,
        ckpt_path: str,
        lora_adapter_path: Optional[str] = None,
        served_model_name: Optional[str] = None,
    ):
        self.port = port
        self.gpu_id = gpu_id
        self.process = process
        self.endpoint = f"http://localhost:{port}"
        self.requests_in_flight = 0
        self.ckpt_path = ckpt_path
        self.lora_adapter_path = lora_adapter_path
        self.served_model_name = served_model_name or ckpt_path
        self.model_version = lora_adapter_path or ckpt_path

    def __repr__(self):
        return (
            f"<ServiceInstance(port={self.port}, gpu_id={self.gpu_id}, pid={self.process.pid}, "
            f"model_version={self.model_version})>"
        )


class GPUInstance:
    """
    Use a data class to uniformly manage the state of each GPU instance, including GPU-level NVML management
    """
    def __init__(
        self,
        gpu_id: int,
        gpu_memory_utilization: float = 0.9,
        nvml_gpu_id: Optional[int] = None,
        cuda_visible_device: Optional[str] = None,
    ):
        self.gpu_id = gpu_id
        self.nvml_gpu_id = gpu_id if nvml_gpu_id is None else nvml_gpu_id
        self.cuda_visible_device = str(self.nvml_gpu_id if cuda_visible_device is None else cuda_visible_device)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.is_available = False
        self._handle = None
        
        # Initialize GPU handle and check availability
        self._initialize_gpu()
        self.check_and_set_availability()

    def _initialize_gpu(self):
        """Initialize GPU handle, lazy load NVML"""
        global NVML_AVAILABLE
        if not NVML_AVAILABLE:
            logger.warning(f"GPU {self.gpu_id}: NVML unavailable, GPU monitoring disabled")
            self._handle = None
            return
            
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.nvml_gpu_id)
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize GPU {self.gpu_id} (NVML index {self.nvml_gpu_id}): {e}")
            self._handle = None

    def check_and_set_availability(self, gpu_memory_utilization: float = None) -> bool:
        """
        Check if GPU has sufficient memory and set is_available flag
        """
        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.gpu_memory_utilization
            
        if self._handle is None:
            self.is_available = False
            return False
            
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_memory = memory_info.total
            free_memory = memory_info.free
            required_memory = int(total_memory * gpu_memory_utilization)
            self.is_available = free_memory >= required_memory
            logger.info(
                f"GPU {self.gpu_id} (NVML index {self.nvml_gpu_id}) memory check: "
                f"free={free_memory // 1024 // 1024}MiB, "
                f"required={required_memory // 1024 // 1024}MiB, "
                f"available={self.is_available}"
            )
            return self.is_available
        except pynvml.NVMLError as e:
            logger.error(f"Error checking GPU {self.gpu_id} (NVML index {self.nvml_gpu_id}) availability: {e}")
            self.is_available = False
            return False

    def __del__(self):
        """GPU instance destructor"""
        try:
            self._handle = None
        except Exception:
            pass

    def __repr__(self):
        return (
            f"<GPUInstance(gpu_id={self.gpu_id}, nvml_gpu_id={self.nvml_gpu_id}, "
            f"cuda_visible_device={self.cuda_visible_device}, is_available={self.is_available})>"
        )
    
    def get_memory_info(self) -> dict:
        """Get current GPU memory information for debugging"""
        if self._handle is None:
            return {}
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                'total': memory_info.total,
                'free': memory_info.free,
                'used': memory_info.used
            }
        except pynvml.NVMLError:
            return {}


class ModelServicePool:
    """
    1. GPU state management
    2. Service instance management
    """
    def __init__(self, model_cfg: ModelConfig):
        logger.info("Initializing ModelServicePool...")
        self.model_cfg = model_cfg
        self.default_ckpt_path = model_cfg.ckpt_path
        self.default_lora_adapter_path = model_cfg.lora_adapter_path if model_cfg.enable_lora else None
        self.last_ckpt_path = model_cfg.ckpt_path
        self.last_lora_adapter_path = self.default_lora_adapter_path
        self.base_port = model_cfg.base_port
        self.replicas = model_cfg.replicas
        self.vllm_params = model_cfg.vllm_params
        self.gpu_memory_utilization = model_cfg.vllm_params.get("gpu_memory_utilization", 0.9)
        self.save_local = model_cfg.save_local
        self.save_path = model_cfg.save_path
        
        # Optimized fine-grained locking mechanism
        self.instances_lock = asyncio.Lock()  # Protect service_instances dictionary
        self.gpu_lock = asyncio.Lock()        # Protect GPU_Instances dictionary
        self.config_lock = asyncio.Lock()     # Protect configuration changes
        self.recovery_lock = asyncio.Lock()   # Avoid concurrent replica recovery storms
        
        # {gpu_id: ServiceInstance}
        self.service_instances: Dict[int, ServiceInstance] = {}
        # {gpu_id: GPUInstance}
        self.gpu_instances: Dict[int, GPUInstance] = {}
        
        self.monitoring_task: Optional[asyncio.Task] = None
        self.recovery_enabled = False
        self.shutting_down = False

    def _validate_model_path(self, ckpt_path: str):
        """Fail fast for local checkpoint paths that vLLM cannot load."""
        if not ckpt_path:
            raise ValueError("Checkpoint path is empty.")

        # Non-absolute values may be Hugging Face repo IDs, so leave them to vLLM.
        if not os.path.isabs(ckpt_path):
            return

        if not os.path.isdir(ckpt_path):
            raise ValueError(f"Checkpoint path does not exist in this container: {ckpt_path}")

        if not self._has_model_config(ckpt_path):
            raise ValueError(
                f"Checkpoint path is missing config.json or params.json: {ckpt_path}"
            )

    def _validate_lora_adapter_path(self, lora_adapter_path: str):
        if not lora_adapter_path:
            raise ValueError("LoRA adapter path is empty.")
        if not os.path.isabs(lora_adapter_path):
            return
        if not os.path.isdir(lora_adapter_path):
            raise ValueError(f"LoRA adapter path does not exist in this container: {lora_adapter_path}")
        if not os.path.isfile(os.path.join(lora_adapter_path, "adapter_config.json")):
            raise ValueError(f"LoRA adapter path is missing adapter_config.json: {lora_adapter_path}")
        adapter_model_path = os.path.join(lora_adapter_path, "adapter_model.safetensors")
        if not os.path.isfile(adapter_model_path):
            raise ValueError(f"LoRA adapter path is missing adapter_model.safetensors: {lora_adapter_path}")
        try:
            with safe_open(adapter_model_path, framework="pt", device="cpu") as f:
                num_keys = len(f.keys())
        except Exception as exc:
            raise ValueError(f"LoRA adapter_model.safetensors cannot be opened: {adapter_model_path}: {exc}") from exc
        if num_keys <= 0:
            raise ValueError(f"LoRA adapter_model.safetensors has no tensor keys: {adapter_model_path}")

    def _has_model_config(self, ckpt_path: str) -> bool:
        """Return whether a local path looks loadable by vLLM."""
        has_config = any(
            os.path.isfile(os.path.join(ckpt_path, filename))
            for filename in ("config.json", "params.json")
        )
        return has_config

    def _checkpoint_sort_key(self, ckpt_path: str):
        """Sort checkpoint candidates by global_step number first, then mtime."""
        match = re.search(r"global_step_(\d+)", ckpt_path)
        step = int(match.group(1)) if match else -1
        try:
            mtime = os.path.getmtime(ckpt_path)
        except OSError:
            mtime = 0
        return step, mtime

    def _checkpoint_search_root(self, ckpt_path: str) -> str:
        """
        Return the directory that contains global_step_* checkpoint folders.
        If ckpt_path is already inside global_step_N/actor/huggingface, this
        returns the parent directory that contains global_step_N.
        """
        parts = os.path.normpath(ckpt_path).split(os.sep)
        for idx, part in enumerate(parts):
            if re.fullmatch(r"global_step_\d+", part):
                root_parts = parts[:idx]
                if not root_parts:
                    return os.sep
                return os.sep.join(root_parts) or os.sep
        return ckpt_path

    def _resolve_checkpoint_path(self, ckpt_path: str) -> str:
        """
        Resolve a checkpoint input to a concrete vLLM-loadable directory.
        If a parent directory is provided, pick the newest global_step*/actor/huggingface.
        """
        if not ckpt_path:
            raise ValueError("Checkpoint path is empty.")

        if not os.path.isabs(ckpt_path):
            return ckpt_path

        if not os.path.isdir(ckpt_path):
            raise ValueError(f"Checkpoint path does not exist in this container: {ckpt_path}")

        candidates = []
        search_root = self._checkpoint_search_root(ckpt_path)
        direct_candidates = [
            ckpt_path,
            os.path.join(ckpt_path, "huggingface"),
            os.path.join(ckpt_path, "actor", "huggingface"),
        ]
        for candidate in direct_candidates:
            if os.path.isdir(candidate) and self._has_model_config(candidate):
                candidates.append(candidate)

        for root, dirs, _ in os.walk(search_root):
            if "huggingface" in dirs:
                candidate = os.path.join(root, "huggingface")
                if self._has_model_config(candidate):
                    candidates.append(candidate)

        if not candidates:
            raise ValueError(
                f"No loadable checkpoint found under {ckpt_path}; "
                "expected config.json or params.json in a checkpoint directory."
            )

        resolved = max(set(candidates), key=self._checkpoint_sort_key)
        if resolved != ckpt_path:
            logger.warning(
                f"Resolved checkpoint input {ckpt_path} to latest loadable checkpoint {resolved} "
                f"by scanning {search_root}"
            )
        else:
            logger.info(
                f"Checkpoint input {ckpt_path} is the latest loadable checkpoint under {search_root}"
            )
        return resolved

    async def initialize(self):    
        """
        Initialize model service pool, including GPU instances and service instances.
        Used for initial startup
        """
        try:
            self.shutting_down = False
            self.recovery_enabled = False
            self._validate_model_path(self.default_ckpt_path)
            if self.default_lora_adapter_path:
                self._validate_lora_adapter_path(self.default_lora_adapter_path)
            gpu_count = self._get_gpu_count()
            if gpu_count == 0:
                raise RuntimeError("No available GPUs found. Cannot start ModelServicePool.")
            
            self.replicas = min(self.model_cfg.replicas, gpu_count)
            if self.replicas < self.model_cfg.replicas:
                logger.warning(f"Warning: Requested replicas ({self.model_cfg.replicas}) > available GPU count ({gpu_count}). "
                      f"Setting replicas to {self.replicas}.")
            
            await self._init_gpu_instances()
            logger.info(await self.get_gpu_info())

            # Start initial services
            instances = await self._start_initial_services()

            logger.info("Waiting for model service pool to become ready...")
            if not await self.wait_for_model_pool_ready(instances):
                raise RuntimeError("Model service pool did not become ready within timeout.")
            else:
                logger.info("Model service pool is ready. Adding instances to pool...")
                async with self.instances_lock:
                    for instance in instances:
                        self.service_instances[instance.gpu_id] = instance

            # Start background monitoring task
            self.recovery_enabled = True
            self.monitoring_task = asyncio.create_task(self._monitor_replicas())
            
            logger.info("ModelServicePool initialized and monitoring started.")
            return True

        except Exception as e:
            logger.error(f"Error during ModelServicePool initialization: {e}")
            await self.shutdown()
            return False

    async def shutdown(self):
        """Safely shutdown the entire model service pool."""
        logger.info("Final shutdown sequence initiated.")
        self.shutting_down = True
        self.recovery_enabled = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        await self._shutdown_all_services()
        logger.info("ModelServicePool has been shut down.")

    async def _init_gpu_instances(self) -> List[GPUInstance]:
        """Initialize all GPU instances"""
        visible_devices = self._get_visible_gpu_devices()
        for logical_gpu_id, cuda_visible_device in enumerate(visible_devices):
            nvml_gpu_id = self._cuda_visible_device_to_nvml_index(cuda_visible_device)
            self.gpu_instances[logical_gpu_id] = GPUInstance(
                logical_gpu_id,
                self.gpu_memory_utilization,
                nvml_gpu_id=nvml_gpu_id,
                cuda_visible_device=cuda_visible_device,
            )

    async def _start_initial_services(self) -> List[ServiceInstance]:
        """Start the initial [replicas] number of service instances."""
        logger.info(f"Starting {self.replicas} initial model services...")
        
        # Use GPU lock to quickly get available GPUs
        async with self.gpu_lock:
            available_gpus = [
                instance.gpu_id for instance in self.gpu_instances.values() 
                if instance.is_available
            ]
        
        if not available_gpus:
            logger.warning("No available GPUs to start initial services. Trying stale model cleanup once.")
            cleaned = await self._cleanup_stale_model_processes()
            if cleaned:
                await asyncio.sleep(5)
                async with self.gpu_lock:
                    available_gpus = [
                        instance.gpu_id for instance in self.gpu_instances.values()
                        if instance.check_and_set_availability()
                    ]

            if not available_gpus:
                logger.warning("No available GPUs to start initial services.")
                return []

        gpus_to_use = available_gpus[:min(self.replicas, len(available_gpus))]
        
        # Asynchronously check port availability
        ports_to_use = await self._find_available_ports(len(gpus_to_use))
        
        if len(ports_to_use) < len(gpus_to_use):
            logger.warning(f"Warning: Only found {len(ports_to_use)} available ports for {len(gpus_to_use)} GPUs")
            gpus_to_use = gpus_to_use[:len(ports_to_use)]
        
        logger.info(f"Using GPUs: {gpus_to_use}, Ports: {ports_to_use}")
        
        # Start service instances (without holding lock)
        tasks = [
            self._add_new_service_instance(port, gpu_id, self.default_ckpt_path, self.default_lora_adapter_path) 
            for port, gpu_id in zip(ports_to_use, gpus_to_use)
        ]
        instances = await asyncio.gather(*tasks)
        
        return instances

    async def add_service_by_id(self, gpu_id: int):
        """Add service instance by GPU ID"""
        ports = await self._find_available_ports(1)
        if not ports:
            return None
        
        port_to_use = ports[0]
        instance = await self._add_new_service_instance(port_to_use, gpu_id, self.default_ckpt_path, self.default_lora_adapter_path)

        if not await self.wait_for_model_pool_ready([instance]):
            return None
        return instance

    async def _add_new_service_instance(
        self,
        port: int,
        gpu_id: int,
        ckpt_path: str,
        lora_adapter_path: Optional[str] = None,
    ) -> ServiceInstance:
        """Start a new service instance and add it to the pool."""
        logger.info(
            f"Attempting to start service on port {port} with GPU {gpu_id} from ckpt_path {ckpt_path}"
            f" lora_adapter_path={lora_adapter_path}..."
        )
        
        # Mark GPU as unavailable
        async with self.gpu_lock:
            if gpu_id in self.gpu_instances:
                self.gpu_instances[gpu_id].is_available = False
        
        proc = self._launch_vllm_process(port, gpu_id, ckpt_path, lora_adapter_path=lora_adapter_path)
        if not proc:
            logger.error(f"Failed to launch process on port {port}.")
            # Restore GPU availability
            async with self.gpu_lock:
                if gpu_id in self.gpu_instances:
                    self.gpu_instances[gpu_id].is_available = True
            raise RuntimeError(f"Failed to launch vLLM process on GPU {gpu_id}")
        
        instance = ServiceInstance(
            port,
            gpu_id,
            proc,
            ckpt_path,
            lora_adapter_path=lora_adapter_path,
            served_model_name="default" if lora_adapter_path else ckpt_path,
        )
        logger.info(f"Successfully started service instance: {instance}")
        
        # Start log monitoring
        asyncio.create_task(self._stream_process_output(instance))

        return instance

    def _launch_vllm_process(
        self,
        port: int,
        gpu_id: int,
        ckpt_path: str,
        lora_adapter_path: Optional[str] = None,
    ) -> Optional[subprocess.Popen]:
        """Launch vLLM subprocess."""
        env = os.environ.copy()
        gpu_instance = self.gpu_instances.get(gpu_id)
        env["CUDA_VISIBLE_DEVICES"] = (
            gpu_instance.cuda_visible_device if gpu_instance else str(gpu_id)
        )
        
        vllm_command = [
            "vllm", "serve", ckpt_path,
            "--trust-remote-code",
            "--port", str(port)
        ]
        if lora_adapter_path:
            self._validate_lora_adapter_path(lora_adapter_path)
            vllm_command.extend([
                "--enable-lora",
                "--max-lora-rank", str(self.model_cfg.max_lora_rank),
                "--lora-modules", f"default={lora_adapter_path}",
            ])

        # Dynamically add parameters from vllm_params
        for key, value in self.vllm_params.items():
            cli_key = f"--{key}"
            
            if value == "store_true":
                vllm_command.append(cli_key)
            elif value is not None:
                # Otherwise, add key-value pairs
                vllm_command.append(cli_key)
                vllm_command.append(str(value))
                
        logger.info(f"Launching vLLM command: {vllm_command}")
        
        try:
            return subprocess.Popen(
                vllm_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                env=env, 
                bufsize=1, 
                universal_newlines=True,
                start_new_session=True,
            )
        except (OSError, FileNotFoundError) as e:
            logger.error(f"Error launching vLLM process on port {port}: {e}")
            return None

    async def _remove_service_instance(
        self,
        gpu_id: int,
        kill: bool = False,
        expected_instance: Optional[ServiceInstance] = None,
    ):
        """Stop and clean up the specified vllm serve instance, and release GPU resources."""
        instance = None
        async with self.instances_lock:
            current = self.service_instances.get(gpu_id)
            if expected_instance is not None and current is None:
                instance = None
            elif expected_instance is not None and current is not expected_instance:
                logger.info(
                    f"Skip removing GPU {gpu_id}: instance changed or was already removed."
                )
                return
            else:
                instance = self.service_instances.pop(gpu_id, None)

        if not instance:
            logger.warning(f"No service instance found on GPU {gpu_id} to remove.")
            if expected_instance is not None:
                async with self.gpu_lock:
                    if gpu_id in self.gpu_instances:
                        self.gpu_instances[gpu_id].is_available = True
            return

        logger.info(f"Shutting down service on GPU {gpu_id}, port {instance.port} (PID: {instance.process.pid if instance.process else 'N/A'})...")
        
        if instance.process and instance.process.poll() is None:
            proc = instance.process
            try:
                # First try graceful termination
                if kill:
                    proc.kill()
                else:
                    proc.terminate()
                
                # Wait for process termination, increase timeout
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=10)
                
            except asyncio.TimeoutError:
                # Force kill process
                logger.warning(f"Process {proc.pid} did not terminate gracefully. Force killing...")
                try:
                    proc.kill()
                    await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    logger.warning(f"Process {proc.pid} already terminated or inaccessible")
            
            # Ensure process group is also cleaned up
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, OSError, PermissionError):
                pass
        
        # Release GPU resources
        async with self.gpu_lock:
            if gpu_id in self.gpu_instances:
                self.gpu_instances[gpu_id].is_available = True
                logger.info(f"GPU {gpu_id} has been marked as available again")
        
        logger.info(f"Service on GPU {gpu_id} has been completely removed and GPU released")

    async def _shutdown_all_services(self):
        """Shutdown all running service instances."""
        logger.info("Shutting down all model services...")
        
        # Get all GPU IDs
        async with self.instances_lock:
            gpu_ids = list(self.service_instances.keys())
        
        shutdown_tasks = [self._remove_service_instance(gpu_id) for gpu_id in gpu_ids]
        await asyncio.gather(*shutdown_tasks)
        
        async with self.instances_lock:
            self.service_instances.clear()

    async def _shutdown_n_services(self, gpu_ids: List[int]):
        """
        Stop and remove the specified number of model service instances.
        """
        logger.info(f"Attempting to remove services on GPUs: {gpu_ids}")
        if not gpu_ids:
            return []

        shutdown_tasks = [self._remove_service_instance(gpu_id) for gpu_id in gpu_ids]
        await asyncio.gather(*shutdown_tasks)
        return gpu_ids
    
    async def _add_n_service_instance(self, count: int):
        """
        Start the specified number of new model service instances.
        """
        logger.info(f"Attempting to add {count} new model service(s)...")
        if count <= 0:
            return []

        # Quickly check available GPUs (hold lock for very short time)
        async with self.gpu_lock:
            available_gpus = [
                gpu.gpu_id for gpu in self.gpu_instances.values()
                if gpu.check_and_set_availability()
            ]

        if not available_gpus:
            logger.warning("Warning: No available GPUs to start new services.")
            return []

        gpus_to_use = available_gpus[:min(count, len(available_gpus))]
        logger.info(f">>>>>>> gpu to run {gpus_to_use}")
        
        if len(gpus_to_use) < count:
            logger.warning(f"Warning: Not enough free GPUs. Will start {len(gpus_to_use)} instead of {count}.")

        # Asynchronously check port availability (without holding lock)
        ports_to_use = await self._find_available_ports(len(gpus_to_use))
        
        if len(ports_to_use) < len(gpus_to_use):
            logger.warning(f"Warning: Not enough free ports. Starting {len(ports_to_use)} services.")
            gpus_to_use = gpus_to_use[:len(ports_to_use)]

        if not gpus_to_use:
            return []

        logger.info(f"Starting services on GPUs: {gpus_to_use}, Ports: {ports_to_use}")
        
        # Start service instances (without holding lock)
        tasks = [
            self._add_new_service_instance(port, gpu_id, self.default_ckpt_path, self.default_lora_adapter_path) 
            for port, gpu_id in zip(ports_to_use, gpus_to_use)
        ]
        instances = await asyncio.gather(*tasks)
        
        # Wait for services to be ready (without holding lock)
        if not await self.wait_for_model_pool_ready(instances):
            raise Exception("add new instance failed")
        
        # Quickly add instances to dictionary
        async with self.instances_lock:
            for instance in instances:
                self.service_instances[instance.gpu_id] = instance
        
        return True

    async def _monitor_replicas(self):
        """Background monitoring task, regularly checks and maintains replica count."""
        await asyncio.sleep(120)
        while True:
            try:
                await self._check_all_service_health()
                await self._recover_replicas_now("periodic monitor")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
            await asyncio.sleep(30)

    async def _recover_replicas_now(self, reason: str):
        """Recover missing replicas immediately, serialized to avoid duplicate startups."""
        if self.shutting_down or not self.recovery_enabled:
            return

        async with self.recovery_lock:
            if self.shutting_down or not self.recovery_enabled:
                return
            logger.info(f"Replica recovery triggered by: {reason}")
            await self._ensure_replicas()

    async def _ensure_replicas(self):
        """Core recovery logic: Check current service status, start missing replicas."""
        # Quickly get current status
        async with self.instances_lock:
            current_service_gpus = set(self.service_instances.keys())
            num_active = len(current_service_gpus)

        if num_active >= self.replicas:
            return

        logger.info(f"Replica check: Found {num_active}/{self.replicas} active instances. Attempting to restore...")
        needed = self.replicas - num_active
        
        # Quickly get available GPUs and ports
        async with self.gpu_lock:
            available_gpus = [
                instance.gpu_id for instance in self.gpu_instances.values() 
                if instance.check_and_set_availability()
            ]
        
        # Asynchronously check ports
        ports_to_create = await self._find_available_ports(min(len(available_gpus), needed))
        
        if not available_gpus:
            if num_active == 0:
                cleaned = await self._cleanup_stale_model_processes()
                if cleaned:
                    await asyncio.sleep(5)
                    async with self.gpu_lock:
                        available_gpus = [
                            instance.gpu_id for instance in self.gpu_instances.values()
                            if instance.check_and_set_availability()
                        ]
                    ports_to_create = await self._find_available_ports(min(len(available_gpus), needed))

                    if available_gpus:
                        logger.info(
                            f"Recovered GPU availability after cleaning stale model processes: {available_gpus}"
                        )
                    else:
                        logger.warning(
                            "Stale model cleanup ran, but GPUs are still not available."
                        )

            if available_gpus:
                num_to_start = min(len(ports_to_create), len(available_gpus), needed)
            else:
                logger.warning("Warning: Cannot restore replicas, no free GPUs available.")
                return

        else:
            num_to_start = min(len(ports_to_create), len(available_gpus), needed)

        if not available_gpus:
            logger.warning("Warning: Cannot restore replicas, no free GPUs available.")
            return

        if num_to_start > 0:
            logger.info(f"Found resources to start {num_to_start} new instance(s).")
            
            # Start instances (without holding lock)
            tasks = [
                self._add_new_service_instance(
                    ports_to_create[i], 
                    available_gpus[i], 
                    self.default_ckpt_path,
                    self.default_lora_adapter_path
                ) for i in range(num_to_start)
            ]
            instances = await asyncio.gather(*tasks)
            
            # Wait for readiness
            if not await self.wait_for_model_pool_ready(instances):
                raise Exception("add new instance failed")
            
            # Quickly add to dictionary
            async with self.instances_lock:
                for instance in instances:
                    self.service_instances[instance.gpu_id] = instance

    async def _check_all_service_health(self):
        """
        Asynchronously check the health status of all service instances.
        If unhealthy instances are found, try to shut down and remove them.
        """
        logger.info("Checking health of all running services...")
        
        # Get instance snapshot (hold lock for very short time)
        async with self.instances_lock:
            current_instances = list(self.service_instances.values())
        
        if not current_instances:
            return
        
        health_checks = [
            self._check_service_health(instance.port) 
            for instance in current_instances
        ]
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        # Handle unhealthy instances (without holding lock)
        unhealthy_instances = []
        for instance, is_healthy in zip(current_instances, results):
            if isinstance(is_healthy, Exception) or not is_healthy:
                logger.warning(f"Service on GPU {instance.gpu_id} (port {instance.port}) is unhealthy. Attempting to remove.")
                unhealthy_instances.append(instance.gpu_id)
            else:
                logger.info(f"Service on GPU {instance.gpu_id} (port {instance.port}) is healthy.")
        
        # Asynchronously remove unhealthy instances
        if unhealthy_instances:
            removal_tasks = [
                self._remove_service_instance(gpu_id) 
                for gpu_id in unhealthy_instances
            ]
            await asyncio.gather(*removal_tasks)

    async def _check_service_health(self, port: int, timeout: int = 5) -> bool:
        """Check the health of a single service through vLLM's /health endpoint."""
        url = f"http://localhost:{port}/health"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False

    def _get_gpu_uuid(self, gpu: GPUInstance) -> Optional[str]:
        """Return the NVML UUID for a GPU instance."""
        if gpu._handle is None:
            return None
        try:
            uuid = pynvml.nvmlDeviceGetUUID(gpu._handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            return uuid
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to read UUID for GPU {gpu.gpu_id}: {e}")
            return None

    def _get_process_command(self, pid: int) -> str:
        """Best-effort command line lookup for process classification."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _is_stale_model_process(self, pid: int, process_name: str) -> bool:
        """Conservatively identify orphaned vLLM workers owned by this model service."""
        current_pid = os.getpid()
        if pid == current_pid:
            return False

        known_pids = {
            inst.process.pid
            for inst in self.service_instances.values()
            if inst.process is not None
        }
        if pid in known_pids:
            return False

        cmd = self._get_process_command(pid)
        process_text = f"{process_name} {cmd}"

        if "ray::WorkerDict" in process_text:
            return False
        if "verl.trainer" in process_text:
            return False
        if "python -m src.run_model" in process_text:
            return False

        if "vllm serve" in process_text:
            return True
        if "multiprocessing.spawn" in process_text and "spawn_main" in process_text:
            return True

        return False

    def _is_likely_stale_model_worker(self, pid: int, process_name: str, used_memory_mib: int) -> bool:
        """
        Fallback for container/NVML UUID mismatches.
        Only matches large standalone Python multiprocessing workers, and still excludes
        trainer/ray/current-service commands in _is_stale_model_process().
        """
        if used_memory_mib < 1024:
            return False
        return self._is_stale_model_process(pid, process_name)

    def _scan_proc_for_stale_model_processes(self) -> List[int]:
        """
        Fallback for containers where nvidia-smi reports memory usage but hides process IDs.
        This scans the container PID namespace for stale vLLM workers.
        """
        stale_pids = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue

            pid = int(name)
            cmd = self._get_process_command(pid)
            if not cmd:
                continue

            process_name = os.path.basename(cmd.split(" ", 1)[0])
            if self._is_stale_model_process(pid, process_name):
                stale_pids.append(pid)
                logger.warning(
                    f"Detected stale model process from /proc scan: pid={pid}, cmd={cmd}"
                )

        return stale_pids

    async def _terminate_stale_processes(self, stale_pids: List[int]) -> bool:
        """Terminate stale backend process groups."""
        if not stale_pids:
            return False

        stale_pids = sorted(set(stale_pids))

        for pid in stale_pids:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, 15)
                logger.warning(f"Sent SIGTERM to stale model process group {pgid} for pid {pid}")
            except ProcessLookupError:
                continue
            except PermissionError as e:
                logger.warning(f"No permission to terminate stale model process {pid}: {e}")
            except OSError as e:
                logger.warning(f"Failed to terminate stale model process {pid}: {e}")

        await asyncio.sleep(5)

        for pid in stale_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue

            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, 9)
                logger.warning(f"Sent SIGKILL to stale model process group {pgid} for pid {pid}")
            except ProcessLookupError:
                continue
            except PermissionError as e:
                logger.warning(f"No permission to kill stale model process {pid}: {e}")
            except OSError as e:
                logger.warning(f"Failed to kill stale model process {pid}: {e}")

        return True

    async def _cleanup_stale_model_processes(self) -> bool:
        """
        Clean orphaned vLLM workers on GPUs visible to this service.
        This handles the case where the pool has 0 instances, no ports are listening,
        but dead backend workers still hold GPU memory and block recovery.
        """
        visible_uuids = {
            uuid for uuid in (
                self._get_gpu_uuid(gpu) for gpu in self.gpu_instances.values()
            )
            if uuid
        }
        if not visible_uuids:
            logger.warning(
                "Stale model cleanup could not read visible GPU UUIDs; falling back to /proc scan."
            )
            return await self._terminate_stale_processes(
                self._scan_proc_for_stale_model_processes()
            )

        logger.warning(
            f"Checking for stale model processes on visible GPU UUIDs: {sorted(visible_uuids)}"
        )

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Failed to query GPU processes for stale cleanup: {e}")
            return await self._terminate_stale_processes(
                self._scan_proc_for_stale_model_processes()
            )

        if result.returncode != 0:
            logger.warning(f"nvidia-smi process query failed: {result.stderr}")
            return await self._terminate_stale_processes(
                self._scan_proc_for_stale_model_processes()
            )

        stale_pids = []
        seen_visible_gpu_process = False
        fallback_stale_pids = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue

            gpu_uuid, pid_text, process_name, used_memory = parts[:4]
            try:
                pid = int(pid_text)
                used_memory_mib = int(used_memory)
            except ValueError:
                continue

            logger.info(
                f"Observed GPU process during stale cleanup: gpu_uuid={gpu_uuid}, "
                f"pid={pid}, process={process_name}, used_memory={used_memory_mib}MiB"
            )

            if gpu_uuid in visible_uuids:
                seen_visible_gpu_process = True
            elif self._is_likely_stale_model_worker(pid, process_name, used_memory_mib):
                fallback_stale_pids.append(pid)
                logger.warning(
                    f"Detected likely stale model process with unmatched GPU UUID: "
                    f"gpu_uuid={gpu_uuid}, pid={pid}, process={process_name}, "
                    f"used_memory={used_memory_mib}MiB"
                )
                continue
            else:
                continue

            if self._is_stale_model_process(pid, process_name):
                stale_pids.append(pid)
                logger.warning(
                    f"Detected stale model process on visible GPU: "
                    f"pid={pid}, process={process_name}, used_memory={used_memory}MiB"
                )

        if not stale_pids and not seen_visible_gpu_process and fallback_stale_pids:
            logger.warning(
                "No GPU process UUID matched the service-visible UUIDs; "
                "using likely stale model worker fallback."
            )
            stale_pids = fallback_stale_pids

        if not stale_pids:
            logger.warning(
                "No stale model processes found from nvidia-smi; falling back to /proc scan."
            )
            stale_pids = self._scan_proc_for_stale_model_processes()

        if not stale_pids:
            logger.warning("No stale model processes found during cleanup.")
            return False

        return await self._terminate_stale_processes(stale_pids)

    async def _wait_for_service_ready(self, instance: ServiceInstance, timeout: int = 1200) -> bool:
        """Wait for the service on the specified GPU to be ready."""
        start_time = time.time()
        logger.info(f"Waiting for service on GPU {instance.gpu_id} to be ready...")
        
        while time.time() - start_time < timeout:
            if not instance.process or instance.process.poll() is not None:
                logger.error(f"Process for GPU {instance.gpu_id} exited prematurely.")
                return False

            if await self._check_service_health(instance.port):
                logger.info(f"Service on GPU {instance.gpu_id} (port {instance.port}) is ready.")
                return True
            
            # Yield control to avoid blocking event loop
            await asyncio.sleep(1)
        
        logger.error(f"Timeout: Service on GPU {instance.gpu_id} did not become ready within {timeout}s.")
        return False

    async def wait_for_model_pool_ready(self, instances: List[ServiceInstance], timeout: int = 600):
        """
        Wait for at least one model service instance to start and be healthy, until the expected replica count is reached.
        """
        if not instances:
            return False
            
        ports = [inst.port for inst in instances]
        
        logger.info(f"Waiting for {len(ports)} service(s) to become ready (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            num_ready = 0
            tasks = [self._check_service_health(port) for port in ports]
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if isinstance(result, bool) and result:
                    num_ready += 1
            
            if num_ready >= len(ports):
                logger.info(f"All {len(ports)} service(s) are ready.")
                return True
            
            # Yield control to avoid blocking event loop
            await asyncio.sleep(1)
        
        logger.error(f"Timeout: {len(instances)} service(s) did not become ready within {timeout}s.")
        return False

    def _get_gpu_count(self) -> int:
        """Get available GPU count."""
        return len(self._get_visible_gpu_devices())

    def _get_visible_gpu_devices(self) -> List[str]:
        """Return CUDA-visible GPU identifiers, preserving CUDA_VISIBLE_DEVICES mapping."""
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if devices:
                return [device.strip() for device in devices.split(",") if device.strip()]
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return [
                    str(index)
                    for index, line in enumerate(lines)
                    if line.startswith('GPU ')
                ]
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return []

    def _cuda_visible_device_to_nvml_index(self, cuda_visible_device: str) -> int:
        """Map CUDA_VISIBLE_DEVICES entries to NVML indices for memory checks."""
        if cuda_visible_device.isdigit():
            return int(cuda_visible_device)

        try:
            target_uuid = cuda_visible_device
            device_count = pynvml.nvmlDeviceGetCount()
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8")
                if uuid == target_uuid:
                    return index
        except pynvml.NVMLError as e:
            logger.error(f"Failed to map CUDA device {cuda_visible_device} to NVML index: {e}")

        logger.warning(
            f"Could not map CUDA_VISIBLE_DEVICES entry {cuda_visible_device}; "
            "falling back to NVML index 0."
        )
        return 0

    def _get_available_gpus(self) -> List[int]:
        """Get list of all GPU IDs."""
        return list(self.gpu_instances.keys())

    async def _stream_process_output(self, instance: ServiceInstance, max_log_length: int = 400):
        """Asynchronously read and print subprocess output stream, truncate overly long content."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        proc = instance.process
        port = instance.port

        async def reader(stream, prefix):
            while True:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break

                clean_line = ansi_escape.sub('', line.rstrip())

                # Only keep first max_log_length characters
                if len(clean_line) > max_log_length:
                    clean_line = f"{clean_line[:max_log_length]}... (truncated {len(clean_line) - max_log_length} chars)"

                logger.info(f"[{prefix}] {clean_line}")

        stdout_task = asyncio.create_task(reader(proc.stdout, f"Port {port} STDOUT"))
        stderr_task = asyncio.create_task(reader(proc.stderr, f"Port {port} STDERR"))

        return_code = await asyncio.to_thread(proc.wait)
        logger.info(f"Process for port {port} terminated with return code {return_code}.")

        stdout_task.cancel()
        stderr_task.cancel()

        if not self.shutting_down:
            logger.warning(
                f"Service process on GPU {instance.gpu_id}, port {port} exited. "
                "Removing it from pool and starting recovery."
            )
            await self._remove_service_instance(
                instance.gpu_id,
                kill=False,
                expected_instance=instance,
            )
            await self._recover_replicas_now(f"process exit on port {port}")
        
    
    @asynccontextmanager
    async def _get_endpoint_for_request(self):
        """Use async context manager to gracefully handle endpoint acquisition and counter management."""
        instance = None
        dead_instances = []
        try:
            async with self.instances_lock:
                active_instances = list(self.service_instances.values())
                for active_instance in active_instances:
                    if active_instance.process and active_instance.process.poll() is not None:
                        dead_instances.append(active_instance)

                active_instances = [
                    active_instance
                    for active_instance in active_instances
                    if active_instance.process and active_instance.process.poll() is None
                ]

                if not active_instances:
                    for dead_instance in dead_instances:
                        asyncio.create_task(
                            self._remove_service_instance(
                                dead_instance.gpu_id,
                                expected_instance=dead_instance,
                            )
                        )
                    asyncio.create_task(self._recover_replicas_now("no live endpoints"))
                    raise Exception("No available endpoints in the pool.")
                
                instance = min(active_instances, key=lambda x: x.requests_in_flight)
                instance.requests_in_flight += 1

            for dead_instance in dead_instances:
                asyncio.create_task(
                    self._remove_service_instance(
                        dead_instance.gpu_id,
                        expected_instance=dead_instance,
                    )
                )
            if dead_instances:
                asyncio.create_task(self._recover_replicas_now("dead endpoint skipped"))
            
            logger.info(f"Routing request to {instance.endpoint} (GPU {instance.gpu_id}, in-flight: {instance.requests_in_flight})")
            yield instance
        
        finally:
            if instance:
                async with self.instances_lock:
                    instance.requests_in_flight = max(0, instance.requests_in_flight - 1)

    async def _quarantine_failed_instance(self, instance: ServiceInstance, reason: str):
        """Remove a failed instance and trigger async replica recovery."""
        logger.warning(
            f"Quarantining service on GPU {instance.gpu_id}, port {instance.port}: {reason}"
        )
        await self._remove_service_instance(
            instance.gpu_id,
            kill=True,
            expected_instance=instance,
        )
        asyncio.create_task(self._recover_replicas_now(reason))

    async def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Use load balancing strategy to send a chat request to the model service pool."""
        try:
            async with self._get_endpoint_for_request() as instance:
                logger.info(f"Using Service Instance ->>> {instance}")
                if not instance:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                        detail="Model service pool is not ready or has no available endpoints."
                    )
                
                url = f"{instance.endpoint}/v1/chat/completions"
                data = {"model": instance.served_model_name, "messages": messages, **kwargs}

                if self.save_local:
                    try:
                        last_image_item = messages[-1]["content"][-1]
                        assert last_image_item.get("type") == "image_url"
                        last_image_url = last_image_item.get("image_url", {}).get("url", "")
                        assert last_image_url.startswith("data:image")
                        header, encoded = last_image_url.split(",", 1)

                        task_id = kwargs.get("task_id")
                        trace_id = kwargs.get("trace_id")
                        step = kwargs.get("step")
                        save_dir = os.path.join(self.save_path, f"{task_id}_{trace_id}")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"image_{int(step)}.png")
                        with open(save_path, "wb") as f:
                            f.write(base64.b64decode(encoded))
                    except Exception as e:
                        logger.error(f"❌ Failed to decode or save image: {e}")
                        raise

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=data) as response:
                            if not response.ok:
                                error_text = await response.text()
                                raise HTTPException(status_code=response.status, detail=f"API call failed: {error_text}")
                            
                            response_data = await response.json()
                            if instance.lora_adapter_path:
                                response_data["model"] = instance.model_version
                except (
                    aiohttp.ClientConnectionError,
                    aiohttp.ServerDisconnectedError,
                    aiohttp.ClientOSError,
                    asyncio.TimeoutError,
                ) as e:
                    await self._quarantine_failed_instance(
                        instance,
                        f"generate connection failure: {e}",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Model backend disconnected; recovery started: {e}",
                    )
                        
                if self.save_local:
                    try:
                        content = response_data["choices"][0]["message"]["content"]
                        model = response_data["model"]

                        logp_list, token_id_list = None, None

                        # If logprobs requested in parameters
                        if kwargs.get("logprobs", False):
                            try:
                                logp_list = [
                                    item["logprob"]
                                    for item in response_data["choices"][0]["logprobs"]["content"]
                                ]
                            except (KeyError, IndexError, TypeError):
                                logp_list = None

                        # If return token_id requested in parameters
                        if kwargs.get("return_tokens_as_token_ids", False):
                            try:
                                token_id_list = [
                                    int(item["token"].split("token_id:")[1])
                                    for item in response_data["choices"][0]["logprobs"]["content"]
                                    if "token_id:" in item["token"]
                                ]
                            except (KeyError, IndexError, TypeError, ValueError):
                                token_id_list = None
                        
                        task_id = kwargs.get("task_id")
                        trace_id = kwargs.get("trace_id")
                        step = kwargs.get("step")
                        save_dir = os.path.join(self.save_path, f"{task_id}_{trace_id}")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"data_for_step_{int(step)+1}.pt")

                        data_to_save = {
                            "logp": torch.tensor(logp_list).cpu() if logp_list is not None else torch.tensor([]).cpu(),
                            "token_ids": torch.tensor(token_id_list).cpu() if token_id_list is not None else torch.tensor([]).cpu(),
                        }

                        torch.save(data_to_save, save_path)
                    except Exception as e:
                        logger.error(f"❌ Failed to save logp/token_id tensors: {e}")
                        raise

                try:
                    return response_data
                except (KeyError, IndexError, TypeError) as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                        detail=f"Failed to parse response: {e}. Full response: {response_data}"
                    )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during generate call: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def save(
        self,
        messages: List[Dict],
        reward: float,
        task_id: str,
        trace_id: str,
        parsed_actions: Optional[List[Dict[str, Any]]] = None,
    ):
        if not self.save_local:
            return {"status": "skipped"}

        try:
            save_dir = os.path.join(self.save_path, f"{task_id}_{trace_id}")
            os.makedirs(save_dir, exist_ok=True)

            # Save messages
            messages_path = os.path.join(save_dir, f"final_messages.json")
            with open(messages_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)

            if parsed_actions is not None:
                parsed_actions_path = os.path.join(save_dir, "parsed_actions.json")
                with open(parsed_actions_path, "w", encoding="utf-8") as f:
                    json.dump(parsed_actions, f, ensure_ascii=False, indent=2)

            # Save reward
            reward_path = os.path.join(save_dir, f"reward.txt")
            with open(reward_path, "w") as f:
                f.write(str(reward))

            return {"status": "success"}

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save data: {e}"
            )
        
    async def tokenize(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Call the /v1/tokenize endpoint of the model service pool"""
        try:
            async with self._get_endpoint_for_request() as instance:
                logger.info(f"Using Service Instance for tokenize ->>> {instance}")
                if not instance:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Model service pool is not ready or has no available endpoints."
                    )

                url = f"{instance.endpoint}/tokenize"
                data = {"model": instance.served_model_name, "prompt": input_text}

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=data) as response:
                            if not response.ok:
                                error_text = await response.text()
                                raise HTTPException(status_code=response.status, detail=f"Tokenize API failed: {error_text}")
                            
                            response_data = await response.json()
                except (
                    aiohttp.ClientConnectionError,
                    aiohttp.ServerDisconnectedError,
                    aiohttp.ClientOSError,
                    asyncio.TimeoutError,
                ) as e:
                    await self._quarantine_failed_instance(
                        instance,
                        f"tokenize connection failure: {e}",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Model backend disconnected; recovery started: {e}",
                    )

                try:
                    return response_data
                except (KeyError, IndexError, TypeError) as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to parse tokenize response: {e}. Full response: {response_data}"
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during tokenize call: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    async def _find_available_ports(self, count: int, start_port: int = None, continuous: bool = False) -> List[int]:
        """
        Asynchronously discover the specified number of available ports.
        """
        if start_port is None:
            start_port = self.base_port
            
        available_ports = []
        port = start_port
        max_port = start_port + 1000  # Prevent infinite loop
        
        def _check_port_bind(port_num: int) -> bool:
            """Synchronously check if port is available"""
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", port_num))
                    return True
            except OSError:
                return False
        
        while len(available_ports) < count and port <= max_port:
            is_free = await asyncio.to_thread(_check_port_bind, port)
            
            if is_free:
                if continuous and available_ports and port != available_ports[-1] + 1:
                    available_ports.clear()
                    available_ports.append(port)
                else:
                    available_ports.append(port)
                
                if len(available_ports) == count:
                    break
            else:
                if continuous and available_ports:
                    available_ports.clear()
            
            port += 1
            
        if len(available_ports) < count:
            logger.warning(f"Warning: Could only find {len(available_ports)}/{count} available ports")
            
        return available_ports[:count]

    async def reload(self, new_ckpt_path: str, lora_adapter_path: Optional[str] = None):
        """
        Full delete and restart mode, temporarily not providing fastapi interface
        """
        new_ckpt_path = self._resolve_checkpoint_path(new_ckpt_path)
        logger.info(f"\n--- Reloading model pool to: {new_ckpt_path}, lora_adapter_path={lora_adapter_path} ---")
        self._validate_model_path(new_ckpt_path)
        if lora_adapter_path:
            self._validate_lora_adapter_path(lora_adapter_path)
        
        # Update configuration
        async with self.config_lock:
            self.default_ckpt_path = new_ckpt_path
            self.default_lora_adapter_path = lora_adapter_path
        
        # Pause background monitoring
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        
        # Shutdown all services
        await self._shutdown_all_services()
        
        # Re-initialize
        await self.initialize()
        
        logger.info("--- Model pool reloaded successfully ---")
        return True

    async def roll_reload(self, new_ckpt_path: str, batch_size: int = 1, lora_adapter_path: Optional[str] = None):
        """
        Smoothly reload the model through rolling updates.
        Update [batch_size] instances at a time.
        """
        requested_ckpt_path = new_ckpt_path
        requested_lora_adapter_path = lora_adapter_path
        new_ckpt_path = self._resolve_checkpoint_path(new_ckpt_path)
        if lora_adapter_path:
            self._validate_lora_adapter_path(lora_adapter_path)

        if self.replicas >= 4:
            batch_size = max(2, batch_size)
        if self.replicas < 4:
            batch_size = 1

        logger.info(
            f"\n--- Rolling reload to new model: {new_ckpt_path} "
            f"lora_adapter_path={lora_adapter_path} "
            f"(requested: {requested_ckpt_path}, requested_lora: {requested_lora_adapter_path}, batch size: {batch_size}) ---"
        )
        self._validate_model_path(new_ckpt_path)
        
        # Pause background monitoring to prevent interference with update process
        if self.monitoring_task and not self.monitoring_task.done():
            logger.info("pause monitoring task")
            self.monitoring_task.cancel()

        # Quickly get current instance information (hold lock for very short time)
        async with self.instances_lock:
            current_instances = list(self.service_instances.values())
            old_instances_gpu_ids = [
                inst.gpu_id for inst in current_instances
                if inst.ckpt_path != new_ckpt_path or inst.lora_adapter_path != lora_adapter_path
            ]
        
        logger.info(f">>>>> old gpu ids : {old_instances_gpu_ids}")

        # Set default model path before any early return. If all backends are down
        # during reload, recovery must still start the requested checkpoint.
        async with self.config_lock:
            if self.default_ckpt_path != new_ckpt_path or self.default_lora_adapter_path != lora_adapter_path:
                self.last_ckpt_path = self.default_ckpt_path
                self.last_lora_adapter_path = self.default_lora_adapter_path
                self.default_ckpt_path = new_ckpt_path
                self.default_lora_adapter_path = lora_adapter_path

        if not current_instances:
            logger.warning(
                "No active model instances during reload; updated default checkpoint "
                f"to {new_ckpt_path} and will restore replicas with this checkpoint."
            )
            try:
                await self._ensure_replicas()
            finally:
                logger.info("restart monitoring task")
                self.monitoring_task = asyncio.create_task(self._monitor_replicas())
            logger.info(f"--- Rolling reload finished with new model: {new_ckpt_path}, lora_adapter_path={lora_adapter_path} ---")
            return True

        if not old_instances_gpu_ids:
            logger.info("All instances are already running the target model. Reload skipped.")
            # Restart monitoring
            self.monitoring_task = asyncio.create_task(self._monitor_replicas())
            return True

        # Update in batches, yield control between batches
        try:
            for i in range(0, len(old_instances_gpu_ids), batch_size):
                batch_gpu_ids = old_instances_gpu_ids[i:i + batch_size]
                logger.info(f"--- Reloading batch: GPUs {batch_gpu_ids} ---")

                # 1. Remove old instances
                logger.info(f"Removing {len(batch_gpu_ids)} old instance(s)...")
                await self._shutdown_n_services(batch_gpu_ids)
                
                # 2. Start new instances
                logger.info(f"Adding {len(batch_gpu_ids)} new instance(s) with new model...")
                await self._add_n_service_instance(len(batch_gpu_ids))
                
                # Yield control to allow other coroutines to execute
                await asyncio.sleep(0)
        except Exception:
            async with self.config_lock:
                self.default_ckpt_path = self.last_ckpt_path
                self.default_lora_adapter_path = self.last_lora_adapter_path
            logger.exception("Rolling reload failed; restored previous default checkpoint path.")
            raise

        # Restart background monitoring
        logger.info("restart monitoring task")
        self.monitoring_task = asyncio.create_task(self._monitor_replicas())
        logger.info(f"--- Rolling reload finished with new model: {new_ckpt_path} ---")
        return True

    async def get_endpoints(self) -> List[str]:
        """
        Get all vllm serve endpoints internally
        Old version ModelServicePool method, will be discarded if not needed
        """
        async with self.instances_lock:
            return [instance.endpoint for instance in self.service_instances.values()]

    async def get_status(self) -> List[Dict]:
        """
        Get status of all service instances
        """
        async with self.instances_lock:
            return [
                {
                    "gpu_id": inst.gpu_id,
                    "port": inst.port,
                    "endpoint": inst.endpoint,
                    "ckpt_path": inst.ckpt_path,
                    "lora_adapter_path": inst.lora_adapter_path,
                    "served_model_name": inst.served_model_name,
                    "model_version": inst.model_version,
                    "requests_in_flight": inst.requests_in_flight,
                    "pid": inst.process.pid if inst.process else None
                }
                for inst in self.service_instances.values()
            ]
    
    async def get_gpu_info(self) -> List[Dict]:
        """Get GPU status"""
        async with self.gpu_lock:
            return [
                {
                    "gpu_id": ginst.gpu_id,
                    "is_available": ginst.is_available
                }
                for ginst in self.gpu_instances.values()
            ]
    
    async def get_checkpoint_info(self) -> Dict[str, Optional[str]]:
        """Get checkpoint path information"""
        async with self.config_lock:
            return {
                "current_ckpt_path": self.default_ckpt_path,
                "current_lora_adapter_path": self.default_lora_adapter_path,
                "current_model_version": self.default_lora_adapter_path or self.default_ckpt_path,
                "last_ckpt_path": self.last_ckpt_path,
                "last_lora_adapter_path": self.last_lora_adapter_path,
                "last_model_version": self.last_lora_adapter_path or self.last_ckpt_path
            }
