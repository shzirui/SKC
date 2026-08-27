import ray
import aiohttp
import asyncio
from typing import List, Dict
from datetime import datetime

def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:
        name_or_path (str): The name or path of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name or path of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor

@ray.remote
class ModelServicePool:
    """
    ModelServicePool is a Ray Actor for interacting with ModelService.
    It encapsulates the communication logic with the remote model service pool.
    """
    
    # --- Initialization Status ---
    def __init__(self, model_cfg):
        """
        Initialize the model service pool.

        Args:
            model_cfg: Model configuration containing service endpoint and checkpoint path
        """
        print("Initializing ModelServicePool...")
        self.service_url = model_cfg.service_endpoint
        print(f"Model Service Endpoint:>>> {self.service_url}")

        self.ckpt_path = model_cfg.ckpt_path

        # self.tokenizer = hf_tokenizer(self.ckpt_path, trust_remote_code=True)
        # self.processor = hf_processor(self.ckpt_path, trust_remote_code=True, use_fast=True)

    async def generate(self, messages: List[Dict[str, str]],  **kwargs) -> str:
        """
        Send a chat request to the model service pool using load balancing strategy.
        """
        payload = {
            "messages": messages,
            "parameters": kwargs
        }
        # print(f"payload:>>> {payload}")
        print(f"begin to generate at {datetime.now()}")
        generate_endpoint = self.service_url + "/generate"
        print(f"generate_endpoint:>>> {generate_endpoint}")

        async with aiohttp.ClientSession() as session:
            async with session.post(generate_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                try:
                    content = response_data["choices"][0]["message"]["content"]
                    model = response_data["model"]

                    logp_list, token_id_list = None, None
                    
                    if kwargs.get("logprobs", False):
                        try:
                            logp_list = [item["logprob"] for item in response_data["choices"][0]["logprobs"]["content"]]
                        except (KeyError, IndexError):
                            logp_list = None

                    if kwargs.get("return_tokens_as_token_ids", False):
                        try:
                            token_id_list = [int(item["token"].split("token_id:")[1]) for item in response_data["choices"][0]["logprobs"]["content"]]
                        except (KeyError, IndexError):
                            token_id_list = None
                        
                    return content, model, logp_list, token_id_list
                        
                except (KeyError, IndexError, TypeError) as e:
                    raise Exception(f"Failed to parse response: {e}. Full response: {response_data}")
                
    async def generate_vllm(self, messages: List[Dict[str, str]],  **kwargs) -> str:
        """
        Send a chat request to the vLLM OpenAI-compatible server.
        """
        # 1. 扁平化 Payload，并强制加入 model 名称（必须与启动脚本的 --served-model-name 一致）
        payload = {
            "model": "qwen3vl-8b",  # 【关键修改】必须带上模型名字
            "messages": messages,
        }
        # 将 kwargs 里的参数平铺放入 payload
        payload.update(kwargs)

        print(f"begin to generate at {datetime.now()}")
        
        # 2. 【关键修改】修正 Endpoint 路径为标准的 OpenAI 格式
        generate_endpoint = self.service_url + "/v1/chat/completions"
        print(f"generate_endpoint:>>> {generate_endpoint}")

        async with aiohttp.ClientSession() as session:
            async with session.post(generate_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                try:
                    content = response_data["choices"][0]["message"]["content"]
                    model = response_data["model"]
                        
                    return content, model
                        
                except (KeyError, IndexError, TypeError) as e:
                    raise Exception(f"Failed to parse response: {e}. Full response: {response_data}")

    async def tokenize(self, messages: str, **kwargs) -> List[int]:
        """
        Call the tokenize interface of the model service pool.
        """
        payload = {
            "prompt": messages,
            "parameters": kwargs
        }
        tokenize_endpoint = self.service_url + "/tokenize"
        async with aiohttp.ClientSession() as session:
            async with session.post(tokenize_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"Tokenize API failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                try:
                    return response_data["tokens"]
                except (KeyError, IndexError, TypeError) as e:
                    raise Exception(f"Failed to parse tokenize response: {e}. Full response: {response_data}")

    async def save_messages_reward(self, messages, reward, task_id, trace_id, parsed_actions=None):
        """
        Save messages and reward to the model service.
        """
        payload = {
            "messages": messages,
            "reward": reward,
            "task_id": task_id,
            "trace_id": trace_id,
        }
        if parsed_actions is not None:
            payload["parsed_actions"] = parsed_actions
        save_endpoint = self.service_url + "/save"
        async with aiohttp.ClientSession() as session:
            async with session.post(save_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"Save API call failed with status {response.status}: {error_text}")

                return await response.json()

    async def reload(self, new_ckpt_path: str, batch_size: int = 1):
        """
        Smoothly reload all model service instances to the new checkpoint path.
        
        Args:
            new_ckpt_path: New model checkpoint path
            batch_size: Number of instances to update at a time (rolling update batch size)
        """
        payload = {
            "new_ckpt_path": new_ckpt_path,
            "batch_size": batch_size
        }
        reload_endpoint = self.service_url + "/reload"
        async with aiohttp.ClientSession() as session:
            async with session.post(reload_endpoint, json=payload) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                return response_data

    async def shutdown(self): 
        """
        Shutdown all model service instances.
        """
        shutdown_endpoint = self.service_url + "/shutdown"
        async with aiohttp.ClientSession() as session:
            async with session.get(shutdown_endpoint) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")

    async def get_checkpoint_info(self) -> Dict[str, str]:
        """
        Get current and previous checkpoint path information.
        
        Returns:
            Dict[str, str]: Dictionary containing current and previous checkpoint paths
        """
        checkpoint_endpoint = self.service_url + "/checkpoint_info"
        async with aiohttp.ClientSession() as session:
            async with session.get(checkpoint_endpoint) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                
                response_data = await response.json()
                return response_data

    async def get_last_model_version(self) -> str:
        """
        Get the previous model version path.
        
        Returns:
            str: Path of the previous model version, or empty string if not exists
        """
        checkpoint_info = await self.get_checkpoint_info()
        return checkpoint_info.get("last_model_version") or checkpoint_info.get("last_ckpt_path", "")

    def process_images_sync(self, images):
        """
        Process images synchronously.
        """
        dummy_texts = [self.processor.image_token] * len(images)
        model_inputs = self.processor(text=dummy_texts, images=images, return_tensors="pt")
        image_grid_thw = model_inputs["image_grid_thw"]
        num_patches_list = (image_grid_thw[:,0]*image_grid_thw[:,1]*image_grid_thw[:,2]).tolist()
        pixel_values = model_inputs["pixel_values"]
        return image_grid_thw, num_patches_list, pixel_values

    async def process_images(self, images):
        """
        Process images asynchronously.
        """
        return await asyncio.to_thread(self.process_images_sync, images)
    
    def process_text_sync(self, messages):
        """
        Process text synchronously.
        """
        formatted = ""
        for m in messages:
            content = m["content"]

            # If content is list (multimodal message)
            if isinstance(content, list):
                # Only take text part
                texts = [c["text"] for c in content if c.get("type") == "text"]
                content_str = "\n".join(texts)
            else:
                # Regular string
                content_str = content

            formatted += f"{content_str}<|im_end|>\\n"
        model_inputs = self.processor(text=[formatted], images=None, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        return input_ids[0]

    async def process_text(self, messages):
        """
        Process text asynchronously.
        """
        return await asyncio.to_thread(self.process_text_sync, messages)