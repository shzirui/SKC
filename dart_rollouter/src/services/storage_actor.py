import torch
import aiofiles, json
from pathlib import Path
import ray, os
import asyncio

@ray.remote
class StorageActor:
    def __init__(self, storage_cfg):
        print(f">>>storage_cfg:>>> {storage_cfg}")
        self.root = Path(storage_cfg.root)
        print(f">>>root:>>> {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- save screenshot ----
    async def save_frame(self, task_root: str, step: int, png_bytes: bytes) -> str:
        """
        Save screenshot frame to file.
        
        Args:
            task_root: Task root directory name
            step: Step number
            png_bytes: PNG image bytes
            
        Returns:
            str: Relative path to the saved file
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"image_{step:04d}.png"
        async with aiofiles.open(fn, "wb") as f:
            await f.write(png_bytes)
        return str(fn.relative_to(self.root / task_root))
    
    # ---- save partial trajectory json ----
    async def save_partial_traj(self, task_root: str, step: int, partial_traj: list[dict]):
        """
        Save partial trajectory JSON file.
        
        Args:
            task_root: Task root directory name
            step: Step number
            partial_traj: Partial trajectory data
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"msg_for_prompt_{step}.json"
        async with aiofiles.open(fn, "w") as f:
            await f.write(json.dumps(partial_traj, ensure_ascii=False, indent=2))

    # ---- save timestamp info ----
    async def save_timestamp(self, task_root: str, trace_id: str, step: int, timestamp_info: dict):
        """
        Save timestamp information.
        
        Args:
            task_root: Task root directory name
            trace_id: Trace identifier
            step: Step number
            timestamp_info: Timestamp information dictionary
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"{trace_id}/{trace_id}/timestamp_info_{step}.json"
        async with aiofiles.open(fn, "w") as f:
            await f.write(json.dumps(timestamp_info, ensure_ascii=False, indent=2))

    # ---- save vllm logp ----
    async def save_partial_pt(self, task_root: str, step: int, logp: list[float], token_ids: list[int] = None, prompt_token_ids: list[int] = None):
        """
        Save vLLM log probabilities as PyTorch tensor file.
        
        Args:
            task_root: Task root directory name
            step: Step number
            logp: Log probability list
            token_ids: Token IDs list
            prompt_token_ids: Prompt token IDs list
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"data_for_step_{step}.pt"

        data_to_save = {
            "logp": torch.tensor(logp).cpu() if logp is not None else torch.tensor([]).cpu(),
            "token_ids": torch.tensor(token_ids).cpu() if token_ids is not None else torch.tensor([]).cpu(),
            "prompt_token_ids": torch.tensor(prompt_token_ids).cpu() if prompt_token_ids is not None else torch.tensor([]).cpu(),
        }

        await asyncio.to_thread(torch.save, data_to_save, fn)

    async def save_img_pt(self, task_root: str, images: list, image_grid_thw: torch.Tensor, num_patches_list: list[int], pixel_values):
        """
        Save image tensors and patch information as .pt file.

        Args:
            task_root: Root directory for saving
            images: Original PIL.Image.Image list
            image_grid_thw: Model processed tensor
            num_patches_list: Number of patches for each image
            pixel_values: Pixel values tensor
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True, parents=True)
        fn = save_dir / "images_data.pt"

        data_to_save = {
            "images": images,
            "image_grid_thw": image_grid_thw.cpu(),
            "num_patches_list": num_patches_list,
            "pixel_values": pixel_values.cpu()
        }

        await asyncio.to_thread(torch.save, data_to_save, fn)

    # ---- save full trajectory json ----
    async def save_episode(self, task_root: str, episode_summary: list[dict]):
        """
        Save full trajectory JSON file.
        
        Args:
            task_root: Task root directory name
            episode_summary: Complete episode summary data
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"final_messages.json"
        async with aiofiles.open(fn, "w") as f:
            await f.write(json.dumps(episode_summary, ensure_ascii=False, indent=2))

    # ---- save reward txt ----
    async def save_reward(self, task_root: str, reward: float):
        """
        Save reward value to text files.
        
        Args:
            task_root: Task root directory name
            reward: Reward value
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        with open(save_dir / "reward.txt", "w") as f:
            f.write(str(reward))
        with open(save_dir / "reward_from_env.txt", "w") as f:
            f.write(str(reward))
            
    # ---- save task config(task info) json ----
    async def save_task_config(self, task_root: str, task_config: dict):
        """
        Save task configuration (task info) JSON file.
        
        Args:
            task_root: Task root directory name
            task_config: Task configuration dictionary
            
        Returns:
            str: Root directory path
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"task_config.json"
        async with aiofiles.open(fn, "w") as f:
            await f.write(json.dumps(task_config, ensure_ascii=False, indent=2))
        return str(self.root)
    
    async def save_prm_detail(self, task_root: str, prm_detail: dict):
        """
        Save the detailed Process Reward Model evaluation JSON file.
        
        Args:
            task_root: Task root directory name
            prm_detail: Detailed evaluation dictionary containing subgroups, reasoning, and step scores
        """
        save_dir = self.root / task_root
        save_dir.mkdir(exist_ok=True)
        fn = save_dir / f"prm_detail.json"
        async with aiofiles.open(fn, "w", encoding="utf-8") as f:
            await f.write(json.dumps(prm_detail, ensure_ascii=False, indent=2))