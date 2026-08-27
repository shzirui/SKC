import asyncio, base64, io, ray
from typing import Dict, Any, List, Tuple, Optional
import os, pathlib, json
from PIL import Image
import time, datetime
import logging
from functools import partial
from io import BytesIO

from desktop_env.desktop_env import DesktopEnv
from src.utils.ui_tars_utils import *
from src.utils.log_config import setup_logging
from src.utils.kill_all_env import stop_emulator
from src.core.group import async_group_steps
from src.core.score_backward_multi import async_score_recursive_step_multi_turn
from src.core.score_backward_single import async_score_recursive_step_single_turn
from copy import deepcopy
from dotenv import load_dotenv
import json
from openai import OpenAI

class ResourceExhaustedException(Exception):
    """Resource exhaustion exception"""
    pass

# Setup unified logging system
setup_logging()
logger = logging.getLogger(__name__)

"""
Refer to run_agent_loop.TrajectoryRunner
"""

def pil_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

@ray.remote
class TrajectoryRunnerActor:
    def __init__(self, task_info: Dict[str, Any], runner_cfg):
        
        # --- load config ---
        self.runner_cfg = runner_cfg
        self.max_images = self.runner_cfg.max_images
        self.max_tests = self.runner_cfg.max_texts
        self.max_steps = self.runner_cfg.max_steps
        self.root = self.runner_cfg.root
        # self.save_img_pt = self.runner_cfg.save_img_pt
        # self.rollout_n = self.runner_cfg.rollout_n

        # --- process task info for running ---
        self.base_messages: List = deepcopy(task_info["messages"])
        self.base_messages_for_save: List = deepcopy(task_info["messages"])
        
        # Remove plan information from final saved messages
        self.base_messages_for_save[1]["content"][0]["text"] = self.base_messages_for_save[1]["content"][0]["text"].split("Here is an instruction to help you complete the task: \n", 1)[0]
        
        self.task_cfg: Dict[str, Any] = task_info["task_config"]
        self.task_id = self.task_cfg["raw"]["task_id"] 
        self.trace_id = task_info.get("trace_id", "unknown_trace")
        
        self.task_root = f"{self.task_id}_{self.trace_id}"

        self.prompt_dialogue: List[Dict] = []   # for model client
        self.save_dialogue:   List[Dict] = []   # for partial storage
        self.save_dialogue_full: List[Dict] = []    # for full storage
        self.image_refs: List[Dict] = []  # record image position

        # --- initialize env ---
        self.env = None    # will be defined in _init_env()     
        self.env_ready = asyncio.Event()
        self.env_init_success = False  # Indicates whether environment initialization was successful
        self.env_init_timeout = 1200.0  # Environment initialization timeout
        self.env_init_start_time = None
        self.env_timeout_task = None  # Environment timeout monitoring task
        # self.env_init_end_time = None
        
    
    def get_service_id(self) -> Optional[str]:
        """Get environment service_id"""
        if not self.env_init_success:
            logger.warning(f"[{self.trace_id}] Environment initialization failed, cannot get service_id - task_id: {self.task_id}")
            return None
        elif self.env is None:
            logger.warning(f"[{self.trace_id}] Environment not initialized, cannot get service_id - task_id: {self.task_id}")
            return None
        elif self.env.provider.emulator_id is None:
            logger.warning(f"[{self.trace_id}] Environment emulator_id is empty - task_id: {self.task_id}")
            return None
        else:
            logger.info(f"[{self.trace_id}] Successfully obtained service_id: {self.env.provider.emulator_id} - task_id: {self.task_id}")
            return self.env.provider.emulator_id

    
    
    async def wait_for_env_ready(self, timeout: float = 120.0) -> bool:
        """Wait for environment initialization to complete (for external calls)"""
        try:
            await asyncio.wait_for(self.env_ready.wait(), timeout=timeout)
            return self.env_init_success
        except asyncio.TimeoutError:
            return False

    def is_env_ready(self) -> bool:
        """Check if environment initialization was successful"""
        return self.env_init_success

    async def monitor_env_timeout(self):
        """Monitor environment initialization timeout, automatically close environment if timeout"""
        if self.env_init_start_time is None:
            logger.warning(f"[{self.trace_id}] env_init_start_time not set, cannot monitor timeout - task_id: {self.task_id}")
            return
        
        check_interval = 60.0  # Check every minute
        while True:
            await asyncio.sleep(check_interval)
            
            if self.env is None:
                # Environment has been closed, exit monitoring
                break
            
            elapsed_time = time.time() - self.env_init_start_time
            if elapsed_time > self.env_init_timeout:
                logger.warning(f"[{self.trace_id}] Environment runtime timeout - task_id: {self.task_id}, "
                            f"Elapsed time: {elapsed_time:.2f}s, Timeout threshold: {self.env_init_timeout}s")
                try:
                    if self.env is not None:
                        self.env.close()
                        logger.info(f"[{self.trace_id}] Environment closed due to timeout - task_id: {self.task_id}")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] Error closing environment due to timeout - task_id: {self.task_id}, Error: {e}")
                finally:
                    self.env = None
                break

    async def run_episode(self, model_pool, storage, mysql_writer, pr_model_pool=None):
        """
        Single task rollout loop
        """
        # --- save task config ---
        storage_root = await storage.save_task_config.remote(self.task_root, self.task_cfg)

        logger.info(f"[{self.trace_id}] TrajectoryRunnerActor initialized - task_id: {self.task_id}, task_root: {self.task_root}")
        
        try:
            # --- initialize env and first frame ----
            try:
                # self._init_env(self.runner_cfg.env)
                await asyncio.to_thread(self._init_env, self.runner_cfg.env)
                self.env_init_success = True
                self.env_init_start_time = time.time()
                # Start environment timeout monitoring task
                # self.env_timeout_task = asyncio.create_task(self.monitor_env_timeout())
                logger.info(f"[{self.trace_id}] Environment initialization successful - task_id: {self.task_id}")
            except Exception as e:
                self.env_init_success = False
                logger.error(f"[{self.trace_id}] Environment initialization failed - task_id: {self.task_id}, Error: {e}")
                # Save failed task info to json file in the storage folder
                if not self.env_init_success:
                    try:
                        task_error_path = os.path.join(storage_root, f"task_{self.task_id}_trace_{self.trace_id}.json")
                        error_info = {
                            "task_id": self.task_id,
                            "trace_id": self.trace_id,
                            "error_type": str(e),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        with open(task_error_path, 'w', encoding='utf-8') as f:
                            json.dump(error_info, f, indent=2, ensure_ascii=False)
                        logger.info(f"[{self.trace_id}] Failed task info saved to: {task_error_path}")
                    except Exception as e:
                        logger.error(f"[{self.trace_id}] Failed to save failed task info - task_id: {self.task_id}, Error: {e}")
            finally:
                # Set event regardless of success or failure so waiting parties know initialization process is complete
                self.env_ready.set()
            # If initialization failed, throw exception
            if not self.env_init_success:
                raise RuntimeError(f"[{self.trace_id}] Environment initialization failed - task_id: {self.task_id}")
            
            # --- get initial observation as first frame ----
            obs = self.env._get_obs()
            obs_img = obs["screenshot"]
            image_size = Image.open(BytesIO(obs["screenshot"])).size
            frame0 = await storage.save_frame.remote(self.task_root, 0, obs_img)
            self._set_first_frame(obs_img, frame0)
            

            step, done = 0, False
            logger.info(f"[{self.trace_id}] Environment initialization complete, starting main loop - task_id: {self.task_id}")

            # def format_chat(messages):
            #     formatted = ""
            #     for m in messages:
            #         content = m["content"]

            #         # If content is list (multimodal message)
            #         if isinstance(content, list):
            #             # Only take text part
            #             texts = [c["text"] for c in content if c.get("type") == "text"]
            #             content_str = "\n".join(texts)
            #         else:
            #             # Regular string
            #             content_str = content

            #         formatted += f"{content_str}<|im_end|>\n"

            #     return formatted

            # base_messages = format_chat(self.base_messages_for_save)

            # tokenize_response = await self._call_model_tokenize(
            #                 model_pool, base_messages,
            #                 self.runner_cfg.model_pool)

            # # prompt_token_ids = tokenize_response
            # prompt_token_ids = await model_pool.process_text.remote(self.base_messages_for_save)
            # --- main loop ----
            # MAX_T, RETRIES, BACKOFF = 10, 3, 2
            prompt_token_ids=None
            flag_vllm_error = False
            while step < self.max_steps and not done:
                
                if step + 1 == self.max_steps:
                    action = "FAIL"
                    logger.warning(f"[{self.trace_id}] Reached maximum step limit, setting action to FAIL - task_id: {self.task_id}, step: {step}")

                else:
                    st_step = time.time()
                    print("step start time: ", st_step)
                    # build prompt
                    messages = self._build_messages()
                    print("len messages: ", len(messages))

                    # ---- call vLLM serve ----

                    # if local model pool not avaiable, use pre launched model server
                    st = time.time()
                    if model_pool:
                        response, model_path, vllm_logp, token_ids = await self._call_model(
                            model_pool, messages,
                            self.runner_cfg.model_pool, step)
                        
                    model_duration = time.time() - st
                    logger.info(f"[{self.trace_id}] Model response - task_id: {self.task_id}, step: {step}, "
                                f"Duration: {model_duration:.2f}s, response_length: {len(response) if response else 0}")
                    logger.debug(f"[{self.trace_id}] Full model response: {response}")
                    
                    if response is None:
                        action = "FAIL"
                        flag_vllm_error = True
                        logger.warning(f"[{self.trace_id}] Model response is empty, setting action to VLLM ERROR - task_id: {self.task_id}, step: {step}")

                    else:
                        self._add_text(response)
                        try:
                            action = self._parse(response, image_size)
                        except Exception as e:
                            logger.warning(f"Action parsing failed! task_id: {self.task_id}, Error: {e}")
                            action = "FAIL"


                # ---- interact with env ----
                if action in ["DONE", "FAIL"]:
                    logger.info(f"[{self.trace_id}] Task completed - task_id: {self.task_id}, step: {step}, action: {action}")
                
                if action == "VLLM ERROR":
                    logger.error(f"[{self.trace_id}] Model call failed, task terminated - task_id: {self.task_id}, step: {step}")
                    # raise RuntimeError(f"Model call failed, task_id: {self.task_id}, step: {step}")

                # Execute the action
                st = time.time()
                obs, reward, done, info = await self._run_step_async(action)
                obs_img = obs["screenshot"]

                env_duration = time.time() - st
                logger.info(f"[{self.trace_id}] Environment step execution complete - task_id: {self.task_id}, step: {step}, "
                            f"Duration: {env_duration:.2f}s, done: {done}")


                # ---- save screenshot ----
                if action not in ["DONE", "FAIL", "VLLM ERROR"]:
                    frame_path = await storage.save_frame.remote(
                        self.task_root, step + 1, obs_img)
                    
                    self._add_image(obs_img, frame_path)
                
                # ---- save current trajectory
                await storage.save_partial_traj.remote(self.task_root, step + 1, self._build_trajectory())

                # ---- save vllm logp ----
                await storage.save_partial_pt.remote(self.task_root, step + 1, vllm_logp, token_ids, prompt_token_ids)
                
                step_duration = time.time() - st_step
                # self._log_latency(step, model_duration, env_duration, step_duration)
                
                step += 1

            

            # calculate and save reward
            """Evaluate whether the task is successfully completed."""
            try:
                # reward = self.env.evaluate()
                reward = await asyncio.to_thread(self.env.evaluate)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                # Re-throw exception so caller knows evaluation failed
                # TODO regression testing
                reward = -1


            # reward = self.env.evaluate()
            # if action == "VLLM ERROR":
            if flag_vllm_error:
                reward = -2
            await storage.save_reward.remote(self.task_root, reward)
            logger.info(f"[{self.trace_id}] Task evaluation complete - task_id: {self.task_id}, reward: {reward}")
            
            # save trajectory json
            full_messages = self.base_messages_for_save + self.save_dialogue_full
            if full_messages:
                last_msg = full_messages[-1]
                if (
                    last_msg.get("role") == "user" and
                    len(last_msg.get("content", [])) == 1 and
                    last_msg["content"][0].get("type") == "image_url"
                ):
                    full_messages.pop()
                    logger.info(f"[{self.trace_id}] Removing last user message containing only image - task_id: {self.task_id}")
            await storage.save_episode.remote(self.task_root, full_messages)

            await model_pool.save_messages_reward.remote(full_messages, reward, self.task_id, self.trace_id)

            process_reward_list = []
            detailed_prm_json = None
            if pr_model_pool and reward >= 0 and step > 0 and not flag_vllm_error:
                logger.info(f"[{self.trace_id}] ⏳ Starting Dual-Pool PRM Evaluation...")
                process_reward_list, detailed_prm_json = await self.evaluate_process_reward(
                    pr_model_pool, full_messages, float(reward)
                )
                logger.info(f"[{self.trace_id}] ✅ PRM Evaluation done. Array: {process_reward_list}")
            else:
                # 降级：如果没有传池子，或任务失败报错，塞入长度对齐的默认分数
                process_reward_list = [-1] * (step if action == 'DONE' else step-1)

            if detailed_prm_json:
                try:
                    # 替换为通过 StorageActor 统一异步保存
                    await storage.save_prm_detail.remote(self.task_root, detailed_prm_json)
                    logger.info(f"[{self.trace_id}] Saved detailed PRM JSON via StorageActor.")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] Failed to save detailed PRM JSON: {e}")
            
            # Insert into mysql database
            if self.runner_cfg.write_to_mysql and not flag_vllm_error:
                if flag_vllm_error:
                    reward = -2
                pr_sql_value = json.dumps(process_reward_list)

                meta = {
                'run_id': storage_root,
                'trajectory_id': self.task_root,
                'task_id': self.task_id,
                'trace_id': self.trace_id,
                'reward': reward,
                'process_reward': pr_sql_value,
                'model_version': model_path,
                'instruction': self.task_cfg["instruction"],
                'num_chunks' : step if action == 'DONE' else step-1     # Spliter function disabled, save step count here
                }
                logger.info(f"[{self.trace_id}] Begin to insert mysql")
                if pr_model_pool:
                    if process_reward_list and all(r >= 0 for r in process_reward_list):
                        if reward>0:
                            logger.info(f"[{self.trace_id}] Stored in mySQL database")
                            await mysql_writer.insert_run.remote(meta)
                        elif reward==0:
                            if step >=4:
                                await mysql_writer.insert_run.remote(meta)
                                logger.info(f"[{self.trace_id}] Stored in mySQL database")
                            else:
                                logger.info(f"[{self.trace_id}] Reward is 0, steps less than 4, not stored in mySQL database")
                        else:
                            logger.info(f"[{self.trace_id}] Reward less than 0, not stored in mySQL database")
                    else:
                        logger.info(f"[{self.trace_id}] Process reward list contains invalid scores, not stored in mySQL database")
                else:
                    if reward>0:
                        logger.info(f"[{self.trace_id}] Stored in mySQL database")
                        await mysql_writer.insert_run.remote(meta)
                    elif reward==0:
                        if step >=4:
                            await mysql_writer.insert_run.remote(meta)
                            logger.info(f"[{self.trace_id}] Stored in mySQL database")
                        else:
                            logger.info(f"[{self.trace_id}] Reward is 0, steps less than 4, not stored in mySQL database")
                    else:
                        logger.info(f"[{self.trace_id}] Reward less than 0, not stored in mySQL database")
            logger.info(f"[{self.trace_id}] Task trajectory execution complete - task_id: {self.task_id}, Total steps: {step}")           
 
        except Exception as e:
            logger.error(f"[{self.trace_id}] Task trajectory execution failed - task_id: {self.task_id}, Error: {e}")
        finally:
            # Cancel environment timeout monitoring task
            if self.env_timeout_task is not None and not self.env_timeout_task.done():
                self.env_timeout_task.cancel()
                try:
                    await self.env_timeout_task
                except asyncio.CancelledError:
                    pass
            # Ensure environment is released regardless of success or failure
            if self.env is not None:
                try:
                    self.env.close()
                    logger.info(f"[{self.trace_id}] Environment closed - Environment emulator_id: {self.env.provider.emulator_id} - task_id: {self.task_id}")
                except Exception as e:
                    logger.error(f"[{self.trace_id}] Error closing environment - task_id: {self.task_id}, Error: {e}")
                finally:
                    self.env = None

       
    
        return True
    
    def _init_env(self, env_cfg):  
        # Add retry logic for RemoteDesktopEnv initialization
        max_retries = env_cfg.max_retries
        for attempt in range(max_retries):
            try:
                load_dotenv()
                self.env = DesktopEnv(
                            action_space=env_cfg.action_space,
                            provider_name=env_cfg.provider_name,
                            os_type=env_cfg.os_type,
                        )
                self.env.reset(task_config=self.task_cfg)
                time.sleep(60)
                logger.info(f"[{self.trace_id}] Environment initialization successful - task_id: {self.task_id}, Attempt: {attempt + 1}")
                break
            except Exception as e:
                
                logger.warning(f"[{self.trace_id}] Environment initialization failed - task_id: {self.task_id}, Attempt: {attempt + 1}, Error: {e} \n Environment initialization failed, environment closed - Environment emulator_id: {self.env.provider.emulator_id}")
                if attempt == max_retries - 1:
                    logger.error(f"[{self.trace_id}] All retry attempts failed - task_id: {self.task_id}")
                    raise
                logger.info(f"[{self.trace_id}] Preparing to retry - task_id: {self.task_id}, Next attempt: {attempt + 2}/{max_retries}")
                self.env.close()


    async def _call_model(self, model_pool, messages: str, model_cfg, step):
        """
        Call model; retry RETRIES times on failure, exponential backoff (backoff**attempt) seconds.
        Return response (str) or None.
        """
        timeout = model_cfg.timeout
        retries = model_cfg.retries
        backoff = model_cfg.backoff
        DEFAULT_MODEL_PATH = model_cfg.ckpt_path
        
        for attempt in range(1, retries + 1):
            try:
                response, model_path, vllm_logp, token_ids = await asyncio.wait_for(
                    model_pool.generate.remote(messages,
                                               frequency_penalty=model_cfg.frequency_penalty,
                                               temperature=model_cfg.temperature,
                                               top_p=model_cfg.top_p,
                                               max_tokens=model_cfg.max_tokens,
                                               seed=model_cfg.seed,
                                               logprobs=model_cfg.logprobs,
                                               return_tokens_as_token_ids=model_cfg.return_tokens_as_token_ids,       
                                               task_id=self.task_id,
                                                trace_id=self.trace_id,
                                                step=step), 
                    timeout=timeout,
             
                )
                
                return response, model_path, vllm_logp, token_ids
            except (asyncio.TimeoutError, ray.exceptions.RayError) as e:
                if attempt == retries:
                    model_path = await asyncio.wait_for(
                        model_pool.get_last_model_version.remote(),
                        timeout=timeout
                    )
                    if not model_path:
                        logger.error(f" Can't get last model version, use default model path in model_cfg")
                        model_path = DEFAULT_MODEL_PATH
                    logger.error(f"[{self.trace_id}] Model call failed - task_id: {self.task_id}, Retry attempts: {attempt}, return last model version Error: {e}")
                    return None, model_path, None, None
                await asyncio.sleep(backoff ** attempt)   # 2s, 4s, 8s …

    async def _call_model_tokenize(self, model_pool, messages: str, model_cfg):
        """
        Call tokenize interface to tokenize formatted base_messages.
        Return token_ids (List[int]).
        """
        timeout = model_cfg.timeout
        retries = model_cfg.retries
        backoff = model_cfg.backoff
        DEFAULT_MODEL_PATH = model_cfg.ckpt_path

        for attempt in range(1, retries + 1):
            try:
                token_ids = await asyncio.wait_for(
                    model_pool.tokenize.remote(messages),
                    timeout=timeout
                )
                return token_ids
            except (asyncio.TimeoutError, ray.exceptions.RayError) as e:
                if attempt == retries:
                    logger.error(f"[{self.trace_id}] Tokenize call failed - task_id: {self.task_id}, Retry attempts: {attempt}, Error: {e}")
                    return None
                await asyncio.sleep(backoff ** attempt) 

    async def _async_call_prm(self, pr_model_pool, messages, temperature=0.1, max_tokens=2048):
        """调用 PRM 服务并清洗 JSON"""
        retries = 3
        for attempt in range(retries):
            try:
                # 复用 model_pool 的底层通讯！
                response, _ = await asyncio.wait_for(
                    pr_model_pool.generate_vllm.remote(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        logprobs=False
                    ),
                    timeout=120.0
                )
                if response:
                    # 清洗 JSON
                    content = re.sub(r"```json\s*", "", response, flags=re.IGNORECASE)
                    content = re.sub(r"```\s*", "", content).strip()
                    return content
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"[{self.trace_id}] PRM Call failed after retries: {e}")
                    return None
                await asyncio.sleep(5)
        return None
    
    async def evaluate_process_reward(self, prm_model_pool, full_messages: List[Dict], outcome_reward: float) -> List[float]:
        # 1. 提取出 task_instruction 和 steps
        task_instruction, steps = self._extract_trace_from_memory(full_messages)
        total_steps = len(steps)
        pr_array = [-1] * total_steps 

        if not steps or not prm_model_pool:
            return pr_array

        # 2. 构造一个能自动绑定当前模型池的 async 回调函数
        async def call_llm_callback(messages):
            return await self._async_call_prm(
                prm_model_pool=prm_model_pool, 
                messages=messages, 
                temperature=0.1, 
                max_tokens=2048
            )

        # 3. ✨ 一行代码搞定复杂的分组逻辑！
        enriched_subgoals = await async_group_steps(task_instruction, steps, call_llm_callback)
        
        if not enriched_subgoals:
            logger.warning(f"[{self.trace_id}] Grouping returned empty, fallback to -1")
            return pr_array

        logger.info(f"[{self.trace_id}] Grouping success, got {len(enriched_subgoals)} parts.")

        current_outcome_reward = outcome_reward
        logger.info(f"[{self.trace_id}] Start backward scoring for {len(enriched_subgoals)} groups...")
        
        # 为了兼容图片路径读取，由于我们的 evaluate 是在当前任务完成后立刻触发的
        # storage 上的文件其实就在你的 self.task_root 对应的目录下 (或者绝对路径)
        # 这里你可以直接传入你保存图片的根目录，如果不确定可以写 "./"
        task_dir = os.path.join(self.root, self.task_root) 

        for idx in range(len(enriched_subgoals) - 1, -1, -1):
            eval_res = await async_score_recursive_step_single_turn(
                task_instruction=task_instruction,
                task_root_path=task_dir,
                all_subgoals=enriched_subgoals,
                current_idx=idx,
                current_outcome_reward=current_outcome_reward,
                async_call_llm_fn=call_llm_callback
            )
            
            if not eval_res: 
                logger.warning(f"[{self.trace_id}] Eval for Group {idx+1} returned None. Keeping defaults for remaining.")
                break 
            
            # 解析打分并填入数组
            enriched_subgoals[idx]['evaluation'] = eval_res

            group_conclusion = eval_res.get("current_group_conclusion", {})
            group_score = float(group_conclusion.get("score", -1))

            step_critics = eval_res.get("step_critic", [])
            for sc in step_critics:
                s_idx = sc.get("step_index", 1) - 1 
                if 0 <= s_idx < total_steps:
                    step_score = float(sc.get("score", -1))
                    
                    if step_score >= 0 and group_score >= 0:
                        pr_array[s_idx] = step_score * group_score
            
            # 传递下一轮(上一组)的 Reward
            prev_judgement = eval_res.get('previous_group_judgment', {})
            current_outcome_reward = float(prev_judgement.get('prev_outcome_reward', current_outcome_reward))
        
        detailed_prm_data = {
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "overall_reward": outcome_reward,
            "task_instruction": task_instruction,
            "subgoals": enriched_subgoals  # 包含了步骤、图文、以及刚才塞进去的 evaluation
        }

        return pr_array, detailed_prm_data
         
    def _add_image(self, img_bytes: bytes, frame_path: str):
        
        self.prompt_dialogue.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image;base64," + pil_to_base64(img_bytes)}
            }]
        })

        self.save_dialogue.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": frame_path
            }]
        })
        
        self.save_dialogue_full.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": frame_path
            }]
        })
        
        self.image_refs.append(
            {"source": "dialogue", "msg_idx": len(self.prompt_dialogue) - 1,
            "content_idx": None}
        )
        
        self._trim()
    
    
    def _add_text(self, assistant_txt: str):
        
        msg = {
            "role": "assistant",
            "content": add_box_token(assistant_txt)
        }
        self.prompt_dialogue.append(msg)
        self.save_dialogue.append(msg)
        self.save_dialogue_full.append(msg)
        # logger.info("Dialogue:", self.save_dialogue)
        self._trim()

        
    def _trim(self):
        """Ensure image num ≤ self.max_images and assistant text num ≤ self.max_tests."""
        img_cnt  = len(self.image_refs)
        txt_cnt  = sum(m["role"] == "assistant" for m in self.prompt_dialogue)

        while img_cnt > self.max_images or txt_cnt > self.max_tests:

            # --- Image limit exceeded: earliest image ---
            if img_cnt > self.max_images:
                ref = self.image_refs.pop(0)
                if ref["source"] == "base":
                    self.base_messages[ref["msg_idx"]]["content"].pop(ref["content_idx"])
                else:               # dialogue image
                    self._remove_dialogue_msg(ref["msg_idx"])
                img_cnt -= 1
                continue

            # --- Text limit exceeded: earliest assistant text ---
            if txt_cnt > self.max_tests:
                for i, m in enumerate(self.prompt_dialogue):
                    if m["role"] == "assistant":
                        self._remove_dialogue_msg(i)
                        txt_cnt -= 1
                        break
                

    def _build_messages(self) -> List[Dict]:
        return self.base_messages + self.prompt_dialogue


    def _build_trajectory(self) -> List[Dict]:
        if len(self.base_messages[1]['content']) == 1:
            return self.base_messages + self.save_dialogue
        else:
            return self.base_messages_for_save + self.save_dialogue
    
    
    def _parse(self, response, image_size):
                
        # Parse the action
        parsed_responses = parse_action_to_structure_output(
            response,
            factor=1000,  # TODO: Make this configurable
            origin_resized_height=image_size[1],
            origin_resized_width=image_size[0],
            model_type="qwen25vl",
            max_pixels=16384*28*28,
            min_pixels=100*28*28
        )
        logger.debug(f"[{self.trace_id}] Parse response result - task_id: {self.task_id}, parsed_responses: {parsed_responses}")

        # Convert to pyautogui code
        action_code = parsing_response_to_pyautogui_code(
            parsed_responses,
            image_height=image_size[1],
            image_width=image_size[0],
            input_swap=False  # TODO: Make this configurable
        )
        
        logger.info(f"[{self.trace_id}] Parse action code - task_id: {self.task_id}, action: {action_code}")
        
        return action_code
    
    
    async def _run_step_async(self, action):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.env.step, action))
    
    def _set_first_frame(self, obs_img, frame0):
        self.base_messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image;base64," + pil_to_base64(obs_img)}
            }
        )
        self.base_messages_for_save[1]["content"].append(
            {
                "type": "image_url",
                "image_url": frame0
            }
        )
        
        self.image_refs.append(
            {"source": "base", "msg_idx": 1,
            "content_idx": len(self.base_messages[1]["content"]) - 1}
        )
        
    def _log_latency(self, step, model_duration, env_duration, step_duration, out_path="timings.jsonl"):
        record = {
            "ts": datetime.datetime.now().isoformat(),
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "step": step,
            "model_duration": round(model_duration, 4),
            "env_duration": round(env_duration, 4),
            "step_duration": round(step_duration, 4),
        }
        pathlib.Path(out_path).parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            
    def _remove_dialogue_msg(self, idx: int):
        """
        Delete the idx-th dialogue message in prompt/save,
        and synchronously correct the msg_idx of dialogue sources in image_refs.
        """
        self.prompt_dialogue.pop(idx)
        self.save_dialogue.pop(idx)

        # Update image_refs
        self.image_refs = [
            r if not (r["source"] == "dialogue" and r["msg_idx"] == idx)
            else None                           # Discard the same image reference being deleted
            for r in self.image_refs
        ]
        self.image_refs = [
            (
                {**r, "msg_idx": r["msg_idx"] - 1}
                if r and r["source"] == "dialogue" and r["msg_idx"] > idx # Image indices after idx all -1
                else r
            )
            for r in self.image_refs
            if r                                 # Remove None
        ]

    def _extract_trace_from_memory(self, full_messages: List[Dict]):
        """从内存中提取指令和步骤 (不落盘)"""
        task_instruction = ""
        steps = []
        step_count = 0
        current_image_file = None 

        for msg in full_messages:
            role = msg.get('role')
            content = msg.get('content')

            if role == 'user' and isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text' and "## User Instruction" in item.get('text', ''):
                        match = re.search(r"## User Instruction\s*(.*)", item['text'], re.DOTALL)
                        if match:
                            task_instruction = match.group(1).strip()
                    if item.get('type') == 'image_url':
                        url_part = item.get('image_url', '')
                        current_image_file = url_part if isinstance(url_part, str) else url_part.get('url', '')

            if role == 'assistant' and isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        step_count += 1
                        steps.append({
                            "step_id": step_count,
                            "raw_content": item.get('text', '').strip(),
                            "image_file": current_image_file
                        })
        return task_instruction, steps