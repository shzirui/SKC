import os
import json
import re
import time
from openai import OpenAI
from typing import List, Dict, Optional

# ================= 配置区域 =================

# 1. 原始数据根目录
SOURCE_ROOT_DIRECTORY = r"pass1_osworldnew_dart"

# 2. 结果保存目录
RESULT_SAVE_DIRECTORY = r"pass1_osworldnew_dart_group"

ENABLE_FORCE_SPLIT = True  # 是否开启强制切分
MAX_STEPS_LIMIT = 5        # 每组最大步数 (超过则切分)

# ================= 核心功能函数 =================

def clean_json_response(content: str) -> str:
    content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```\s*", "", content)
    return content.strip()

def get_next_image_filename(current_filename: str) -> Optional[str]:
    if not current_filename:
        return None
    match = re.search(r'(\d+)(?=\D*$)', current_filename)
    if match:
        num_str = match.group(1)
        num_val = int(num_str)
        next_val = num_val + 1
        next_num_str = str(next_val).zfill(len(num_str))
        start, end = match.span(1)
        next_filename = current_filename[:start] + next_num_str + current_filename[end:]
        return next_filename
    return None

async def async_group_steps(
    task_instruction: str, 
    steps: List[Dict], 
    async_call_llm_fn  # 这是一个回调函数，传入我们 Actor 里的 _async_call_prm
) -> List[Dict]:
    """
    核心分组逻辑：接收指令和步骤，调用异步 LLM 接口，并执行强制切分逻辑。
    返回 enriched_subgoals
    """
    if not steps:
        return []

    # 1. 组装 prompt
    steps_text = ""
    for step in steps:
        steps_text += f"Step {step['step_id']}:\n{step['raw_content']}\n\n"

    system_prompt = """
    You are an expert in Hierarchical Reinforcement Learning.
    Decompose the GUI trajectory into logical subgoals based on the User Instruction.
    
    Return ONLY a valid JSON list. No extra text.
    
    Format:
    [
        {
            "subgoal_name": "Summary",
            "start_step": <int>,
            "end_step": <int>,
            "reasoning": "Why these steps are grouped"
        }
    ]
    """

    user_prompt = f"""
    ## User Task Instruction
    {task_instruction}

    ## Trajectory Steps
    {steps_text}

    Analyze the steps and group them. Return ONLY JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 2. 调用外部传入的异步 LLM 方法
    json_result_str = await async_call_llm_fn(messages)
    
    if not json_result_str:
        return []

    # 3. 解析与强制切分 (原汁原味复用你原来的核心逻辑)
    enriched_subgoals = []
    try:
        json_result_str = clean_json_response(json_result_str)
        subgoals_structure = json.loads(json_result_str)
        
        for item in subgoals_structure:
            original_start = int(item.get('start_step', 1))
            original_end = int(item.get('end_step', 1))
            
            steps_count = original_end - original_start + 1
            should_split = ENABLE_FORCE_SPLIT and (steps_count > MAX_STEPS_LIMIT)
            
            if not should_split:
                # === 不切分 ===
                start_idx = max(0, original_start - 1)
                end_idx = min(len(steps), original_end)
                
                if start_idx < len(steps):
                    # 构建当前组的基本 Steps
                    current_segment = [
                        {
                            "step_idx": s['step_id'],
                            "content": s['raw_content'],
                            "image": s['image_file'],
                            "next_image": get_next_image_filename(s['image_file'])
                        } 
                        for s in steps[start_idx : end_idx]
                    ]

                    # 如果有上一步，插到最前面
                    if start_idx > 0:
                        prev_s = steps[start_idx - 1]
                        context_step = {
                            "step_idx": prev_s['step_id'],
                            "content": f"[PREVIOUS CONTEXT] {prev_s['raw_content']}", 
                            "image": prev_s['image_file'],
                            "next_image": get_next_image_filename(prev_s['image_file'])
                        }
                        item['steps_data'] = [context_step] + current_segment
                    else:
                        item['steps_data'] = current_segment

                    enriched_subgoals.append(item)
            else:
                # === 需要切分 ===
                current_chunk_start = original_start
                part_counter = 1
                
                while current_chunk_start <= original_end:
                    current_chunk_end = min(current_chunk_start + MAX_STEPS_LIMIT - 1, original_end)
                    
                    chunk_start_idx = max(0, current_chunk_start - 1)
                    chunk_end_idx = min(len(steps), current_chunk_end)
                    
                    if chunk_start_idx < chunk_end_idx:
                        # 构建当前组的基本 Steps
                        current_segment = [
                            {
                                "step_idx": s['step_id'],
                                "content": s['raw_content'],
                                "image": s['image_file'],
                                "next_image": get_next_image_filename(s['image_file'])
                            } 
                            for s in steps[chunk_start_idx : chunk_end_idx]
                        ]

                        # 如果有上一步，插到最前面
                        final_segment_data = current_segment
                        if chunk_start_idx > 0:
                            prev_s = steps[chunk_start_idx - 1]
                            context_step = {
                                "step_idx": prev_s['step_id'],
                                "content": f"[PREVIOUS CONTEXT] {prev_s['raw_content']}",
                                "image": prev_s['image_file'],
                                "next_image": get_next_image_filename(prev_s['image_file'])
                            }
                            final_segment_data = [context_step] + current_segment

                        new_item = item.copy()
                        new_item['subgoal_name'] = f"{item['subgoal_name']} (Part {part_counter})"
                        new_item['start_step'] = current_chunk_start
                        new_item['end_step'] = current_chunk_end
                        new_item['reasoning'] = f"{item.get('reasoning','')} [Auto-split]"
                        new_item['steps_data'] = final_segment_data
                        
                        enriched_subgoals.append(new_item)
                    
                    current_chunk_start = current_chunk_end + 1
                    part_counter += 1
                    
    except json.JSONDecodeError as e:
        # 这里你可以把日志抛给上层，或者直接 print
        print(f"JSON 解析失败: {e}")
    except Exception as e:
        print(f"分组逻辑出错: {e}")

    return enriched_subgoals

