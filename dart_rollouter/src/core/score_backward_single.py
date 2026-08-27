import os
import json
import re
import base64
from typing import List, Dict, Optional

def clean_json_response(content: str) -> str:
    content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```\s*", "", content)
    return content.strip()

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path): return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except: return None

def get_base_subgoal_name(name: str) -> str:
    """
    识别 Part 标记。
    Input: "Open File (Part 1)" -> Output: "Open File"
    """
    return re.sub(r"\s*\(Part\s*\d+\)$", "", name).strip()

async def async_score_recursive_step_single_turn(
    task_instruction: str,
    task_root_path: str,
    all_subgoals: List[Dict],
    current_idx: int,
    current_outcome_reward: float,
    async_call_llm_fn  # [关键] 传入回调函数
) -> Optional[Dict]:
    """
    统一 Prompt 打分逻辑 (异步版)
    """
    current_sg_info = all_subgoals[current_idx]
    raw_subgoal_name = current_sg_info.get('subgoal_name', 'Unknown')
    current_base_name = get_base_subgoal_name(raw_subgoal_name)
    current_reasoning = current_sg_info.get('reasoning', 'N/A')
    
    # 向前找 Previous Subgoal
    prev_subgoal_str = "N/A (Start of Task)"
    temp_prev_idx = current_idx - 1
    while temp_prev_idx >= 0:
        p_item = all_subgoals[temp_prev_idx]
        p_base = get_base_subgoal_name(p_item.get('subgoal_name', ''))
        if p_base != current_base_name:
            prev_subgoal_str = f"Subgoal: {p_base}\n  Reasoning: {p_item.get('reasoning', 'N/A')}"
            break
        temp_prev_idx -= 1

    # 向后找 Next Subgoal
    next_subgoal_str = "N/A (End of Task)"
    temp_next_idx = current_idx + 1
    while temp_next_idx < len(all_subgoals):
        n_item = all_subgoals[temp_next_idx]
        n_base = get_base_subgoal_name(n_item.get('subgoal_name', ''))
        if n_base != current_base_name:
            next_subgoal_str = f"Subgoal: {n_base}\n  Reasoning: {n_item.get('reasoning', 'N/A')}"
            break
        temp_next_idx += 1

    # 提取 Evidence
    raw_steps_data = current_sg_info.get('steps_data', [])
    evidence_b64 = None
    evidence_text = "Unknown"
    start_b64 = None
    
    if raw_steps_data:
        first_step = raw_steps_data[0]
        if "[PREVIOUS CONTEXT]" in first_step.get('content', ''):
            evidence_img_path = os.path.join(task_root_path, first_step['image'])
            evidence_b64 = encode_image_to_base64(evidence_img_path)
            start_img_path = os.path.join(task_root_path, raw_steps_data[1]['image'])
            start_b64 = encode_image_to_base64(start_img_path)
            evidence_text = first_step['content'].replace("[PREVIOUS CONTEXT]", "").strip()

    system_prompt = """
    You are an expert Visual Reward Model for GUI Agents. 
    Your goal is to evaluate a specific 'Subgoal Group' of actions within a global task.

    **YOUR CONTEXT:**
    - **Global Task Instruction**: The overall task the agent is trying to accomplish.
    - **Subgoal Trace**:
        - **Previous Group Subgoal**: What was just finished.
        - **Current Group Subgoal**: The nominal goal of the Current Group.
        - **Next Group Subgoal**: What comes next.
    - **Current Group Trace**: The steps executed in this group.
    - **Current Group Outcome**: The Success (1.0) or Failure (0.0) signal from the Future.

    **HOW TO USE 'CURRENT GROUP OUTCOME' (Global Constraint):**
    - **If Outcome is LOW (0.0 - 0.5)**: **Be a Detective.** Reward correct steps, BUT explicitly identify the **FATAL STEP** (Logic Error or Execution Failure) that caused the group to fail.
    - **If Outcome is HIGH (0.6 - 1.0)**: **Be a Strict Critic.** Penalize steps that were redundant or inefficient.

    Return ONLY a valid JSON object.
    """

    user_content_list = []
    
    user_content_list.append({
        "type": "text",
        "text": f"""
    **YOUR AUDIT PROCESS:**

    1. **Phase 1: Step-Level Critic**: For EACH step, evaluate and assign a score

       - **Step 0: Action Type**: Determine if this step is a standard action or the special 'finished()' action.
       
       # === BRANCH A: IF Action is 'finished()' ===
       - Look STRICTLY at the **Current Group Outcome**.
         - **Score = Current Group Outcome**: Outcome > 0.0 + Agent Claims Success.
         - **Score = 0.0**: Outcome = 0.0 OR Agent Claims Failure.

       # === BRANCH B: IF Action is NOT 'finished()' (Standard action) ===
       - **Step 1: Verify Execution (Execution vs Intent)**:
           - **Context**: Look at the **Screenshot**, the **Agent Response**, AND the **Next Screenshot / Next Agent Response**.
           - **Visual Delta**: Compare **Screenshot** vs **Next Screenshot**. Did the UI/Value *actually* change to match the specific target in the Thought?
           - **Agent Feedback**: Compare **Agent Response** vs **Next Agent Response**. Did the Agent's next response reflect progress, or did it admit failure/repeat the exact same intent? 
           - **Judgment**: "Did the action successfully execute AND achieve the intended result?"
              - **Completed**: UI changed matching intent.
              - **Failed**: No change, mismatch, error, admit failure, or repeated intent.

       - **Step 2: Verify Logic (Intent vs Subgoal)**: "Assuming execution was successful, is this specific **Intent** a **LOGICALLY CORRECT** step towards the Subgoal?"
           - **Context**: Review the Entire **Current Group Trace** and **Current Group Subgoal**. Does this step enable *subsequent* steps in this group?
           - **The Enabler Test**: Did this action produce a state change that was *actively utilized* by subsequent steps, or did it lead to a dead end?
           - **Judgment**:
              - **Useful (The Enabler)**: The action created a necessary precondition for subsequent steps (e.g., Opened a menu -> Next step selected an item from that menu).
              - **Plausible (Dead End / Info Gain)**: The action was executed, but the trace shows it led to a **Dead End** or required a **Rollback**. This is a valid "Trial & Error".
              - **Useless (Noise / Error)**: The action had NO impact on the trace (e.g., hovered, clicked blank space) OR introduced a state that was ignored or immediately undone without information gain.
       
       - **Step 3: Score**: 
           - **1. Effective Progress (0.8 - 1.0)**: Execution is **Completed**, AND the intent is **Useful**.
           - **2. Valid Exploration (0.2 - 0.7)**: Execution is **Completed**, BUT the intent is **Plausible**.
           - **3. Redundant / Error (0.0 - 0.1)**: Execution is **Completed**, BUT the intent is **Useless**.
           - **4. Execution Failure (0.0)**: Execution **Failed**.

    2. **Phase 2: Current Group Conclusion**: Evaluate the Group's total contribution to the **Global Task**.
       - **Step 1: Subgoal Execution Status**: 
           - **Context**: Review the Entire **Step-Level Critic**. Does steps of this group complete the subgoal?
           - **Judgment**: (Completed/Partial/Failed)
       - **Step 2: Global Goal Alignment**:
           - **Context**: Review the Entire **Subgoal Trace**.
           - **The Bridge Test**: Independent of whether it succeeded, was *attempting* this Subgoal the logically correct move? Does it bridge the gap between Previous and Next?
           - **Judgment**:
               - **Useful (The Bridge)**: The Subgoal acts as a necessary **PRECONDITION** or **FIX** that enables the **Next Subgoal** (or Final Goal). It logically advances the Global Task state.
               - **Misaligned (Broken Bridge)**: The Subgoal is relevant but **Ill-Timed** (e.g., trying to run code before installing libs). It disrupts the logical flow.
               - **Useless (Noise)**: The Subgoal is **Redundant** (already done) or **Irrelevant** to the Global Task.

       - **Step 3: Score**: 
            - **1. Perfect Success (1.0)**: Subgoal is **Completed**, AND it makes a solid **Useful** contribution to the Global Task.
            - **2. Partial Success (0.8 - 1.0)**: Subgoal is **Partial Completed** or currently in progress, AND the actions taken are **Useful** and aligned.
            - **3. Valuable Failure (0.3 - 0.7)**: Subgoal explicitly **Failed** (objective not met), BUT the attempt is **Useful** that pushed the task forward.
            - **4. Misaligned Focus (0.1 - 0.2)**: Subgoal is **Completed or Partial**, BUT the intention was **Misaligned** with the Global Task (e.g., wrong object, wrong constraints).
            - **5. Complete Failure (0.0)**: Subgoal **Failed**, AND the actions were **Useless** or had no impact.

    3. **PHASE 3 INSTRUCTIONS:**
       - **Step 1: Inspect Evidence**: Look at the **[EVIDENCE]** (Last Action and Screen State), "does this evidence PROVE that the Previous Subgoal was actually met?"
       - **Step 2: Score**: 
          - **1. Perfect Success (1.0)**: Evidence CLEARLY shows the subgoal was **COMPLETED**.
          - **2. Partial Success (0.8 - 1.0)**: The state shows **PARTIAL COMPLETION** or progress.
          - **3. Valuable Failure (0.3 - 0.7)**: The subgoal **FAILED**, BUT the Last Action performed was **CORRECT/VALUABLE** (Salvage value).
          - **4. Misaligned Focus (0.1 - 0.2)**: The subgoal was technically achieved but was **MISALIGNED**.
          - **5. Complete Failure (0.0)**: The state is wrong, unchanged, or action was useless.

## Global Task Instruction:
{task_instruction}

## Subgoal Trace (Context):
**1. Previous Group Subgoal:**
{prev_subgoal_str}

**2. Current Group Subgoal (TARGET):**
Subgoal: {current_base_name}
Reasoning: {current_reasoning}

**3. Next Group Subgoal:**
{next_subgoal_str}

## Current Group Outcome: {current_outcome_reward}

## Current Group Trace:
"""
    })

    # --- Phase 1: 循环构建图文 Trace (识别 Part 逻辑) ---
    sibling_indices = []
    l = current_idx
    while l >= 0 and get_base_subgoal_name(all_subgoals[l].get('subgoal_name', '')) == current_base_name:
        sibling_indices.insert(0, l)
        l -= 1
    r = current_idx + 1
    while r < len(all_subgoals) and get_base_subgoal_name(all_subgoals[r].get('subgoal_name', '')) == current_base_name:
        sibling_indices.append(r)
        r += 1

    user_content_list.append({"type": "text", "text": "\n[EXECUTION STEPS (Full Context of this Subgoal)]:"})
    
    target_step_indices = []
    
    for s_idx in sibling_indices:
        is_current_part = (s_idx == current_idx)
        is_immediate_next = (s_idx == current_idx + 1)
        sg_item = all_subgoals[s_idx]
        sg_steps = sg_item.get('steps_data', [])
        filtered_steps = [s for s in sg_steps if "[PREVIOUS CONTEXT]" not in s.get('content', '')]

        for i, step in enumerate(filtered_steps):
            real_step_id = step.get('step_idx')
            content_text = step.get('content', '')
            step_display_text = f"Step {real_step_id}: {content_text}"
            img_path = os.path.join(task_root_path, step.get('image', ''))
            
            if is_current_part:
                target_step_indices.append(real_step_id)
                if i == 0:
                    user_content_list.append({"type": "text", "text": f"\n[START STATE] (Current Group Start):"})
                
                b64 = encode_image_to_base64(img_path)
                if b64:
                    user_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                user_content_list.append({"type": "text", "text": step_display_text})
                
                if i == len(filtered_steps) - 1 and s_idx == sibling_indices[-1]:
                    next_img_path = os.path.join(task_root_path, step.get('next_image', ''))
                    b64_end = encode_image_to_base64(next_img_path)
                    if b64_end:
                        user_content_list.append({"type": "text", "text": f"\n[FINAL RESULT STATE] (Result of Current Group):"})
                        user_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_end}"}})

            elif is_immediate_next and i == 0:
                user_content_list.append({"type": "text", "text": f"\n[FINAL RESULT STATE] (Result of Current Group):"})
                b64 = encode_image_to_base64(img_path)
                if b64:
                    user_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                user_content_list.append({"type": "text", "text": f"{step_display_text} (Future Action)"})
            else:
                user_content_list.append({"type": "text", "text": f"\n[Context Text Only]: {step_display_text}"})
    
    # [关键] 动态 Prompt 替换
    target_indices_str = ", ".join(map(str, target_step_indices))
    if user_content_list and 'text' in user_content_list[0]:
        original_text = user_content_list[0]['text']
        old_phrase = "For EACH step, evaluate and assign a score:"
        new_phrase = f"For EACH target step ({target_indices_str}), evaluate and assign a score:"
        user_content_list[0]['text'] = original_text.replace(old_phrase, new_phrase)

    # 补充 Evidence 逻辑
    target_prev_idx = current_idx - 1
    if target_prev_idx >= 0:
        prev_item = all_subgoals[target_prev_idx]
        prev_raw_name = prev_item.get('subgoal_name', 'Unknown')
        prev_base_name = get_base_subgoal_name(prev_raw_name)
        evidence_subgoal = f"{prev_base_name}"
    else:
        evidence_subgoal = "Task Start (No Previous Action)"

    if evidence_b64:
        user_content_list.append({
            "type": "text", 
            "text": f"""## [EVIDENCE FOR PHASE 3] PREVIOUS GROUP END STATE:\n**Previous Subgoal:** "{evidence_subgoal}" """
        })
        user_content_list.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{evidence_b64}"}
        })
        user_content_list.append({
            "type": "text", 
            "text": f"""\n**Last Action Performed:** {evidence_text}"""
        })
        user_content_list.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{start_b64}"}
        })
    else:
        user_content_list.append({"type": "text", "text": f"""## [EVIDENCE FOR PHASE 3] PREVIOUS GROUP END STATE:\n**Previous Subgoal:** "{evidence_subgoal}" """})

    # 添加输出格式要求
    user_content_list.append({
        "type": "text",
        "text": f"""
    \n===================================================
    ## Output Format (JSON)
    
    {{
        "trace_causal_analysis": {{
             "logic_chain_summary": "<string> Briefly analyze the causal chain. Which steps enabled others, and which were dead ends?"
        }},
        "step_critic": [
            {{
                "step_index": <int> (MUST be one of: [{target_indices_str}]),
                "is_finished_action": <bool>,
                "intent_vs_execution": "<string> 'Completed' or 'Failed'.",
                "intent_vs_subgoal": "<string> 'Useful', 'Plausible', or 'Useless'.",
                "score": <float>,
                "reason": "<string>"
            }},
            ...
        ],
        "current_group_conclusion": {{
            "subgoal_status": "<string> 'Completed', 'Partial', or 'Failed'.",
            "global_alignment": "<string> 'Useful', 'Misaligned', or 'Useless'.",
            "score": <float>,
            "reason": "<string>"
        }},
        "previous_group_judgment": {{
            "subgoal_status": "<string> 'Completed', 'Partial', or 'Failed'.",
            "global_alignment": "<string> 'Useful', 'Misaligned', or 'Useless'.",
            "prev_outcome_reward": <float>,
            "reason": "<string>"
        }}
    }}
    """
    })
    
    # 组合最终对话
    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content_list}
    ]
    
    # ==========================================
    # 🚀 发起单一的异步 LLM 调用
    # ==========================================
    response_content = await async_call_llm_fn(conversation_history, max_tokens=4096)
    
    if not response_content:
        return None
        
    try:
        parsed_json = json.loads(clean_json_response(response_content))
        
        # [关键] 后处理: 强力清洗 & 校验
        if "step_critic" in parsed_json:
            original_list = parsed_json["step_critic"]
            cleaned_list = [
                item for item in original_list 
                if isinstance(item, dict) and item.get("step_index") in target_step_indices
            ]
            if len(cleaned_list) != len(target_step_indices):
                # 记录警告并返回
                print(f"Step count mismatch! Expected {len(target_step_indices)} steps {target_step_indices}, but got {len(cleaned_list)} valid steps.")
                
            parsed_json["step_critic"] = cleaned_list
        
        return parsed_json
        
    except Exception as e:
        print(f"LLM 解析 JSON 失败: {e}")
        return None