import pytest
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.core.group import async_group_steps
from src.core.score_backward_multi import async_score_recursive_step_multi_turn

@pytest.mark.asyncio
async def test_prm_group_and_score_logic():
    # 1. 构造假数据
    dummy_instruction = "Open terminal and type 'hello'"
    dummy_steps = [
        {"step_id": 1, "raw_content": "Click Start", "image_file": "img1.png"},
        {"step_id": 2, "raw_content": "Type terminal", "image_file": "img2.png"}
    ]
    
    # 2. Mock: 伪造一个能返回完美 JSON 的大模型回调函数
    async def mock_llm_for_group(messages, **kwargs):
        # 模拟模型返回的分组 JSON
        fake_response = """
        ```json
        [
            {
                "subgoal_name": "Open App",
                "start_step": 1,
                "end_step": 2,
                "reasoning": "Standard procedure"
            }
        ]
        ```
        """
        return fake_response

    # ==========================================
    # 测试 Group 逻辑
    # ==========================================
    enriched_subgoals = await async_group_steps(dummy_instruction, dummy_steps, mock_llm_for_group)
    
    assert len(enriched_subgoals) == 1
    assert enriched_subgoals[0]["subgoal_name"] == "Open App"
    assert len(enriched_subgoals[0]["steps_data"]) == 2
    assert enriched_subgoals[0]["steps_data"][0]["step_idx"] == 1
    print("✅ Grouping 逻辑单测通过！")

    # ==========================================
    # 测试 Score 逻辑
    # ==========================================
    async def mock_llm_for_score(messages, **kwargs):
        # 模拟模型返回的打分 JSON
        fake_response = """
        {
            "trace_causal_analysis": {"logic_chain_summary": "looks good"},
            "step_critic": [
                {"step_index": 1, "is_finished_action": false, "score": 0.8, "reason": "ok"},
                {"step_index": 2, "is_finished_action": false, "score": 1.0, "reason": "perfect"}
            ],
            "current_group_conclusion": {"score": 0.9},
            "previous_group_judgment": {"prev_outcome_reward": 0.5}
        }
        """
        return fake_response

    eval_result = await async_score_recursive_step_multi_turn(
        task_instruction=dummy_instruction,
        task_root_path="./",
        all_subgoals=enriched_subgoals,
        current_idx=0,
        current_outcome_reward=1.0,
        async_call_llm_fn=mock_llm_for_score
    )
    
    assert eval_result is not None
    assert len(eval_result["step_critic"]) == 2
    assert eval_result["step_critic"][0]["score"] == 0.8
    assert eval_result["previous_group_judgment"]["prev_outcome_reward"] == 0.5
    print("✅ Scoring 逻辑与 JSON 解析单测通过！")

@pytest.mark.asyncio
async def test_score_step_mismatch_fallback():
    # 测试大模型如果出现幻觉（比如造了不存在的 step_index），我们的清洗逻辑是否能拦截
    # (你可以自己实现这个 case，把 mock 返回里的 step_index 改成 99，看看断言会不会抛出 Mismatch 警告)
    pass