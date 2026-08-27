import pytest
import torch
import numpy as np
from collections import defaultdict

# =========================================================================
# 模拟代码：思路一 (Approach 1) - 奖励融合 + 全局 GRPO
# =========================================================================
def approach1_global_grpo(token_level_rewards: torch.Tensor, uid_index: np.ndarray):
    """
    原生 GRPO 逻辑：只按任务 (uid) 分组，所有步骤混在一起算优势。
    """
    returns_scalar = token_level_rewards.sum(dim=-1)
    adv_scalars = torch.zeros_like(returns_scalar)
    
    unique_uids = set(uid_index)
    for uid in unique_uids:
        # 找出当前任务的所有样本（不论是第几步）
        uid_mask = [i for i, x in enumerate(uid_index) if x == uid]
        tensor_idxs = torch.tensor(uid_mask, device=returns_scalar.device)
        group_returns = returns_scalar[tensor_idxs]
        
        # 全局均值和方差
        mean_r = group_returns.mean()
        std_r = group_returns.std(unbiased=True) + 1e-8
        
        adv_scalars[tensor_idxs] = (group_returns - mean_r) / std_r
        
    return adv_scalars

# =========================================================================
# 模拟代码：思路二 (Approach 2) - 密集收益 + 步级二维 GRPO (用 trace 动态推导)
# =========================================================================
def approach2_stepwise_grpo(token_level_rewards: torch.Tensor, uid_index: np.ndarray, trace_index: np.ndarray):
    """
    重写的 GRPO 逻辑：按任务 (uid) 和 步骤 (通过 trace_index 推导) 强行二维正交分组。
    """
    returns_scalar = token_level_rewards.sum(dim=-1)
    B = returns_scalar.shape[0]
    
    # 🌟 核心：动态推导真实的 step_idx
    trace_step_counts = defaultdict(int)
    step_indices = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        trace = trace_index[i]
        step_indices[i] = trace_step_counts[trace]
        trace_step_counts[trace] += 1

    adv_scalars = torch.zeros_like(returns_scalar)
    unique_uids = set(uid_index)
    
    for uid in unique_uids:
        uid_mask = [i for i, x in enumerate(uid_index) if x == uid]
        
        # 二次分组：按步数切分
        step_to_idxs = defaultdict(list)
        for idx in uid_mask:
            step = step_indices[idx].item()
            step_to_idxs[step].append(idx)
            
        for step, idxs in step_to_idxs.items():
            tensor_idxs = torch.tensor(idxs, device=returns_scalar.device)
            step_returns = returns_scalar[tensor_idxs]
            
            if len(step_returns) > 1:
                mean_r = step_returns.mean()
                std_r = step_returns.std(unbiased=True) + 1e-8
                step_adv = (step_returns - mean_r) / std_r
            else:
                step_adv = torch.zeros_like(step_returns)
                
            adv_scalars[tensor_idxs] = step_adv
            
    return adv_scalars

# =========================================================================
# 单元测试用例
# =========================================================================
def test_two_approaches_difference():
    """
    测试场景：
    同一个任务 (uid="Task_A") 跑了 2 次，产生了 2 条轨迹 (trace_1 和 trace_2)。
    每条轨迹都被切成了 2 步，所以 Batch Size 为 4。
    """
    # 1. 伪造非张量数据 (从数据库读取出来的 meta 数据)
    uid_index = np.array(["Task_A", "Task_A", "Task_A", "Task_A"])
    # 关键点：数据是按照轨迹顺序平铺的
    trace_index = np.array(["trace_1", "trace_1", "trace_2", "trace_2"])
    
    # 2. 伪造张量数据 (Reward)
    # 假设 trace_1 表现较好，trace_2 表现较差
    token_level_rewards = torch.tensor([
        [2.0],  # trace_1 第 0 步收益
        [1.0],  # trace_1 第 1 步收益
        [1.0],  # trace_2 第 0 步收益
        [0.0],  # trace_2 第 1 步收益
    ])

    print("\n" + "="*50)
    print("📊 思路一 (全局混合 GRPO) 的计算结果：")
    adv1 = approach1_global_grpo(token_level_rewards, uid_index).numpy()
    
    # 【修改点】去掉多余的 [0]
    print(f"  trace_1_step_0 (R=2.0) -> Advantage: {adv1[0]:.4f}")
    print(f"  trace_1_step_1 (R=1.0) -> Advantage: {adv1[1]:.4f}")
    print(f"  trace_2_step_0 (R=1.0) -> Advantage: {adv1[2]:.4f}")
    print(f"  trace_2_step_1 (R=0.0) -> Advantage: {adv1[3]:.4f}")
    
    # 思路一的断言验证：全局均值为 1.0，标准差为 0.8165
    # trace_1_step_1 的得分为 1.0，因为被全局平局值抵消，优势被算成了 0.0 (被埋没了！)
    assert np.isclose(adv1[1], 0.0, atol=1e-3)

    print("-" * 50)
    print("📈 思路二 (步级二维 GRPO) 的计算结果：")
    adv2 = approach2_stepwise_grpo(token_level_rewards, uid_index, trace_index).numpy()
    
    # 【修改点】去掉多余的 [0]
    print(f"  trace_1_step_0 (R=2.0) -> Advantage: {adv2[0]:.4f}")
    print(f"  trace_1_step_1 (R=1.0) -> Advantage: {adv2[1]:.4f}")
    print(f"  trace_2_step_0 (R=1.0) -> Advantage: {adv2[2]:.4f}")
    print(f"  trace_2_step_1 (R=0.0) -> Advantage: {adv2[3]:.4f}")
    
    # 思路二的断言验证：
    # Step 0 组 [2.0, 1.0]，均值 1.5，标准差 0.7071
    # Step 1 组 [1.0, 0.0]，均值 0.5，标准差 0.7071
    # trace_1_step_1 在“第 1 步”的较量中是赢家，因此它应该和 trace_1_step_0 拿到相同的正向 Advantage！
    assert np.isclose(adv2[0], 0.7071, atol=1e-3)
    assert np.isclose(adv2[1], 0.7071, atol=1e-3)
    assert np.isclose(adv2[2], -0.7071, atol=1e-3)
    assert np.isclose(adv2[3], -0.7071, atol=1e-3)
    
    print("✅ 两种思路的数学断言全部通过！思路二完美实现了步级正交对抗！")
    print("="*50)

if __name__ == "__main__":
    test_two_approaches_difference()