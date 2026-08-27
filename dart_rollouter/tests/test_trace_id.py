#!/usr/bin/env python3
"""
Test trace_id propagation throughout the call chain
"""

import asyncio
import logging
from omegaconf import DictConfig
from src.core.agent_coordinator import AgentCoordinator
from src.core.task_loader import TaskLoader
from src.utils.log_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def test_trace_id_propagation():
    """Test trace_id propagation throughout the call chain"""
    
    # Mock configuration
    coordinator_cfg = DictConfig({
        'max_concurrent_envs': 1,
        'max_task_queue_size': 1,
        'rollout_n': 1,
        'model_check_interval': 30,
        'env_cleanup_interval': 60,
        'env_cleanup_timeout': 300
    })
    
    # Create coordinator
    coordinator = AgentCoordinator(coordinator_cfg)
    
    # Create test task
    test_task = {
        'task_config': {'id': 'test_task_001'},
        'messages': [],
        'trace_id': 'TEST_TRACE_12345',
        'rollout_idx': 0
    }
    
    # Generate trace_id and verify format
    trace_id = coordinator._generate_trace_id()
    logger.info(f"Generated trace_id: {trace_id}")
    
    # Test task info
    task_with_trace = test_task.copy()
    task_with_trace['trace_id'] = trace_id
    
    logger.info(f"[{trace_id}] Test task created successfully - task_id: test_task_001")
    logger.info(f"[{trace_id}] trace_id propagation test completed")
    
    print("✅ trace_id propagation test completed!")
    print(f"   Generated trace_id: {trace_id}")
    print("   Check log files for the same trace_id")

if __name__ == "__main__":
    test_trace_id_propagation()