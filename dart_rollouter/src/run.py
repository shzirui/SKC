import ray
import asyncio
from pathlib import Path
import time
import copy
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Updated import paths based on new project structure
from src.core.task_loader import TaskLoader
from src.core.agent_coordinator import AgentCoordinator
from src.services.model_service_pool import ModelServicePool
from src.services.storage_actor import StorageActor
from src.services.mysql_writer import MySQLWriterActor
from src.utils.log_config import setup_logging
from src.utils.kill_all_env import clean_env

# Setup unified logging system
setup_logging()
logger = logging.getLogger(__name__)

# Set environment variables
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    asyncio.run(async_main(config))

async def async_main(config=None):
    ray.init(log_to_driver=True, dashboard_host='0.0.0.0')

    print(f">>>taskloader begin to load tasks")
    task_loader = TaskLoader(config.task, config.storage.root)
    print(f">>>coordinator begin to init")
    coordinator = AgentCoordinator(config.coordinator)
    print(f">>>storage begin to init")
    print(f">>>config.storage:>>> {config.storage}")
    storage = StorageActor.remote(config.storage)
  
    mysql_writer = MySQLWriterActor.remote(config.mysql)
   
    model_pool = None
    model_pool = ModelServicePool.remote(model_cfg=config.model)

    if config.pr_model.use_pr:
        pr_model_pool = ModelServicePool.remote(model_cfg=config.pr_model)
    else:
        pr_model_pool = None
    
    # Clear environment
    clean_env()

    while True:
        try:
            completed = await coordinator.start_rollout(
                task_loader=task_loader,
                runner_cfg=config.runner,
                model_pool=model_pool,
                storage=storage,
                mysql_writer=mysql_writer,
                pr_model_pool=pr_model_pool
            )
            
            # Exit main loop if all tasks are completed
            if completed:
                logger.info("All tasks completed, program exits normally")
                break
                
        except KeyboardInterrupt:
            logger.info("Interrupt signal received, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error occurred in main loop: {e}")
            break

if __name__ == "__main__":
    main()