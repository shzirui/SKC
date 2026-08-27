import asyncio
from src.core.task_loader import TaskLoader
from src.core.trajectory_runner import TrajectoryRunnerActor
from src.services.storage_actor import StorageActor

async def test_run_episode(config=None): 
    """Test running an episode"""
    task_loader = TaskLoader("./evaluation_examples", "./evaluation_examples/examples")
    tasks_batch = task_loader.poll_for_tasks()
    storage = StorageActor.remote("./results")
    t = TrajectoryRunnerActor(tasks_batch[0])
    await t.run_episode(None, storage)

if __name__ == "__main__":
    asyncio.run(test_run_episode())