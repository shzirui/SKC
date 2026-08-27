from src.core.task_loader import TaskLoader
from src.core.trajectory_runner import TrajectoryRunnerActor

def test_env():
    """Test environment initialization"""
    task_loader = TaskLoader("./evaluation_examples", "./evaluation_examples/examples")
    tasks_batch = task_loader.poll_for_tasks()
    t = TrajectoryRunnerActor(tasks_batch[0])
    t._init_env(tasks_batch[0]["task_config"])

if __name__ == "__main__":
    test_env()