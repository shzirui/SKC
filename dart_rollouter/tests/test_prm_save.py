import pytest
import os
import json
import tempfile
import ray
from types import SimpleNamespace
from src.services.storage_actor import StorageActor # 请确保这里导入路径与你的项目一致

# 注意：这里我们去掉了 @pytest.mark.asyncio，因为对付 Ray Actor，用同步的 ray.get() 阻塞等待更简单稳定
def test_storage_save_prm_detail():
    # 1. 启动一个本地的微型 Ray 实例来进行测试
    ray.init(ignore_reinit_error=True)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 2. 使用 SimpleNamespace 模拟 config。
            # (千万不要在函数内部定义 class DummyCfg，因为局部类无法被 Ray 序列化并发送给 Actor)
            cfg = SimpleNamespace(root=temp_dir)
            
            # 3. 按照标准方式创建 Actor
            storage = StorageActor.remote(cfg)
            
            task_root = "task123_trace456"
            dummy_detail = {"test": "data", "score": [0.8, 1.0]}
            
            # 4. 调用 remote 并用 ray.get() 同步阻塞等待它内部的 async 逻辑执行完毕！
            ray.get(storage.save_prm_detail.remote(task_root, dummy_detail))
            
            # 5. 验证文件是否确实写到了磁盘上
            expected_path = os.path.join(temp_dir, task_root, "prm_detail.json")
            assert os.path.exists(expected_path)
            
            with open(expected_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                assert saved_data["score"][0] == 0.8
            print("✅ StorageActor 保存 PRM 详情单测通过！")
            
    finally:
        # 6. 测试结束后必须关闭 Ray，以免卡死后续的其他测试
        ray.shutdown()