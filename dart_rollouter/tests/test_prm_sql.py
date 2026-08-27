import pytest
import json
from unittest.mock import MagicMock

# =================================================================
# ⚡ 核心黑科技：在导入真实模块前，把 ray.remote 变成一个透明的空壳
# 这样被 @ray.remote 装饰的类就会保持为普通的 Python 类，方便完美 Mock！
# =================================================================
import ray

def dummy_remote(*args, **kwargs):
    # 处理 @ray.remote 和 @ray.remote(max_concurrency=1) 两种情况
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def decorator(cls_or_func):
        return cls_or_func
    return decorator

# 偷天换日：替换真实的 ray.remote
original_remote = getattr(ray, "remote", None)
ray.remote = dummy_remote

# -----------------------------------------------------------------
# 此时再导入你的 Actor，它就是一个纯净的本地类了！
from src.services.mysql_writer import MySQLWriterActor
# -----------------------------------------------------------------

# 恢复原状，避免影响后续的其他测试文件
if original_remote:
    ray.remote = original_remote

def test_mysql_insert_with_process_reward():
    class DummyCfg:
        host = "localhost"
        port = 3306
        username = "root"
        password = "pwd"
        database = "db"

    # 1. 像普通 Python 类一样直接实例化（不需要 .remote()，也不需要 __wrapped__）
    writer = MySQLWriterActor(DummyCfg())
    
    # 2. 伪造数据库连接和游标
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # 强制替换内部方法，防止它真的去连数据库
    writer._get_new_conn = MagicMock(return_value=mock_conn)
    
    # 3. 准备带有 process_reward 的假数据
    dummy_meta = {
        'run_id': "run_1",
        'trajectory_id': "traj_1",
        'task_id': "task_1",
        'trace_id': "trace_1",
        'reward': 1.0,
        'model_version': "v1",
        'instruction': "do it",
        'num_chunks': 5,
        'process_reward': json.dumps([0.5, 0.8, 1.0]) # 核心测试点
    }
    
    # 4. 执行写入逻辑
    writer.insert_run(dummy_meta)
    
    # 5. 断言验证
    assert mock_cursor.execute.called, "MySQL execute 方法未被调用！"
    
    # 取出真正执行的 SQL 语句和参数元组
    args, kwargs = mock_cursor.execute.call_args
    sql_query = args[0]
    sql_params = args[1]
    
    # 🌟 终极验证：检查 SQL 语句和参数里有没有我们新加的字段！
    assert "process_reward" in sql_query, "SQL 语句中缺少 process_reward 字段！"
    assert '[0.5, 0.8, 1.0]' in sql_params, "传递给 MySQL 的参数里缺少 PR 数组！"
    
    print("\n✅ MySQLWriterActor 插入逻辑与字段对齐单测完美通过！")