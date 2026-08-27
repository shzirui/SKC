"""
vLLM Service Main Entry Point
"""

import asyncio
import os
import uvicorn
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import hydra
from omegaconf import DictConfig, OmegaConf
from fastapi import FastAPI, HTTPException, status, Depends

from src.services.model_service import (
    ModelServicePool, 
    ModelConfig,
    GenerationRequest,
    TokenizeRequest,
    SaveRequest,
    ReloadRequest
)

# Setup logging
from src.utils.log_config import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# 使用Hydra装饰器，它会自动处理配置加载
@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """主函数，用于解析命令行参数并启动 FastAPI 应用。"""
    
    # 打印解析后的配置
    logger.info(OmegaConf.to_yaml(cfg))

    # 创建 ModelConfig，直接从 Hydra 的配置对象中获取值
    model_cfg = ModelConfig(
        ckpt_path=cfg.model.ckpt_path,
        base_port=cfg.model.base_port,
        replicas=cfg.model.replicas,
        vllm_params=OmegaConf.to_container(cfg.model.vllm_params),
        save_local=cfg.model.save_local,
        save_path=cfg.storage.root,
        enable_lora=cfg.model.get("enable_lora", False),
        lora_adapter_path=cfg.model.get("lora_adapter_path", None),
        max_lora_rank=cfg.model.get("max_lora_rank", 32),
    )

    # 定义并创建 lifespan 函数的闭包
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_pool = ModelServicePool(model_cfg)
        
        if not await app.state.model_pool.initialize():
            logger.error("Fatal error: ModelServicePool initialization failed. Exiting.")
            raise RuntimeError("ModelServicePool initialization failed.")
        
        yield
        
        await app.state.model_pool.shutdown()

    # 创建 FastAPI 应用实例
    app = FastAPI(lifespan=lifespan)
    
    # 定义依赖注入函数
    def get_model_pool() -> ModelServicePool:
        """获取应用状态中的模型服务池实例"""
        return app.state.model_pool

    # 定义 API 路由
    @app.get("/")
    async def read_root():
        return {"message": "Model Service is running."}

    @app.post("/generate")
    async def generate_text(request: GenerationRequest, pool: ModelServicePool = Depends(get_model_pool)):
        """生成文本的API路由。"""
        try:
            kwargs = request.parameters or {}
            response_data = await pool.generate(request.messages, **kwargs)
            return response_data
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"An unexpected error occurred: {str(e)}"
            )

    @app.post("/tokenize")
    async def tokenize_text(request: TokenizeRequest, pool: ModelServicePool = Depends(get_model_pool)):
        kwargs = request.parameters or {}
        return await pool.tokenize(request.prompt, **kwargs)

    @app.post("/save")
    async def save_messages_reward(request: SaveRequest, pool: ModelServicePool = Depends(get_model_pool)):
        return await pool.save(
            messages=request.messages,
            reward=request.reward,
            task_id=request.task_id,
            trace_id=request.trace_id,
            parsed_actions=request.parsed_actions,
        )

    @app.post("/reload", summary="Reload model with rolling update")
    async def reload_model(request: ReloadRequest, pool: ModelServicePool = Depends(get_model_pool)):
        """通过滚动更新的方式重新加载模型。"""
        asyncio.create_task(pool.roll_reload(request.new_ckpt_path, request.batch_size, request.lora_adapter_path))
        return {"message": "Model rolling reload process initiated successfully."}

    @app.post("/remove_service_by_id", summary="Remove a service instance by gpu id")
    async def remove_service(gpu_id: int, pool: ModelServicePool = Depends(get_model_pool)):
        """通过GPU ID移除一个服务实例。"""
        try:
            await pool._remove_service_instance(gpu_id)
            return {"message": f"Service instance on GPU {gpu_id} removed successfully."}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/add_service_by_id", summary="Add a service instance by gpu id")
    async def add_service(gpu_id: int, pool: ModelServicePool = Depends(get_model_pool)):
        """通过GPU ID添加一个服务实例。"""
        try:
            await pool.add_service_by_id(gpu_id)
            return {"message": f"Service instance on GPU {gpu_id} added successfully."}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.get("/status", summary="Get service pool status")
    async def get_status(pool: ModelServicePool = Depends(get_model_pool)):
        """获取服务池的状态信息。"""
        status_info = await pool.get_status()
        return {
            "replicas_configured": pool.replicas,
            "active_services_count": len(status_info),
            "instances": status_info
        }
    
    @app.get("/gpu_info", summary="All GPU information")
    async def get_gpu_info(pool: ModelServicePool = Depends(get_model_pool)):
        gpu_info = await pool.get_gpu_info()
        return gpu_info
    
    @app.get("/shutdown")
    async def shutdown(pool: ModelServicePool = Depends(get_model_pool)):
        """关闭所有模型服务实例。"""
        await pool.shutdown()
        return {"message": "All model services have been shut down."}
    
    @app.get("/endpoints")
    async def get_endpoints(pool: ModelServicePool = Depends(get_model_pool)):
        """获取所有模型服务实例的端点。"""
        endpoints = await pool.get_endpoints()
        return {"endpoints": endpoints}
    
    @app.get("/checkpoint_info", summary="Get checkpoint path information")
    async def get_checkpoint_info(pool: ModelServicePool = Depends(get_model_pool)):
        """获取当前和上一个检查点路径信息。"""
        checkpoint_info = await pool.get_checkpoint_info()
        return checkpoint_info
    
    uvicorn.run(app, host=cfg.model.host, port=cfg.model.service_port)


if __name__ == "__main__":
    main()