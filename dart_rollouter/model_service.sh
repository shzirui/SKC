#!/usr/bin/env bash
# start_model.sh - Start the model service in the current container/image.
# Usage: CUDA_VISIBLE_DEVICES=6,7,8,9 bash dart_rollouter/model_service.sh

set -euo pipefail

if [ "${INSTALL_HYDRA_CORE:-1}" = "1" ]; then
    pip install hydra-core -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}

echo "[INFO] Starting model service: python -m src.run_model"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec python -m src.run_model
