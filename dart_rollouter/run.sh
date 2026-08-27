#!/usr/bin/env bash
# start_run.sh - Wait for model service, then start main run.py
# Usage: chmod +x start_run.sh

source ~/miniconda3/bin/activate 
conda activate verl
# cd /workspace/codes/verl/rollouter/
set -euo pipefail

export PYTHONUNBUFFERED=1

# ===== 2) Run main program =====
echo "[INFO] Starting main program: python src/run.py $*"

exec python -u -m src.run "$@"
