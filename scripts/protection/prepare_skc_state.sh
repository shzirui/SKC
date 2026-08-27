#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Configuration
MODEL_PATH="<model_path>"
SAMPLES_JSONL="<samples_jsonl>"
PROTECTION_ROOT="<protection_root>"
APPLICATION_NAME="<application_name>"
INPUT_FIELD="<input_field>"
TOPK_RATIO="<topk_ratio>"

CHECKPOINT_DIRS=(
  "<checkpoint_before>"
  "<checkpoint_after>"
)
TASK_SPECS=(
  "<task_name>=<protection_folder>"
)

STATE_OUTPUT_DIR="<state_output_dir>"
LAYER_INDEX="<layer_index>"
NUM_DIRECTIONS="<num_directions>"
CURRENT_STATE_PATH="${STATE_OUTPUT_DIR}/gradient_surgery_state.pt"

# Leave empty when there is no previous state to merge.
PREVIOUS_STATE_PATH=""
MERGED_STATE_PATH="<merged_state_path>"

require_config() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == \<*\> ]]; then
    echo "Configure ${name} at the top of ${BASH_SOURCE[0]}." >&2
    exit 1
  fi
}

for config_name in \
  MODEL_PATH SAMPLES_JSONL PROTECTION_ROOT APPLICATION_NAME INPUT_FIELD \
  TOPK_RATIO STATE_OUTPUT_DIR LAYER_INDEX NUM_DIRECTIONS; do
  require_config "${config_name}" "${!config_name}"
done

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
  require_config "CHECKPOINT_DIRS" "${checkpoint_dir}"
done

for task_spec in "${TASK_SPECS[@]}"; do
  require_config "TASK_SPECS" "${task_spec}"
done

if (( ${#CHECKPOINT_DIRS[@]} != ${#TASK_SPECS[@]} + 1 )); then
  echo "CHECKPOINT_DIRS must contain one more entry than TASK_SPECS." >&2
  exit 1
fi

TASK_ARGS=()
for task_spec in "${TASK_SPECS[@]}"; do
  TASK_ARGS+=(--task "${task_spec}")
done

echo "[1/4] Generating protection artifacts"
"${PYTHON_BIN}" "${SCRIPT_DIR}/protection.py" \
  --model "${MODEL_PATH}" \
  --data_path "${SAMPLES_JSONL}" \
  --output_dir "${PROTECTION_ROOT}" \
  --output_name "${APPLICATION_NAME}" \
  --input_field "${INPUT_FIELD}" \
  --topk_ratio "${TOPK_RATIO}"

echo "[2/4] Building historical dSVD directions"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_layer_hhist.py" \
  --project-root "${PROJECT_ROOT}" \
  --model-dirs "${CHECKPOINT_DIRS[@]}" \
  "${TASK_ARGS[@]}" \
  --protection-root "${PROTECTION_ROOT}" \
  --output-dir "${STATE_OUTPUT_DIR}" \
  --layer "${LAYER_INDEX}" \
  --num-directions "${NUM_DIRECTIONS}"

echo "[3/4] Building the gradient-surgery state"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_gradient_surgery_state.py" \
  --input-dir "${STATE_OUTPUT_DIR}" \
  --output-state "${CURRENT_STATE_PATH}"

echo "[4/4] Merging states"
if [[ -n "${PREVIOUS_STATE_PATH}" ]]; then
  require_config "MERGED_STATE_PATH" "${MERGED_STATE_PATH}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/merge_gradient_surgery_state.py" \
    --prev-state "${PREVIOUS_STATE_PATH}" \
    --current-state "${CURRENT_STATE_PATH}" \
    --output-state "${MERGED_STATE_PATH}"
  echo "State ready: ${MERGED_STATE_PATH}"
else
  echo "PREVIOUS_STATE_PATH is empty; merge skipped."
  echo "State ready: ${CURRENT_STATE_PATH}"
fi
