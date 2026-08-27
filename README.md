# Selective Knowledge Control for Continual Learning of GUI Agents
### Selective Knowledge Control for Continual Learning of GUI Agents Over Application Streams

<p align="center">
&nbsp&nbsp🌐 <a href="https://github.com/computer-use-agents/dart-gui">DART-GUI</a>&nbsp&nbsp | &nbsp&nbsp📑 Paper: Coming Soon&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://huggingface.co/XinShu3047/gui-agent-checkpoints">Model</a>&nbsp&nbsp | &nbsp&nbsp📊 Data: Not Included&nbsp&nbsp
</p>

This repository extends the [DART-GUI](https://github.com/computer-use-agents/dart-gui) training infrastructure with **Selective Knowledge Control (SKC)**, a lightweight continual-learning method for GUI agents trained over a stream of applications.

For the full DART-GUI setup and system details, please refer to the official DART-GUI repository.

## 📢 Updates

- SKC training logic and configuration switches are added to the DART-GUI / verl training path.
- SKC state construction and merging utilities are added under `scripts/protection/`.
- Baseline utilities for replay, EWC, and LoRA-based continual-learning experiments are included for comparison.
- Runtime artifacts are not included: model weights, checkpoints, generated trajectories, MySQL dumps, logs, and experiment outputs should be prepared separately.

## 🔨 TODO

- [ ] Release public checkpoints and processed benchmark artifacts.
- [ ] Add final paper citation after public release.
- [ ] Further clean and document environment-specific launch scripts.

## 🚀 Quick Start

This guide follows the original DART-GUI structure: prepare containers and database, start the rollout service, start the GUI environment runner, then start training. The new SKC-specific part is the `gradient_surgery` configuration and the SKC state file used during training.

### 1. Preparation

#### Download Docker Images

Use DART-GUI-compatible images for the rollout service, trainer, and GUI environment. The exact image may depend on your cluster, CUDA/vLLM version, and accelerator type.

For the original DART-GUI image and environment instructions, see:

- DART-GUI: https://github.com/computer-use-agents/dart-gui
- GUI-Docker-Env: `GUI-Docker-Env/`

A MySQL container is also required:

```bash
docker pull mysql:8.0.44-debian
```

#### Prepare Model Checkpoints

Download the base GUI-agent model separately. For example:

```bash
huggingface-cli download ByteDance-Seed/UI-TARS-1.5-7B --local-dir /path/to/UI-TARS-1.5-7B
```

Set this path in both the rollout model-service config and the trainer launch script.

#### Prepare SKC State Artifacts

SKC uses protection artifacts and ordered checkpoints to build a historical state. First, generate `protected_neurons.json` for each completed application:

```bash
python scripts/protection/protection.py \
  --model <model_path> \
  --data_path <samples_jsonl> \
  --output_dir <protection_root> \
  --output_name <application_name> \
  --input_field <input_field> \
  --topk_ratio <topk_ratio>
```

Then build the historical dSVD directions. Repeat `--task` in history order and provide one more checkpoint than the number of tasks.

```bash
python scripts/protection/build_layer_hhist.py \
  --project-root <project_root> \
  --model-dirs <checkpoint_before> <checkpoint_after> \
  --task <task_name>=<protection_folder> \
  --protection-root <protection_root> \
  --output-dir <state_output_dir> \
  --layer <layer_index> \
  --num-directions <num_directions> \
  --write-gradient-surgery-state
```

This writes `gradient_surgery_state.pt` directly. Existing dSVD artifacts can also be converted separately:

```bash
python scripts/build_gradient_surgery_state.py \
  --input-dir <state_artifact_dir> \
  --output-state <output_state_path>
```

States can be merged between stages:

```bash
python scripts/merge_gradient_surgery_state.py \
  --prev-state <previous_state_path> \
  --current-state <current_state_path> \
  --output-state <merged_state_path>
```

### 2. Docker Initialization

Initialize the same components as DART-GUI: rollout service, trainer, MySQL, and GUI environment workers.

#### Rollouter Container

Used for the vLLM rollout model service.

```bash
docker run -dit \
  --name rollouter \
  --gpus all \
  --network=host \
  --shm-size=128g \
  -v /path/to/workspace:/workspace \
  <dart-gui-compatible-rollout-image> \
  sleep infinity
```

#### Trainer Container

Used for verl/FSDP training.

```bash
docker run -dit \
  --name trainer \
  --gpus all \
  --network=host \
  --shm-size=128g \
  -v /path/to/workspace:/workspace \
  <dart-gui-compatible-trainer-image> \
  sleep infinity
```

#### MySQL Container

Database server for rollout records and checkpoint versions.

```bash
docker run -dit \
  --name mysql-server \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD} \
  -v /path/to/mysql:/var/lib/mysql \
  mysql:8.0.44-debian
```

### 3. Database Configuration

The pipeline uses MySQL in the same way as DART-GUI. The most important tables are:

- `rollout_run`: stores trajectory metadata, rewards, task IDs, split paths, usage state, and model version.
- `checkpoint`: stores checkpoint paths and deployment status for model synchronization.

Set database credentials before launching rollout and training processes:

```bash
export DB_HOST="127.0.0.1"
export DB_USER="root"
export DB_PASSWORD="${DB_PASSWORD}"
export DB_DATABASE="dart"
export DB_PORT="3306"
export DB_CHARSET="utf8mb4"
```

For the full original SQL schema, see `README.md.infra_backup` or the official DART-GUI repository.

### 4. Environment Setup (GUI Workers)

Follow the DART-GUI environment setup for OSWorld-style GUI workers:

- `GUI-Docker-Env/`
- https://github.com/computer-use-agents/dart-gui

The GUI workers execute application tasks, save trajectory files, and write rollout metadata into MySQL.

### 5. Execution

#### Step 1: Start Rollouter

Inside the rollout container:

```bash
cd dart_rollouter
bash model_service.sh
```

The script starts:

```bash
python -m src.run_model
```

Configure the model service in `dart_rollouter/config/config.yaml`, especially:

- `model.ckpt_path`
- `model.service_port`
- `model.service_endpoint`
- `model.enable_lora` and `model.lora_adapter_path` when running LoRA baselines
- `mysql.*`

#### Step 2: Start Agent Runner

Inside the GUI environment:

```bash
cd dart_rollouter
bash run.sh
```

#### Step 3: Start Training

Inside the trainer container:

```bash
bash examples/osworld/async/debug.sh
```

Edit the launch script before running. The most commonly changed fields are:

- `MODEL_PATH`: base model or previous-stage checkpoint.
- `RUN_ID`: rollout run ID consumed from MySQL.
- `ROLLOUT_SERVER_URL`: rollout service endpoint.
- `N_GPUS_PER_NODE`, `N_NODES`, `CUDA_VISIBLE_DEVICES`, FSDP size, and batch sizes.
- SKC switches listed below.

## 🧠 Using Selective Knowledge Control

SKC is enabled from the trainer launch script and implemented in `verl/workers/actor/dp_actor.py`.

```bash
gradient_surgery=True
gradient_surgery_state_path="/path/to/gradient_surgery_state.pt"
gradient_surgery_all_project=False
gradient_surgery_all_zero=False
```

The flags mean:

- `gradient_surgery=True`: enable SKC. If set to `False`, the training logic follows the normal DART-GUI/verl update path.
- `gradient_surgery_state_path`: historical knowledge state used by SKC.
- `gradient_surgery_all_project=True`: ablation that projects all protected gradients.
- `gradient_surgery_all_zero=True`: ablation that zeros all protected gradients.

The default SKC implementation applies gradient control to a configurable MLP layer. Related optional parameters include:

- `gradient_surgery_layer`
- `gradient_surgery_hidden_size`
- `gradient_surgery_intermediate_size`
- `gradient_surgery_realtime_top_ratio`

A typical application-stream workflow is:

1. Train the first application with `gradient_surgery=False` because there is no historical state yet.
2. Build a SKC state from the first application stage.
3. Train the next application with `gradient_surgery=True` and `gradient_surgery_state_path` pointing to that state.
4. After the stage finishes, build and merge the new state.
5. Repeat for later applications.

## 🔬 Baselines and Ablations

This codebase also contains switches and scripts for comparison methods.

Replay:

```bash
use_replay=true
replay_run_ids='[app_a,app_b]'
```

EWC:

```bash
use_ewc=true
ewc_state_path="/path/to/ewc_state.pt"
ewc_coef=100
```

Build an EWC state with:

```bash
python scripts/build_ewc_state.py --help
```

LoRA continual-learning baseline:

```bash
use_lora_setting=true
lora_rank=32
lora_alpha=32
lora_target_modules=all-linear
```

LoRA rollout loading is configured in `dart_rollouter/config/config.yaml`.

## 📈 Efficiency Logging

Optional local efficiency logging can be enabled in the trainer script:

```bash
profile_efficiency_metrics=True
efficiency_metrics_path="logs/${EXPERIMENT_NAME}_efficiency_metrics.jsonl"
```

The log records per-step training time and SKC overhead. When remote experiment logging is unavailable, local JSONL logging is used as a fallback.

## 🤝 Acknowledgments

This repository is built on top of DART-GUI and verl, and uses vLLM for rollout model serving. Please refer to the official DART-GUI repository for the original training infrastructure and setup details:

- https://github.com/computer-use-agents/dart-gui

## 📝 Citation

The SKC paper citation will be added after public release.

```bibtex
@misc{skc_gui_agent_continual_learning,
  title = {Selective Knowledge Control for Continual Learning of GUI Agents Over Application Streams},
  year = {2027},
  note = {Anonymous submission artifact}
}
```
