#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG=$EXP_ROOT/configs/experiments.json
RESULT_ROOT=${ZIP_RESULT_ROOT:-$EXP_ROOT/results}
TUNING_DIR=${ZIP_TUNING_DIR:-}

usage() {
  cat <<'EOF'
Usage: slurm/run_zipserv.sh [OPTIONS]

Submit one Slurm job per model/block and one dependent collector job.

Options:
  --mode tune|run|both       Split-K tuning, performance run, or both in sequence
                             (default: run)
  --partial-dtype bf16|fp32  Split-K partial-sum storage (default: bf16)
                             FP32 run requires an explicit --tuning-dir to reuse
  --models MODEL [...]       Models to run (default: all enabled models)
  --blocks MODEL=IDX,...     Override real-weight blocks; one job per model/block
                             (with --models all, selects the listed models)
  --layers LAYER [...]       Layers to run (default: all)
  --output-root DIR          Parent directory for all experiment directories
  --tuning-dir DIR           Persistent tuning directory
  --run-dir DIR              Exact run directory (use to resume a run)
  --time HH:MM:SS            Wall time for each model job (default: 05:00:00)
  --dry-run                  Print commands without submitting
  -h, --help                 Show this help

Examples:
  slurm/run_zipserv.sh --mode tune
  slurm/run_zipserv.sh --mode both --models llama3.1-8b llama3.1-70b
  slurm/run_zipserv.sh --models llama3.1-8b --layers qkv_proj o_proj
  slurm/run_zipserv.sh --mode run --blocks llama3.1-8b=16,31 llama3.1-70b=40,79
  slurm/run_zipserv.sh --run-dir results/run-20260822-153000 --models llama3.1-8b
EOF
}

MODE=run
PARTIAL_DTYPE=bf16
MODELS=(all)
BLOCK_SPECS=()
LAYERS=(all)
RUN_DIR=
WALL_TIME=05:00:00
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partial-dtype)
      [[ $# -ge 2 ]] || { echo "--partial-dtype requires bf16 or fp32." >&2; exit 2; }
      PARTIAL_DTYPE=$2
      [[ "$PARTIAL_DTYPE" == bf16 || "$PARTIAL_DTYPE" == fp32 ]] || {
        echo "Invalid partial dtype: $PARTIAL_DTYPE (expected bf16 or fp32)." >&2; exit 2;
      }
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires tune, run, or both." >&2; exit 2; }
      MODE=$2
      shift 2
      [[ "$MODE" == tune || "$MODE" == run || "$MODE" == both ]] || {
        echo "Invalid --mode: $MODE (expected tune, run, or both)." >&2
        exit 2
      }
      ;;
    --models|--model)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
      [[ ${#MODELS[@]} -gt 0 ]] || { echo "--models requires a value." >&2; exit 2; }
      ;;
    --blocks|--block)
      shift
      BLOCK_SPECS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do BLOCK_SPECS+=("$1"); shift; done
      [[ ${#BLOCK_SPECS[@]} -gt 0 ]] || {
        echo "--blocks requires MODEL=IDX[,IDX...] values." >&2
        exit 2
      }
      ;;
    --layers|--layer)
      shift
      LAYERS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do LAYERS+=("$1"); shift; done
      [[ ${#LAYERS[@]} -gt 0 ]] || { echo "--layers requires a value." >&2; exit 2; }
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "--output-root requires a directory." >&2; exit 2; }
      RESULT_ROOT=$2
      shift 2
      ;;
    --tuning-dir)
      [[ $# -ge 2 ]] || { echo "--tuning-dir requires a directory." >&2; exit 2; }
      TUNING_DIR=$2
      shift 2
      ;;
    --run-dir)
      [[ $# -ge 2 ]] || { echo "--run-dir requires a directory." >&2; exit 2; }
      RUN_DIR=$2
      shift 2
      ;;
    --time)
      [[ $# -ge 2 ]] || { echo "--time requires HH:MM:SS." >&2; exit 2; }
      WALL_TIME=$2
      shift 2
      [[ "$WALL_TIME" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || {
        echo "Invalid --time: $WALL_TIME (expected HH:MM:SS)." >&2
        exit 2
      }
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == run && "$PARTIAL_DTYPE" == fp32 && -z "$TUNING_DIR" ]]; then
  echo "FP32 run requires --tuning-dir; explicitly choose the Split-K selection to reuse." >&2
  exit 2
fi

mapfile -t PLAN < <(
  python3 - "$CONFIG" "${MODELS[*]}" "${LAYERS[*]}" "${BLOCK_SPECS[*]}" <<'PY'
import json
import struct
import sys
from pathlib import Path

config_path, model_arg, layer_arg, block_arg = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
is_synthetic = "synthetic" in config["input"].get("weight_source", "").lower()
models = {
    model["id"]: model
    for model in config["models"]
    if model.get("enabled", True)
    and (is_synthetic or isinstance(model.get("weight"), dict))
}
requested_models, requested_layers = model_arg.split(), layer_arg.split()
if "all" in requested_models and requested_models != ["all"]:
    raise SystemExit("'all' cannot be combined with specific models")
if "all" in requested_layers and requested_layers != ["all"]:
    raise SystemExit("'all' cannot be combined with specific layers")
block_specs = {}
for spec in block_arg.split():
    if "=" not in spec:
        raise SystemExit(f"invalid --blocks value {spec!r}; expected MODEL=IDX[,IDX...]")
    model_id, values = spec.split("=", 1)
    if not model_id or not values or model_id in block_specs:
        raise SystemExit(f"invalid or duplicate --blocks value: {spec!r}")
    try:
        blocks = [int(value) for value in values.split(",")]
    except ValueError:
        raise SystemExit(f"invalid block index in {spec!r}") from None
    if any(block < 0 for block in blocks) or len(set(blocks)) != len(blocks):
        raise SystemExit(f"block indices must be unique and non-negative: {spec!r}")
    block_specs[model_id] = blocks

if block_specs and is_synthetic:
    raise SystemExit("--blocks is only valid for real-weight experiments")

if requested_models == ["all"] and block_specs:
    selected_models = list(block_specs)
else:
    selected_models = list(models) if requested_models == ["all"] else requested_models
unknown = [model for model in selected_models if model not in models]
if unknown:
    raise SystemExit(f"unknown model(s): {', '.join(unknown)}; available: {', '.join(models)}")
unknown_block_models = [model for model in block_specs if model not in selected_models]
if unknown_block_models:
    raise SystemExit(f"--blocks contains unselected model(s): {', '.join(unknown_block_models)}")
missing_block_models = [model for model in selected_models if block_specs and model not in block_specs]
if missing_block_models:
    raise SystemExit(f"--blocks is missing selected model(s): {', '.join(missing_block_models)}")

def required_weight_files(model_id, model_config, layers, block):
    weight = model_config.get("weight")
    if not isinstance(weight, dict):
        raise SystemExit(f"real-weight configuration is missing for {model_id}")
    model_dir = Path(weight.get("model_dir", "")).expanduser()
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
    elif single_path.is_file():
        with single_path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        weight_map = {
            tensor_name: single_path.name
            for tensor_name in header
            if tensor_name != "__metadata__"
        }
    else:
        raise SystemExit(
            f"safetensors checkpoint is missing for {model_id}: "
            f"expected {index_path} or {single_path}"
        )

    tensor_map = {}
    layout_id = weight.get("layout")
    if layout_id:
        layout = config.get("weight_layouts", {}).get(layout_id)
        if not isinstance(layout, dict):
            raise SystemExit(f"unknown weight layout for {model_id}: {layout_id!r}")
        tensor_map.update(layout)
    overrides = weight.get("tensors", {})
    if overrides:
        if not isinstance(overrides, dict):
            raise SystemExit(f"weight.tensors must be an object for {model_id}")
        tensor_map.update(overrides)

    missing_tensors = []
    missing_shards = set()
    for layer in layers:
        templates = tensor_map.get(layer)
        if not isinstance(templates, list) or not templates:
            raise SystemExit(
                f"no tensor mapping for {model_id}/{layer}; layout={layout_id!r}"
            )
        try:
            tensor_names = [
                template.format(model=model_id, block=block)
                for template in templates
            ]
        except (AttributeError, KeyError, ValueError) as exc:
            raise SystemExit(f"invalid tensor template for {model_id}/{layer}: {exc}")
        for tensor_name in tensor_names:
            shard_name = weight_map.get(tensor_name)
            if not shard_name:
                missing_tensors.append(tensor_name)
            elif not (model_dir / shard_name).is_file():
                missing_shards.add(str(model_dir / shard_name))
    if missing_tensors:
        raise SystemExit(
            f"tensor(s) missing from {model_id} index: {', '.join(missing_tensors)}"
        )
    if missing_shards:
        raise SystemExit(
            f"required shard(s) missing for {model_id}: {', '.join(sorted(missing_shards))}"
        )

for model in selected_models:
    model_config = models[model]
    available_layers = [layer["id"] for layer in model_config["layers"]]
    layers = available_layers if requested_layers == ["all"] else requested_layers
    unknown = [layer for layer in layers if layer not in available_layers]
    if unknown:
        raise SystemExit(f"unknown layer(s) for {model}: {', '.join(unknown)}")
    weight = model_config.get("weight", {})
    blocks = block_specs.get(model, ["default"])
    resolved_blocks = [
        int(weight.get("block_index", 0)) if block == "default" else block
        for block in blocks
    ]
    if not is_synthetic:
        model_config_path = Path(weight.get("model_dir", "")) / "config.json"
        if model_config_path.is_file():
            with model_config_path.open(encoding="utf-8") as handle:
                num_hidden_layers = json.load(handle).get("num_hidden_layers")
            if num_hidden_layers is not None:
                invalid = [
                    block for block in resolved_blocks
                    if block >= int(num_hidden_layers)
                ]
                if invalid:
                    raise SystemExit(
                        f"block(s) out of range for {model}: {invalid}; "
                        f"available: 0..{int(num_hidden_layers) - 1}"
                    )
        for resolved_block in resolved_blocks:
            required_weight_files(model, model_config, layers, resolved_block)
    block_scoped = {
        layer["id"]: layer.get("block_scoped", True)
        for layer in model_config["layers"]
    }
    for position, block in enumerate(blocks):
        # Global tensors such as lm_head are measured once in the first job of
        # a block sweep instead of being redundantly repeated for every block.
        block_layers = [
            layer for layer in layers
            if position == 0 or block_scoped[layer]
        ]
        if block_layers:
            print("\t".join([model, str(block), *block_layers]))
PY
)

[[ ${#PLAN[@]} -gt 0 ]] || { echo "No jobs selected." >&2; exit 2; }

read -r GPU_TAG WEIGHT_TAG < <(
  python3 - "$CONFIG" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)

def tag(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower())

gpu = tag(config["gpu"].get("expected_name_substring", "gpu")) or "gpu"
weight_source = config["input"].get("weight_source", "unknown")
weight = "synthetic" if "synthetic" in weight_source.lower() else tag(weight_source)
print(gpu, weight or "unknown")
PY
)

BLOCK_TAG=
if [[ ${#BLOCK_SPECS[@]} -gt 0 ]]; then
  BLOCK_TAG=$(python3 - "${BLOCK_SPECS[@]}" <<'PY'
import re
import sys

parts = []
for spec in sys.argv[1:]:
    model, blocks = spec.split("=", 1)
    model = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    parts.append(f"{model}-b{blocks.replace(',', '-')}")
print("-blocks-" + "-".join(parts))
PY
  )
fi
RUN_WEIGHT_TAG=${WEIGHT_TAG}${BLOCK_TAG}-partial-${PARTIAL_DTYPE}

snapshot_config() {
  python3 - "$CONFIG" "$1/experiments.json" "$PARTIAL_DTYPE" <<'PY'
import csv
import json
import sys
from pathlib import Path
source, destination, dtype = sys.argv[1:]
path = Path(destination)
if path.exists() and json.loads(path.read_text()).get('partial_dtype', 'bf16') != dtype:
    raise SystemExit('Refusing to overwrite a result directory with a different partial dtype')
for result in (path.parent / 'raw').rglob('result.csv'):
    with result.open() as handle:
        if any((row.get('partial_dtype') or 'bf16') != dtype for row in csv.DictReader(handle)):
            raise SystemExit(f'Mixed partial dtype: use a new result directory, not {path.parent}')
config = json.loads(Path(source).read_text())
config['partial_dtype'] = dtype
path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
PY
}

mkdir -p "$RESULT_ROOT"

submit_both_model_jobs() {
  local phase=$1
  local phase_dir=$2
  local selection_file=$3
  local dependency_id=$4

  BOTH_MODEL_JOB_IDS=()
  for item in "${PLAN[@]}"; do
    IFS=$'\t' read -r -a fields <<< "$item"
    local model=${fields[0]}
    local block_index=${fields[1]}
    local layers=("${fields[@]:2}")
    local block_suffix=
    local block_path=
    if [[ "$block_index" != default ]]; then
      block_suffix=-block-$block_index
      block_path=/block-$block_index
    fi
    local model_dir=$phase_dir/raw/$model$block_path
    local model_log=$phase_dir/logs/$model$block_suffix.log
    local safe_model=${model//./_}
    mkdir -p "$model_dir"

    local command=(sbatch --parsable)
    if [[ -n "$dependency_id" ]]; then
      command+=(--dependency="afterok:$dependency_id")
    fi
    command+=(
      --job-name="zip-${safe_model}${block_suffix}-${phase}-${PARTIAL_DTYPE}"
      --time="$WALL_TIME"
      --output="$phase_dir/logs/${model}${block_suffix}-slurm-%j.out"
      "$SCRIPT_DIR/run_zipserv.sbatch"
      "$phase"
      --config "$phase_dir/experiments.json"
      --partial-dtype "$PARTIAL_DTYPE"
      --models "$model"
      --layers "${layers[@]}"
      --output-dir "$model_dir"
      --log-file "$model_log"
    )
    if [[ "$block_index" != default ]]; then
      command+=(--block-index "$block_index")
    fi
    if [[ "$phase" == run ]]; then
      command+=(--selection-file "$selection_file")
    fi

    printf '  %s %s%s: layers=%s\n' "$phase" "$model" "$block_suffix" "${layers[*]}"
    if [[ "$DRY_RUN" == 1 ]]; then
      printf '    '; printf '%q ' "${command[@]}"; printf '\n'
    else
      local submission
      submission=$("${command[@]}")
      local job_id=${submission%%;*}
      BOTH_MODEL_JOB_IDS+=("$job_id")
      printf '    job=%s\n' "$job_id"
    fi
  done

  if [[ "$DRY_RUN" == 1 ]]; then
    BOTH_MODEL_DEP="<${phase}-model-job-ids>"
  else
    BOTH_MODEL_DEP=$(IFS=:; echo "${BOTH_MODEL_JOB_IDS[*]}")
  fi
}

submit_both_collector() {
  local phase=$1
  local phase_dir=$2
  local dependency_type=$3
  local dependency_id=$4
  local command=(
    sbatch --parsable
    --job-name="zip-collect-${phase}"
    --output="$phase_dir/logs/collector-slurm-%j.out"
    --dependency="${dependency_type}:${dependency_id}"
    "$SCRIPT_DIR/collect_results.sbatch"
    "$phase"
    "$phase_dir"
    "$phase_dir/experiments.json"
  )

  if [[ "$DRY_RUN" == 1 ]]; then
    printf '  %s collector: ' "$phase"; printf '%q ' "${command[@]}"; printf '\n'
    BOTH_COLLECTOR_JOB_ID="<${phase}-collector-job-id>"
  else
    local submission
    submission=$("${command[@]}")
    BOTH_COLLECTOR_JOB_ID=${submission%%;*}
    printf '  %s collector job=%s\n' "$phase" "$BOTH_COLLECTOR_JOB_ID"
  fi
}

if [[ "$MODE" == both ]]; then
  if [[ -z "$TUNING_DIR" ]]; then
    TUNING_DIR=$RESULT_ROOT/zipserv-${GPU_TAG}-${RUN_WEIGHT_TAG}-$(date +%Y%m%d)-tuning
  fi
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$RESULT_ROOT/zipserv-${GPU_TAG}-${RUN_WEIGHT_TAG}-$(date +%Y%m%d-%H%M%S)-run
  fi
  [[ "$TUNING_DIR" != "$RUN_DIR" ]] || {
    echo "--tuning-dir and --run-dir must be different in --mode both." >&2
    exit 2
  }

  mkdir -p "$TUNING_DIR/raw" "$TUNING_DIR/logs" "$RUN_DIR/raw" "$RUN_DIR/logs"
  snapshot_config "$TUNING_DIR"
  snapshot_config "$RUN_DIR"

  echo "Mode: both"
  echo "Tuning directory: $TUNING_DIR"
  echo "Run directory: $RUN_DIR"
  echo "Model/block jobs: ${#PLAN[@]}"

  submit_both_model_jobs tune "$TUNING_DIR" "" ""
  submit_both_collector tune "$TUNING_DIR" afterok "$BOTH_MODEL_DEP"
  tune_collector_id=$BOTH_COLLECTOR_JOB_ID

  submit_both_model_jobs \
    run "$RUN_DIR" "$TUNING_DIR/selected_splitk.csv" "$tune_collector_id"
  submit_both_collector run "$RUN_DIR" afterany "$BOTH_MODEL_DEP"

  echo "Tuning result will be written to: $TUNING_DIR/result_all.csv"
  echo "Split-K selection will be written to: $TUNING_DIR/selected_splitk.csv"
  echo "Final result will be written to: $RUN_DIR/result_all.csv"
  echo "Final summary will be written to: $RUN_DIR/summary.csv"
  exit 0
fi

if [[ "$MODE" == tune ]]; then
  [[ -z "$RUN_DIR" ]] || { echo "--run-dir cannot be used with --mode tune; use --tuning-dir." >&2; exit 2; }
  if [[ -z "$TUNING_DIR" ]]; then
    TUNING_DIR=$RESULT_ROOT/zipserv-${GPU_TAG}-${RUN_WEIGHT_TAG}-$(date +%Y%m%d)-tuning
  fi
  RUN_DIR=$TUNING_DIR
else
  if [[ -z "$TUNING_DIR" ]]; then
    shopt -s nullglob
    tuning_candidates=("$RESULT_ROOT"/zipserv-${GPU_TAG}-${WEIGHT_TAG}-*-tuning)
    shopt -u nullglob
    for ((index=${#tuning_candidates[@]} - 1; index >= 0; index--)); do
      if [[ -f "${tuning_candidates[index]}/selected_splitk.csv" ]]; then
        # Automatic reuse is only within the same storage precision. Cross-dtype
        # comparisons must name --tuning-dir explicitly.
        if ! python3 - "${tuning_candidates[index]}/experiments.json" "$PARTIAL_DTYPE" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
dtype = json.loads(path.read_text()).get('partial_dtype', 'bf16') if path.is_file() else 'bf16'
raise SystemExit(0 if dtype == sys.argv[2] else 1)
PY
        then
          continue
        fi
        TUNING_DIR=${tuning_candidates[index]}
        break
      fi
    done
  fi
  if [[ -z "$TUNING_DIR" ]]; then
    echo "No tuning directory found for gpu=$GPU_TAG weight=$WEIGHT_TAG under $RESULT_ROOT" >&2
    echo "Run once with: slurm/run_zipserv.sh --mode tune" >&2
    exit 2
  fi
  if [[ ! -f "$TUNING_DIR/selected_splitk.csv" ]]; then
    echo "Missing tuning result: $TUNING_DIR/selected_splitk.csv" >&2
    echo "Run once with: slurm/run_zipserv.sh --mode tune" >&2
    exit 2
  fi
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$RESULT_ROOT/zipserv-${GPU_TAG}-${RUN_WEIGHT_TAG}-$(date +%Y%m%d-%H%M%S)-run
  fi
fi

mkdir -p "$RUN_DIR/raw" "$RUN_DIR/logs"
snapshot_config "$RUN_DIR"

echo "Mode: $MODE"
echo "Partial dtype: $PARTIAL_DTYPE"
echo "Run directory: $RUN_DIR"
echo "Model/block jobs: ${#PLAN[@]}"

job_ids=()
for item in "${PLAN[@]}"; do
  IFS=$'\t' read -r -a fields <<< "$item"
  model=${fields[0]}
  block_index=${fields[1]}
  layers=("${fields[@]:2}")
  block_suffix=
  block_path=
  if [[ "$block_index" != default ]]; then
    block_suffix=-block-$block_index
    block_path=/block-$block_index
  fi
  model_dir=$RUN_DIR/raw/$model$block_path
  model_log=$RUN_DIR/logs/$model$block_suffix.log
  mkdir -p "$model_dir"
  safe_model=${model//./_}
  command=(
    sbatch --parsable
    --job-name="zip-${safe_model}${block_suffix}-${MODE}-${PARTIAL_DTYPE}"
    --time="$WALL_TIME"
    --output="$RUN_DIR/logs/${model}${block_suffix}-slurm-%j.out"
    "$SCRIPT_DIR/run_zipserv.sbatch"
    "$MODE"
    --config "$RUN_DIR/experiments.json"
    --partial-dtype "$PARTIAL_DTYPE"
    --models "$model"
    --layers "${layers[@]}"
    --output-dir "$model_dir"
    --log-file "$model_log"
  )
  if [[ "$block_index" != default ]]; then
    command+=(--block-index "$block_index")
  fi
  if [[ "$MODE" == run ]]; then
    command+=(--selection-file "$TUNING_DIR/selected_splitk.csv")
  fi

  printf '  %s%s: layers=%s\n' "$model" "$block_suffix" "${layers[*]}"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '    '; printf '%q ' "${command[@]}"; printf '\n'
  else
    submission=$("${command[@]}")
    job_id=${submission%%;*}
    job_ids+=("$job_id")
    printf '    job=%s\n' "$job_id"
  fi
done

collector=(
  sbatch --parsable
  --job-name="zip-collect-${MODE}"
  --output="$RUN_DIR/logs/collector-slurm-%j.out"
)
if [[ "$DRY_RUN" == 1 ]]; then
  collector+=(--dependency="afterany:<model-job-ids>")
else
  dependency=$(IFS=:; echo "${job_ids[*]}")
  collector+=(--dependency="afterany:$dependency")
fi
collector+=(
  "$SCRIPT_DIR/collect_results.sbatch" "$MODE" "$RUN_DIR" "$RUN_DIR/experiments.json"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf '  collector: '; printf '%q ' "${collector[@]}"; printf '\n'
else
  collector_submission=$("${collector[@]}")
  echo "Collector job: ${collector_submission%%;*}"
fi

echo "Combined result will be written to: $RUN_DIR/result_all.csv"
if [[ "$MODE" == tune ]]; then
  echo "Split-K selection will be written to: $RUN_DIR/selected_splitk.csv"
fi
