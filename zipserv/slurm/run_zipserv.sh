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

Submit one Slurm job per model and one dependent collector job.

Options:
  --mode tune|run            One-time Split-K tuning or performance run
                             (default: run)
  --models MODEL [...]       Models to run (default: all enabled models)
  --layers LAYER [...]       Layers to run (default: all)
  --output-root DIR          Parent directory for all experiment directories
  --tuning-dir DIR           Persistent tuning directory
  --run-dir DIR              Exact run directory (use to resume a run)
  --time HH:MM:SS            Wall time for each model job (default: 05:00:00)
  --dry-run                  Print commands without submitting
  -h, --help                 Show this help

Examples:
  slurm/run_zipserv.sh --mode tune
  slurm/run_zipserv.sh --models llama3.1-8b --layers qkv_proj o_proj
  slurm/run_zipserv.sh --run-dir results/run-20260822-153000 --models llama3.1-8b
EOF
}

MODE=run
MODELS=(all)
LAYERS=(all)
RUN_DIR=
WALL_TIME=05:00:00
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires tune or run." >&2; exit 2; }
      MODE=$2
      shift 2
      [[ "$MODE" == tune || "$MODE" == run ]] || {
        echo "Invalid --mode: $MODE (expected tune or run)." >&2
        exit 2
      }
      ;;
    --models|--model)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
      [[ ${#MODELS[@]} -gt 0 ]] || { echo "--models requires a value." >&2; exit 2; }
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

mapfile -t PLAN < <(
  python3 - "$CONFIG" "${MODELS[*]}" "${LAYERS[*]}" <<'PY'
import json
import sys

config_path, model_arg, layer_arg = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
models = {
    model["id"]: [layer["id"] for layer in model["layers"]]
    for model in config["models"] if model.get("enabled", True)
}
requested_models, requested_layers = model_arg.split(), layer_arg.split()
if "all" in requested_models and requested_models != ["all"]:
    raise SystemExit("'all' cannot be combined with specific models")
if "all" in requested_layers and requested_layers != ["all"]:
    raise SystemExit("'all' cannot be combined with specific layers")
selected_models = list(models) if requested_models == ["all"] else requested_models
unknown = [model for model in selected_models if model not in models]
if unknown:
    raise SystemExit(f"unknown model(s): {', '.join(unknown)}; available: {', '.join(models)}")
for model in selected_models:
    layers = models[model] if requested_layers == ["all"] else requested_layers
    unknown = [layer for layer in layers if layer not in models[model]]
    if unknown:
        raise SystemExit(f"unknown layer(s) for {model}: {', '.join(unknown)}")
    print("\t".join([model, *layers]))
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

mkdir -p "$RESULT_ROOT"
if [[ "$MODE" == tune ]]; then
  [[ -z "$RUN_DIR" ]] || { echo "--run-dir cannot be used with --mode tune; use --tuning-dir." >&2; exit 2; }
  if [[ -z "$TUNING_DIR" ]]; then
    TUNING_DIR=$RESULT_ROOT/zipserv-${GPU_TAG}-${WEIGHT_TAG}-$(date +%Y%m%d)-tuning
  fi
  RUN_DIR=$TUNING_DIR
else
  if [[ -z "$TUNING_DIR" ]]; then
    shopt -s nullglob
    tuning_candidates=("$RESULT_ROOT"/zipserv-${GPU_TAG}-${WEIGHT_TAG}-*-tuning)
    shopt -u nullglob
    for ((index=${#tuning_candidates[@]} - 1; index >= 0; index--)); do
      if [[ -f "${tuning_candidates[index]}/selected_splitk.csv" ]]; then
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
    RUN_DIR=$RESULT_ROOT/zipserv-${GPU_TAG}-${WEIGHT_TAG}-$(date +%Y%m%d-%H%M%S)-run
  fi
fi

mkdir -p "$RUN_DIR/raw" "$RUN_DIR/logs"
cp -p "$CONFIG" "$RUN_DIR/experiments.json"

echo "Mode: $MODE"
echo "Run directory: $RUN_DIR"
echo "Models: ${#PLAN[@]}"

job_ids=()
for item in "${PLAN[@]}"; do
  IFS=$'\t' read -r -a fields <<< "$item"
  model=${fields[0]}
  layers=("${fields[@]:1}")
  model_dir=$RUN_DIR/raw/$model
  model_log=$RUN_DIR/logs/$model.log
  mkdir -p "$model_dir"
  safe_model=${model//./_}
  command=(
    sbatch --parsable
    --job-name="zip-${safe_model}-${MODE}"
    --time="$WALL_TIME"
    --output="$RUN_DIR/logs/${model}-slurm-%j.out"
    "$SCRIPT_DIR/run_zipserv.sbatch"
    "$MODE"
    --models "$model"
    --layers "${layers[@]}"
    --output-dir "$model_dir"
    --log-file "$model_log"
  )
  if [[ "$MODE" == run ]]; then
    command+=(--selection-file "$TUNING_DIR/selected_splitk.csv")
  fi

  printf '  %s: layers=%s\n' "$model" "${layers[*]}"
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
collector+=("$SCRIPT_DIR/collect_results.sbatch" "$MODE" "$RUN_DIR")

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
