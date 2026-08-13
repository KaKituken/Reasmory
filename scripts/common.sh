#!/usr/bin/env bash
# Shared setup sourced by the other scripts in this directory.
# Every value below can be overridden from the environment.

set -euo pipefail

# Always run from the repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME}/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

# Load API keys and path configuration from .env if present (see .env.example).
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

# Activate the conda environment unless the caller already has one.
CONDA_ENV="${CONDA_ENV:-qwen}"
if [ "${SKIP_CONDA:-0}" != "1" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  if command -v conda >/dev/null 2>&1; then
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  fi
fi

# Models used by the pipeline stages.
DECOMPOSER_MODEL="${DECOMPOSER_MODEL:-gemini-3-flash-preview}"
PLANNER_MODEL="${PLANNER_MODEL:-gemini-3-flash-preview}"
REASONER_MODEL="${REASONER_MODEL:-gemini-3-flash-preview}"
FALLBACK_MODEL="${FALLBACK_MODEL:-}"

TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_IDX="${START_IDX:-0}"

# Optional --max_samples passthrough.
MAX_SAMPLES_ARG=""
if [ -n "${MAX_SAMPLES:-}" ]; then
  MAX_SAMPLES_ARG="--max_samples ${MAX_SAMPLES}"
fi

# "dynamic" for moving-scene benchmarks (Flow3R), "static" otherwise (Pi3/Pi3x).
scene_of() {
  case "$1" in
    vlm4d*) echo dynamic ;;
    *)      echo static ;;
  esac
}

# Tag used in output filenames: eval_results/eval_<benchmark>/results_<tag>.json
tag_for() {
  local stage="$1" model="$2"
  echo "${MODEL_TAG:-${model//[^A-Za-z0-9]/_}_${stage}}"
}
