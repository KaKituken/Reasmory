#!/usr/bin/env bash
# Run a single pipeline stage, for iterating on one prompt without paying for the rest.
#
#   scripts/run_stage.sh decomposer mindcube_tiny_no_text_shuffled
#   scripts/run_stage.sh planner    mindcube_tiny_no_text_shuffled      # reuses the decomposer output
#
# The planner stage expects the decomposer results to exist; point at a specific
# file with DECOMPOSITION_RESULTS=<path> if you want a different one.

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

stage="${1:-}"
benchmark="${2:-}"
if [ -z "$stage" ] || [ -z "$benchmark" ]; then
  echo "usage: $0 {decomposer|planner} <benchmark>" >&2
  exit 1
fi
shift 2

case "$stage" in
  decomposer)
    python evaluation/eval_decomposer.py \
      --benchmark "$benchmark" \
      --model_type "$(tag_for decomposer "$DECOMPOSER_MODEL")" \
      --model_path "$DECOMPOSER_MODEL" \
      --temp "$TEMP" --top_p "$TOP_P" \
      --num_workers "$NUM_WORKERS" --start_idx "$START_IDX" $MAX_SAMPLES_ARG "$@"
    ;;
  planner)
    decomp="${DECOMPOSITION_RESULTS:-eval_results/eval_${benchmark}/results_$(tag_for decomposer "$DECOMPOSER_MODEL").json}"
    if [ ! -f "$decomp" ]; then
      echo "missing decomposer results: $decomp" >&2
      echo "run '$0 decomposer $benchmark' first, or set DECOMPOSITION_RESULTS" >&2
      exit 1
    fi
    python evaluation/eval_dsl_planner.py \
      --benchmark "$benchmark" \
      --model_type "$(tag_for planner "$PLANNER_MODEL")" \
      --model_path "$PLANNER_MODEL" \
      --decomposition_results "$decomp" \
      --temp "$TEMP" --top_p "$TOP_P" \
      --num_workers "$NUM_WORKERS" --start_idx "$START_IDX" $MAX_SAMPLES_ARG "$@"
    ;;
  *)
    echo "unknown stage '$stage' (expected decomposer or planner)" >&2
    exit 1
    ;;
esac
