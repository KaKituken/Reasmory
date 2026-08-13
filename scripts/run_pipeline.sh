#!/usr/bin/env bash
# Full pipeline on one or more benchmarks.
#
# Static scenes run decomposer -> planner -> executor+reasoner, reusing each
# stage's cached output. Dynamic scenes run the decomposer + dynamic-evidence
# reasoner. The scene type is inferred from the benchmark name.
#
#   scripts/run_pipeline.sh mindcube_tiny_no_text_shuffled
#   scripts/run_pipeline.sh vsibench_tiny --bev_fallback
#   scripts/run_pipeline.sh vlm4d_ego_50 vlm4d_allo_50
#
# Override anything through the environment, e.g.
#   REASONER_MODEL=gpt-5-mini MAX_SAMPLES=20 scripts/run_pipeline.sh mindcube_tiny_no_text_shuffled
#
# Stage reuse: if a stage's result file already exists it is reused; set
# FORCE_STAGES=1 to recompute.

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BENCHMARKS=()
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    -*) EXTRA_ARGS+=("$arg") ;;
    *)  BENCHMARKS+=("$arg") ;;
  esac
done
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
  echo "usage: $0 <benchmark> [<benchmark>...] [extra args forwarded to the backend]" >&2
  echo "       run 'python evaluation/run_eval.py --list-benchmarks' to see the options" >&2
  exit 1
fi

for benchmark in "${BENCHMARKS[@]}"; do
  scene="$(scene_of "$benchmark")"
  echo "=== ${benchmark} (${scene}) ==="

  if [ "$scene" = "dynamic" ]; then
    python evaluation/run_eval.py \
      --benchmark "$benchmark" \
      --model_type "$(tag_for pipeline "$REASONER_MODEL")" \
      --decomposer_model_path "$DECOMPOSER_MODEL" \
      --reasoner_model_path "$REASONER_MODEL" \
      --temp "$TEMP" --top_p "$TOP_P" \
      --num_workers "$NUM_WORKERS" --start_idx "$START_IDX" \
      $MAX_SAMPLES_ARG "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    continue
  fi

  out_dir="eval_results/eval_${benchmark}"
  decomp_tag="$(tag_for decomposer "$DECOMPOSER_MODEL")"
  planner_tag="$(tag_for planner "$PLANNER_MODEL")"
  decomp_results="${out_dir}/results_${decomp_tag}.json"
  planner_results="${out_dir}/results_${planner_tag}.json"

  echo "[1/3] decomposer"
  if [ -f "$decomp_results" ] && [ "${FORCE_STAGES:-0}" != "1" ]; then
    echo "      reusing ${decomp_results}"
  else
    python evaluation/eval_decomposer.py \
      --benchmark "$benchmark" --model_type "$decomp_tag" \
      --model_path "$DECOMPOSER_MODEL" \
      --temp "$TEMP" --top_p "$TOP_P" \
      --num_workers "$NUM_WORKERS" --start_idx "$START_IDX" $MAX_SAMPLES_ARG
  fi

  echo "[2/3] planner"
  if [ -f "$planner_results" ] && [ "${FORCE_STAGES:-0}" != "1" ]; then
    echo "      reusing ${planner_results}"
  else
    python evaluation/eval_dsl_planner.py \
      --benchmark "$benchmark" --model_type "$planner_tag" \
      --model_path "$PLANNER_MODEL" \
      --decomposition_results "$decomp_results" \
      --temp "$TEMP" --top_p "$TOP_P" \
      --num_workers "$NUM_WORKERS" --start_idx "$START_IDX" $MAX_SAMPLES_ARG \
      ${VIDEO_PREVIEW_FRAMES:+--video_preview_frames $VIDEO_PREVIEW_FRAMES} \
      ${VIDEO_RECONSTRUCTION_FRAMES:+--video_reconstruction_frames $VIDEO_RECONSTRUCTION_FRAMES}
  fi

  echo "[3/3] executor + reasoner"
  python evaluation/run_eval.py \
    --benchmark "$benchmark" \
    --model_type "$(tag_for pipeline "$REASONER_MODEL")" \
    --planner_model_path "$PLANNER_MODEL" \
    --reasoner_model_path "$REASONER_MODEL" \
    --import_existing_plan "$planner_results" \
    --planner_temp "$TEMP" --reasoner_temp "$TEMP" \
    --num_workers "$NUM_WORKERS" --start_idx "$START_IDX" $MAX_SAMPLES_ARG \
    ${FALLBACK_MODEL:+--fallback_model_path "$FALLBACK_MODEL"} \
    ${SPATIAL_MEMORY_CACHE_ROOT:+--spatial_memory_cache_root "$SPATIAL_MEMORY_CACHE_ROOT"} \
    ${MEASURING_CACHE_ROOT:+--measuring_cache_root "$MEASURING_CACHE_ROOT"} \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
done

echo "Done. Results under eval_results/eval_<benchmark>/"
