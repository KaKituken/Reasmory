#!/usr/bin/env bash
# Direct-VLM baseline: answer straight from the frames, no tools or reconstruction.
# Useful as the reference point the tool pipeline is compared against.
#
#   scripts/run_baseline.sh mindcube_tiny_no_text_shuffled
#   BASELINE_MODEL=gpt-5-mini scripts/run_baseline.sh vsibench_tiny
#   scripts/run_baseline.sh vlm4d_ego_50 vlm4d_allo_50
#
# BASELINE_MODEL accepts any model tools/llm_cfg.py knows, e.g.
#   gemini-3-flash-preview | gpt-5-mini | claude-sonnet-4-6
#   Qwen3-VL-4B-Instruct | SpaceOm | SpatialLadder-3B

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BASELINE_MODEL="${BASELINE_MODEL:-gemini-3-flash-preview}"
NFRAMES="${NFRAMES:-16}"

if [ $# -eq 0 ]; then
  echo "usage: $0 <benchmark> [<benchmark>...]" >&2
  exit 1
fi

for benchmark in "$@"; do
  echo "=== ${benchmark} | baseline ${BASELINE_MODEL} ==="
  # Video benchmarks need a media root; images are resolved from the annotations.
  video_root_arg=""
  if [ "$(scene_of "$benchmark")" = "dynamic" ]; then
    video_root_arg="--video_root ${REASMORY_DATA_ROOT:-./data}/VLM4D/videos_real"
  fi
  python evaluation/eval_agent.py \
    --benchmark "$benchmark" \
    --model_path "$BASELINE_MODEL" \
    --model_type "$(tag_for baseline "$BASELINE_MODEL")" \
    --temp "$TEMP" --top_p "$TOP_P" \
    --use_cot \
    --nframes "$NFRAMES" \
    --start_idx "$START_IDX" $MAX_SAMPLES_ARG \
    ${GEMINI_VIDEO_INPUT_MODE:+--gemini_video_input_mode "$GEMINI_VIDEO_INPUT_MODE"} \
    $video_root_arg
done
