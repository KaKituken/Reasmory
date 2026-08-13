#!/usr/bin/env bash
# Precompute the 3D reconstruction caches. Reconstruction dominates runtime, so
# build the cache once and every later evaluation reuses it.
#
#   scripts/precompute_cache.sh static  vsibench_tiny
#   scripts/precompute_cache.sh dynamic vlm4d_ego_50 vlm4d_allo_50
#
# Cache locations come from REASMORY_SPATIAL_MEMORY_CACHE /
# REASMORY_DYNAMIC_MEMORY_CACHE (see .env.example).

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

kind="${1:-}"
shift || true
if [ -z "$kind" ]; then
  echo "usage: $0 {static|dynamic} [<benchmark>...]" >&2
  exit 1
fi

case "$kind" in
  static)
    python evaluation/precompute_vsibench_tiny_spatial_memory_cache.py \
      ${REASMORY_SPATIAL_MEMORY_CACHE:+--cache_root "$REASMORY_SPATIAL_MEMORY_CACHE"} \
      --num_workers "$NUM_WORKERS" --start_idx "$START_IDX" $MAX_SAMPLES_ARG "$@"
    ;;
  dynamic)
    benchmarks=("$@")
    if [ ${#benchmarks[@]} -eq 0 ]; then benchmarks=(vlm4d_ego_50 vlm4d_allo_50); fi
    python evaluation/precompute_vlm4d_dynamic_memory_cache.py \
      --benchmarks "${benchmarks[@]}" \
      --fps "${FPS:-4.0}" --max_frames "${MAX_FRAMES:-32}"
    ;;
  *)
    echo "unknown kind '$kind' (expected static or dynamic)" >&2
    exit 1
    ;;
esac
