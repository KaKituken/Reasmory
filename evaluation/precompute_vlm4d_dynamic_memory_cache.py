"""
Precompute Flow3r dynamic-memory caches for VLM4D videos.

Mirrors the static-memory precompute pattern in
`precompute_vsibench_tiny_spatial_memory_cache.py`. We hash each video on
(path, mtime, size, fps, n_frames) to get a stable cache key, then run
Flow3r in batch mode (model loaded once) to fill an .npz cache.

The .npz format is the same SpatialMemory cache format the static path uses,
so `runtime.load_spatial_memory_cache(session_id, cache_path)` can pick up
the dynamic memory transparently at eval time.

Usage:
    python evaluation/precompute_vlm4d_dynamic_memory_cache.py \
        --benchmarks vlm4d_ego_50 vlm4d_allo_50 \
        --fps 1.0 --max_frames 32

Cache root:  $FLOW3R_CACHE_ROOT (default ./data/flow3r_cache)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from tools.spatial_memory import SpatialMemory  # noqa: E402

FLOW3R_PYTHON = os.environ.get(
    "FLOW3R_PYTHON", "./data/baselines/envs/flow3r/bin/python"
)
FLOW3R_RUNNER = os.environ.get(
    "FLOW3R_RUNNER", "./data/baselines/Flow3r/flow3r_runner.py"
)
FLOW3R_CHECKPOINT = os.environ.get(
    "FLOW3R_CHECKPOINT", "./data/flow3r_weights/flow3r.bin"
)
FLOW3R_CACHE_ROOT = Path(os.environ.get("FLOW3R_CACHE_ROOT", "./data/flow3r_cache"))


def _video_cache_key(video_path: str, fps: float, max_frames: int, pixel_limit: int) -> str:
    stat = os.stat(video_path)
    raw = f"{os.path.abspath(video_path)}::{stat.st_mtime_ns}::{stat.st_size}::{fps}::{max_frames}::{pixel_limit}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _npz_to_spatial_memory_cache(npz_path: str, sm_cache_path: str, *, align_xz_with_pca: bool = True):
    """Load a Flow3r-runner .npz, run the same post-processing as the dynamic
    memory tool (two_stage_up_estimation), and save a SpatialMemory-format cache.

    This way `runtime.load_spatial_memory_cache(sm_cache_path)` returns a ready-to-use
    SpatialMemory whose schema is identical to the static-memory cache.
    """
    import torch
    from agent_tools import two_stage_up_estimation  # type: ignore  # late import

    d = np.load(npz_path)
    images = d["images"]
    points = d["points"].astype(np.float32)
    conf = d["conf"].astype(np.float32)
    poses = d["camera_poses"].astype(np.float32)
    K_raw = d["intrinsics"]
    frame_idx = d["frame_index"]

    rotated_points, R, global_up = two_stage_up_estimation(
        points.copy(), images, poses.copy(), target_axis="-y"
    )
    camera_traj = poses.copy()
    cams_rot = camera_traj[:, :3, :3]
    cams_transl = camera_traj[:, :3, 3]
    camera_traj[:, :3, :3] = R @ cams_rot
    camera_traj[:, :3, 3] = cams_transl @ R.T

    K_norm = K_raw[0] if K_raw.ndim == 3 else K_raw
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).contiguous().float()

    sm = SpatialMemory(
        rgb_images=rgb_tensor,
        position_3d=rotated_points.astype(np.float32),
        confidence=conf,
        camera_trajectory=camera_traj.astype(np.float32),
        intrinsics=K_norm.astype(np.float32),
        global_up=np.asarray(global_up, dtype=np.float32),
        align_xz_with_pca=align_xz_with_pca,
    )
    meta = {
        "backend": "flow3r",
        "is_dynamic": True,
        "frame_index": frame_idx.tolist(),
        "memory_resolution_hw": list(sm.memory_3d_map_size),
    }
    sm.save(sm_cache_path, metadata=meta)
    return sm_cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["vlm4d_ego_50", "vlm4d_allo_50"])
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--pixel_limit", type=int, default=200000)
    ap.add_argument("--cache_root", default=str(FLOW3R_CACHE_ROOT))
    ap.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    ap.add_argument("--gpu", type=int, default=None, help="CUDA index for the Flow3r runner")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_npz_dir = cache_root / "_raw_npz"
    raw_npz_dir.mkdir(exist_ok=True)
    sm_dir = cache_root / "spatial_memory"
    sm_dir.mkdir(exist_ok=True)
    index_path = cache_root / "index.json"

    # 1) Collect unique videos from the benchmarks
    items: List[Dict] = []
    for bench in args.benchmarks:
        path = REPO_ROOT / "evaluation" / "annotation" / f"eval_{bench}.json"
        items.extend(json.load(open(path)))

    seen, jobs = {}, []
    for item in items:
        v = item["path"]
        if v in seen:
            continue
        if not os.path.exists(v):
            print(f"[precompute] skip missing: {v}")
            continue
        seen[v] = True
        key = _video_cache_key(v, args.fps, args.max_frames, args.pixel_limit)
        raw_npz = raw_npz_dir / f"{key}.npz"
        sm_npz = sm_dir / f"{key}.npz"
        if not args.force and sm_npz.exists():
            continue
        jobs.append({
            "video": v,
            "cache_key": key,
            "raw_npz": str(raw_npz),
            "sm_npz": str(sm_npz),
            "manifest": {
                "input_type": "video",
                "video_path": v,
                "fps": args.fps,
                "max_frames": args.max_frames,
                "pixel_limit": args.pixel_limit,
                "checkpoint_path": FLOW3R_CHECKPOINT,
                "device": "cuda",
            },
        })

    print(f"[precompute] {len(seen)} unique videos in benchmarks; {len(jobs)} need building.")
    if not jobs:
        return

    # 2) Phase 1: batch-call the Flow3r runner with a single model load
    batch_path = cache_root / "_batch_jobs.json"
    with open(batch_path, "w") as f:
        json.dump([{"manifest": j["manifest"], "output": j["raw_npz"]} for j in jobs], f)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[precompute] launching Flow3r batch on {len(jobs)} jobs")
    t0 = time.time()
    subprocess.run(
        [FLOW3R_PYTHON, FLOW3R_RUNNER, "--batch", str(batch_path)],
        env=env, check=True,
    )
    print(f"[precompute] Flow3r batch finished in {time.time()-t0:.1f}s")

    # 3) Phase 2: for each raw .npz, run the same post-processing as the live tool
    #    and save a SpatialMemory cache file.
    print(f"[precompute] post-processing {len(jobs)} raw npz files → SpatialMemory cache")
    t1 = time.time()
    for j in jobs:
        if not os.path.exists(j["raw_npz"]):
            print(f"[precompute] raw npz missing for {j['video']}; runner failure?")
            continue
        try:
            _npz_to_spatial_memory_cache(j["raw_npz"], j["sm_npz"])
        except Exception as e:
            print(f"[precompute] FAIL post-process {j['video']}: {e}")
    print(f"[precompute] post-processing done in {time.time()-t1:.1f}s")

    # 4) Write/update the lookup index keyed by absolute video path → SpatialMemory cache path
    if index_path.exists():
        index = json.load(open(index_path))
    else:
        index = {}
    for j in jobs:
        if os.path.exists(j["sm_npz"]):
            index[j["video"]] = {
                "sm_cache": j["sm_npz"],
                "raw_npz": j["raw_npz"],
                "cache_key": j["cache_key"],
                "fps": args.fps,
                "max_frames": args.max_frames,
                "pixel_limit": args.pixel_limit,
            }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[precompute] index → {index_path}  (now {len(index)} entries)")


if __name__ == "__main__":
    main()
