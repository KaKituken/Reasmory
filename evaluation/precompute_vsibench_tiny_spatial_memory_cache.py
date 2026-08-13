import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from tqdm import tqdm

from evaluation.eval_three_step import prepare_eval_media_messages
from tools.agent_tools import construct_3d_spatial_memory, construct_3d_spatial_memory_metric
from tools.run_time import Runtime
from tools.spatial_memory import SpatialMemory
from pi3.utils.basic import load_images_as_tensor


def _default_cache_root(using_pi3x=False) -> Path:
    data_disk = os.environ.get("DATA_DISK")
    if not data_disk:
        raise EnvironmentError("DATA_DISK is not set.")
    cache_dir = "spatial_memory_cache_pi3x" if using_pi3x else "spatial_memory_cache"
    return Path(data_disk) / cache_dir


def _load_annotation(annotation_path: str) -> List[Dict[str, Any]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_identifier(sample: Dict[str, Any]) -> str:
    for key in ("problem_id", "id", "sample_id", "question_id"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return hashlib.md5(
        json.dumps(sample, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]


def _video_identifier(video_path: str) -> str:
    return Path(video_path).stem


def _video_cache_key(video_path: str, reconstruction_frames: int, fps: float) -> str:
    stat = os.stat(video_path)
    raw = f"{os.path.abspath(video_path)}::{stat.st_mtime_ns}::{stat.st_size}::{fps}::{reconstruction_frames}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_unique_video_jobs(
    annotation: List[Dict[str, Any]],
    start_idx: int,
    max_samples: Optional[int],
    reconstruction_frames: int,
    fps: float,
    cache_root: Path,
    overwrite: bool,
) -> tuple[list[Dict[str, Any]], int]:
    end_idx = None if max_samples is None else start_idx + max_samples
    subset = annotation[start_idx:end_idx]

    jobs_by_video_id: Dict[str, Dict[str, Any]] = {}
    skipped_existing = 0

    for item in subset:
        video_path = item.get("path")
        if not isinstance(video_path, str) or not video_path.endswith((".mp4", ".avi")):
            continue

        video_id = _video_identifier(video_path)
        sample_id = _sample_identifier(item)
        sample_dir = cache_root / video_id
        memory_path = sample_dir / "spatial_memory.npz"
        metadata_path = sample_dir / "spatial_memory.json"

        if video_id not in jobs_by_video_id:
            jobs_by_video_id[video_id] = {
                "video_id": video_id,
                "video_path": video_path,
                "sample_dir": str(sample_dir),
                "memory_path": str(memory_path),
                "metadata_path": str(metadata_path),
                "cache_key": _video_cache_key(
                    video_path=video_path,
                    reconstruction_frames=reconstruction_frames,
                    fps=fps,
                ),
                "source_problem_ids": [],
                "source_sample_ids": [],
                "representative_item": item,
            }

        jobs_by_video_id[video_id]["source_problem_ids"].append(item.get("problem_id"))
        jobs_by_video_id[video_id]["source_sample_ids"].append(sample_id)

    jobs: List[Dict[str, Any]] = []
    for video_id, job in jobs_by_video_id.items():
        memory_path = Path(job["memory_path"])
        metadata_path = Path(job["metadata_path"])
        if memory_path.exists() and metadata_path.exists() and not overwrite:
            print(f"[skip] video_id={video_id} cache exists: {memory_path}")
            skipped_existing += 1
            continue
        jobs.append(job)

    jobs.sort(key=lambda x: x["video_id"])
    return jobs, skipped_existing


@ray.remote
class SpatialMemoryPrecomputeWorker:
    def __init__(self, use_pi3x=False):
        self.runtime = Runtime()
        if use_pi3x:
            self.runtime.ensure_pi3x_metric()
        else:
            self.runtime.ensure_pi3()

    def process_job(
        self,
        job: Dict[str, Any],
        annotation_path: str,
        video_preview_frames: int,
        video_reconstruction_frames: int,
        fps: float,
    ) -> Dict[str, Any]:
        if fps != 1.0:
            raise ValueError(
                "This script intentionally matches eval_three_step's current video logic and requires fps=1.0."
            )

        item = job["representative_item"]
        video_id = job["video_id"]
        video_path = job["video_path"]
        sample_dir = Path(job["sample_dir"])
        memory_path = Path(job["memory_path"])
        cleanup_frame_paths: List[str] = []
        start_time = time.perf_counter()

        try:
            media_inputs = prepare_eval_media_messages(
                item,
                video_preview_frames=video_preview_frames,
                video_reconstruction_frames=video_reconstruction_frames,
            )
            cleanup_frame_paths = media_inputs["cleanup_frame_paths"]
            reconstruction_frame_paths = media_inputs["reconstruction_frame_paths"]
            preview_frame_paths = media_inputs["preview_frame_paths"]
            preview_indices = media_inputs["preview_frame_indices_in_reconstruction"]

            images = load_images_as_tensor(reconstruction_frame_paths)
            with self.runtime._lock:
                if self.runtime.pi3:
                    position, confidence, camera_trajectory, intrinsics, global_up = (
                        construct_3d_spatial_memory(images, pi3=self.runtime.pi3)
                    )
                elif self.runtime.pi3x_metric:
                    position, confidence, camera_trajectory, intrinsics, global_up = (
                        construct_3d_spatial_memory_metric(
                            images, pi3x=self.runtime.pi3x_metric
                        )
                    )
                else:
                    raise RuntimeError("No suitable runtime available for spatial memory construction.")
                

            memory = SpatialMemory(
                rgb_images=images,
                position_3d=position,
                confidence=confidence,
                camera_trajectory=camera_trajectory,
                intrinsics=intrinsics,
                global_up=global_up,
            )
            sample_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "cache_format_version": SpatialMemory.CACHE_FORMAT_VERSION,
                "annotation_path": os.path.abspath(annotation_path),
                "problem_id": item.get("problem_id"),
                "sample_id": _sample_identifier(item),
                "video_id": video_id,
                "video_path": video_path,
                "cache_key": job["cache_key"],
                "source_problem_ids": job["source_problem_ids"],
                "source_sample_ids": job["source_sample_ids"],
                "sampling": {
                    "fps": fps,
                    "video_reconstruction_frames": video_reconstruction_frames,
                    "video_preview_frames": video_preview_frames,
                    "preview_frame_indices_in_reconstruction": preview_indices,
                },
                "reconstruction_frame_paths": reconstruction_frame_paths,
                "preview_frame_paths": preview_frame_paths,
            }
            memory.save(str(memory_path), metadata=metadata)
            return {
                "status": "ok",
                "video_id": video_id,
                "memory_path": str(memory_path),
                "source_problem_ids": job["source_problem_ids"],
                "elapsed_sec": round(time.perf_counter() - start_time, 3),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "video_id": video_id,
                "video_path": video_path,
                "error": f"{type(exc).__name__}: {exc}",
                "source_problem_ids": job["source_problem_ids"],
                "elapsed_sec": round(time.perf_counter() - start_time, 3),
            }
        finally:
            for frame_path in cleanup_frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="evaluation/annotation/eval_vsibench_tiny.json",
    )
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--video_preview_frames", type=int, default=16)
    parser.add_argument("--video_reconstruction_frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpus_per_worker", type=float, default=1.0)
    parser.add_argument("--use_pi3x", action="store_true")
    args = parser.parse_args()

    if args.fps != 1.0:
        raise ValueError(
            "This script intentionally matches eval_three_step's current video logic and requires --fps 1.0."
        )
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive.")
    if args.num_gpus_per_worker <= 0:
        raise ValueError("--num_gpus_per_worker must be positive.")

    annotation = _load_annotation(args.annotation_path)
    cache_root = Path(args.cache_root) if args.cache_root else _default_cache_root(args.use_pi3x)
    cache_root.mkdir(parents=True, exist_ok=True)

    jobs, skipped_existing = _build_unique_video_jobs(
        annotation=annotation,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
        reconstruction_frames=args.video_reconstruction_frames,
        fps=args.fps,
        cache_root=cache_root,
        overwrite=args.overwrite,
    )

    if not jobs:
        print(
            json.dumps(
                {
                    "processed": 0,
                    "skipped_existing": skipped_existing,
                    "failed": 0,
                    "cache_root": str(cache_root),
                },
                indent=2,
            )
        )
        return

    num_workers = min(args.num_workers, len(jobs))
    ray.init(ignore_reinit_error=True)

    WorkerRemote = SpatialMemoryPrecomputeWorker.options(
        num_gpus=args.num_gpus_per_worker,
        max_concurrency=1,
    )
    workers = [WorkerRemote.remote(args.use_pi3x) for _ in range(num_workers)]

    pending: Dict[Any, Dict[str, Any]] = {}
    job_queue = list(jobs)

    for worker in workers:
        if not job_queue:
            break
        job = job_queue.pop(0)
        ref = worker.process_job.remote(
            job=job,
            annotation_path=args.annotation_path,
            video_preview_frames=args.video_preview_frames,
            video_reconstruction_frames=args.video_reconstruction_frames,
            fps=args.fps,
        )
        pending[ref] = {"worker": worker, "job": job}

    processed = 0
    failed = 0
    elapsed_times: List[float] = []
    pbar = tqdm(total=len(jobs), desc="Precomputing spatial memory", unit="video")

    while pending:
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = ready[0]
        assignment = pending.pop(ref)
        worker = assignment["worker"]
        result = ray.get(ref)
        video_id = result["video_id"]
        elapsed_sec = float(result.get("elapsed_sec", 0.0))
        elapsed_times.append(elapsed_sec)

        if result["status"] == "ok":
            print(
                f"[ok] video_id={video_id} saved to {result['memory_path']} "
                f"({elapsed_sec:.2f}s)"
            )
            processed += 1
        else:
            print(
                f"[fail] video_id={video_id} "
                f"video={result.get('video_path')} error={result.get('error')} "
                f"({elapsed_sec:.2f}s)"
            )
            failed += 1
        avg_sec = sum(elapsed_times) / max(len(elapsed_times), 1)
        pbar.update(1)
        pbar.set_postfix(
            processed=processed,
            failed=failed,
            avg_sec=f"{avg_sec:.2f}",
            last_sec=f"{elapsed_sec:.2f}",
        )

        if job_queue:
            next_job = job_queue.pop(0)
            next_ref = worker.process_job.remote(
                job=next_job,
                annotation_path=args.annotation_path,
                video_preview_frames=args.video_preview_frames,
                video_reconstruction_frames=args.video_reconstruction_frames,
                fps=args.fps,
            )
            pending[next_ref] = {"worker": worker, "job": next_job}

    pbar.close()
    print(
        json.dumps(
            {
                "processed": processed,
                "skipped_existing": skipped_existing,
                "failed": failed,
                "cache_root": str(cache_root),
                "num_workers": num_workers,
                "num_gpus_per_worker": args.num_gpus_per_worker,
                "unique_videos_considered": len(jobs) + skipped_existing,
                "avg_elapsed_sec": (
                    round(sum(elapsed_times) / len(elapsed_times), 3)
                    if elapsed_times
                    else 0.0
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
