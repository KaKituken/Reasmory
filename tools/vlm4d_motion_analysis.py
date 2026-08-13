"""
Motion-analysis primitives on top of a Flow3r-backed dynamic SpatialMemory.

Phase B's most useful single piece of evidence for VLM4D is a textual
summary of CAMERA EGO-MOTION: Gemini-3 / GPT-5 often misattribute pixel
motion to the scene when the camera itself is moving. By telling them
"the camera is panning right at ~10 deg/frame and translating ~0.3 m
forward" they can correctly subtract that motion when answering.

This module reads a SpatialMemory (statically or dynamically built) and
produces:
  * `summarize_camera_egomotion(sm)` -> short structured text
  * `summarize_scene_extent(sm)` -> hint about how big the scene is
  * `summarize_per_axis_camera_motion(sm)` -> components in the camera frame
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, Optional, Tuple

import numpy as np


_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from spatial_memory import SpatialMemory  # noqa: E402


def _so3_log(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (axis, angle) for a 3x3 rotation matrix in world frame."""
    # robust rotation-vector extraction
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-6:
        return np.array([0.0, 0.0, 0.0]), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2.0 * np.sin(theta) + 1e-12)
    return axis, theta


def _relative_pose(c2w_a: np.ndarray, c2w_b: np.ndarray) -> np.ndarray:
    """world_T_b in frame of camera a:  T_ba = inv(c2w_a) @ c2w_b ."""
    inv_a = np.linalg.inv(c2w_a)
    return inv_a @ c2w_b


def summarize_camera_egomotion(sm: SpatialMemory) -> Dict[str, Any]:
    """Aggregate camera_trajectory into a few high-signal numbers + a natural-
    language sentence. The world frame here is the one used inside SpatialMemory
    (after two_stage_up_estimation), so +y is gravity (down)."""
    poses = np.asarray(sm.camera_trajectory)
    if poses is None or len(poses) < 2:
        return {"available": False, "summary": "Camera motion unknown."}

    # Pairwise relative motion summed → cumulative rotation + translation in the FIRST camera frame.
    cam0_inv = np.linalg.inv(poses[0])
    trans_cam0 = np.zeros((len(poses), 3), dtype=np.float64)
    rot_cam0 = np.zeros((len(poses),), dtype=np.float64)        # cumulative |angle|
    for i, p in enumerate(poses):
        T = cam0_inv @ p
        trans_cam0[i] = T[:3, 3]
        _, ang = _so3_log(T[:3, :3])
        rot_cam0[i] = ang

    last = trans_cam0[-1]
    rot_deg = float(np.degrees(rot_cam0[-1]))

    # Decompose translation in cam-0 frame: +x right, +y down, +z forward (OpenCV).
    tx, ty, tz = float(last[0]), float(last[1]), float(last[2])
    speed_xz = float(np.hypot(tx, tz))

    # Per-frame angular rate
    if len(poses) > 1:
        per_step = np.zeros(len(poses) - 1, dtype=np.float64)
        for i in range(1, len(poses)):
            T_rel = _relative_pose(poses[i - 1], poses[i])
            _, a = _so3_log(T_rel[:3, :3])
            per_step[i - 1] = np.degrees(a)
        mean_rot_per_frame = float(per_step.mean())
        max_rot_per_frame = float(per_step.max())
    else:
        mean_rot_per_frame = 0.0
        max_rot_per_frame = 0.0

    # Build phrases
    phrases = []
    # Translation magnitude relative to scene extent so we can call it "small" / "large"
    extent = _scene_extent(sm)
    extent_scale = max(extent, 1e-3)
    translation_ratio = float(np.linalg.norm(last) / extent_scale)
    if translation_ratio < 0.05 and rot_deg < 5:
        phrases.append("the camera is approximately static (no panning or translation)")
    else:
        # Rotation direction
        if rot_deg >= 5:
            # Project rotation axis onto camera-y (yaw=pan) and camera-x (pitch=tilt).
            T_full = cam0_inv @ poses[-1]
            axis, _ = _so3_log(T_full[:3, :3])
            yaw = axis[1] * rot_deg   # +y is down (OpenCV), so positive yaw with axis=+y → camera turns LEFT in image
            pitch = axis[0] * rot_deg
            # Pan
            if abs(yaw) >= 5:
                # Convention: axis +y (down) rotation → world rotates left-to-right past camera ⇒ camera pans RIGHT.
                pan = "right" if yaw > 0 else "left"
                phrases.append(f"the camera pans to the {pan} by ~{abs(yaw):.0f}° overall (~{mean_rot_per_frame:.1f}°/frame, peak {max_rot_per_frame:.1f}°/frame)")
            if abs(pitch) >= 5:
                tilt = "down" if pitch > 0 else "up"
                phrases.append(f"the camera tilts {tilt} by ~{abs(pitch):.0f}°")
        if translation_ratio >= 0.05:
            parts = []
            if abs(tx) > 0.05 * extent_scale:
                parts.append(f"~{abs(tx):.2f}m to the {'right' if tx > 0 else 'left'}")
            if abs(tz) > 0.05 * extent_scale:
                parts.append(f"~{abs(tz):.2f}m {'forward' if tz > 0 else 'backward'}")
            if abs(ty) > 0.05 * extent_scale:
                parts.append(f"~{abs(ty):.2f}m {'down' if ty > 0 else 'up'}")
            if parts:
                phrases.append(f"the camera translates {', '.join(parts)} (scene extent ≈ {extent:.2f}m)")

    summary = ". ".join(phrases) + "." if phrases else "Camera motion negligible."
    return {
        "available": True,
        "summary": summary,
        "rot_deg_total": rot_deg,
        "mean_rot_per_frame_deg": mean_rot_per_frame,
        "max_rot_per_frame_deg": max_rot_per_frame,
        "translation_xyz_cam0": [tx, ty, tz],
        "translation_norm": float(np.linalg.norm(last)),
        "translation_ratio_to_extent": translation_ratio,
        "scene_extent_m": extent,
        "num_frames": int(len(poses)),
    }


def _scene_extent(sm: SpatialMemory) -> float:
    pts = np.asarray(sm.position_3d).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return 1.0
    qs = np.quantile(pts, [0.05, 0.95], axis=0)
    return float(np.linalg.norm(qs[1] - qs[0]))


def summarize_scene_extent(sm: SpatialMemory) -> Dict[str, Any]:
    e = _scene_extent(sm)
    return {"scene_extent_m": e, "summary": f"Scene extent ≈ {e:.2f} m (diagonal of 90% point quantile)."}


def render_evidence_block(sm: SpatialMemory, is_egocentric: bool = False) -> str:
    """Compose the final text block that gets injected into the VLM prompt.

    The framing depends on whether the source video is egocentric (camera is on the
    actor's head, so its motion IS the actor's body motion) or allocentric (third-person
    camera, whose motion is a confounder when judging in-scene object motion)."""
    ego = summarize_camera_egomotion(sm)
    lines = [
        "[Dynamic-memory evidence — derived from Flow3r 3D reconstruction]",
        f"  Frames analysed: {ego.get('num_frames', '?')}",
        f"  Camera motion: {ego['summary']}",
    ]
    rot_tot = ego.get("rot_deg_total", 0.0)
    trans_ratio = ego.get("translation_ratio_to_extent", 0.0)
    if is_egocentric:
        if rot_tot >= 8 or trans_ratio >= 0.08:
            lines.append(
                "  ▸ This is egocentric video. The camera is mounted on the actor's head, "
                "so the camera motion ABOVE IS the actor's head/body motion (turning, "
                "tilting, walking)."
            )
        else:
            lines.append(
                "  ▸ This is egocentric video. The camera (actor's head) is essentially still, "
                "so motion in the frames is action by the actor's hands or other entities."
            )
    else:
        if rot_tot >= 5 or trans_ratio >= 0.05:
            lines.append(
                "  ⚠ The camera is moving. Background pixel motion is partly from this camera "
                "motion. When judging object motion, mentally subtract the camera ego-motion."
            )
        else:
            lines.append(
                "  ✓ Camera is essentially static — pixel motion in the video reflects real "
                "motion in the scene, not camera ego-motion."
            )
    return "\n".join(lines)


def evidence_from_cache(video_path: str, *, cache_index_path: str = "./data/flow3r_cache/index.json"
                        ) -> Optional[str]:
    """Look up the cached dynamic memory for a video and produce the prompt evidence block."""
    import json
    if not os.path.exists(cache_index_path):
        return None
    index = json.load(open(cache_index_path))
    entry = index.get(video_path) or index.get(os.path.abspath(video_path))
    if not entry:
        return None
    sm_cache = entry.get("sm_cache")
    if not sm_cache or not os.path.exists(sm_cache):
        return None
    sm = SpatialMemory.load(sm_cache, align_xz_with_pca=False)
    return render_evidence_block(sm)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to a VLM4D video already in the cache")
    args = ap.parse_args()
    txt = evidence_from_cache(args.video)
    print(txt or "(no cache for this video yet)")
