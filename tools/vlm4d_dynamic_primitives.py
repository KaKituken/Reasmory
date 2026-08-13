"""
Named primitives that turn a Flow3r-backed dynamic SpatialMemory into compact
textual evidence for the VLM reasoner.

Design choice (see Phase-A finalisation in conversation):
    * No LLM planner.  The decomposer's `query_archetype` is sufficient to
      decide which primitives to invoke; we map archetype → list-of-primitives
      with a static dispatch table.
    * Each primitive is a self-contained function `f(sm, is_egocentric) -> str`
      so it can be unit-tested and reused.
    * Phase-B (SAM3-based object trajectory) plugs in as new primitives that
      take an extra `entity: str` argument; the dispatch table just gets a new
      row.

The legacy `render_evidence_block(sm, is_egocentric)` (in
`vlm4d_motion_analysis.py`) remains unchanged — the new path is opt-in via
`compose_evidence_for_archetype`.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from spatial_memory import SpatialMemory                         # noqa: E402
from vlm4d_motion_analysis import (                              # noqa: E402
    summarize_camera_egomotion,
    _scene_extent,
)


# ---------------------------------------------------------------------------
# Helpers for converting world-frame points to camera-frame depth (z forward).
# ---------------------------------------------------------------------------

def _world_to_camera_z(points_world: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Map world-frame points (N, 3) to camera-frame z component."""
    w2c = np.linalg.inv(c2w)
    homo = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    return (w2c @ homo.T).T[:, 2]


def _per_frame_depth_stats(sm: SpatialMemory) -> Dict[str, np.ndarray]:
    """For each frame i, summarise scene depth (camera-frame z) over its 3D map.

    Returns dict with arrays of length N:
        median_z, p10_z, p90_z, center_z
    """
    N, H, W, _ = sm.position_3d.shape
    out_median = np.zeros(N, dtype=np.float32)
    out_p10 = np.zeros(N, dtype=np.float32)
    out_p90 = np.zeros(N, dtype=np.float32)
    out_center = np.zeros(N, dtype=np.float32)
    cu, cv = W // 2, H // 2
    for i in range(N):
        world = sm.position_3d[i].reshape(-1, 3)
        finite = np.isfinite(world).all(axis=1)
        world = world[finite]
        if world.size == 0:
            out_median[i] = np.nan
            out_p10[i] = np.nan
            out_p90[i] = np.nan
            out_center[i] = np.nan
            continue
        z = _world_to_camera_z(world, sm.camera_trajectory[i])
        out_median[i] = np.median(z)
        out_p10[i] = np.percentile(z, 10)
        out_p90[i] = np.percentile(z, 90)
        center = sm.position_3d[i, cv, cu]
        if np.all(np.isfinite(center)):
            out_center[i] = _world_to_camera_z(center[None, :], sm.camera_trajectory[i])[0]
        else:
            # fall back to a small patch around the centre
            patch = sm.position_3d[i, max(cv - 2, 0):cv + 3, max(cu - 2, 0):cu + 3].reshape(-1, 3)
            patch = patch[np.isfinite(patch).all(axis=1)]
            if patch.size == 0:
                out_center[i] = np.nan
            else:
                out_center[i] = float(np.median(_world_to_camera_z(patch, sm.camera_trajectory[i])))
    return {"median_z": out_median, "p10_z": out_p10, "p90_z": out_p90, "center_z": out_center}


# ---------------------------------------------------------------------------
# Primitive: camera ego-motion summary
#   (thin wrapper around the legacy function so it lives in the same registry)
# ---------------------------------------------------------------------------

def primitive_camera_egomotion(sm: SpatialMemory, *, is_egocentric: bool = False) -> str:
    ego = summarize_camera_egomotion(sm)
    rot_tot = ego.get("rot_deg_total", 0.0)
    trans_ratio = ego.get("translation_ratio_to_extent", 0.0)
    lines = [
        "[Camera ego-motion]",
        f"  Frames analysed: {ego.get('num_frames', '?')}",
        f"  Summary: {ego['summary']}",
    ]
    if is_egocentric:
        if rot_tot >= 8 or trans_ratio >= 0.08:
            lines.append(
                "  ▸ Egocentric camera = actor's head. The motion above IS the actor's "
                "head/body motion (turning, tilting, walking)."
            )
        else:
            lines.append(
                "  ▸ Egocentric camera (actor's head) is essentially still — pixel motion is action by the hands or other entities."
            )
    else:
        if rot_tot >= 5 or trans_ratio >= 0.05:
            lines.append(
                "  ⚠ The camera itself is moving. Background pixel motion is partly from this. "
                "When judging object motion, mentally subtract camera ego-motion."
            )
        else:
            lines.append(
                "  ✓ Camera is essentially static — pixel motion reflects real scene motion."
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primitive: scene-level depth trajectory
# ---------------------------------------------------------------------------

def _describe_trend(delta_abs: float, threshold_close: float, threshold_strong: float) -> str:
    """Map a |delta| → qualitative word."""
    if abs(delta_abs) < threshold_close:
        return "essentially unchanged"
    if delta_abs > 0:
        return "increasing strongly" if delta_abs > threshold_strong else "increasing slightly"
    return "decreasing strongly" if -delta_abs > threshold_strong else "decreasing slightly"


def primitive_scene_depth_trajectory(sm: SpatialMemory, *, is_egocentric: bool = False) -> str:
    stats = _per_frame_depth_stats(sm)
    med = stats["median_z"]
    p10 = stats["p10_z"]
    N = len(med)
    if N < 2:
        return "[Scene depth trajectory]\n  Not enough frames to compute trajectory."
    first, last = med[0], med[-1]
    delta = float(last - first)
    extent = _scene_extent(sm)
    close_thr = max(0.05, 0.03 * extent)
    strong_thr = max(0.30, 0.15 * extent)
    trend = _describe_trend(delta, close_thr, strong_thr)
    # Use 5-sample summary so reasoner sees the shape
    sample_idx = [0, N // 4, N // 2, (3 * N) // 4, N - 1]
    sample_idx = sorted(set(i for i in sample_idx if 0 <= i < N))
    sample = "  ".join(f"f{idx}:{med[idx]:.2f}m" for idx in sample_idx)
    near = float(np.nanmin(p10))
    far = float(np.nanmax(stats["p90_z"]))
    lines = [
        "[Scene depth trajectory  (camera-frame z, median over scene)]",
        f"  Per-frame median depth: {sample}",
        f"  Net Δ over the video: {delta:+.2f} m → scene is {trend}.",
        f"  Depth range (near 10%, far 90% percentiles): {near:.2f} m → {far:.2f} m.",
        f"  Threshold for 'essentially unchanged': ±{close_thr:.2f} m (3% of scene extent {extent:.2f} m).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primitive: center-pixel depth trajectory (proxy for centered subject)
# ---------------------------------------------------------------------------

def primitive_center_pixel_depth(sm: SpatialMemory, *, is_egocentric: bool = False) -> str:
    stats = _per_frame_depth_stats(sm)
    cz = stats["center_z"]
    N = len(cz)
    if N < 2 or not np.isfinite(cz).any():
        return "[Center-pixel depth trajectory]\n  Centre depth unavailable."
    first = float(np.nan)
    last = float(np.nan)
    # robust first / last: first valid forwards, last valid backwards
    for v in cz:
        if np.isfinite(v):
            first = float(v); break
    for v in cz[::-1]:
        if np.isfinite(v):
            last = float(v); break
    delta = last - first
    extent = _scene_extent(sm)
    close_thr = max(0.05, 0.03 * extent)
    strong_thr = max(0.30, 0.15 * extent)
    trend = _describe_trend(delta, close_thr, strong_thr)
    sample_idx = [0, N // 4, N // 2, (3 * N) // 4, N - 1]
    sample_idx = sorted(set(i for i in sample_idx if 0 <= i < N))
    sample = "  ".join(
        f"f{idx}:{cz[idx]:.2f}m" if np.isfinite(cz[idx]) else f"f{idx}:--" for idx in sample_idx
    )
    lines = [
        "[Centre-pixel depth trajectory  (camera-frame z at image centre)]",
        "  Useful when the question is about a centred subject (e.g. 'the dog on the scale',",
        "  'the bus', the person currently being filmed).",
        f"  Per-frame centre depth: {sample}",
        f"  Net Δ over the video: {delta:+.2f} m → centred content is {trend}.",
        f"    interpretation: positive Δ (further) ⇒ subject moved away or camera moved backward;",
        f"                    negative Δ (closer)  ⇒ subject moved toward camera or camera moved forward.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry + archetype dispatch
# ---------------------------------------------------------------------------

def _try_import_object_primitives():
    try:
        from vlm4d_object_primitives import (
            primitive_object_trajectory,
            primitive_object_motion_in_camera_frame,
            primitive_object_depth,
            primitive_object_motion_in_own_frame,
        )
        return {
            "object_trajectory":              primitive_object_trajectory,
            "object_motion_in_camera_frame":  primitive_object_motion_in_camera_frame,
            "object_depth":                   primitive_object_depth,
            "object_motion_in_own_frame":     primitive_object_motion_in_own_frame,
        }
    except Exception as e:                    # pragma: no cover
        return {}


PRIMITIVES: Dict[str, Callable[..., str]] = {
    "camera_egomotion":        primitive_camera_egomotion,
    "scene_depth_trajectory":  primitive_scene_depth_trajectory,
    "center_pixel_depth":      primitive_center_pixel_depth,
    **_try_import_object_primitives(),
}


# Per (archetype, ref_frame_group) dispatch.  Falls back to (archetype, '*').
# Lookup is `_lookup_primitives(archetype, ref_group)`.
PRIMITIVES_BY_ARCH_REF: Dict[tuple, List[str]] = {
    ("DIRECTION", "camera"):  ["camera_egomotion", "object_motion_in_camera_frame"],
    ("DIRECTION", "object"):  ["camera_egomotion", "object_trajectory",
                               "object_motion_in_camera_frame", "object_motion_in_own_frame"],
    ("DIRECTION", "world"):   ["camera_egomotion", "object_trajectory"],
    ("DEPTH_CHANGE", "camera"): ["camera_egomotion", "scene_depth_trajectory",
                                  "center_pixel_depth", "object_depth"],
    ("DEPTH_CHANGE", "object"): ["camera_egomotion", "object_depth", "object_motion_in_own_frame"],
    ("ROTATION_SENSE", "*"):  ["camera_egomotion"],
    ("COUNT_EVENTS", "*"):    ["camera_egomotion"],
    ("PRESENCE_CHECK", "*"):  [],
    ("HAND_OR_SIDE", "*"):    [],
    ("*", "*"):               ["camera_egomotion"],
}


ARCHETYPE_TO_PRIMITIVES: Dict[str, List[str]] = {
    "DIRECTION":      ["camera_egomotion"],
    "DEPTH_CHANGE":   ["camera_egomotion", "scene_depth_trajectory", "center_pixel_depth"],
    "ROTATION_SENSE": ["camera_egomotion"],
    "COUNT_EVENTS":   ["camera_egomotion"],
    "PRESENCE_CHECK": [],
    "HAND_OR_SIDE":   [],
    "UNKNOWN":        ["camera_egomotion"],  # safe default
}


def _ref_group_of(ref: Optional[str]) -> str:
    if not isinstance(ref, str): return "UNK"
    if ref.startswith("object"): return "object"
    if ref in ("camera", "world"): return ref
    return "UNK"


def _lookup_primitives(archetype: str, ref_group: str) -> List[str]:
    a = (archetype or "UNKNOWN").upper()
    rg = ref_group or "UNK"
    return (PRIMITIVES_BY_ARCH_REF.get((a, rg))
            or PRIMITIVES_BY_ARCH_REF.get((a, "*"))
            or PRIMITIVES_BY_ARCH_REF.get(("*", "*"))
            or [])


def compose_evidence_for_archetype(
    sm: SpatialMemory,
    archetype: Optional[str],
    *,
    is_egocentric: bool = False,
    entity: Optional[str] = None,
    ref_frame: Optional[str] = None,
    legacy_archetype_only: bool = False,
) -> str:
    """Run the primitives chosen for `archetype` (and `ref_frame` when given)
    and concatenate their text. With `entity` provided, object-level primitives
    are wired in for the relevant cells."""
    arch = (archetype or "UNKNOWN").upper()
    if legacy_archetype_only:
        names = ARCHETYPE_TO_PRIMITIVES.get(arch, ARCHETYPE_TO_PRIMITIVES["UNKNOWN"])
    else:
        names = _lookup_primitives(arch, _ref_group_of(ref_frame))
    if not names:
        return "[Dynamic memory evidence — no archetype-specific primitives invoked.]"
    parts: List[str] = ["[Dynamic-memory evidence — archetype="
                        f"{arch}, ref_frame={ref_frame!r} ⇒ "
                        f"primitives: {', '.join(names)}]"]
    for name in names:
        fn = PRIMITIVES.get(name)
        if fn is None:
            parts.append(f"[Primitive '{name}' not registered; skipped]")
            continue
        try:
            # Object-level primitives accept an `entity` kwarg; scene-level ones don't.
            if name.startswith("object_"):
                parts.append(fn(sm, is_egocentric=is_egocentric, entity=entity))
            else:
                parts.append(fn(sm, is_egocentric=is_egocentric))
        except Exception as e:
            parts.append(f"[Primitive '{name}' failed: {e}]")
    return "\n\n".join(parts)
