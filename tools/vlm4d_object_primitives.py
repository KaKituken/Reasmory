"""
Phase-B primitives — SAM3-backed object-level evidence.

These run a single SAM3 text query against a few keyframes of the cached
SpatialMemory, backproject the masks to 3D via Flow3r's per-pixel position
map, and summarise the object's trajectory / camera-frame displacement /
depth-change in natural text.

The primitives degrade gracefully: if SAM3 finds nothing for the queried
entity, the primitive returns a short note saying so, which is fine for
the reasoner (it will fall back to visual analysis).
"""
from __future__ import annotations

import gc
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from spatial_memory import SpatialMemory          # noqa: E402
from vlm4d_motion_analysis import _scene_extent  # noqa: E402

# --- SAM3 imports happen lazily, only when a primitive is actually invoked.
# (importing them triggers heavy module init).
_RUNTIME = None
_BATCH_CHUNK = int(os.environ.get("SAM3_CHUNK_SIZE", "2"))

# --- Dynamic single-target temporal merge config ---
# Adapted from the static multi-view voxel merge (Query3DObjectPosition.aggregate_3d_positions):
# static links per-frame detections of ONE physical instance by SPATIAL voxel co-occupancy across
# all views. For a MOVING object that fails, so we instead link adjacent-frame detections by
# TEMPORAL voxel IoU and pick the most temporally-coherent chain (single target -> a DP, not a graph).
_TEMPORAL_MERGE = os.environ.get("VLM4D_TEMPORAL_MERGE", "1") == "1"
_TOPK = int(os.environ.get("VLM4D_TOPK", "3"))               # SAM3 candidates kept per keyframe
_VOXEL_EPS = float(os.environ.get("VLM4D_VOXEL_EPS", "0.10"))  # coarse voxel so adjacent frames of a moving obj overlap
_W_TEMPORAL = float(os.environ.get("VLM4D_W_TEMPORAL", "3.0"))  # weight of temporal-IoU vs SAM3 score in the DP
# Keyframes sampled from the cached memory for object queries. Flow3r caches hold ~5-32 frames
# (median 15) for the short VLM4D clips, so 16 uses ~all frames for most videos (was 5 -> too coarse
# for fast motion). Cost is ~linear in SAM3 queries. `_pick_keyframes` caps at the available N.
_MAX_KEYFRAMES = int(os.environ.get("VLM4D_MAX_KEYFRAMES", "16"))


def _ensure_sam3():
    """Lazy-load the shared agent_tools runtime + SAM3."""
    global _RUNTIME
    if _RUNTIME is None:
        from agent_tools import runtime as rt  # noqa
        _RUNTIME = rt
    _RUNTIME.ensure_sam3()
    return _RUNTIME


def _pick_keyframes(N: int, max_kf: int) -> List[int]:
    if N <= max_kf:
        return list(range(N))
    return [int(round(i * (N - 1) / (max_kf - 1))) for i in range(max_kf)]


def _segment_entity_in_keyframes_multi(
    sm: SpatialMemory,
    entity_text: str,
    frame_indices: List[int],
    topk: int = 1,
) -> List[List[Tuple[np.ndarray, float]]]:
    """For each requested frame, return up to `topk` (bool_mask, score) candidates
    of the entity at memory resolution (matching position_3d). Empty list if none.

    Uses the SAM3 text-query path borrowed from Query3DObjectPosition's
    `collect_sam3_text_query_instances`. One text per frame, so per-call cost is small.
    """
    rt = _ensure_sam3()
    transform = rt.sam3_transform
    postproc = rt.sam3_postprocessor
    sam3 = rt.sam3_model

    # Import the SAM3 helpers from agent_tools (the classes already wrap them tidily)
    from agent_tools import Query3DObjectPosition  # noqa
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device

    pil_images = sm.rgb_images_pil
    targets = [pil_images[i] for i in frame_indices]

    datapoints, prompt_ids = [], []
    for img in targets:
        dp = Query3DObjectPosition.create_empty_datapoint()
        Query3DObjectPosition.set_image(dp, img)
        prompt_ids.append(Query3DObjectPosition.add_text_prompt(dp, entity_text))
        datapoints.append(transform(dp))

    processed = {}
    chunk = max(1, _BATCH_CHUNK)
    for s in range(0, len(datapoints), chunk):
        batch = collate(datapoints[s:s + chunk], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device(sam3.device), non_blocking=True)
        with torch.no_grad():
            out = sam3(batch)
        processed.update(postproc.process_results(out, batch.find_metadatas))
        del batch, out
        gc.collect()
        torch.cuda.empty_cache()

    mem_h, mem_w = sm.memory_3d_map_size
    out_all: List[List[Tuple[np.ndarray, float]]] = []
    import torchvision.transforms.functional as TF
    for pid in prompt_ids:
        result = processed.get(pid, {})
        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None or masks.numel() == 0 or scores is None or scores.numel() == 0:
            out_all.append([])
            continue
        # keep the top-k instances by score for this query
        k = min(int(topk), int(scores.numel()))
        top_idx = torch.topk(scores, k).indices.tolist()
        cands: List[Tuple[np.ndarray, float]] = []
        for ti in top_idx:
            mask = masks[ti]
            if mask.ndim == 3:
                mask = mask[0]
            mask_mem = TF.resize(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(mem_h, mem_w),
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0).squeeze(0).bool().cpu().numpy()
            if mask_mem.sum() < 30:           # very small / spurious
                continue
            cands.append((mask_mem, float(scores[ti].item())))
        out_all.append(cands)
    return out_all


def _segment_entity_in_keyframes(
    sm: SpatialMemory,
    entity_text: str,
    frame_indices: List[int],
) -> List[Optional[np.ndarray]]:
    """Backward-compatible top-1 wrapper: one bool mask per frame (or None)."""
    multi = _segment_entity_in_keyframes_multi(sm, entity_text, frame_indices, topk=1)
    return [(cl[0][0] if cl else None) for cl in multi]


def _centroid_per_frame(
    sm: SpatialMemory,
    masks: List[Optional[np.ndarray]],
    frame_indices: List[int],
) -> List[Optional[np.ndarray]]:
    """For each frame, average the world-frame 3D points inside the mask."""
    centroids: List[Optional[np.ndarray]] = []
    for fi, mask in zip(frame_indices, masks):
        if mask is None:
            centroids.append(None); continue
        pts = sm.position_3d[fi][mask]               # (K, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            centroids.append(None); continue
        centroids.append(np.mean(pts, axis=0))
    return centroids


# Tiny per-process cache so that multiple object primitives invoked on the same
# (sm, entity, idxs) only segment the video once.  Bounded; OK to never evict.
_OBJ_CACHE: dict = {}


def _voxel_hashset(propagator, pts: np.ndarray) -> frozenset:
    """Voxelize 3D points and return the set of occupied voxel hashes.
    Reuses the SAME voxel machinery as the static merge (VoxelPropagator)."""
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return frozenset()
    vox = propagator.voxelize(pts)
    return frozenset(propagator.hash_voxels(vox).astype(np.int64).tolist())


def _temporal_iou(a: frozenset, b: frozenset) -> float:
    """Temporal IoU = Jaccard of two ADJACENT-frame candidates' voxel sets.
    (Static merge uses spatial multi-view co-occupancy; here we compare frame t vs t+1.)"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def _build_frame_candidates(
    sm: SpatialMemory,
    entity: str,
    frame_indices: List[int],
    topk: int,
    voxel_eps: float,
) -> List[List[dict]]:
    """Per keyframe: list of candidate dicts {score, voxset, centroid} for the entity."""
    from agent_tools import Query3DObjectPosition  # noqa

    cand_masks = _segment_entity_in_keyframes_multi(sm, entity, frame_indices, topk=topk)
    propagator = Query3DObjectPosition.build_voxel_propagator(sm, eps=voxel_eps)
    frames: List[List[dict]] = []
    for fi, cl in zip(frame_indices, cand_masks):
        items = []
        for mask, score in cl:
            pts = sm.position_3d[fi][mask]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) == 0:
                continue
            items.append({
                "score": float(score),
                "voxset": _voxel_hashset(propagator, pts),
                "centroid": np.mean(pts, axis=0),
            })
        frames.append(items)
    return frames


def merge_targets_temporal(
    sm: SpatialMemory,
    entity: str,
    frame_indices: List[int],
    topk: int = _TOPK,
    voxel_eps: float = _VOXEL_EPS,
    w_temporal: float = _W_TEMPORAL,
    max_objects: int = 1,
    min_mean_score: float = 0.3,
) -> Tuple[List[int], List[List[Optional[np.ndarray]]]]:
    """Multi-target dynamic instance merge = several temporally-coherent chains.

    Adapts the static multi-view voxel merge: keep top-k SAM3 candidates per keyframe,
    link ADJACENT frames by temporal voxel IoU, and extract the best chain via DP
    (score on nodes + w * IoU on edges). To support multiple objects of the same
    label we GREEDILY extract chains one at a time, removing each chain's chosen
    candidates before the next pass — the temporal analogue of the static merge's
    per-frame mutual-exclusion constraint. Returns (frame_indices, tracks) where each
    track is a per-frame list of centroids (None where absent).
    """
    frames = _build_frame_candidates(sm, entity, frame_indices, topk, voxel_eps)
    remaining = [list(items) for items in frames]  # mutable copies
    tracks: List[List[Optional[np.ndarray]]] = []

    for _ in range(max(1, max_objects)):
        valid, path = _dp_best_chain(remaining, w_temporal)
        if not valid:
            break
        chosen = [(k, path[oi]) for oi, k in enumerate(valid)]
        mean_score = float(np.mean([remaining[k][ci]["score"] for k, ci in chosen]))
        if mean_score < min_mean_score:
            break  # leftover candidates are just noise -> stop adding chains
        track: List[Optional[np.ndarray]] = [None] * len(frame_indices)
        for k, ci in chosen:
            track[k] = remaining[k][ci]["centroid"]
        tracks.append(track)
        # remove used candidates (per-frame mutual exclusion) so the next chain is disjoint
        for k, ci in sorted(chosen, key=lambda x: -x[1]):
            del remaining[k][ci]

    return frame_indices, tracks


def merge_single_target_temporal(
    sm: SpatialMemory,
    entity: str,
    frame_indices: List[int],
    topk: int = _TOPK,
    voxel_eps: float = _VOXEL_EPS,
    w_temporal: float = _W_TEMPORAL,
) -> Tuple[List[int], List[Optional[np.ndarray]]]:
    """Single-target convenience wrapper: the single best temporally-coherent chain."""
    fi, tracks = merge_targets_temporal(
        sm, entity, frame_indices, topk=topk, voxel_eps=voxel_eps,
        w_temporal=w_temporal, max_objects=1, min_mean_score=0.0,
    )
    return fi, (tracks[0] if tracks else [None] * len(frame_indices))


def _dp_best_chain(frames: List[List[dict]], w_temporal: float) -> Tuple[List[int], List[int]]:
    """Pick the single most temporally-coherent chain across frames.

    frames[k] = list of candidate dicts with keys 'score' (float) and 'voxset' (frozenset).
    Returns (valid_frame_positions, chosen_candidate_index_per_valid_frame).
    Reward maximized = sum(node score) + w_temporal * sum(temporal IoU on consecutive edges).
    """
    valid = [k for k, items in enumerate(frames) if items]
    if not valid:
        return [], []
    dp: List[List[float]] = []
    bp: List[List[int]] = []
    for oi, k in enumerate(valid):
        items = frames[k]
        dp.append([0.0] * len(items))
        bp.append([-1] * len(items))
        for ci, it in enumerate(items):
            if oi == 0:
                dp[oi][ci] = it["score"]
            else:
                best, best_p = -1e9, -1
                for pci, pit in enumerate(frames[valid[oi - 1]]):
                    val = dp[oi - 1][pci] + w_temporal * _temporal_iou(pit["voxset"], it["voxset"])
                    if val > best:
                        best, best_p = val, pci
                dp[oi][ci] = it["score"] + best
                bp[oi][ci] = best_p
    ci = int(np.argmax(dp[-1]))
    path = [0] * len(valid)
    for oi in range(len(valid) - 1, -1, -1):
        path[oi] = ci
        nxt = bp[oi][ci]
        ci = nxt if nxt >= 0 else (int(np.argmax(dp[oi - 1])) if oi > 0 else 0)
    return valid, path


def get_or_compute_object_centroids(
    sm: SpatialMemory,
    entity: str,
    frame_indices: List[int],
) -> Tuple[List[int], List[Optional[np.ndarray]]]:
    """Returns (frame_indices, centroids) — using cached SAM3 segmentation
    when called repeatedly with the same args.

    When temporal merge is enabled (default), the single moving target's per-frame
    centroid is chosen by the temporally-coherent chain (fixes top-1 identity drift
    when several same-label objects exist). Falls back to plain top-1 otherwise.
    """
    key = (id(sm), entity, tuple(frame_indices))
    if key in _OBJ_CACHE:
        return _OBJ_CACHE[key]
    if _TEMPORAL_MERGE and _TOPK > 1:
        try:
            res = merge_single_target_temporal(sm, entity, frame_indices)
        except Exception:
            masks = _segment_entity_in_keyframes(sm, entity, frame_indices)
            res = (frame_indices, _centroid_per_frame(sm, masks, frame_indices))
    else:
        masks = _segment_entity_in_keyframes(sm, entity, frame_indices)
        res = (frame_indices, _centroid_per_frame(sm, masks, frame_indices))
    _OBJ_CACHE[key] = res
    # crude size limit
    if len(_OBJ_CACHE) > 256:
        _OBJ_CACHE.pop(next(iter(_OBJ_CACHE)))
    return _OBJ_CACHE[key]


def _world_to_camera(point_w: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    homo = np.concatenate([point_w, [1.0]])
    return (np.linalg.inv(c2w) @ homo)[:3]


def _format_keyframe_summary(label: str, idxs: List[int], values: List[Optional[float]]) -> str:
    parts = []
    for idx, v in zip(idxs, values):
        parts.append(f"f{idx}:{v:.2f}m" if (v is not None and np.isfinite(v)) else f"f{idx}:--")
    return f"  {label}: " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Primitive: object trajectory (the core piece — used by the others below)
# ---------------------------------------------------------------------------

def primitive_object_trajectory(
    sm: SpatialMemory,
    *,
    entity: Optional[str] = None,
    is_egocentric: bool = False,
    max_keyframes: int = _MAX_KEYFRAMES,
) -> str:
    """Segment the queried entity in a few keyframes and report its 3D centroid
    trajectory in BOTH world and the FIRST camera's frame."""
    if not entity:
        return "[Object trajectory]\n  (no moving_entity in decomposition; primitive skipped)"
    N = len(sm.rgb_images)
    if N < 2:
        return "[Object trajectory]\n  (need ≥ 2 frames)"
    idxs = _pick_keyframes(N, max_keyframes)
    try:
        idxs, centroids = get_or_compute_object_centroids(sm, entity, idxs)
    except Exception as e:
        return f"[Object trajectory]\n  SAM3 failed: {e}"
    valid = [(i, c) for i, c in zip(idxs, centroids) if c is not None]
    if len(valid) < 2:
        return f"[Object trajectory]\n  SAM3 found '{entity}' in {len(valid)} of {len(idxs)} keyframes — not enough to derive a trajectory."

    # World-frame summary
    first_idx, first_c = valid[0]
    last_idx, last_c = valid[-1]
    disp_world = last_c - first_c
    norm_w = float(np.linalg.norm(disp_world))

    # Camera-frame: project displacement using the FIRST camera pose where entity was visible
    c2w_first = sm.camera_trajectory[first_idx]
    R0 = c2w_first[:3, :3]
    # The world-displacement expressed in cam_first's basis is R0.T @ disp_world.
    disp_cam = R0.T @ disp_world

    extent = _scene_extent(sm)
    eps = max(0.05, 0.03 * extent)

    lines = [
        f"[Object trajectory  — entity='{entity}']",
        f"  SAM3 segmented the entity in {len(valid)}/{len(idxs)} keyframes "
        f"({', '.join('f' + str(i) for i,_ in valid)}).",
        f"  World-frame net displacement (first→last visible frame): "
        f"{disp_world[0]:+.2f}, {disp_world[1]:+.2f}, {disp_world[2]:+.2f} m  (norm={norm_w:.2f} m).",
        f"  Same displacement expressed in the FIRST camera's frame "
        f"(x=right, y=down, z=forward):",
        f"    Δx={disp_cam[0]:+.2f} m   Δy={disp_cam[1]:+.2f} m   Δz={disp_cam[2]:+.2f} m",
        f"  Threshold for 'no motion' on each axis: ±{eps:.2f} m (3% of scene extent {extent:.2f} m).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primitive: object motion classification in CAMERA frame
# ---------------------------------------------------------------------------

def primitive_object_motion_in_camera_frame(
    sm: SpatialMemory,
    *,
    entity: Optional[str] = None,
    is_egocentric: bool = False,
    max_keyframes: int = _MAX_KEYFRAMES,
) -> str:
    if not entity:
        return "[Object motion in camera frame]\n  (no entity)"
    N = len(sm.rgb_images)
    idxs = _pick_keyframes(N, max_keyframes)
    try:
        idxs, centroids = get_or_compute_object_centroids(sm, entity, idxs)
    except Exception as e:
        return f"[Object motion in camera frame]\n  SAM3 failed: {e}"
    valid = [(i, c) for i, c in zip(idxs, centroids) if c is not None]
    if len(valid) < 2:
        return f"[Object motion in camera frame]\n  '{entity}' not reliably segmented (only {len(valid)} frames)."
    first_idx, first_c = valid[0]
    last_idx, last_c = valid[-1]
    R0 = sm.camera_trajectory[first_idx][:3, :3]
    disp_cam = R0.T @ (last_c - first_c)
    dx, dy, dz = float(disp_cam[0]), float(disp_cam[1]), float(disp_cam[2])
    extent = _scene_extent(sm)
    eps = max(0.05, 0.03 * extent)

    def _axis_word(v: float, pos_label: str, neg_label: str) -> str:
        if abs(v) < eps: return "no_change"
        return pos_label if v > 0 else neg_label

    x_word = _axis_word(dx, "right", "left")
    y_word = _axis_word(dy, "down", "up")
    z_word = _axis_word(dz, "away_from_camera", "toward_camera")
    lines = [
        f"[Object motion in CAMERA frame — '{entity}']",
        f"  Δx={dx:+.2f}m → {x_word} ;  Δy={dy:+.2f}m → {y_word} ;  Δz={dz:+.2f}m → {z_word}",
        f"  (threshold for 'no_change' per axis: ±{eps:.2f}m)",
        f"  ⇒ The dominant axis (by magnitude) is: "
        f"{'x (left/right)' if abs(dx)>=max(abs(dy),abs(dz)) else 'y (up/down)' if abs(dy)>=abs(dz) else 'z (toward/away)'}.",
        "  These signs are CAMERA-relative ; for screen-space directions "
        "(\"left\"/\"right\" in the question), positive Δx = camera-right.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primitive: object depth (towards/away) trajectory
# ---------------------------------------------------------------------------

def primitive_object_depth(
    sm: SpatialMemory,
    *,
    entity: Optional[str] = None,
    is_egocentric: bool = False,
    max_keyframes: int = _MAX_KEYFRAMES,
) -> str:
    if not entity:
        return "[Object depth trajectory]\n  (no entity)"
    N = len(sm.rgb_images)
    idxs = _pick_keyframes(N, max_keyframes)
    try:
        idxs, centroids = get_or_compute_object_centroids(sm, entity, idxs)
    except Exception as e:
        return f"[Object depth trajectory]\n  SAM3 failed: {e}"
    valid = [(i, c) for i, c in zip(idxs, centroids) if c is not None]
    if len(valid) < 2:
        return f"[Object depth trajectory]\n  '{entity}' not reliably segmented."
    # camera-frame z of centroid at each visible frame, using THAT frame's camera pose
    zs = []
    for i, c in valid:
        zs.append(float(_world_to_camera(c, sm.camera_trajectory[i])[2]))
    first_z, last_z = zs[0], zs[-1]
    delta = last_z - first_z
    extent = _scene_extent(sm)
    eps = max(0.05, 0.03 * extent)
    if abs(delta) < eps:
        verdict = "essentially the same distance from the camera (no significant depth change)"
    elif delta > 0:
        verdict = "moving AWAY from the camera"
    else:
        verdict = "moving TOWARD the camera"
    sample = "  ".join(f"f{idx}:{z:.2f}m" for (idx, _), z in zip(valid, zs))
    lines = [
        f"[Object depth trajectory  — '{entity}'  (z in camera frame; z=0 at camera, z>0 in front)]",
        f"  Per-keyframe depth: {sample}",
        f"  Δz from first→last visible frame: {delta:+.2f} m  → the entity is {verdict}.",
        f"  Threshold for 'no significant change': ±{eps:.2f} m.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primitive: object motion in its OWN frame
#   forward-axis estimated by average displacement direction; works best for
#   things that translate without rotating much
# ---------------------------------------------------------------------------

def primitive_object_motion_in_own_frame(
    sm: SpatialMemory,
    *,
    entity: Optional[str] = None,
    is_egocentric: bool = False,
    max_keyframes: int = _MAX_KEYFRAMES,
) -> str:
    if not entity:
        return "[Object motion in own frame]\n  (no entity)"
    N = len(sm.rgb_images)
    idxs = _pick_keyframes(N, max_keyframes)
    try:
        idxs, centroids = get_or_compute_object_centroids(sm, entity, idxs)
    except Exception as e:
        return f"[Object motion in own frame]\n  SAM3 failed: {e}"
    valid = [(i, c) for i, c in zip(idxs, centroids) if c is not None]
    if len(valid) < 3:
        return f"[Object motion in own frame]\n  '{entity}' not segmented in enough frames."
    pts = np.stack([c for _, c in valid], axis=0)
    disps = np.diff(pts, axis=0)
    norms = np.linalg.norm(disps, axis=1) + 1e-9
    avg_dir = (disps / norms[:, None]).mean(axis=0)
    if np.linalg.norm(avg_dir) < 1e-3:
        return f"[Object motion in own frame]\n  '{entity}' net motion is nearly zero — own-frame direction is undefined."
    forward = avg_dir / np.linalg.norm(avg_dir)
    # Assume world-up is the SpatialMemory's global_up (probably -y after two_stage_up).
    up = np.asarray(sm.global_up) if sm.global_up is not None else np.array([0., -1., 0.])
    up = up / (np.linalg.norm(up) + 1e-9)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-3:
        # forward parallel to up; fall back to arbitrary perpendicular
        tmp = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, tmp)
    right = right / (np.linalg.norm(right) + 1e-9)
    up2 = np.cross(right, forward)
    # net displacement in own basis
    disp_world = pts[-1] - pts[0]
    own = np.array([float(np.dot(disp_world, forward)),
                    float(np.dot(disp_world, right)),
                    float(np.dot(disp_world, up2))])
    extent = _scene_extent(sm)
    eps = max(0.05, 0.03 * extent)

    def _word(v: float, pos: str, neg: str) -> str:
        if abs(v) < eps: return "no_change"
        return pos if v > 0 else neg

    lines = [
        f"[Object motion in OWN frame  — '{entity}']",
        f"  forward axis estimated from the object's average velocity direction.",
        f"  ΔFwd ={own[0]:+.2f}m → {_word(own[0],'forward','backward')}",
        f"  ΔRight={own[1]:+.2f}m → {_word(own[1],'its right','its left')}",
        f"  ΔUp  ={own[2]:+.2f}m → {_word(own[2],'up','down')}",
        f"  (threshold per axis: ±{eps:.2f}m)",
    ]
    return "\n".join(lines)
