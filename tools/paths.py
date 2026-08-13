"""Central, environment-driven filesystem layout.

Everything machine-specific lives here so the rest of the code carries no
absolute paths. Configure via environment variables (see `.env.example`):

    REASMORY_DATA_ROOT                  benchmark videos / images
    REASMORY_WORKSPACE_ROOT             scratch space for rendered tool outputs
    REASMORY_SPATIAL_MEMORY_CACHE       precomputed static Pi3 reconstructions
    REASMORY_SPATIAL_MEMORY_CACHE_PI3X  precomputed metric Pi3x reconstructions
    REASMORY_DYNAMIC_MEMORY_CACHE       precomputed dynamic Flow3R reconstructions

The released benchmark annotation files still contain the absolute media paths
from the machine they were generated on. Rather than force everyone to rewrite
those JSONs, `resolve_media_path` re-roots any such legacy prefix under
`REASMORY_DATA_ROOT`, so pointing that one variable at your own copy of the data
is enough.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

# Absolute prefixes baked into the released annotation files. A media path that
# starts with one of these is treated as "<prefix>/<rel>" and re-rooted to
# "<REASMORY_DATA_ROOT>/<rel>".
LEGACY_DATA_PREFIXES: tuple[str, ...] = (
    "/ssd3/jxhe_cache",
    "/ssd2/jxhe",
)

# Some annotations reference assets that live inside the repository itself, via an
# absolute path from the machine that generated them. Anything after this marker
# is re-rooted at REPO_ROOT instead of REASMORY_DATA_ROOT.
LEGACY_REPO_MARKER = "/qwen-vl-finetune/"


def _env_path(var: str, default: Path | str) -> Path:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else Path(default)


def data_root() -> Path:
    """Root of the benchmark media (videos, image folders)."""
    return _env_path("REASMORY_DATA_ROOT", REPO_ROOT / "data")


def workspace_root() -> Path:
    """Scratch space for rendered tool artifacts."""
    return _env_path("REASMORY_WORKSPACE_ROOT", REPO_ROOT / "workspace")


def spatial_memory_cache_root(metric: bool = False) -> Path:
    """Precomputed static reconstruction cache (Pi3, or metric Pi3x)."""
    if metric:
        return _env_path(
            "REASMORY_SPATIAL_MEMORY_CACHE_PI3X", REPO_ROOT / "cache" / "spatial_memory_cache_pi3x"
        )
    return _env_path(
        "REASMORY_SPATIAL_MEMORY_CACHE", REPO_ROOT / "cache" / "spatial_memory_cache"
    )


def dynamic_memory_cache_root() -> Path:
    """Precomputed dynamic (Flow3R) reconstruction cache, keyed by video."""
    return _env_path(
        "REASMORY_DYNAMIC_MEMORY_CACHE", REPO_ROOT / "cache" / "flow3r_cache" / "by_video"
    )


def resolve_media_path(path: Any) -> Any:
    """Re-root a legacy absolute media path under `REASMORY_DATA_ROOT`.

    Non-strings, relative paths and paths that already exist are returned
    unchanged, so this is safe to apply blanket-wise to loaded annotations.
    """
    if not isinstance(path, str) or not path:
        return path
    # Repo-relative asset paths: resolve against the repository, not the caller's cwd.
    if not path.startswith("/"):
        candidate = REPO_ROOT / path
        return str(candidate) if candidate.exists() else path
    if os.path.exists(path):
        return path
    for prefix in LEGACY_DATA_PREFIXES:
        if path.startswith(prefix + "/"):
            return str(data_root() / path[len(prefix) + 1:])
    # Assets shipped inside the repository (e.g. the MindCube image subset).
    if path.startswith("/") and LEGACY_REPO_MARKER in path:
        return str(REPO_ROOT / path.split(LEGACY_REPO_MARKER, 1)[1])
    return path


def resolve_media_paths(value: Any) -> Any:
    """Recursively apply `resolve_media_path` to strings inside lists/dicts."""
    if isinstance(value, str):
        return resolve_media_path(value)
    if isinstance(value, list):
        return [resolve_media_paths(v) for v in value]
    if isinstance(value, tuple):
        return tuple(resolve_media_paths(v) for v in value)
    if isinstance(value, dict):
        return {k: resolve_media_paths(v) for k, v in value.items()}
    return value


# Keys in a benchmark sample that point at media on disk.
MEDIA_KEYS: tuple[str, ...] = ("path", "image", "images", "video", "video_path", "image_paths")


def resolve_sample_media(sample: dict, media_keys: Iterable[str] = MEDIA_KEYS) -> dict:
    """Re-root the media fields of one benchmark sample, in place."""
    for key in media_keys:
        if key in sample:
            sample[key] = resolve_media_paths(sample[key])
    return sample
