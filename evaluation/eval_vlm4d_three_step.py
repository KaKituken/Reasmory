"""
VLM4D dynamic-branch end-to-end evaluator.

Three-stage system that mirrors the static-scene pipeline structure
(decomposer → executor → reasoner) but uses the Flow3r-backed dynamic
SpatialMemory cache as the source of geometric evidence.

Per VLM4D item:
  Stage 1 (decomposer) : Gemini extracts a JSON {moving_entity, reference_frame,
                         query_archetype, temporal_scope, counting_constraint,
                         expected_answer_shape}.
  Stage 2 (executor)   : load cached dynamic SpatialMemory; compute camera
                         ego-motion + scene-extent evidence; render anything
                         else useful for the archetype (future work).
  Stage 3 (reasoner)   : Gemini answers the original multi-choice question
                         using frames + decomposition JSON + executor evidence.

The pipeline is additive: it lives in a new evaluator file, leaves the static
decomposer/planner/executor/reasoner code untouched, and is invoked through a
dedicated script (`05_28_eval_end2end_vlm4d.sh`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sibling modules importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from tqdm import tqdm

from evaluation.utils import clean_text, extract_video_frames, reward_fn
from tools.llm_cfg import build_llm_cfg
from tools.paths import resolve_sample_media, dynamic_memory_cache_root
from tools.spatial_memory import SpatialMemory
from tools.vlm4d_motion_analysis import render_evidence_block
from tools.vlm4d_dynamic_primitives import compose_evidence_for_archetype
from tools.vlm4d_prompt import (
    VLM4D_DECOMPOSER_SYSTEM,
    VLM4D_DECOMPOSER_PROMPT,
    VLM4D_REASONER_SYSTEM,
    build_vlm4d_reasoner_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_benchmark(name: str) -> List[Dict[str, Any]]:
    path = REPO_ROOT / "evaluation" / "annotation" / f"eval_{name}.json"
    with open(path) as f:
        samples = json.load(f)
    # Re-root the shipped absolute media paths under REASMORY_DATA_ROOT.
    for sample in samples:
        if isinstance(sample, dict):
            resolve_sample_media(sample)
    return samples


def _format_options(options: List[str]) -> str:
    return "\n".join(options) if options else ""


def _extract_text(response: Any) -> str:
    """qwen_agent.Assistant.run_nonstream returns a list of Message-like objects.
    Pull the last assistant message's text content out robustly."""
    if not isinstance(response, list) or not response:
        return str(response)
    last = response[-1]
    # qwen_agent Message has .content; dict path is for vanilla dicts
    content = last.content if hasattr(last, "content") else (last.get("content", "") if isinstance(last, dict) else "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # content can be a list of {text:...} or {image:...} items
        parts: List[str] = []
        for c in content:
            t = c.get("text") if isinstance(c, dict) else None
            if t is None and hasattr(c, "text"):
                t = c.text
            if t:
                parts.append(t)
        return "\n".join(parts)
    return str(content)


def _parse_decomposition_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    s = raw.strip()
    # strip markdown fences if any
    if s.startswith("```"):
        s = s.strip("`")
        # remove leading "json\n"
        if s.startswith("json"):
            s = s[len("json"):]
    # find first { and matching }
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end < start:
        return None, "no JSON object found"
    try:
        obj = json.loads(s[start:end + 1])
        return obj, None
    except Exception as e:
        return None, f"json parse error: {e}"


def _resolve_dynamic_cache(video_path: str, cache_root: str) -> Optional[str]:
    """Mirror static memory's resolve pattern: <root>/<video_stem>/spatial_memory.npz."""
    stem = Path(video_path).stem
    p = Path(cache_root) / stem / "spatial_memory.npz"
    return str(p) if p.exists() else None


def _build_executor_evidence(item: Dict[str, Any], cache_root: str,
                             archetype: Optional[str] = None,
                             entity: Optional[str] = None,
                             ref_frame: Optional[str] = None) -> str:
    cache_path = _resolve_dynamic_cache(item["path"], cache_root)
    if not cache_path:
        return "(No dynamic memory cache for this video; reasoning from frames only.)"
    try:
        sm = SpatialMemory.load(cache_path, align_xz_with_pca=False)
        is_egocentric = item.get("original_question_type") == "vlm4d_ego"
        if archetype is not None:
            return compose_evidence_for_archetype(
                sm, archetype, is_egocentric=is_egocentric,
                entity=entity, ref_frame=ref_frame,
            )
        # Backwards-compatible default
        return render_evidence_block(sm, is_egocentric=is_egocentric)
    except Exception as e:
        return f"(Failed to load dynamic memory: {e})"


def _call_with_retry(fn, *, label: str, max_attempts: int = 6, base_sleep: float = 4.0):
    """Retry around remote-API calls; backs off on 429 / transient errors."""
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e)
            is_rate = "429" in msg or "rate" in msg.lower() or "too many requests" in msg.lower()
            if attempt == max_attempts - 1:
                break
            sleep_s = base_sleep * (2 ** attempt) if is_rate else base_sleep
            print(f"[retry] {label} attempt {attempt+1} hit error ({msg[:80]}); sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"{label} failed after {max_attempts} attempts: {last_err}")


def _run_decomposer(agent: Assistant, item: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    options_text = _format_options(item.get("options", []))
    prompt_text = VLM4D_DECOMPOSER_PROMPT.format(
        question_description=item["problem"], options=options_text
    )
    messages = [Message("user", [{"text": prompt_text}])]
    response = _call_with_retry(lambda: agent.run_nonstream(messages=messages), label="decomposer")
    raw = _extract_text(response)
    parsed, err = _parse_decomposition_json(raw)
    if parsed is None:
        # graceful fallback: leave empty decomposition
        return {"_parse_error": err}, raw
    return parsed, raw


def _run_reasoner(
    agent: Assistant,
    item: Dict[str, Any],
    decomposition: Dict[str, Any],
    evidence: str,
    frame_paths: List[str],
    include_rubric: bool = True,
) -> str:
    is_egocentric = item.get("original_question_type") == "vlm4d_ego"
    prompt_text = build_vlm4d_reasoner_prompt(
        question=item["problem"],
        options_block=_format_options(item.get("options", [])),
        decomposition=decomposition,
        evidence_block=evidence,
        use_cot=True,
        is_egocentric=is_egocentric,
        include_rubric=include_rubric,
    )
    content: List[Dict[str, Any]] = [{"image": p} for p in frame_paths]
    content.append({"text": prompt_text})
    response = _call_with_retry(
        lambda: agent.run_nonstream(messages=[Message("user", content)]),
        label="reasoner",
    )
    return _extract_text(response)


def _evaluate_one(
    item: Dict[str, Any],
    decomposer: Assistant,
    reasoner: Assistant,
    cache_root: str,
    nframes: int,
    include_rubric: bool = True,
) -> Dict[str, Any]:
    t0 = time.time()
    raw_video = item["path"]
    if not (isinstance(raw_video, str) and raw_video.endswith(".mp4")):
        return {"sample": item, "skipped": True, "reason": "non-video input"}
    frame_paths = extract_video_frames(raw_video, num_frames=nframes)

    # Stage 1: decomposer
    decomposition, decomposer_raw = _run_decomposer(decomposer, item)

    # Stage 2: executor — primitives selected by (archetype, ref_frame); pass entity
    archetype = (decomposition or {}).get("query_archetype")
    entity = (decomposition or {}).get("moving_entity")
    ref_frame = (decomposition or {}).get("reference_frame")
    evidence = _build_executor_evidence(item, cache_root, archetype=archetype,
                                        entity=entity, ref_frame=ref_frame)

    # Stage 3: reasoner
    try:
        reasoner_raw = _run_reasoner(reasoner, item, decomposition, evidence, frame_paths,
                                     include_rubric=include_rubric)
    except Exception as e:
        reasoner_raw = f"<error>{e}</error>"
        traceback.print_exc()
    elapsed = time.time() - t0

    pred = clean_text(reasoner_raw)
    gt = clean_text(item.get("solution", ""))
    reward = reward_fn(gt, pred, item.get("problem_type", "multiple choice"))

    # cleanup extracted frames
    for fp in frame_paths:
        try:
            os.remove(fp)
        except OSError:
            pass

    return {
        "sample": item,
        "decomposition": decomposition,
        "decomposer_raw": decomposer_raw,
        "evidence_block": evidence,
        "model_output": reasoner_raw,
        "cleaned_model_output": pred,
        "cleaned_gt_answer": gt,
        "reward": float(reward),
        "correct": float(reward) == 1.0,
        "elapsed_s": elapsed,
    }


def _aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    accs = [r["reward"] for r in results if "reward" in r]
    return {
        "mean_acc": sum(accs) / len(accs) if accs else 0.0,
        "mean_all": sum(accs) / len(accs) if accs else 0.0,
        "n": len(accs),
        "skipped": sum(1 for r in results if r.get("skipped")),
    }


def _save(output_path: str, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": results, "final_metrics": [metrics]}, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="Dynamic-scene benchmark name; see evaluation/annotation/")
    ap.add_argument("--model_type", required=True, help="output filename tag")
    ap.add_argument("--decomposer_model_path", default="gemini-3-flash-preview")
    ap.add_argument("--reasoner_model_path",  default="gemini-3-flash-preview")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--dynamic_memory_cache_root", default=str(dynamic_memory_cache_root()),
                    help="Precomputed Flow3R cache; defaults to $REASMORY_DYNAMIC_MEMORY_CACHE")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_rubric", action="store_true",
                    help="Drop the per-archetype reasoning rubric; let the reasoner figure out the analysis.")
    args = ap.parse_args()

    decomposer_cfg = build_llm_cfg(args.decomposer_model_path, args.temp, args.top_p)
    reasoner_cfg = build_llm_cfg(args.reasoner_model_path, args.temp, args.top_p)

    items = _load_benchmark(args.benchmark)
    if args.start_idx:
        items = items[args.start_idx:]
    if args.max_samples is not None:
        items = items[: args.max_samples]
    print(f"[vlm4d-3step] benchmark={args.benchmark}  items={len(items)}  cache_root={args.dynamic_memory_cache_root}")

    output_dir = REPO_ROOT / "eval_results" / f"eval_{args.benchmark}"
    output_path = output_dir / f"results_{args.model_type}.json"

    # Build one Assistant per role (the Assistant is stateless across calls)
    decomposer = Assistant(llm=decomposer_cfg, function_list=[], system_message=VLM4D_DECOMPOSER_SYSTEM)
    reasoner = Assistant(llm=reasoner_cfg,  function_list=[], system_message=VLM4D_REASONER_SYSTEM)

    results: List[Dict[str, Any]] = []
    include_rubric = not args.no_rubric
    if args.num_workers <= 1:
        for it in tqdm(items, desc=args.benchmark):
            results.append(_evaluate_one(it, decomposer, reasoner, args.dynamic_memory_cache_root,
                                         args.nframes, include_rubric=include_rubric))
            _save(str(output_path), results, _aggregate_metrics(results))
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(_evaluate_one, it, decomposer, reasoner,
                                 args.dynamic_memory_cache_root, args.nframes,
                                 include_rubric): it for it in items}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=args.benchmark):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"item failed: {e}")
                _save(str(output_path), results, _aggregate_metrics(results))

    final = _aggregate_metrics(results)
    _save(str(output_path), results, final)
    print(f"[vlm4d-3step] done. metrics={final}\n  → {output_path}")


if __name__ == "__main__":
    main()
