import argparse
import json
import os
import re
import time
import sys
import traceback
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_agent.agents import Assistant
from tqdm import tqdm

from evaluation.eval_three_step import (
    build_planner_messages,
    get_sample_identifier,
    load_benchmark,
    make_json_safe,
    prepare_eval_media_messages,
    save_results,
    _extract_last_text_from_response,
    _extract_user_question,
)
from tools.llm_cfg import build_llm_cfg
from tools.plan_auto_repair import try_auto_repair_plan
from tools.plan_verifier import (
    ALLOWED_TOOL_NAMES,
    PlanVerificationError,
    PythonPlanVerifier,
    extract_python_code,
    format_verifier_feedback,
)
from tools.prompt import PLANNER_SYSTEM_MESSAGE

PLANNER_RUNTIME_MAX_RETRIES = 3
PLANNER_RUNTIME_BACKOFF_SECONDS = 5


def calculate_planner_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    verified = sum(1 for r in results if r.get("verify_passed"))
    metrics = {
        "total_samples": total,
        "verified_count": verified,
        "verified_rate": verified / total if total else 0.0,
    }
    return metrics


def _parse_decomposition_from_text(raw_output: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_output:
        return None

    text = raw_output.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def load_decomposition_results(path: Optional[str]) -> Dict[Any, Dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(open(path).read())
    results = data.get("results", []) if isinstance(data, dict) else data
    mapping = {}
    for item in results:
        sample = item.get("sample", {})
        problem_id = sample.get("problem_id")
        if problem_id is None:
            continue
        parsed_output = item.get("parsed_output")
        if isinstance(parsed_output, dict):
            mapping[problem_id] = parsed_output
            continue
        parsed_from_text = _parse_decomposition_from_text(item.get("model_output"))
        if parsed_from_text is not None:
            mapping[problem_id] = parsed_from_text
    return mapping


def run_planner_round(
    planner: Assistant,
    planner_messages: List[Any],
) -> Any:
    return planner.run_nonstream(messages=planner_messages)


def run_planner_round_with_retry(
    planner: Assistant,
    planner_messages: List[Any],
    attempt: int,
) -> Any:
    last_exc: Optional[Exception] = None
    for runtime_try in range(1, PLANNER_RUNTIME_MAX_RETRIES + 1):
        try:
            return run_planner_round(planner, planner_messages)
        except Exception as exc:
            last_exc = exc
            if runtime_try == PLANNER_RUNTIME_MAX_RETRIES:
                raise
            sleep_seconds = PLANNER_RUNTIME_BACKOFF_SECONDS * (2 ** (runtime_try - 1))
            print(
                f"[planner retry] attempt={attempt} runtime_try={runtime_try} "
                f"error_type={type(exc).__name__} sleep={sleep_seconds}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    if last_exc is not None:
        raise last_exc


def build_planner_result(
    sample: Dict[str, Any],
    prompt: Any,
    decomposition: Optional[Dict[str, Any]],
    planner_output: str,
    candidate_code: str,
    verify_passed: bool,
    verification_error: Optional[str],
    attempt_logs: List[Dict[str, Any]],
    planner_history: Any,
) -> Dict[str, Any]:
    return {
        "sample": sample.copy(),
        "prompt": make_json_safe(prompt),
        "decomposition": make_json_safe(decomposition),
        "decomposition_loaded": decomposition is not None,
        "planner_output": planner_output,
        "candidate_code": candidate_code,
        "verify_passed": verify_passed,
        "verification_error": verification_error,
        "attempt_logs": make_json_safe(attempt_logs),
        "planner_history": make_json_safe(planner_history),
    }


def evaluate_partition(
    eval_data: List[Dict[str, Any]],
    output_path: str,
    llm_cfg: Dict[str, Any],
    max_plan_retries: int,
    decomposition_by_problem_id: Optional[Dict[Any, Dict[str, Any]]] = None,
    video_preview_frames: int = 16,
    video_reconstruction_frames: int = 64,
) -> List[Dict[str, Any]]:
    try:
        planner = Assistant(
            llm=llm_cfg,
            function_list=[],
            system_message=PLANNER_SYSTEM_MESSAGE,
        )
    except Exception as exc:
        init_error = {
            "status": "planner_init_failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        final_output = [
            build_planner_result(
                sample=item,
                prompt=[],
                decomposition=(decomposition_by_problem_id or {}).get(item.get("problem_id")),
                planner_output="",
                candidate_code="",
                verify_passed=False,
                verification_error=f"{type(exc).__name__}: {exc}",
                attempt_logs=[init_error],
                planner_history=None,
            )
            for item in eval_data
        ]
        save_results(output_path, final_output, calculate_planner_metrics(final_output))
        return final_output

    verifier = PythonPlanVerifier(ALLOWED_TOOL_NAMES)
    final_output: List[Dict[str, Any]] = []

    for item in tqdm(eval_data, desc=f"Planning {os.path.basename(output_path)}"):
        try:
            media_inputs = prepare_eval_media_messages(
                item,
                video_preview_frames=video_preview_frames,
                video_reconstruction_frames=video_reconstruction_frames,
            )
        except ValueError:
            continue
        extract_frame = media_inputs["extract_frame"]
        cleanup_frame_paths = media_inputs["cleanup_frame_paths"]
        messages_input = media_inputs["vlm_messages"]
        input_image_count = len(media_inputs["preview_frame_paths"])
        question = _extract_user_question(messages_input)
        decomposition = (decomposition_by_problem_id or {}).get(item.get("problem_id"))

        planner_output = ""
        candidate_code = ""
        verification_error = None
        planner_history = None
        attempt_logs: List[Dict[str, Any]] = []
        verify_passed = False
        repair_feedback = None

        try:
            for attempt in range(1, max_plan_retries + 2):
                planner_messages = build_planner_messages(
                    messages_input,
                    question,
                    repair_feedback=repair_feedback,
                )
                planner_history = run_planner_round_with_retry(
                    planner,
                    planner_messages,
                    attempt=attempt,
                )
                planner_output = _extract_last_text_from_response(planner_history)
                candidate_code = extract_python_code(planner_output)
                try:
                    verifier.verify(
                        candidate_code,
                        decomposition=decomposition,
                        input_image_count=input_image_count,
                    )
                    attempt_logs.append(
                        {
                            "attempt": attempt,
                            "status": "verified",
                            "candidate_code": candidate_code,
                        }
                    )
                    verify_passed = True
                    verification_error = None
                    break
                except PlanVerificationError as exc:
                    verification_error = str(exc)
                    repaired_code, repair_info = try_auto_repair_plan(
                        candidate_code,
                        verification_error,
                        decomposition,
                        input_image_count=input_image_count,
                    )
                    if repaired_code is not None:
                        try:
                            verifier.verify(
                                repaired_code,
                                decomposition=decomposition,
                                input_image_count=input_image_count,
                            )
                            candidate_code = repaired_code
                            verify_passed = True
                            verification_error = None
                            attempt_logs.append(
                                {
                                    "attempt": attempt,
                                    "status": "auto_repaired_verified",
                                    "candidate_code": candidate_code,
                                    "auto_repair": make_json_safe(repair_info),
                                }
                            )
                            break
                        except PlanVerificationError as repaired_exc:
                            verification_error = str(repaired_exc)
                    attempt_logs.append(
                        {
                            "attempt": attempt,
                            "status": "verification_failed",
                            "candidate_code": candidate_code,
                            "error_message": verification_error,
                            "auto_repair": make_json_safe(repair_info),
                        }
                    )
                    repair_feedback = format_verifier_feedback(candidate_code, verification_error)
        except Exception as exc:
            verification_error = f"{type(exc).__name__}: {exc}"
            attempt_logs.append(
                {
                    "status": "planner_runtime_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        result = build_planner_result(
            sample=item,
            prompt=messages_input[0]["content"],
            decomposition=decomposition,
            planner_output=planner_output,
            candidate_code=candidate_code,
            verify_passed=verify_passed,
            verification_error=verification_error,
            attempt_logs=attempt_logs,
            planner_history=planner_history,
        )
        result["media_input_info"] = {
            "extract_frame": extract_frame,
            "preview_frame_count": len(media_inputs["preview_frame_paths"]),
            "reconstruction_frame_count": len(media_inputs["reconstruction_frame_paths"]),
            "preview_frame_indices_in_reconstruction": media_inputs["preview_frame_indices_in_reconstruction"],
        }
        final_output.append(result)
        save_results(output_path, final_output, calculate_planner_metrics(final_output))
        if extract_frame:
            for frame_path in cleanup_frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

    return final_output


def main(args: argparse.Namespace) -> None:
    import ray

    llm_cfg = build_llm_cfg(args.model_path, args.temp, args.top_p)
    eval_data = load_benchmark(args.benchmark)
    eval_data = eval_data[args.start_idx : (args.start_idx + args.max_samples) if args.max_samples else None]
    decomposition_by_problem_id = load_decomposition_results(args.decomposition_results)
    output_dir = os.path.join("eval_results", f"eval_{args.benchmark}")
    os.makedirs(output_dir, exist_ok=True)

    num_workers = min(args.num_workers, len(eval_data))
    per_worker = len(eval_data) // num_workers
    chunks: List[List[Dict[str, Any]]] = []
    for i in range(num_workers):
        if i == num_workers - 1:
            chunk = eval_data[i * per_worker :]
        else:
            chunk = eval_data[i * per_worker : (i + 1) * per_worker]
        if chunk:
            chunks.append(chunk)

    ray.init(ignore_reinit_error=True)
    evaluate_partition_remote = ray.remote(evaluate_partition)
    futures = []
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"results_{args.model_type}_{i}.json")
        futures.append(
            evaluate_partition_remote.options(num_gpus=args.num_gpus_per_worker).remote(
                eval_data=chunk,
                output_path=output_path,
                llm_cfg=llm_cfg,
                max_plan_retries=args.max_plan_retries,
                decomposition_by_problem_id=decomposition_by_problem_id,
                video_preview_frames=args.video_preview_frames,
                video_reconstruction_frames=args.video_reconstruction_frames,
            )
        )

    ret = ray.get(futures)
    final_output: List[Dict[str, Any]] = []
    for part in ret:
        final_output.extend(part)

    save_results(
        os.path.join(output_dir, f"results_{args.model_type}.json"),
        final_output,
        calculate_planner_metrics(final_output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the DSL planner only.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_plan_retries", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpus_per_worker", type=float, default=0.0)
    parser.add_argument("--decomposition_results", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--video_preview_frames", type=int, default=16)
    parser.add_argument("--video_reconstruction_frames", type=int, default=64)
    args = parser.parse_args()
    main(args)
