import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from tqdm import tqdm

from evaluation.eval_three_step import (
    _extract_last_text_from_response,
    _extract_user_question,
    load_benchmark,
    make_json_safe,
    prepare_eval_message,
    save_results,
)
from tools.decomposition_verifier import (
    DecompositionVerificationError,
    format_decomposer_feedback,
    parse_decomposition_output,
    verify_decomposition_structure,
)
from tools.llm_cfg import build_llm_cfg, is_remote_api_model
from tools.prompt import INITIAL_DECOMPOSER_PROMPT


DECOMPOSER_SYSTEM_MESSAGE = """
You are a decomposition assistant for spatial reasoning questions.
Your job is to analyze the question and decompose it into the requested structured parts.
Do not answer the original question directly.
Return only the decomposition result in JSON format.
"""

def calculate_decomposer_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_samples": len(results),
    }


def run_decomposer_round(
    decomposer: Assistant,
    decomposer_messages: List[Any],
) -> Any:
    return decomposer.run_nonstream(messages=decomposer_messages)


def build_decomposer_messages(
    question: str,
    repair_feedback: str | None = None,
) -> List[Message]:
    prompt_text = INITIAL_DECOMPOSER_PROMPT.format(question_description=question)
    if repair_feedback:
        prompt_text = f"{prompt_text}\n\n[Verifier Feedback]\n{repair_feedback}"
    return [Message("user", [{"text": prompt_text}])]


def build_decomposer_result(
    sample: Dict[str, Any],
    prompt: Any,
    model_output: str,
    decomposer_history: Any,
    parsed_output: Dict[str, Any] | None = None,
    verification_passed: bool = False,
    error_message: str | None = None,
    attempt_logs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "sample": sample.copy(),
        "prompt": make_json_safe(prompt),
        "model_output": model_output,
        "decomposer_history": make_json_safe(decomposer_history),
        "parsed_output": make_json_safe(parsed_output),
        "verification_passed": verification_passed,
        "error_message": error_message,
        "attempt_logs": attempt_logs or [],
    }


def evaluate_partition(
    eval_data: List[Dict[str, Any]],
    output_path: str,
    llm_cfg: Dict[str, Any],
    max_retries: int,
) -> List[Dict[str, Any]]:
    decomposer = Assistant(
        llm=llm_cfg,
        function_list=[],
        system_message=DECOMPOSER_SYSTEM_MESSAGE,
    )
    final_output: List[Dict[str, Any]] = []

    for item in tqdm(eval_data, desc=f"Decomposing {os.path.basename(output_path)}"):
        question_message = prepare_eval_message(item, [])
        question = _extract_user_question(question_message)
        repair_feedback = None
        model_output = ""
        decomposer_history = None
        parsed_output = None
        attempt_logs: List[Dict[str, Any]] = []
        verification_passed = False
        error_message = None
        prompt_payload = None

        for attempt in range(1, max_retries + 2):
            decomposer_messages = build_decomposer_messages(
                question,
                repair_feedback=repair_feedback,
            )
            prompt_payload = decomposer_messages[0]["content"]
            decomposer_history = run_decomposer_round(decomposer, decomposer_messages)
            model_output = _extract_last_text_from_response(decomposer_history)
            try:
                parsed_output = parse_decomposition_output(model_output)
                verify_decomposition_structure(parsed_output)
                attempt_logs.append(
                    {"attempt": attempt, "status": "success"}
                )
                verification_passed = True
                error_message = None
                break
            except DecompositionVerificationError as exc:
                error_message = str(exc)
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "status": "verification_failed",
                        "error_message": error_message,
                    }
                )
                repair_feedback = format_decomposer_feedback(model_output, error_message)

        result = build_decomposer_result(
            sample=item,
            prompt=prompt_payload,
            model_output=model_output,
            decomposer_history=decomposer_history,
            parsed_output=parsed_output,
            verification_passed=verification_passed,
            error_message=error_message,
            attempt_logs=attempt_logs,
        )
        final_output.append(result)
        save_results(output_path, final_output, calculate_decomposer_metrics(final_output))

    return final_output


def main(args: argparse.Namespace) -> None:
    import ray

    if not is_remote_api_model(args.model_path):
        if args.num_gpus_per_worker <= 0:
            raise ValueError(
                "Local Qwen models require GPU-backed Ray workers. "
                f"Received --num_gpus_per_worker={args.num_gpus_per_worker} for model_path={args.model_path}."
            )
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count <= 0:
            raise RuntimeError(
                "No CUDA GPUs are visible in the current process. "
                "Please check CUDA availability and CUDA_VISIBLE_DEVICES."
            )

    llm_cfg = build_llm_cfg(args.model_path, args.temp, args.top_p)
    eval_data = load_benchmark(args.benchmark)
    if args.start_idx is not None:
        eval_data = eval_data[args.start_idx:]
    if args.max_samples is not None:
        eval_data = eval_data[:args.max_samples]
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
                max_retries=args.max_retries,
            )
        )

    ret = ray.get(futures)
    final_output: List[Dict[str, Any]] = []
    for part in ret:
        final_output.extend(part)

    save_results(
        os.path.join(output_dir, f"results_{args.model_type}.json"),
        final_output,
        calculate_decomposer_metrics(final_output),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the question decomposer.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpus_per_worker", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
