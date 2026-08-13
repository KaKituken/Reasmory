import argparse
import copy
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message
from tqdm import tqdm

from evaluation.utils import clean_text, extract_video_frames, reward_fn
from tools.plan_executor import (
    PlanExecutionError,
    execute_plan,
    format_execution_feedback,
)
from tools.llm_cfg import build_llm_cfg
from tools.plan_verifier import (
    ALLOWED_TOOL_NAMES,
    PlanVerificationError,
    PythonPlanVerifier,
    extract_python_code,
    format_verifier_feedback,
)
from tools.prompt import PLANNER_PROMPT_CODE, PLANNER_SYSTEM_MESSAGE, TOOL_CATALOG_CODE, PLANNER_PROMPT_CODE_TRANSLATOR, SUMMARY_CONTEXT_FALLBACK

# VSI-Bench question types that need absolute metric scale -> pi3x reconstruction.
MEASURING_QUESTION_TYPES = {
    "object_size_estimation",
    "room_size_estimation",
    "object_abs_distance",
}

# Question types better answered by looking at the ordered video frames directly
# (temporal / non-spatial), bypassing the spatial-tool plan pipeline.
DEFAULT_DIRECT_VLM_QUESTION_TYPES = {
    "obj_appearance_order",
}

DIRECT_VLM_SYSTEM_MESSAGE = """
You are a spatial-video question answering assistant.
You are given a sequence of frames sampled in TEMPORAL ORDER from a video, followed by a question.
Reason about what appears in the frames and their order over time, then answer.
If options are provided, answer with the option letter. Put the final answer inside <answer>...</answer>.
"""

# Canonical BEV-only plan used as a fallback when the planner/executor fails all attempts.
BEV_FALLBACK_PLAN_CODE = '''
def plan_to_solve_problem(input_images: List[str]):
    memory = build_static_spatial_memory(input_type="images", image_paths=input_images)
    session_id = memory["session_id"]
    layout = render_rgb_bev(session_id=session_id)
    useful_observation = [layout]
    return useful_observation
'''
from tools.reasoner_prompt_builder import build_reasoner_messages
from tools.paths import resolve_sample_media, spatial_memory_cache_root as default_spatial_cache_root


# REASONER_SYSTEM_MESSAGE = """
# You are a helpful assistant specialized in spatial reasoning tasks.
# You are given the original question, the input images, and useful observations produced by a verified tool-execution pipeline.
# Use those observations to answer the question.
# Do not invent missing observations.
# If a tool-generated view directly corresponds to the target direction or transformed viewpoint in the question, prioritize that view over guessing from the original input image alone.
# When options are provided, compare the plausible options against the observations and eliminate the inconsistent ones before answering.
# Be careful with semantically similar categories (for example sofa vs bench, wall vs window, rack vs hanging clothes). Do not stop at the first plausible object.
# If options are provided in the question, answer with the exact option text inside <answer>...</answer>.
# If no options are provided, answer concisely inside <answer>...</answer>.
# """

REASONER_SYSTEM_MESSAGE = """
You are a helpful assistant specialized in spatial reasoning tasks.
You are given the original question, the input images, and useful observations produced by a verified tool-execution pipeline.
Use those observations to answer the question.
Do not invent missing observations.
If a tool-generated view directly corresponds to the target direction or transformed viewpoint in the question, prioritize that view over guessing from the original input image alone.
When options are provided, compare the plausible options against the observations and eliminate the inconsistent ones before answering.
Be careful with semantically similar categories (for example sofa vs bench, wall vs window, rack vs hanging clothes). Do not stop at the first plausible object.
"""

SFT_TYPE_TEMPLATE = {
    "multiple choice": "Please answer with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags. Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should be this format: '<think>(replace with your reasoning here)</think><answer>A</answer>'.",
    "numerical": "Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags. Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is '42', your response should be this format: '<think>(replace with your reasoning here)</think><answer>42</answer>'.",
    "regression": "Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags. Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is '3.14', your response should be this format: '<think>(replace with your reasoning here)</think><answer>3.14</answer>'.",
    "verbal": "Please answer the question simply within the <answer> </answer> tags",
}

REASONER_DEBUG_SYSTEM_MESSAGE = """
You are a helpful assistant specialized in spatial reasoning tasks.
You are given the original question, the input images, and useful observations produced by a verified tool-execution pipeline.
Use those observations to answer the question.
Do not invent missing observations.
If a tool-generated view directly corresponds to the target direction or transformed viewpoint in the question, prioritize that view over guessing from the original input image alone.
When options are provided, compare the plausible options against the observations and eliminate the inconsistent ones before answering.
Be careful with semantically similar categories (for example sofa vs bench, wall vs window, rack vs hanging clothes). Do not stop at the first plausible object.
First write a short reasoning note inside <analysis>...</analysis>.
Then give the final answer inside <answer>...</answer>.
If options are provided in the question, the final answer must use the exact option text.
Keep the analysis concise and grounded in the provided observations.
"""

FALLBACK_SYSTEM_MESSAGE = """
You are a helpful AI assistant for spatial reasoning.
Use the available tools pragmatically to answer the question.
Prefer short tool trajectories that directly resolve the spatial relation.
Return the final answer inside <answer>...</answer>.
"""

SFT_QUESTION_TEMPLATE = "{Question}"
# MINDCUBE_HEADER = """
# [Task]
# Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
# [Answer Instruction]
# You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
# """

# MINDCUBE_HEADER = """
# [Task]
# Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
# [Answer Instruction]
# Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should be this format: '<think>(replace with your reasoning here)</think><answer>A. Above</answer>'.
# """

MINDCUBE_HEADER = """
[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
"""

VSI_QUESTION_TYPE = {
    "obj_appearance_order",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
    "object_abs_distance",
    "object_rel_direction_hard",
    "object_rel_direction_medium",
    "object_rel_direction_easy",
    "object_rel_distance",
    "route_planning",
}

# SUMMARY_CONTEXT = """
# Please answer the original question based on the verified tool execution outputs.
# Useful observations are attached below as images and structured text.
# Each observation description explains whether it is an original input view or a 
# tool-generated view, and under which transformed viewpoint it was produced.
# Use the observation descriptions to align the evidence before deciding the answer.
# """

# SUMMARY_CONTEXT = """
# Please answer the original question based on the verified tool execution outputs.
# Useful observations are attached below as images and structured text.
# Since the rendered views have already been transformed to align with the question's target viewpoint, you can directly use them as evidence without needing to mentally transform the original input views.
# Therefore, to answer the question, please answer:

# 1. What I'm facing in the rendered views?
# 2. Identify which input view has the most similar visible layout to the rendered views.
# 3. Use the answer from step 1 as the final answer.
# """


def load_benchmark(benchmark_name: str) -> List[Dict[str, Any]]:
    file_path = os.path.abspath(__file__)
    benchmark_annotation_path = os.path.join(
        os.path.dirname(file_path), "annotation", f"eval_{benchmark_name}.json"
    )
    with open(benchmark_annotation_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    # Annotations ship with the absolute media paths of the machine that produced
    # them; re-root them under REASMORY_DATA_ROOT so a fresh checkout can run.
    for sample in samples:
        if isinstance(sample, dict):
            resolve_sample_media(sample)
    return samples


def get_sample_identifier(sample: Dict[str, Any]) -> Optional[Any]:
    for key in ("problem_id", "id", "sample_id", "question_id"):
        value = sample.get(key)
        if value is not None:
            return value
    return None


def load_imported_plans(plan_path: str) -> Dict[Any, Dict[str, Any]]:
    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plans_by_id: Dict[Any, Dict[str, Any]] = {}
    for result in data.get("results", []):
        sample = result.get("sample", {})
        sample_id = get_sample_identifier(sample)
        if sample_id is None or sample_id in plans_by_id:
            continue
        raw_plan = (
            result.get("candidate_code")
            or result.get("planner_output")
            or result.get("model_output")
            or result.get("cleaned_model_output")
            or ""
        )
        if raw_plan:
            plans_by_id[sample_id] = {
                "plan": raw_plan,
                "decomposition": result.get("decomposition"),
            }
    return plans_by_id


def prepare_eval_message(item: Dict[str, Any], frame_paths: List[str]) -> List[Message]:
    if item["original_question_type"] in VSI_QUESTION_TYPE:
        question = (
            "[Task]\nYour task is to analyze the spatial arrangement of objects in the scene "
            "by examining the provided images, which show the scene from different viewpoints."
        )
        question += "\n[Question]\n"
        if item["problem_type"] == "multiple choice":
            question += item["problem"] + " " + " ".join(item["options"])
        else:
            question += item["problem"]
    elif item["original_question_type"] == "spatial reasoning":
        question = MINDCUBE_HEADER + "[Question]\n"
        if item["problem_type"] == "multiple choice":
            question += item["problem"] + " " + " ".join(item["options"])
        else:
            question += item["problem"]
    else:
        question = item["problem"]

    content: List[Dict[str, str]] = [{"image": frame_path} for frame_path in frame_paths]
    content.append({"text": SFT_QUESTION_TEMPLATE.format(Question=question) + SFT_TYPE_TEMPLATE[item["problem_type"]]})
    return [Message("user", content)]


def _uniform_subsample_indices(total_count: int, target_count: int) -> List[int]:
    if total_count <= 0:
        return []
    if target_count >= total_count:
        return list(range(total_count))
    if target_count <= 1:
        return [0]

    indices: List[int] = []
    for i in range(target_count):
        idx = round(i * (total_count - 1) / (target_count - 1))
        indices.append(int(idx))

    deduped: List[int] = []
    for idx in indices:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)

    if len(deduped) < target_count:
        used = set(deduped)
        for idx in range(total_count):
            if idx not in used:
                deduped.append(idx)
                used.add(idx)
            if len(deduped) == target_count:
                break

    deduped.sort()
    return deduped[:target_count]


def prepare_eval_media_messages(
    item: Dict[str, Any],
    video_preview_frames: int = 16,
    video_reconstruction_frames: int = 64,
) -> Dict[str, Any]:
    if not isinstance(item["path"], list) and item["path"].endswith((".mp4", ".avi")):
        reconstruction_frame_paths = extract_video_frames(
            item["path"],
            fps=1.0,
            num_frames=video_reconstruction_frames,
        )
        preview_indices = _uniform_subsample_indices(
            len(reconstruction_frame_paths),
            min(video_preview_frames, len(reconstruction_frame_paths)),
        )
        preview_frame_paths = [reconstruction_frame_paths[idx] for idx in preview_indices]
        return {
            "extract_frame": True,
            "cleanup_frame_paths": reconstruction_frame_paths,
            "preview_frame_paths": preview_frame_paths,
            "reconstruction_frame_paths": reconstruction_frame_paths,
            "preview_frame_indices_in_reconstruction": preview_indices,
            "vlm_messages": prepare_eval_message(item, preview_frame_paths),
            "runtime_messages": prepare_eval_message(item, reconstruction_frame_paths),
        }

    if isinstance(item["path"], list) and item["path"] and all(
        p.endswith((".jpg", ".png")) for p in item["path"]
    ):
        frame_paths = item["path"]
        return {
            "extract_frame": False,
            "cleanup_frame_paths": [],
            "preview_frame_paths": frame_paths,
            "reconstruction_frame_paths": frame_paths,
            "preview_frame_indices_in_reconstruction": list(range(len(frame_paths))),
            "vlm_messages": prepare_eval_message(item, frame_paths),
            "runtime_messages": prepare_eval_message(item, frame_paths),
        }

    raise ValueError(f"Unsupported input path format: {item['path']}")


def resolve_precomputed_spatial_memory_path(
    item: Dict[str, Any],
    spatial_memory_cache_root: Optional[str],
    measuring_cache_root: Optional[str] = None,
) -> Optional[str]:
    # Route measuring (metric) questions to the pi3x cache, others to the default (pi3) cache.
    root = spatial_memory_cache_root
    if measuring_cache_root and item.get("original_question_type") in MEASURING_QUESTION_TYPES:
        root = measuring_cache_root
    if not root:
        return None
    video_path = item.get("path")
    if not isinstance(video_path, str) or not video_path.endswith((".mp4", ".avi")):
        return None
    cache_path = os.path.join(
        root,
        os.path.splitext(os.path.basename(video_path))[0],
        "spatial_memory.npz",
    )
    return cache_path if os.path.exists(cache_path) else None


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _message_text(msg: Any) -> str:
    content = _get_attr_or_key(msg, "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for item in content:
        text = _get_attr_or_key(item, "text")
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _extract_last_text_from_response(response: Optional[List[Any]]) -> str:
    if not response:
        return ""
    texts = []
    for msg in response:
        text = _message_text(msg)
        if text:
            texts.append(text)
    return texts[-1] if texts else ""


def _extract_user_question(messages: List[Message]) -> str:
    last_content = messages[-1]["content"]
    for item in last_content:
        text = _get_attr_or_key(item, "text")
        if text:
            return str(text)
    return ""


def _extract_input_images(messages: List[Message]) -> List[str]:
    images: List[str] = []
    last_content = messages[-1]["content"]
    for item in last_content:
        image = _get_attr_or_key(item, "image")
        if image:
            images.append(str(image))
    return images


def build_planner_messages(
    original_messages: List[Message],
    question: str,
    imported_plan_decomposition: Optional[str] = None,
    repair_feedback: Optional[str] = None,
) -> List[Message]:
    planner_messages = copy.deepcopy(original_messages)
    code_prompt = PLANNER_PROMPT_CODE_TRANSLATOR.format(
        tool_catalog=TOOL_CATALOG_CODE,
        problem_description=question,
        decomposition_json=imported_plan_decomposition,
    )
    # code_prompt = PLANNER_PROMPT_CODE.format(
    #     tool_catalog=TOOL_CATALOG_CODE,
    #     problem_description=question,
    # )
    if repair_feedback:
        code_prompt = f"{code_prompt}\n\n[Verifier Feedback]\n{repair_feedback}"

    for item in planner_messages[-1]["content"]:
        if _get_attr_or_key(item, "text"):
            item["text"] = code_prompt
            break
    return planner_messages


def postprocess_result(sample: Dict[str, Any], model_output: str, prompt: Any) -> Dict[str, Any]:
    result_sample: Dict[str, Any] = {
        "sample": sample.copy(),
        "prompt": make_json_safe(prompt),
        "model_output": model_output,
    }
    clean_ans = clean_text(model_output)
    result_sample["cleaned_model_output"] = clean_ans
    clean_ans_gt = clean_text(sample.get("solution", ""))
    result_sample["cleaned_gt_answer"] = clean_ans_gt
    result_sample["reward"] = reward_fn(clean_ans_gt, clean_ans, sample["problem_type"])
    result_sample["correct"] = result_sample["reward"] == 1.0
    return result_sample


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    mean_acc_rewards = [
        s["reward"]
        for s in results
        if s["sample"].get("problem_type") != "regression" and "reward" in s
    ]
    mean_mra_rewards = [
        s["reward"]
        for s in results
        if s["sample"].get("problem_type") == "regression"
        and "reward" in s
        and s.get("prediction") != "error"
    ]

    final_metrics = {"mean_acc": 0.0, "mean_mra": 0.0, "mean_all": 0.0}
    if mean_acc_rewards:
        final_metrics["mean_acc"] = torch.tensor(
            mean_acc_rewards, dtype=torch.float32
        ).mean().item()
    if mean_mra_rewards:
        final_metrics["mean_mra"] = torch.tensor(
            mean_mra_rewards, dtype=torch.float32
        ).mean().item()
    if mean_acc_rewards or mean_mra_rewards:
        all_rewards = torch.cat(
            [
                torch.tensor(mean_acc_rewards, dtype=torch.float32),
                torch.tensor(mean_mra_rewards, dtype=torch.float32),
            ]
        )
        final_metrics["mean_all"] = all_rewards.mean().item()
    return final_metrics


def calculate_stage_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    failure_counts: Dict[str, int] = {}
    for result in results:
        stage = result.get("failure_stage") or "success"
        failure_counts[stage] = failure_counts.get(stage, 0) + 1

    metrics = {
        "total_samples": total,
        "failure_counts": failure_counts,
        "failure_rates": {},
    }
    if total > 0:
        metrics["failure_rates"] = {
            key: value / total for key, value in failure_counts.items()
        }
    return metrics


def save_results(output_path: str, results: List[Dict[str, Any]], final_acc: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "final_acc": [final_acc]},
            f,
            indent=2,
            ensure_ascii=False,
        )


def _response_to_serializable(response: Any) -> Any:
    if response is None:
        return None
    try:
        return json.loads(json.dumps(response, ensure_ascii=False, default=str))
    except Exception:
        return str(response)


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}

    serialized_fields: Dict[str, Any] = {}
    for field_name in ("role", "name", "text", "image", "content"):
        field_value = getattr(value, field_name, None)
        if field_value is not None:
            serialized_fields[field_name] = make_json_safe(field_value)
    if serialized_fields:
        return serialized_fields

    return str(value)


def _build_failure_result(
    sample: Dict[str, Any],
    prompt: Any,
    planner_source: str,
    planner_output: str,
    candidate_code: str,
    compiled_code: Optional[str],
    failure_stage: str,
    error_message: str,
    planner_history: Any = None,
    attempt_logs: Optional[List[Dict[str, Any]]] = None,
    useful_observation: Any = None,
    execution_trace: Optional[List[Dict[str, Any]]] = None,
    reasoner_input_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = postprocess_result(
        sample=sample,
        model_output=f"<answer>ERROR</answer>\n{error_message}",
        prompt=prompt,
    )
    result.update(
        {
            "planner_source": planner_source,
            "planner_output": planner_output,
            "candidate_code": candidate_code,
            "compiled_code": compiled_code,
            "failure_stage": failure_stage,
            "error_message": error_message,
            "planner_history": _response_to_serializable(planner_history),
            "attempt_logs": attempt_logs or [],
            "useful_observation": make_json_safe(useful_observation),
            "execution_trace": make_json_safe(execution_trace or []),
            "reasoner_input_package": make_json_safe(reasoner_input_package),
        }
    )
    return result


def extract_tag_text(text: str, tag_name: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(
        rf"<{tag_name}>(.*?)</{tag_name}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


def run_planner_round(
    planner: Assistant,
    planner_messages: List[Dict[str, Any]],
    total_attempts: int = 3,
) -> Any:
    for attempt in range(total_attempts):
        try:
            return planner.run_nonstream(messages=planner_messages)
        except Exception as exc:
            if attempt == total_attempts - 1:
                raise exc
            time.sleep(5)
    raise RuntimeError("Unexpected planner retry failure.")


def run_reasoner_round(
    reasoner: Assistant,
    reasoner_messages: List[Dict[str, Any]],
    total_attempts: int = 3,
) -> Any:
    for attempt in range(total_attempts):
        try:
            return reasoner.run_nonstream(messages=reasoner_messages)
        except Exception as exc:
            if attempt == total_attempts - 1:
                raise exc
            time.sleep(5)
    raise RuntimeError("Unexpected reasoner retry failure.")


def build_fallback_messages(
    original_messages: List[Message],
    planner_output: str,
    candidate_code: str,
    failure_stage: str,
    error_message: str,
) -> List[Message]:
    messages = copy.deepcopy(original_messages)
    guidance = [
        {
            "text": (
                "The primary DSL execution pipeline failed.\n"
                f"Failure stage: {failure_stage}\n"
                f"Error: {error_message}\n\n"
                "Use tools directly as a fallback executor and answer the original question.\n"
                "You may reuse the failed plan as weak guidance, but do not follow it if it is clearly wrong.\n\n"
                "Failed planner output:\n"
                f"{planner_output}\n\n"
                "Extracted candidate code:\n"
                f"```python\n{candidate_code}\n```"
            )
        }
    ]
    messages.append(Message("user", guidance))
    return messages


def build_tool_fallback_agent(llm_cfg: Dict[str, Any]) -> Assistant:
    from tools.agent_tools import (
        BuildStaticSpatialMemory,
        QueryCameraPose,
        Query3DObjectPosition,
        SetViewpoint,
        StepCamera,
        TurnCamera,
        RenderEgoRGB,
        RenderRGBBEV,
        RenderSemanticBEV,
    )

    del (
        BuildStaticSpatialMemory,
        QueryCameraPose,
        Query3DObjectPosition,
        SetViewpoint,
        StepCamera,
        TurnCamera,
        RenderEgoRGB,
        RenderRGBBEV,
        RenderSemanticBEV,
    )

    return Assistant(
        llm=llm_cfg,
        function_list=list(ALLOWED_TOOL_NAMES),
        system_message=FALLBACK_SYSTEM_MESSAGE,
    )


def run_fallback_round(
    fallback_agent: Assistant,
    fallback_messages: List[Message],
    total_attempts: int = 2,
) -> Any:
    for attempt in range(total_attempts):
        try:
            return fallback_agent.run_nonstream(messages=fallback_messages)
        except Exception as exc:
            if attempt == total_attempts - 1:
                raise exc
            time.sleep(5)
    raise RuntimeError("Unexpected fallback retry failure.")


class FallbackWorker:
    def __init__(self, llm_cfg: Dict[str, Any]):
        self.fallback_agent = build_tool_fallback_agent(llm_cfg)

    def run(
        self,
        fallback_messages: List[Message],
        total_attempts: int = 2,
    ) -> Any:
        return run_fallback_round(
            self.fallback_agent,
            fallback_messages,
            total_attempts=total_attempts,
        )


def evaluate_partition(
    eval_data: List[Dict[str, Any]],
    output_path: str,
    benchmark_name: str,
    planner_llm_cfg: Optional[Dict[str, Any]],
    reasoner_llm_cfg: Dict[str, Any],
    imported_plans_by_id: Optional[Dict[Any, Dict[str, Any]]] = None,
    max_plan_retries: int = 2,
    repair_imported_plan: bool = False,
    fallback_llm_cfg: Optional[Dict[str, Any]] = None,
    fallback_worker = None,
    debug_reasoning: bool = False,
    reasoner_include_original_inputs: bool = True,
    reasoner_drop_input_observations: bool = False,
    video_preview_frames: int = 16,
    video_reconstruction_frames: int = 64,
    spatial_memory_cache_root: Optional[str] = None,
    measuring_cache_root: Optional[str] = None,
    bev_fallback: bool = False,
    direct_vlm_types: Optional[set] = None,
) -> List[Dict[str, Any]]:
    del benchmark_name
    direct_vlm_types = direct_vlm_types or set()

    planner = None
    if planner_llm_cfg is not None:
        planner = Assistant(
            llm=planner_llm_cfg,
            function_list=[],
            system_message=PLANNER_SYSTEM_MESSAGE,
        )
    reasoner = Assistant(
        llm=reasoner_llm_cfg,
        function_list=[],
        system_message=(
            REASONER_DEBUG_SYSTEM_MESSAGE if debug_reasoning else REASONER_SYSTEM_MESSAGE
        ),
    )
    direct_vlm_agent = Assistant(
        llm=reasoner_llm_cfg,
        function_list=[],
        system_message=DIRECT_VLM_SYSTEM_MESSAGE,
    )
    fallback_agent = (
        build_tool_fallback_agent(fallback_llm_cfg)
        if fallback_llm_cfg is not None and fallback_worker is None
        else None
    )
    verifier = PythonPlanVerifier(ALLOWED_TOOL_NAMES)

    final_output: List[Dict[str, Any]] = []

    for idx, item in enumerate(tqdm(eval_data, desc=f"Processing {os.path.basename(output_path)}")):
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
        runtime_messages = media_inputs["runtime_messages"]
        media_input_info = {
            "extract_frame": extract_frame,
            "preview_frame_count": len(media_inputs["preview_frame_paths"]),
            "reconstruction_frame_count": len(media_inputs["reconstruction_frame_paths"]),
            "preview_frame_indices_in_reconstruction": media_inputs["preview_frame_indices_in_reconstruction"],
        }
        question = _extract_user_question(messages_input)
        input_images = _extract_input_images(messages_input)
        sample_id = get_sample_identifier(item)
        precomputed_spatial_memory_path = resolve_precomputed_spatial_memory_path(
            item,
            spatial_memory_cache_root=spatial_memory_cache_root,
            measuring_cache_root=measuring_cache_root,
        )
        print("spatial_memory_cache_root:", spatial_memory_cache_root)
        print("precomputed_spatial_memory_path:", precomputed_spatial_memory_path)
        media_input_info["precomputed_spatial_memory_path"] = precomputed_spatial_memory_path

        # Direct-VLM bypass for temporal / non-spatial question types: answer straight
        # from the ordered video frames instead of the spatial-tool plan pipeline.
        if item.get("original_question_type") in direct_vlm_types:
            try:
                direct_response = run_reasoner_round(direct_vlm_agent, runtime_messages)
                direct_output = _extract_last_text_from_response(direct_response)
                result = postprocess_result(
                    sample=item,
                    model_output=direct_output,
                    prompt=messages_input[0]["content"],
                )
                result.update(
                    {
                        "planner_source": "direct_vlm",
                        "failure_stage": None,
                        "error_message": None,
                        "reasoner_history": _response_to_serializable(direct_response),
                        "direct_vlm_used": True,
                    }
                )
            except Exception as direct_exc:
                result = _build_failure_result(
                    sample=item,
                    prompt=messages_input[0]["content"],
                    planner_source="direct_vlm",
                    planner_output="",
                    candidate_code="",
                    compiled_code=None,
                    failure_stage="direct_vlm",
                    error_message=f"{type(direct_exc).__name__}: {direct_exc}",
                    planner_history=None,
                    attempt_logs=[],
                    useful_observation=None,
                    execution_trace=[],
                    reasoner_input_package=None,
                )
                result["direct_vlm_used"] = True
            result["media_input_info"] = media_input_info
            final_output.append(result)
            current_metrics = calculate_metrics(final_output)
            current_metrics["stage_metrics"] = calculate_stage_metrics(final_output)
            save_results(output_path, final_output, current_metrics)
            if extract_frame:
                for frame_path in cleanup_frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
            continue

        planner_history = None
        attempt_logs: List[Dict[str, Any]] = []
        planner_output = ""
        candidate_code = ""
        compiled_code = None
        useful_observation = None
        execution_trace: List[Dict[str, Any]] = []
        reasoner_input_package = None
        planner_source = "model"
        repair_feedback = None

        imported_plan = None
        imported_plan_decomposition = None
        if imported_plans_by_id is not None and sample_id in imported_plans_by_id:
            imported_plan_info = imported_plans_by_id[sample_id]
            imported_plan = imported_plan_info.get("plan")
            imported_plan_decomposition = imported_plan_info.get("decomposition")

        success = False
        max_attempts = 1 if imported_plan and not repair_imported_plan else max_plan_retries + 1
        if imported_plan:
            planner_source = "imported_plan"

        for attempt in range(1, max_attempts + 1):
            try:
                if imported_plan and attempt == 1:
                    planner_output = imported_plan
                    planner_history = [{"role": "assistant", "content": imported_plan}]
                else:
                    if planner is None:
                        raise RuntimeError("No planner model configured for replanning.")
                    planner_messages = build_planner_messages(
                        messages_input,
                        question,
                        imported_plan_decomposition=imported_plan_decomposition,
                        repair_feedback=repair_feedback,
                    )
                    planner_history = run_planner_round(planner, planner_messages)
                    planner_output = _extract_last_text_from_response(planner_history)
                    planner_source = "planner_model"

                candidate_code = extract_python_code(planner_output)
                verifier.verify(
                    candidate_code,
                    decomposition=imported_plan_decomposition,
                    input_image_count=len(input_images),
                )
                useful_observation, execution_trace, compiled_code = execute_plan(
                    candidate_code,
                    input_images,
                    runtime_messages,
                    decomposition=imported_plan_decomposition,
                    precomputed_spatial_memory_path=precomputed_spatial_memory_path,
                )
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "planner_source": planner_source,
                        "status": "success",
                        "candidate_code": candidate_code,
                    }
                )
                success = True
                break
            except PlanVerificationError as exc:
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "planner_source": planner_source,
                        "status": "verification_failed",
                        "candidate_code": candidate_code,
                        "error_message": str(exc),
                    }
                )
                repair_feedback = format_verifier_feedback(candidate_code, str(exc))
                last_error = ("verification", str(exc))
            except PlanExecutionError as exc:
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "planner_source": planner_source,
                        "status": "execution_failed",
                        "candidate_code": candidate_code,
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
                repair_feedback = format_execution_feedback(
                    candidate_code, f"{type(exc).__name__}: {exc}"
                )
                last_error = ("execution", f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                attempt_logs.append(
                    {
                        "attempt": attempt,
                        "planner_source": planner_source,
                        "status": "runtime_failed",
                        "candidate_code": candidate_code,
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
                last_error = ("runtime", f"{type(exc).__name__}: {exc}")
                break

        if not success:
            failure_stage, error_message = last_error
            if bev_fallback:
                # After the planner/executor fails all attempts, fall back to a plain
                # BEV of the scene and let the reasoner answer from it.
                try:
                    bev_obs, bev_trace, bev_compiled = execute_plan(
                        BEV_FALLBACK_PLAN_CODE,
                        input_images,
                        runtime_messages,
                        decomposition=imported_plan_decomposition,
                        precomputed_spatial_memory_path=precomputed_spatial_memory_path,
                    )
                    reasoner_messages, reasoner_input_package = build_reasoner_messages(
                        messages_input,
                        bev_obs,
                        input_images,
                        bev_trace,
                        include_original_inputs=reasoner_include_original_inputs,
                        drop_original_input_observations=reasoner_drop_input_observations,
                        summary_context=SUMMARY_CONTEXT_FALLBACK,
                    )
                    reasoner_response = run_reasoner_round(reasoner, reasoner_messages)
                    bev_output = _extract_last_text_from_response(reasoner_response)
                    result = postprocess_result(
                        sample=item,
                        model_output=bev_output,
                        prompt=messages_input[0]["content"],
                    )
                    result.update(
                        {
                            "planner_source": planner_source,
                            "planner_output": planner_output,
                            "candidate_code": candidate_code,
                            "compiled_code": compiled_code,
                            "failure_stage": failure_stage,
                            "error_message": error_message,
                            "planner_history": _response_to_serializable(planner_history),
                            "attempt_logs": attempt_logs,
                            "reasoner_history": _response_to_serializable(reasoner_response),
                            "useful_observation": make_json_safe(bev_obs),
                            "execution_trace": make_json_safe(bev_trace),
                            "reasoner_input_package": make_json_safe(reasoner_input_package),
                            "fallback_used": True,
                            "fallback_reason": failure_stage,
                            "bev_fallback_used": True,
                        }
                    )
                except Exception as bev_exc:
                    result = _build_failure_result(
                        sample=item,
                        prompt=messages_input[0]["content"],
                        planner_source=planner_source,
                        planner_output=planner_output,
                        candidate_code=candidate_code,
                        compiled_code=compiled_code,
                        failure_stage=failure_stage,
                        error_message=error_message,
                        planner_history=planner_history,
                        attempt_logs=attempt_logs,
                        useful_observation=useful_observation,
                        execution_trace=execution_trace,
                        reasoner_input_package=reasoner_input_package,
                    )
                    result["fallback_used"] = True
                    result["fallback_reason"] = failure_stage
                    result["bev_fallback_used"] = True
                    result["bev_fallback_error"] = f"{type(bev_exc).__name__}: {bev_exc}"
                result["media_input_info"] = media_input_info
                final_output.append(result)
                current_metrics = calculate_metrics(final_output)
                current_metrics["stage_metrics"] = calculate_stage_metrics(final_output)
                save_results(output_path, final_output, current_metrics)
                if extract_frame:
                    for frame_path in cleanup_frame_paths:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                continue
            if fallback_agent is not None:
                try:
                    fallback_messages = build_fallback_messages(
                        messages_input,
                        planner_output=planner_output,
                        candidate_code=candidate_code,
                        failure_stage=failure_stage,
                        error_message=error_message,
                    )
                    if fallback_worker is not None:
                        import ray
                        fallback_response = ray.get(
                            fallback_worker.run.remote(
                                fallback_messages,
                            )
                        )
                    else:
                        fallback_response = run_fallback_round(
                            fallback_agent,
                            fallback_messages,
                        )
                    fallback_output = _extract_last_text_from_response(fallback_response)
                    result = postprocess_result(
                        sample=item,
                        model_output=fallback_output,
                        prompt=messages_input[0]["content"],
                    )
                    result.update(
                        {
                            "planner_source": planner_source,
                            "planner_output": planner_output,
                            "candidate_code": candidate_code,
                            "compiled_code": compiled_code,
                            "failure_stage": failure_stage,
                            "error_message": error_message,
                            "planner_history": _response_to_serializable(planner_history),
                            "attempt_logs": attempt_logs,
                            "useful_observation": make_json_safe(useful_observation),
                            "execution_trace": make_json_safe(execution_trace),
                            "fallback_used": True,
                            "fallback_reason": failure_stage,
                            "fallback_history": _response_to_serializable(fallback_response),
                        }
                    )
                except Exception as fallback_exc:
                    result = _build_failure_result(
                        sample=item,
                        prompt=messages_input[0]["content"],
                        planner_source=planner_source,
                        planner_output=planner_output,
                        candidate_code=candidate_code,
                        compiled_code=compiled_code,
                        failure_stage=failure_stage,
                        error_message=error_message,
                        planner_history=planner_history,
                        attempt_logs=attempt_logs,
                        useful_observation=useful_observation,
                        execution_trace=execution_trace,
                        reasoner_input_package=reasoner_input_package,
                    )
                    result["fallback_used"] = True
                    result["fallback_reason"] = failure_stage
                    result["fallback_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"
            elif fallback_worker is not None:
                try:
                    fallback_messages = build_fallback_messages(
                        messages_input,
                        planner_output=planner_output,
                        candidate_code=candidate_code,
                        failure_stage=failure_stage,
                        error_message=error_message,
                    )
                    import ray
                    fallback_response = ray.get(
                        fallback_worker.run.remote(
                            fallback_messages,
                        )
                    )
                    fallback_output = _extract_last_text_from_response(fallback_response)
                    result = postprocess_result(
                        sample=item,
                        model_output=fallback_output,
                        prompt=messages_input[0]["content"],
                    )
                    result.update(
                        {
                            "planner_source": planner_source,
                            "planner_output": planner_output,
                            "candidate_code": candidate_code,
                            "compiled_code": compiled_code,
                            "failure_stage": failure_stage,
                            "error_message": error_message,
                            "planner_history": _response_to_serializable(planner_history),
                            "attempt_logs": attempt_logs,
                            "useful_observation": make_json_safe(useful_observation),
                            "execution_trace": make_json_safe(execution_trace),
                            "fallback_used": True,
                            "fallback_reason": failure_stage,
                            "fallback_history": _response_to_serializable(fallback_response),
                        }
                    )
                except Exception as fallback_exc:
                    result = _build_failure_result(
                        sample=item,
                        prompt=messages_input[0]["content"],
                        planner_source=planner_source,
                        planner_output=planner_output,
                        candidate_code=candidate_code,
                        compiled_code=compiled_code,
                        failure_stage=failure_stage,
                        error_message=error_message,
                        planner_history=planner_history,
                        attempt_logs=attempt_logs,
                        useful_observation=useful_observation,
                        execution_trace=execution_trace,
                        reasoner_input_package=reasoner_input_package,
                    )
                    result["fallback_used"] = True
                    result["fallback_reason"] = failure_stage
                    result["fallback_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"
            else:
                result = _build_failure_result(
                    sample=item,
                    prompt=messages_input[0]["content"],
                    planner_source=planner_source,
                    planner_output=planner_output,
                    candidate_code=candidate_code,
                    compiled_code=compiled_code,
                    failure_stage=failure_stage,
                    error_message=error_message,
                    planner_history=planner_history,
                    attempt_logs=attempt_logs,
                    useful_observation=useful_observation,
                    execution_trace=execution_trace,
                    reasoner_input_package=reasoner_input_package,
                )
                result["fallback_used"] = False
            result["media_input_info"] = media_input_info
            final_output.append(result)
            current_metrics = calculate_metrics(final_output)
            current_metrics["stage_metrics"] = calculate_stage_metrics(final_output)
            save_results(output_path, final_output, current_metrics)
            if extract_frame:
                for frame_path in cleanup_frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
            continue

        try:
            reasoner_messages, reasoner_input_package = build_reasoner_messages(
                messages_input,
                useful_observation,
                input_images,
                execution_trace,
                include_original_inputs=reasoner_include_original_inputs,
                drop_original_input_observations=reasoner_drop_input_observations,
            )
            reasoner_response = run_reasoner_round(reasoner, reasoner_messages)
            model_output = _extract_last_text_from_response(reasoner_response)
            result = postprocess_result(
                sample=item,
                model_output=model_output,
                prompt=messages_input[0]["content"],
            )
            result.update(
                {
                    "planner_source": planner_source,
                    "planner_output": planner_output,
                    "candidate_code": candidate_code,
                    "compiled_code": compiled_code,
                    "failure_stage": None,
                    "error_message": None,
                    "planner_history": _response_to_serializable(planner_history),
                    "attempt_logs": attempt_logs,
                    "reasoner_history": _response_to_serializable(reasoner_response),
                    "reasoning_analysis": extract_tag_text(model_output, "analysis"),
                    "useful_observation": make_json_safe(useful_observation),
                    "execution_trace": make_json_safe(execution_trace),
                    "reasoner_input_package": make_json_safe(reasoner_input_package),
                    "fallback_used": False,
                }
            )
        except Exception as exc:
            result = _build_failure_result(
                sample=item,
                prompt=messages_input[0]["content"],
                planner_source=planner_source,
                planner_output=planner_output,
                candidate_code=candidate_code,
                compiled_code=compiled_code,
                failure_stage="reasoner",
                error_message=f"{type(exc).__name__}: {exc}",
                planner_history=planner_history,
                attempt_logs=attempt_logs,
                useful_observation=useful_observation,
                execution_trace=execution_trace,
                reasoner_input_package=reasoner_input_package,
            )
            result["fallback_used"] = False
        result["media_input_info"] = media_input_info

        final_output.append(result)
        current_metrics = calculate_metrics(final_output)
        current_metrics["stage_metrics"] = calculate_stage_metrics(final_output)
        save_results(output_path, final_output, current_metrics)

        if extract_frame:
            for frame_path in cleanup_frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

    return final_output


def main(args: argparse.Namespace) -> None:
    if args.import_existing_plan and args.planner_model_path is None and args.repair_imported_plan:
        raise ValueError("--repair_imported_plan requires --planner_model_path.")

    raw_direct = (args.direct_vlm_types or "").strip()
    if raw_direct.lower() == "default":
        _direct_vlm_types = set(DEFAULT_DIRECT_VLM_QUESTION_TYPES)
    elif raw_direct:
        _direct_vlm_types = {t.strip() for t in raw_direct.split(",") if t.strip()}
    else:
        _direct_vlm_types = set()
    print("direct_vlm_types:", _direct_vlm_types)

    planner_llm_cfg = (
        build_llm_cfg(args.planner_model_path, args.planner_temp, args.planner_top_p)
        if args.planner_model_path
        else None
    )
    reasoner_llm_cfg = build_llm_cfg(
        args.reasoner_model_path, args.reasoner_temp, args.reasoner_top_p
    )
    fallback_llm_cfg = (
        build_llm_cfg(args.fallback_model_path, args.fallback_temp, args.fallback_top_p)
        if args.fallback_model_path
        else None
    )

    eval_data = load_benchmark(args.benchmark)
    if args.start_idx < 0:
        raise ValueError("--start_idx must be non-negative.")
    eval_data = eval_data[args.start_idx :]
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive when provided.")
        eval_data = eval_data[: args.max_samples]
    imported_plans_by_id = None
    if args.import_existing_plan:
        imported_plans_by_id = load_imported_plans(args.import_existing_plan)

    output_dir = os.path.join("eval_results", f"eval_{args.benchmark}")
    os.makedirs(output_dir, exist_ok=True)

    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive.")
    num_workers = min(args.num_workers, len(eval_data))

    import ray

    ray.init(ignore_reinit_error=True)
    fallback_worker = None
    if fallback_llm_cfg is not None and args.fallback_num_gpus > 0:
        FallbackWorkerRemote = ray.remote(max_concurrency=1)(FallbackWorker)
        fallback_worker = FallbackWorkerRemote.options(
            num_gpus=args.fallback_num_gpus
        ).remote(fallback_llm_cfg)

    chunks: List[List[Dict[str, Any]]] = []
    per_worker = len(eval_data) // num_workers
    for i in range(num_workers):
        if i == num_workers - 1:
            chunk = eval_data[i * per_worker :]
        else:
            chunk = eval_data[i * per_worker : (i + 1) * per_worker]
        if chunk:
            chunks.append(chunk)

    futures = []
    evaluate_partition_remote = ray.remote(evaluate_partition)
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"results_{args.model_type}_{i}.json")
        futures.append(
            evaluate_partition_remote.options(num_gpus=args.num_gpus_per_worker).remote(
                eval_data=chunk,
                output_path=output_path,
                benchmark_name=args.benchmark,
                planner_llm_cfg=planner_llm_cfg,
                reasoner_llm_cfg=reasoner_llm_cfg,
                imported_plans_by_id=imported_plans_by_id,
                max_plan_retries=args.max_plan_retries,
                repair_imported_plan=args.repair_imported_plan,
                fallback_llm_cfg=fallback_llm_cfg,
                fallback_worker=fallback_worker,
                debug_reasoning=args.debug_reasoning,
                reasoner_include_original_inputs=(
                    not args.reasoner_without_original_inputs
                ),
                reasoner_drop_input_observations=(
                    args.reasoner_drop_input_observations
                ),
                video_preview_frames=args.video_preview_frames,
                video_reconstruction_frames=args.video_reconstruction_frames,
                spatial_memory_cache_root=args.spatial_memory_cache_root,
                measuring_cache_root=args.measuring_cache_root,
                bev_fallback=args.bev_fallback,
                direct_vlm_types=_direct_vlm_types,
            )
        )

    ret = ray.get(futures)
    final_output: List[Dict[str, Any]] = []
    for item in ret:
        final_output.extend(item)

    final_metrics = calculate_metrics(final_output)
    final_metrics["stage_metrics"] = calculate_stage_metrics(final_output)
    save_results(
        os.path.join(output_dir, f"results_{args.model_type}.json"),
        final_output,
        final_metrics,
    )
    print(f"Finished evaluation for {args.benchmark}.")
    print(final_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the three-step DSL spatial reasoning pipeline.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--planner_model_path", type=str, default=None)
    parser.add_argument("--reasoner_model_path", type=str, required=True)
    parser.add_argument("--planner_temp", type=float, default=1.0)
    parser.add_argument("--planner_top_p", type=float, default=1.0)
    parser.add_argument("--reasoner_temp", type=float, default=1.0)
    parser.add_argument("--reasoner_top_p", type=float, default=1.0)
    parser.add_argument("--import_existing_plan", type=str, default=None)
    parser.add_argument("--repair_imported_plan", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_plan_retries", type=int, default=2)
    parser.add_argument("--fallback_model_path", type=str, default=None)
    parser.add_argument("--fallback_temp", type=float, default=0.0)
    parser.add_argument("--fallback_top_p", type=float, default=1.0)
    parser.add_argument("--fallback_num_gpus", type=float, default=0.0)
    parser.add_argument("--debug_reasoning", action="store_true")
    parser.add_argument("--reasoner_without_original_inputs", action="store_true")
    parser.add_argument("--reasoner_drop_input_observations", action="store_true")
    parser.add_argument("--video_preview_frames", type=int, default=16)
    parser.add_argument("--video_reconstruction_frames", type=int, default=64)
    parser.add_argument("--spatial_memory_cache_root", type=str, default=None)
    parser.add_argument("--measuring_cache_root", type=str, default=None,
                        help="Cache root (e.g. pi3x) for measuring/metric questions; other types use --spatial_memory_cache_root.")
    parser.add_argument("--bev_fallback", action="store_true",
                        help="On planner/executor failure after all retries, render a BEV of the scene and answer with SUMMARY_CONTEXT_FALLBACK.")
    parser.add_argument("--direct_vlm_types", type=str, default="",
                        help="Comma-separated question types answered directly from ordered frames (bypass plan). "
                             "Pass 'default' to use the built-in set (obj_appearance_order).")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpus_per_worker", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
