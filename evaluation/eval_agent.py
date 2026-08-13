import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from copy import deepcopy
# add workspace to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import ray
import requests
import torch
from tqdm import tqdm
from utils import clean_text, extract_video_frames, reward_fn
from qwen_agent.agents import Assistant
# add workspace to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from tools.llm_cfg import build_llm_cfg
from tools.prompt import PLANNER_SYSTEM_MESSAGE, PLANNER_PROMPT, TOOL_CATALOG


SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = {
    "multiple choice": " Please answer with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags.",
    "numerical": " Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags.",
    "regression": " Please answer with the only numerical value (e.g., 42, 3.14, etc.) within the <answer> </answer> tags.",
    "verbal": " Please answer the question simply within the <answer> </answer> tags",
}
MINDCUBE_HEADER = """
[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.
"""

MINDCUBE_COT_HEADER = """
[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
Please do step by step reasoning first, then give your final answer. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should be this format: '<think>(replace with your reasoning here)</think><answer>A. Above</answer>'.
"""

# MINDCUBE_COT_HEADER = """
# What are in the scene?
# """

CGMAP_GENERATION_PROMPT = """
[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. 
You will then create a detailed cognitive map representing the scene using a **10x10 grid coordinate system**. 
[Rules]
1. Focus ONLY on these categories of objects in the scene and mentioned in the question
2. Create a cognitive map with the following structure in the bird's view:
   - A 10x10 grid where [0, 0] is at the top-left corner and [9, 9] is at the bottom-right corner
   - up = towards the top of the grid (decreasing y)
   - right = towards the right of the grid (increasing x)
   - down = towards the bottom of the grid (increasing y)
   - left = towards the left of the grid (decreasing x)
   - Include positions of all objects from the specified categories
   - Estimate the center location (coordinates [x, y]) of each instance within provided categories
   - If a category contains multiple instances, include all of them
   - Object positions must maintain accurate relative spatial relationships
   - Combine and merge information from the images since they are pointing to the same scene, calibrating the object locations with grid coordinates accordingly
3. Carefully integrate information from all views to create a single coherent spatial representation.

[Answer Instruction]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in the following JSON format **before your reasoning**:
```json
{
    \"object_category_1\": {\"position\": [x, y]},
    \"object_category_2\": {\"position\": [x, y], \"facing\": \"direction\"}, # if the object is asked for orientation
    ...
}
```
2. Next, please also provide your reasons step by step in details, then provide *ONE* correct answer selecting from the options.
3. In general, your response's format should be like \"Based on my observation, the answer is:
<cogmap>(Replace with your cogmap here)</cogmap><think>(Replace with your reasoning here)</think><answer>(Replace with your answer here)</answer>\". Your option must be from the available options.\n
"""

TOOL_PLAN_PROMPT = """
Please plan how to use the existing tool to solve this problem, then call the tools accordingly. Try to make full use of these tools to avoid complex mathematical calculation. 
Tips:
Please use `set_reference_pose` properly to help your analysis. By setting a observation perspective and query more information from that point could help you to reduce a lot calculation. 
You might need to update the position for each entities after setting a reference pose by querying their pose again.
In the end, you can render different information ( (scene visualization, entities locations, camera trajectories, etc.) in a proper form (BEV, Perspective RGB, etc.) to help visualize the result.
"""

VSI_QUESTION_TYPE = {'obj_appearance_order', 'object_counting', 'object_size_estimation', 'room_size_estimation',
                     'object_abs_distance', 'object_rel_direction_hard', 'object_rel_direction_medium', 'object_rel_direction_easy',
                     'object_rel_distance', 'route_planning'}

SPATIALLADDER_THINK_PROMPT = (
    " Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', "
    "'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural "
    "language thought expressions. It's encouraged to include self-reflection or "
    "verification in the reasoning process."
)
SPATIALLADDER_POST_PROMPTS = {
    "multiple choice": (
        " Please provide your detailed reasoning between the <think> </think> tags, "
        "and then answer the question with the option's letter from the given choices "
        "(e.g., A, B, etc.) within the <answer> </answer> tags."
    ),
    "numerical": (
        " Please provide your detailed reasoning between the <think> </think> tags, "
        "and then answer the question with a numerical value (e.g., 42 or 3.1) "
        "within the <answer> </answer> tags."
    ),
    "regression": (
        " Please provide your detailed reasoning between the <think> </think> tags, "
        "and then answer the question with a numerical value (e.g., 42 or 3.1) "
        "within the <answer> </answer> tags."
    ),
}

def load_benchmark(benchmark_name):
    """Load benchmark dataset based on the provided name."""
    file_path = os.path.abspath(__file__)
    benchmark_annotation_path = os.path.join(os.path.dirname(file_path), "annotation", f"eval_{benchmark_name}.json")
    with open(benchmark_annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def is_spatialladder_model() -> bool:
    model_path = getattr(args, "model_path", "")
    model_type = getattr(args, "model_type", "")
    return "spatialladder" in model_path.lower() or "spatialladder" in model_type.lower()


def build_spatialladder_prompt(question: str, problem_type: str) -> str:
    post_prompt = SPATIALLADDER_POST_PROMPTS.get(
        problem_type,
        SPATIALLADDER_POST_PROMPTS["numerical"],
    )
    return question + SPATIALLADDER_THINK_PROMPT + post_prompt


def build_spatialladder_question(item) -> str:
    question = item["problem"]
    if item["problem_type"] == "multiple choice":
        question += " " + " ".join(item["options"])
    return question


def build_eval_question(item, use_cot=False, gen_cg_map=False, plan_tool=False, eval_planner=False):
    if item['original_question_type'] in VSI_QUESTION_TYPE:
        if gen_cg_map:
            question = CGMAP_GENERATION_PROMPT
        else:
            question = '[Task]\nYour task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.'
        # question += '\n[Answer Instruction]\n'
        # question += 'You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is ’A. Above’ from ’A. Above B. Under C. Front D. Behind’, your response should **only** be ’<answer>A. Above</answer>’.'
        question += '[Question]\n'
        if item["problem_type"] == "multiple choice":
            question += item["problem"] + " "
            for op in item["options"]:
                question += op + " "
        else:
            question += item["problem"]
        if use_cot:
            question += " You should first think about the reasoning process and then provide the answer. Use <answer>...</answer> to provide your final answer."
    elif item['original_question_type'] == "spatial reasoning":
        if not eval_planner:
            question = MINDCUBE_HEADER if not use_cot else MINDCUBE_COT_HEADER
            question += '[Question]\n'
            if item["problem_type"] == "multiple choice":
                question += item["problem"] + " "
                for op in item["options"]:
                    question += op + " "
            else:
                question += item["problem"]
        else:
            question = PLANNER_PROMPT.format(problem_description=item["problem"], tool_catalog=TOOL_CATALOG)
    else:
        if item["problem_type"] == "multiple choice":
            question = item["problem"] + "Options:\n"
            for op in item["options"]:
                question += op + "\n"
        else:
            question = item["problem"]

        if use_cot:
            question += " You should first think about the reasoning process and then provide the answer. Use <think>...</think> and <answer>...</answer> tags. "
    if plan_tool:
        question += TOOL_PLAN_PROMPT
    return question


def prepare_single_message_eval(item, frame_paths, use_cot=False, gen_cg_map=False, max_pixels=28*28*1280, plan_tool=False, eval_planner=False):
    """Prepare message structure for a single eval data point."""
    question = build_eval_question(
        item,
        use_cot=use_cot,
        gen_cg_map=gen_cg_map,
        plan_tool=plan_tool,
        eval_planner=eval_planner,
    )
    content = []
    if not args.drop_image:
        for frame_path in frame_paths:
            content.append({"image": frame_path})

    if is_spatialladder_model():
        prompt_text = build_spatialladder_prompt(
            build_spatialladder_question(item),
            item["problem_type"],
        )
    else:
        prompt_text = (
            SFT_QUESTION_TEMPLATE.format(Question=question)
            + SFT_TYPE_TEMPLATE[item["problem_type"]]
        )

    content.append({"text": prompt_text})
    msg = [{"role": "user", "content": content}]
    return msg


def prepare_single_message_eval_gemini_video(item, video_path, use_cot=False, gen_cg_map=False, plan_tool=False, eval_planner=False):
    question = build_eval_question(
        item,
        use_cot=use_cot,
        gen_cg_map=gen_cg_map,
        plan_tool=plan_tool,
        eval_planner=eval_planner,
    )
    prompt_text = (
        build_spatialladder_prompt(
            build_spatialladder_question(item),
            item["problem_type"],
        )
        if is_spatialladder_model()
        else SFT_QUESTION_TEMPLATE.format(Question=question)
    )
    content = [
        {"video": video_path},
        {"text": prompt_text},
    ]
    return [{"role": "user", "content": content}]


def _extract_text_from_gemini_generate_content_response(payload):
    texts = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return texts[-1] if texts else json.dumps(payload, ensure_ascii=False)


def run_gemini_direct_video_inference(llm_cfg, video_path, question):
    mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": video_b64,
                    }
                },
                {"text": question},
            ]
        }],
        "generationConfig": {
            "temperature": llm_cfg.get("generate_cfg", {}).get("temperature", 0.0),
            "topP": llm_cfg.get("generate_cfg", {}).get("top_p", 0.8),
        },
    }
    response = requests.post(
        f"{llm_cfg['model_server'].rstrip('/')}/models/{llm_cfg['model']}:generateContent",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": llm_cfg["api_key"],
        },
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    text = _extract_text_from_gemini_generate_content_response(data)
    return [
        {
            "role": "user",
            "content": [
                {"video": video_path},
                {"text": question},
            ],
        },
        {"role": "assistant", "content": text},
    ]
    

def postprocess_batch(batch_data, batch_output_text, prompts_text):
    batch_results = []
    for batch_idx, sample in enumerate(batch_data):
        model_output = batch_output_text[batch_idx]
        result_sample = {}
        result_sample['sample'] = sample.copy()
        result_sample["prompt"] = prompts_text[batch_idx]
        result_sample["model_output"] = model_output
        
        # --- if contains <answer> tags, extract answer, else use model_output as answer ---
        # if not isinstance(model_output, str):
        #     print(model_output)
        #     exit(0)
        clean_ans = clean_text(model_output)
        result_sample["cleaned_model_output"] = clean_ans
        
        # --- get cleaned gt answer ---
        clean_ans_gt = clean_text(sample.get("solution", ""))
        result_sample["cleaned_gt_answer"] = clean_ans_gt
        
        # --- calculate reward ---
        result_sample["reward"] = reward_fn(clean_ans_gt, clean_ans, sample['problem_type'])
        result_sample["correct"] = result_sample["reward"] == 1.0
        batch_results.append(result_sample)
    return batch_results

def calculate_metrics(results, skipped_count: int = 0):
    """Calculate metrics from a list of results."""
    mean_acc_rewards = [s["reward"] for s in results if s["sample"].get("problem_type") != "regression" and "reward" in s]
    mean_mra_rewards = [s["reward"] for s in results if s["sample"].get("problem_type") == "regression" and "reward" in s and s.get("prediction") != "error"]

    final_metrics = {"mean_acc": 0.0, "mean_mra": 0.0, "mean_all": 0.0, "skipped_count": skipped_count}
    if mean_acc_rewards:
            final_metrics["mean_acc"] = torch.tensor(mean_acc_rewards, dtype=torch.float32).mean().item()
    if mean_mra_rewards:
            final_metrics["mean_mra"] = torch.tensor(mean_mra_rewards, dtype=torch.float32).mean().item()
    if mean_acc_rewards or mean_mra_rewards:
        all_rewards = torch.cat([torch.tensor(mean_acc_rewards, dtype=torch.float32), torch.tensor(mean_mra_rewards, dtype=torch.float32)])
        final_metrics["mean_all"] = all_rewards.mean().item()
    return final_metrics

def save_results(output_path: str, results, final_acc, skipped_instances=None):
    """Save evaluation results to file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": results,
                    "final_acc": [final_acc],
                    "skipped_instances": skipped_instances or [],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing results to output file: {e}")

def clean_temp_image_files(response):
    """Clean up temporary image files generated during evaluation."""
    for msg in response:
        content = msg.get("content", [])
        for item in content:
            if isinstance(item, dict) and "image" in item:
                img_path = item["image"]
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        print(f"Deleted temporary image file: {img_path}")
                    except Exception as e:
                        print(f"Error deleting temporary image file {img_path}: {e}")

@ray.remote(num_gpus=1)
def evaluate_benchmark(eval_data, llm_cfg, output_path, benchmark_name, 
                       use_cot=False, gen_cg_map=False, use_tool=False, plan_tool=False, eval_planner=False): 
    """Evaluate model on a specific dataset. Batching rule: accumulate up to BATCH_SIZE samples; if a video's resolution is different from the first video in the current batch, flush the batch before this sample."""
    system_instruction = '''You are an helpful AI assistant. Please use the tools to help you analyze the spatial arrangement in the scene and then provide the answer.'''
    if is_spatialladder_model():
        system_instruction = "You are a helpful assistant."
    if eval_planner:
        system_instruction = PLANNER_SYSTEM_MESSAGE
        tools = []
    elif use_tool:
        from tools.agent_tools import FrameSelection, Query3DObjectPosition, BuildStaticSpatialMemory, QueryCameraPose, RenderCameraPoseEgocentricBEV, RenderObjectPoseEgocentricBEV, SetViewpoint, RenderSemanticBEV
        tools = getattr(args, "tool_list", 'frame_selection').split(',')   # `code_interpreter` is a built-in tool for executing code.
        # import ipdb; ipdb.set_trace()  # Debug
        # tools = ['build_static_spatial_memory', 'query_camera_pose', 'render_camera_pose_egocentric_bev']  # `code_interpreter` is a built-in tool for executing code.
        # tools = ['build_static_spatial_memory', 'query_camera_pose', 'query_3d_object_position', 'render_object_pose_egocentric_bev']  # `code_interpreter` is a built-in tool for executing code.
        # tools = ['build_static_spatial_memory', 'query_camera_pose', 'set_reference_pose', 'render_bev']  # `code_interpreter` is a built-in tool for executing code.
        # tools = ['frame_selection']
        # tools = []
    else:
        tools = []  # `code_interpreter` is a built-in tool for executing code.
   
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=system_instruction,
        # [!Optional] We provide `analysis_prompt` to enable VL conduct deep analysis. Otherwise use system_message='' to simply enable the tools.
    )
    
    final_output = []
    skipped_instances = []
    gemini_direct_video_enabled = (
        getattr(args, "gemini_video_input_mode", "frames") == "video"
        and "gemini" in args.model_path.lower()
    )

    # Helper function to process the accumulated batch and flush results
    def handle_current_instance(
        current_instance,
        processed_idx,
        frame_paths,
        use_cot=False,
        gen_cg_map=False,
        plan_tool=False,
        raw_video_path=None,
    ):
        """Run inference on one accumulated batch, update metrics & save."""
        nonlocal final_output, skipped_instances
        if not current_instance:
            return
        use_gemini_direct_video = (
            gemini_direct_video_enabled
            and raw_video_path is not None
            and not use_tool
            and not args.eval_planner
        )
        if use_gemini_direct_video:
            messages_input = prepare_single_message_eval_gemini_video(
                current_instance,
                raw_video_path,
                use_cot=use_cot,
                gen_cg_map=gen_cg_map,
                plan_tool=plan_tool,
                eval_planner=args.eval_planner,
            )
            question_text = build_eval_question(
                current_instance,
                use_cot=use_cot,
                gen_cg_map=gen_cg_map,
                plan_tool=plan_tool,
                eval_planner=args.eval_planner,
            )
        else:
            messages_input = prepare_single_message_eval(
                current_instance,  
                frame_paths,
                use_cot=use_cot,
                gen_cg_map=gen_cg_map,
                plan_tool=plan_tool,
                eval_planner=args.eval_planner
            )
        # import ipdb; ipdb.set_trace()  # Debug
        total_attempts = 3
        for attempt in range(total_attempts):
            try:
                if use_gemini_direct_video:
                    response = run_gemini_direct_video_inference(llm_cfg, raw_video_path, question_text)
                else:
                    response = agent.run_nonstream(messages=messages_input)
                break  # If successful, exit the retry loop
            except Exception as e:
                print(f"Error during agent inference on index {processed_idx}, attempt {attempt + 1}/{total_attempts}: {e}")
                if attempt == total_attempts - 1:
                    print(f"Failed to get response after {total_attempts} attempts for index {processed_idx}. Skipping this sample.")
                    skipped_instances.append({
                        "index": processed_idx,
                        "problem_id": current_instance.get("problem_id"),
                        "path": current_instance.get("path"),
                        "error": str(e),
                    })
                    current_metrics = calculate_metrics(final_output, skipped_count=len(skipped_instances))
                    save_results(output_path, final_output, current_metrics, skipped_instances=skipped_instances)
                    return
                else:
                    time.sleep(5)  # Wait before retrying
        # clean_temp_image_files(response)
        # import ipdb; ipdb.set_trace()  # Debug
        result = response[-1]['content']
        if not isinstance(result, str):
            print(f"Warning: Model output for index {processed_idx} is not a string. Converting to string for post-processing.")
            result = str(result)
        result_batched = postprocess_batch([current_instance], [result], [messages_input[0]['content']])[0]
        result_batched["full_log"] = response  # Store the full conversation log for later analysis
        final_output.extend([result_batched])

        # --- calculate metrics ---
        current_metrics = calculate_metrics(final_output, skipped_count=len(skipped_instances))
        save_results(output_path, final_output, current_metrics, skipped_instances=skipped_instances)
        processed_count = len(final_output)
        print(
            f"Processed up to overall index {processed_idx}, saved {processed_count} samples."
        )

    for idx, item in enumerate(tqdm(eval_data, desc=f"Processing {benchmark_name} batches")):
        # if not item['original_question_type'].startswith('object_rel_direction'):
        #     continue
        extract_frame = False
        raw_video_path = None
        if not isinstance(item['path'], list) and (item['path'].endswith('.mp4') or item['path'].endswith('.avi')):
            video_path = item['path']
            raw_video_path = video_path
            if gemini_direct_video_enabled and not use_tool and not args.eval_planner:
                frame_paths = []
            else:
                num_frames = args.nframes
                frame_paths = extract_video_frames(video_path, num_frames)
                extract_frame = True
        elif isinstance(item['path'], list) and item['path'] and all(p.endswith(('.jpg', '.png')) for p in item['path']):
            frame_paths = item['path']
        else:
            print(f"Warning: Unsupported media type for item at index {idx}, skipping.")
            continue
        # import ipdb; ipdb.set_trace()  # Debug
        handle_current_instance(
            item,
            idx,
            frame_paths,
            use_cot=use_cot,
            gen_cg_map=gen_cg_map,
            plan_tool=plan_tool,
            raw_video_path=raw_video_path,
        )
        # clear temp files
        if extract_frame:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

    return {"results": final_output, "skipped_instances": skipped_instances}


def main(args):
    llm_cfg = build_llm_cfg(args.model_path, args.temp, args.top_p)
    output_dir = os.path.join("eval_results", f"eval_{args.benchmark}")
    os.makedirs(output_dir, exist_ok=True)
    eval_data = load_benchmark(args.benchmark)
    
    if args.start_idx is not None:
        eval_data = eval_data[args.start_idx:]
    if args.max_samples is not None:
        eval_data = eval_data[:args.max_samples]

    n_gpu = torch.cuda.device_count()
    
    ray.init()
    features = []
    per_gpu_data_length = len(eval_data) // n_gpu
    for i in range(n_gpu):
        if i == n_gpu - 1:  # last gpu takes the remaining data
            data_gpu = eval_data[i * per_gpu_data_length :]
        else:
            data_gpu = eval_data[i * per_gpu_data_length : (i + 1) * per_gpu_data_length]
        output_path_gpu = os.path.join(output_dir, f"results_{args.model_type}_{i}.json")
        features.append(evaluate_benchmark.remote(
            data_gpu, 
            llm_cfg=llm_cfg, 
            output_path=output_path_gpu,
            benchmark_name=args.benchmark, 
            use_cot=args.use_cot,
            gen_cg_map=args.gen_cg_map,   # whether to generate cognitive map first
            use_tool=args.use_tool,  # whether to use tools during evaluation
            plan_tool=args.plan_tool and args.use_tool,  # whether to plan tool usage during evaluation, only valid when use_tool is True
            eval_planner=args.eval_planner  # whether to evaluate planner
        )
    )

    ret = ray.get(features)
    final_output = []
    skipped_instances = []
    for item in ret:
        final_output.extend(item["results"])
        skipped_instances.extend(item.get("skipped_instances", []))
        
    # --- calculate final metrics ---
    final_acc_dict = calculate_metrics(final_output, skipped_count=len(skipped_instances))
    save_results(
        os.path.join(output_dir, f"results_{args.model_type}.json"),
        final_output,
        final_acc_dict,
        skipped_instances=skipped_instances,
    )
    print(f"Finished evaluation for {args.benchmark}.")
    print(f"Final Metrics: {final_acc_dict}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on Standard Benchmark dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory for video files.")
    parser.add_argument("--model_type", type=str, default="spatial-mllm-tiny", help="Type of the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--nframes", type=int, default=16, help="Number of frames to sample from each video.")
    parser.add_argument("--benchmark", type=str, default="vsibench", help="Benchmark dataset to evaluate on.")
    parser.add_argument("--vggt_ratio", type=float, default=1.0, help="Ratio of VGG features to image features.")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.001, help="Top-p sampling parameter.")
    parser.add_argument("--use_cot", action="store_true", help="Use Chain of Thought (CoT) reasoning in evaluation.")
    parser.add_argument("--plan_tool", action="store_true", help="Whether to plan tool usage during evaluation.")
    parser.add_argument("--gen_cg_map", action="store_true", help="Whether to generate cognitive map first before answering.")
    parser.add_argument("--use_tool", action="store_true", help="Whether to use tools during evaluation.")
    parser.add_argument("--tool_list", type=str, default="", help="Comma-separated list of tools to use during evaluation, e.g., 'build_static_spatial_memory,query_camera_pose'.")
    parser.add_argument("--eval_planner", action="store_true", help="Whether to evaluate a planner.")
    parser.add_argument("--frame_selector", type=str, default=None, help="Frame selection strategy: uniform, topk_clip, etc.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for evaluation.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate.")
    parser.add_argument("--drop_image", action="store_true", help="Whether to drop image input and only evaluate the question answering ability.")
    parser.add_argument(
        "--gemini_video_input_mode",
        type=str,
        default="frames",
        choices=["frames", "video"],
        help="For Gemini models only: choose between frame extraction and direct video input.",
    )

    args = parser.parse_args()


    main(args)
