import re
import torch
from typing import Optional, Dict, Any, Callable, List
# from Levenshtein import ratio
# from qwen_vl_utils import process_vision_info, extract_vision_info
# from functools import wraps
import cv2
from pathlib import Path


def extract_think(output_str: str) -> str:
    """Extract the thinking process from model output."""
    pattern = r"<think>\s*(.*?)\s*</think>"
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer(text: str) -> str:
    """Extract the answer from model output."""
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def clean_text(text, exclue_chars=["\n", "\r"]):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ["\n", "\r"]:
            # If there is a space before the newline, remove the newline
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    # Remove leading and trailing spaces and convert to lowercase
    cleaned_ans = text.strip().rstrip(".").lower()
    # import ipdb; ipdb.set_trace()
    # only keep the options in the answer if it is a multiple choice question
    if len(cleaned_ans) > 1 and cleaned_ans[1] == "." and not str.isnumeric(cleaned_ans[0]):
        # Deal with cases like "A. 1, B. 2, C. 3"
        cleaned_ans = cleaned_ans[0]
    return cleaned_ans

def normalize_number(num_str: str) -> Optional[float]:
    """Convert string number to float, handling commas."""
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """Calculate mean relative accuracy for regression tasks."""
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

    thresholds = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


def vsi_reward(clean_ans_gt: str, clean_ans_pred: str, question_type: str) -> float:
    """Calculate reward based on question type and model output."""
    if question_type == "multiple choice":
        return 1.0 if clean_ans_pred.strip() == clean_ans_gt.strip() else 0.0
    elif question_type == "regression" or question_type == "numerical":
        gt_number = normalize_number(clean_ans_gt)
        pred_number = normalize_number(clean_ans_pred)
        if gt_number is None or pred_number is None:
            return 0.0
        return mean_relative_accuracy(pred_number, gt_number)
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


def reward_fn(clean_ans_gt: str, clean_ans_pred: str, question_type: str) -> float:
    try:
        if question_type == "multiple choice":
            return 1.0 if clean_ans_pred.strip() == clean_ans_gt.strip() else 0.0
        elif question_type == "numerical":
            gt_has_decimal = ("." in clean_ans_gt) or ("," in clean_ans_gt)
            out_has_decimal = ("." in clean_ans_pred) or ("," in clean_ans_pred)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(clean_ans_gt)
            out_number = normalize_number(clean_ans_pred)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
        elif question_type == "regression":
            gt_number = normalize_number(clean_ans_gt)
            out_number = normalize_number(clean_ans_pred)
            if gt_number is None or out_number is None:
                return 0.0
            mra = mean_relative_accuracy(out_number, gt_number)
            return mra
        else:
            return 0.0
    except Exception as e:
        return 0.0

def extract_video_frames(
    video_path: str,
    fps: Optional[float] = None,
    num_frames: Optional[int] = None,
) -> List[str]:
    """Extract frames from video with optional FPS and frame-count budgets.

    Rules:
    - If only one of `fps` / `num_frames` is provided, sampling is controlled by that value.
    - If both are provided, sample count is the minimum implied by the two budgets.
    - Backward compatibility: `extract_video_frames(video_path, 7)` is interpreted as
      `num_frames=7`, not `fps=7`.
    """
    if num_frames is None and isinstance(fps, int):
        num_frames = fps
        fps = None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid video with zero frames: {video_path}")
    if original_fps <= 0:
        original_fps = 30.0

    total_duration = total_frames / original_fps
    fps_limited_count = None
    if fps is not None:
        if fps <= 0:
            cap.release()
            raise ValueError(f"`fps` must be positive, got {fps}")
        fps_frame_step = max(int(round(original_fps / fps)), 1)
        fps_limited_count = max((total_frames + fps_frame_step - 1) // fps_frame_step, 1)

    target_frames = None
    if num_frames is not None:
        if num_frames <= 0:
            cap.release()
            raise ValueError(f"`num_frames` must be positive, got {num_frames}")
        target_frames = num_frames

    if fps_limited_count is not None and target_frames is not None:
        target_frames = min(fps_limited_count, target_frames)
    elif fps_limited_count is not None:
        target_frames = fps_limited_count
    elif target_frames is not None:
        target_frames = target_frames
    else:
        target_frames = 10

    target_frames = max(min(target_frames, total_frames), 1)
    frame_interval = total_frames / target_frames

    frame_paths = []
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    video_filename = Path(video_path).stem

    for i in range(target_frames):
        frame_idx = int(i * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"{video_filename}_frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

    cap.release()
    print(
        f"Extracted {len(frame_paths)} frames from video "
        f"(duration: {total_duration:.2f}s, original fps: {original_fps:.2f}, "
        f"fps budget: {fps}, num_frames budget: {num_frames}, final sampled: {target_frames})"
    )
    return frame_paths
