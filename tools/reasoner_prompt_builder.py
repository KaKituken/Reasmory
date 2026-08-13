import ast
import copy
import json
from typing import Any, Dict, List, Optional

from qwen_agent.llm.schema import Message

from tools.prompt import SUMMARY_CONTEXT


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_user_question(messages: List[Message]) -> str:
    last_content = messages[-1]["content"]
    for item in last_content:
        text = _get_attr_or_key(item, "text")
        if text:
            return str(text)
    return ""


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}

    serialized_fields: Dict[str, Any] = {}
    for field_name in ("role", "name", "text", "image", "content"):
        field_value = getattr(value, field_name, None)
        if field_value is not None:
            serialized_fields[field_name] = _make_json_safe(field_value)
    if serialized_fields:
        return serialized_fields

    return str(value)


def _observation_item_to_content(item: Any) -> List[Dict[str, str]]:
    content: List[Dict[str, str]] = []

    if isinstance(item, str):
        if item.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            content.append({"image": item})
        else:
            content.append({"text": item})
        return content

    if isinstance(item, list):
        for sub_item in item:
            content.extend(_observation_item_to_content(sub_item))
        return content

    if isinstance(item, dict):
        if "image" in item and isinstance(item["image"], str):
            content.append({"image": item["image"]})
        if "images" in item and isinstance(item["images"], list):
            for image in item["images"]:
                if isinstance(image, str):
                    content.append({"image": image})
        meta_payload = {k: v for k, v in item.items() if k not in {"image", "images"}}
        if meta_payload:
            content.append({"text": json.dumps(meta_payload, ensure_ascii=False, indent=2)})
        return content

    content.append({"text": json.dumps(item, ensure_ascii=False, indent=2)})
    return content


def _find_trace_for_image(
    image_path: str, execution_trace: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    for trace in reversed(execution_trace):
        result = trace.get("result")
        if isinstance(result, str) and result == image_path:
            return trace
        if isinstance(result, list) and image_path in result:
            return trace
        if isinstance(result, dict):
            if result.get("image") == image_path:
                return trace
            if image_path in result.get("images", []):
                return trace
    return None


def _find_trace_for_object(
    item: Any, execution_trace: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    for trace in reversed(execution_trace):
        if trace.get("result") == item:
            return trace
    return None


def _find_trace_index(
    trace: Optional[Dict[str, Any]], execution_trace: List[Dict[str, Any]]
) -> Optional[int]:
    if trace is None:
        return None
    step_index = trace.get("step_index")
    for idx, cur_trace in enumerate(execution_trace):
        if cur_trace.get("step_index") == step_index:
            return idx
    return None


def _normalize_trace_argument(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized: Any = value
    for _ in range(2):
        if not isinstance(normalized, str):
            break
        try:
            parsed = ast.literal_eval(normalized)
        except Exception:
            break
        if parsed == normalized:
            break
        normalized = parsed
    return normalized


def _trace_motion_to_text(trace: Dict[str, Any]) -> str:
    tool_name = trace.get("tool_name")
    args = trace.get("arguments", {})
    if tool_name == "turn_camera":
        direction = _normalize_trace_argument(args.get("direction", "unknown"))
        angle = _normalize_trace_argument(args.get("angle"))
        if angle is not None:
            return f"turn {direction} by {angle} degrees"
        return f"turn {direction}"
    if tool_name == "step_camera":
        direction = _normalize_trace_argument(args.get("direction", "unknown"))
        return f"step {direction}"
    return trace.get("call_signature", tool_name or "unknown_action")


def _motion_sequence_to_natural_language(motion_traces: List[Dict[str, Any]]) -> str:
    if not motion_traces:
        return "no additional camera motion"
    actions = [_trace_motion_to_text(trace) for trace in motion_traces]
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"first {actions[0]}, then {actions[1]}"
    return "first " + ", then ".join(actions[:-1]) + f", and finally {actions[-1]}"


def _format_frame_indices_as_images(frame_indices: Any) -> str:
    if not isinstance(frame_indices, list) or not frame_indices:
        return "queried input images"
    # frame_indices are 1-based (see planner prompt: "Frame indices are 1-based"),
    # so map directly to the 1-based "image N" label -- do NOT add 1.
    image_ids = [f"image {int(frame_idx)}" for frame_idx in frame_indices]
    if len(image_ids) == 1:
        return image_ids[0]
    if len(image_ids) == 2:
        return f"{image_ids[0]} and {image_ids[1]}"
    return ", ".join(image_ids[:-1]) + f", and {image_ids[-1]}"


def _format_category_names(category_names: Any) -> str:
    if not isinstance(category_names, list) or not category_names:
        return "queried objects"
    names = [str(name) for name in category_names]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _infer_initial_position_text(
    trace_idx: Optional[int], execution_trace: List[Dict[str, Any]]
) -> str:
    if trace_idx is None:
        return "Initial position: an original camera view from the input."

    nearest_set_idx = None
    for idx in range(trace_idx, -1, -1):
        if execution_trace[idx].get("tool_name") == "set_viewpoint":
            nearest_set_idx = idx
            break

    if nearest_set_idx is None:
        return "Initial position: the default active viewpoint in the current reference frame."

    source_text = "Initial position: a manually specified reference viewpoint."
    for idx in range(nearest_set_idx - 1, -1, -1):
        trace = execution_trace[idx]
        tool_name = trace.get("tool_name")
        if tool_name == "query_camera_pose":
            frame_indices = trace.get("arguments", {}).get("frame_indices")
            image_text = _format_frame_indices_as_images(frame_indices)
            source_text = (
                f"Initial position: aligned to the queried camera viewpoint from {image_text}, "
                "facing that camera's forward direction."
            )
            break
        if tool_name == "query_3d_object_position":
            category_names = trace.get("arguments", {}).get("category_names")
            category_text = _format_category_names(category_names)
            source_text = (
                f"Initial position: aligned to a queried object-centered viewpoint around {category_text}."
            )
            break
        if tool_name == "set_viewpoint":
            break

    return source_text


def _describe_motion_before_observation(
    trace_idx: Optional[int], execution_trace: List[Dict[str, Any]]
) -> str:
    if trace_idx is None:
        return "Camera motion before this image: none."

    nearest_set_idx = None
    for idx in range(trace_idx, -1, -1):
        if execution_trace[idx].get("tool_name") == "set_viewpoint":
            nearest_set_idx = idx
            break

    start_idx = 0 if nearest_set_idx is None else nearest_set_idx + 1
    motion_traces = [
        trace
        for trace in execution_trace[start_idx : trace_idx + 1]
        if trace.get("tool_name") in {"turn_camera", "step_camera"}
    ]
    return (
        "Camera motion before this image: "
        f"{_motion_sequence_to_natural_language(motion_traces)}."
    )


def _collect_future_motion_traces(
    trace_idx: Optional[int], execution_trace: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if trace_idx is None:
        return []

    future_traces: List[Dict[str, Any]] = []
    for idx in range(trace_idx + 1, len(execution_trace)):
        trace = execution_trace[idx]
        tool_name = trace.get("tool_name")
        if tool_name in {"turn_camera", "step_camera"}:
            future_traces.append(trace)
    return future_traces


def _merge_future_rotation_direction(motion_traces: List[Dict[str, Any]]) -> Optional[str]:
    yaw_turns = 0
    has_rotation = False

    for trace in motion_traces:
        if trace.get("tool_name") != "turn_camera":
            continue
        args = trace.get("arguments", {})
        direction = _normalize_trace_argument(args.get("direction"))
        angle = _normalize_trace_argument(args.get("angle"))
        has_rotation = True

        if direction == "left":
            quarter_turns = max(1, int(round((angle or 90) / 90.0)))
            yaw_turns -= quarter_turns
        elif direction == "right":
            quarter_turns = max(1, int(round((angle or 90) / 90.0)))
            yaw_turns += quarter_turns
        elif direction == "back":
            yaw_turns += 2
        else:
            continue

    if not has_rotation:
        return None

    yaw_turns %= 4
    direction_map = {0: "front", 1: "right", 2: "back", 3: "left"}
    return direction_map[yaw_turns]


def _build_usage_instruction_debug(
    trace: Optional[Dict[str, Any]],
    execution_trace: List[Dict[str, Any]],
    observation_kind: str,
) -> Dict[str, Any]:
    trace_idx = _find_trace_index(trace, execution_trace)
    future_motion_traces = _collect_future_motion_traces(trace_idx, execution_trace)
    merged_direction = _merge_future_rotation_direction(future_motion_traces)
    has_future_rotation = any(
        motion_trace.get("tool_name") == "turn_camera" for motion_trace in future_motion_traces
    )
    return {
        "observation_kind": observation_kind,
        "trace_idx": trace_idx,
        "matched_step_index": None if trace is None else trace.get("step_index"),
        "matched_tool_name": None if trace is None else trace.get("tool_name"),
        "future_motion_traces": [
            {
                "step_index": motion_trace.get("step_index"),
                "tool_name": motion_trace.get("tool_name"),
                "call_signature": motion_trace.get("call_signature"),
                "arguments": _make_json_safe(motion_trace.get("arguments", {})),
                "normalized_direction": _normalize_trace_argument(
                    motion_trace.get("arguments", {}).get("direction")
                ),
                "normalized_angle": _normalize_trace_argument(
                    motion_trace.get("arguments", {}).get("angle")
                ),
            }
            for motion_trace in future_motion_traces
        ],
        "has_future_rotation": has_future_rotation,
        "merged_direction": merged_direction,
    }


def _build_usage_instruction(
    trace: Optional[Dict[str, Any]],
    execution_trace: List[Dict[str, Any]],
    observation_kind: str,
) -> str:
    debug_info = _build_usage_instruction_debug(trace, execution_trace, observation_kind)
    merged_direction = debug_info["merged_direction"]
    has_future_rotation = debug_info["has_future_rotation"]

    if not has_future_rotation or merged_direction in {None, "front"}:
        if observation_kind == "render_rgb_bev_group":
            return (
                "How to use this observation: after this observation, there is no remaining camera "
                "rotation that changes the final facing direction. To answer the final question, you "
                "should observe what is in front of the ego marker."
            )
        if observation_kind == "render_ego_rgb":
            return (
                "How to use this observation: this ego-centric view has ALREADY been rotated to face "
                "the direction the question asks about, so read the salient object ahead in this image "
                "(a wall / floor filling the frame is usually just the backdrop, not the answer). Do "
                "NOT mentally rotate the image further. If this render is blurry or ambiguous, confirm "
                "the object against the original input view that faces the same direction."
            )
        return (
            "How to use this observation: after this observation, there is no remaining camera "
            "rotation that changes the final facing direction. To answer the final question, you "
            "should observe what is in front of you."
        )

    if observation_kind == "render_rgb_bev_group":
        return (
            "How to use this observation: the remaining camera rotation sequence after this "
            f"observation reduces to facing {merged_direction}. This is a grouped BEV zoom series, "
            f"so merge the scales mentally and inspect the {merged_direction} direction relative to "
            "the ego marker."
        )
    if observation_kind == "render_ego_rgb":
        return (
            "How to use this observation: the remaining camera rotation sequence after this "
            f"observation reduces to facing {merged_direction}. This ego-centric RGB view is an "
            f"intermediate product, so to answer the final question, focus on what would be to your "
            f"{merged_direction}."
        )
    return (
        "How to use this observation: the remaining camera rotation sequence after this "
        f"observation reduces to facing {merged_direction}. Focus on the {merged_direction} "
        "direction."
    )


def _describe_observation(
    item: Any,
    observation_index: int,
    input_images: List[str],
    execution_trace: List[Dict[str, Any]],
) -> str:
    prefix = f"Observation {observation_index}:"

    if isinstance(item, dict) and "compiler_shortcut" in item:
        shortcut = item.get("compiler_shortcut") or {}
        frame_index = shortcut.get("frame_index")
        trace = _find_trace_for_object(item, execution_trace)
        return (
            f"{prefix}\n"
            f"Source: original input image {frame_index} reused as a compiler shortcut for "
            "a `render_ego_rgb` request.\n"
            f"{_infer_initial_position_text(_find_trace_index(trace, execution_trace), execution_trace)}\n"
            f"{_describe_motion_before_observation(_find_trace_index(trace, execution_trace), execution_trace)}\n"
            f"{_build_usage_instruction(trace, execution_trace, observation_kind='render_ego_rgb')}"
        )

    if isinstance(item, str):
        if item in input_images:
            image_index = input_images.index(item) + 1
            return (
                f"{prefix}\n"
                f"Source: original input image {image_index} from the question.\n"
                "Initial position: this is one of the given camera views, before any tool-driven "
                "viewpoint transformation.\n"
                "Camera motion before this image: none.\n"
                "How to use this observation: use it as an original visual anchor when comparing "
                "against transformed or rendered views."
            )
        trace = _find_trace_for_image(item, execution_trace)
        if trace:
            tool_name = trace.get("tool_name", "tool")
            call_signature = trace.get("call_signature", tool_name)
            trace_idx = _find_trace_index(trace, execution_trace)
            return (
                f"{prefix}\n"
                f"Source: rendered image produced at execution step {trace['step_index']} by "
                f"`{call_signature}`.\n"
                f"{_infer_initial_position_text(trace_idx, execution_trace)}\n"
                f"{_describe_motion_before_observation(trace_idx, execution_trace)}\n"
                f"{_build_usage_instruction(trace, execution_trace, observation_kind=tool_name)}"
            )
        return f"{prefix} useful image artifact."

    trace = _find_trace_for_object(item, execution_trace)
    if trace:
        trace_idx = _find_trace_index(trace, execution_trace)
        return (
            f"{prefix}\n"
            f"Source: structured result produced at execution step {trace['step_index']} by "
            f"`{trace['call_signature']}`.\n"
            f"{_infer_initial_position_text(trace_idx, execution_trace)}\n"
            f"{_describe_motion_before_observation(trace_idx, execution_trace)}\n"
            "How to use this observation: use this structured output to understand the scene layout "
            "or queried geometry under the transformed frame."
        )

    if isinstance(item, dict) and ("image" in item or "images" in item):
        return f"{prefix} rendered observation with attached metadata."

    return f"{prefix} structured useful observation."


def _describe_grouped_observation(
    items: List[Any],
    observation_index: int,
    input_images: List[str],
    execution_trace: List[Dict[str, Any]],
) -> Optional[str]:
    if not items or not all(isinstance(item, str) for item in items):
        return None
    if any(item in input_images for item in items):
        return None

    traces = [_find_trace_for_image(item, execution_trace) for item in items]
    if any(trace is None for trace in traces):
        return None

    first_trace = traces[0]
    if not all(
        trace.get("step_index") == first_trace.get("step_index")
        and trace.get("tool_name") == first_trace.get("tool_name")
        for trace in traces
    ):
        return None

    prefix = f"Observation {observation_index}:"
    call_signature = first_trace.get("call_signature", first_trace.get("tool_name", "tool"))
    tool_name = first_trace.get("tool_name")
    trace_idx = _find_trace_index(first_trace, execution_trace)

    if tool_name == "render_rgb_bev":
        return (
            f"{prefix}\n"
            f"Source: grouped RGB BEV observation produced at execution step "
            f"{first_trace['step_index']} by `{call_signature}`. This observation contains "
            f"{len(items)} BEV images at different scales.\n"
            f"{_infer_initial_position_text(trace_idx, execution_trace)}\n"
            f"{_describe_motion_before_observation(trace_idx, execution_trace)}\n"
            f"{_build_usage_instruction(first_trace, execution_trace, observation_kind='render_rgb_bev_group')}"
        )

    return (
        f"{prefix}\n"
        f"Source: this grouped observation contains {len(items)} images produced together at execution step "
        f"{first_trace['step_index']} by `{call_signature}`.\n"
        f"{_infer_initial_position_text(trace_idx, execution_trace)}\n"
        f"{_describe_motion_before_observation(trace_idx, execution_trace)}\n"
        "How to use this observation: interpret these images as closely related outputs from the same step."
    )


def _summarize_trace_step(trace: Dict[str, Any]) -> str:
    step_index = trace.get("step_index", "?")
    tool_name = trace.get("tool_name", "unknown_tool")
    call_signature = trace.get("call_signature") or tool_name
    view_context = trace.get("view_context") or []

    if tool_name == "query_camera_pose":
        frame_indices = trace.get("arguments", {}).get("frame_indices")
        return (
            f"Step {step_index}: queried camera poses for frames {frame_indices} via "
            f"`{call_signature}`."
        )
    if tool_name == "set_viewpoint":
        return (
            f"Step {step_index}: aligned the active viewpoint with a reference pose using "
            f"`{call_signature}`."
        )
    if tool_name in {"turn_camera", "step_camera"}:
        return f"Step {step_index}: refined the active viewpoint with `{call_signature}`."
    if tool_name.startswith("render_"):
        if view_context:
            return (
                f"Step {step_index}: rendered evidence with `{call_signature}` after viewpoint "
                f"context {' -> '.join(view_context)}."
            )
        return f"Step {step_index}: rendered evidence with `{call_signature}`."
    if view_context:
        return (
            f"Step {step_index}: executed `{call_signature}` under viewpoint context "
            f"{' -> '.join(view_context)}."
        )
    return f"Step {step_index}: executed `{call_signature}`."


def _summarize_execution_trace(execution_trace: List[Dict[str, Any]]) -> str:
    if not execution_trace:
        return "No execution trace is available."
    return "\n".join(_summarize_trace_step(trace) for trace in execution_trace)


def _build_observation_content(
    useful_observation: Any,
    input_images: List[str],
    execution_trace: List[Dict[str, Any]],
    drop_original_input_observations: bool = False,
) -> List[Dict[str, str]]:
    content: List[Dict[str, str]] = []
    items = useful_observation if isinstance(useful_observation, list) else [useful_observation]
    observation_index = 1

    for item in items:
        if isinstance(item, list):
            filtered_group = item
            if drop_original_input_observations:
                filtered_group = [
                    sub_item
                    for sub_item in item
                    if not (isinstance(sub_item, str) and sub_item in input_images)
                ]
            if not filtered_group:
                continue

            grouped_text = _describe_grouped_observation(
                filtered_group,
                observation_index=observation_index,
                input_images=input_images,
                execution_trace=execution_trace,
            )
            if grouped_text is not None:
                content.append({"text": grouped_text})
                for sub_item in filtered_group:
                    content.extend(_observation_item_to_content(sub_item))
                observation_index += 1
                continue

            for sub_item in filtered_group:
                content.append(
                    {
                        "text": _describe_observation(
                            sub_item,
                            observation_index=observation_index,
                            input_images=input_images,
                            execution_trace=execution_trace,
                        )
                    }
                )
                content.extend(_observation_item_to_content(sub_item))
                observation_index += 1
            continue

        if drop_original_input_observations and isinstance(item, str) and item in input_images:
            continue

        content.append(
            {
                "text": _describe_observation(
                    item,
                    observation_index=observation_index,
                    input_images=input_images,
                    execution_trace=execution_trace,
                )
            }
        )
        content.extend(_observation_item_to_content(item))
        observation_index += 1

    return content


def _build_observation_debug_info(
    useful_observation: Any,
    input_images: List[str],
    execution_trace: List[Dict[str, Any]],
    drop_original_input_observations: bool = False,
) -> List[Dict[str, Any]]:
    debug_items: List[Dict[str, Any]] = []
    items = useful_observation if isinstance(useful_observation, list) else [useful_observation]
    observation_index = 1

    for item in items:
        if isinstance(item, list):
            filtered_group = item
            if drop_original_input_observations:
                filtered_group = [
                    sub_item
                    for sub_item in item
                    if not (isinstance(sub_item, str) and sub_item in input_images)
                ]
            if not filtered_group:
                continue

            grouped_text = _describe_grouped_observation(
                filtered_group,
                observation_index=observation_index,
                input_images=input_images,
                execution_trace=execution_trace,
            )
            if grouped_text is not None:
                first_trace = _find_trace_for_image(filtered_group[0], execution_trace)
                debug_items.append(
                    {
                        "observation_index": observation_index,
                        "grouped": True,
                        "kind": "render_rgb_bev_group",
                        "matched_trace": None if first_trace is None else _make_json_safe(first_trace),
                        "usage_debug": _build_usage_instruction_debug(
                            first_trace,
                            execution_trace,
                            observation_kind="render_rgb_bev_group",
                        ),
                        "items": _make_json_safe(filtered_group),
                    }
                )
                observation_index += 1
                continue

            for sub_item in filtered_group:
                trace = (
                    _find_trace_for_image(sub_item, execution_trace)
                    if isinstance(sub_item, str)
                    else _find_trace_for_object(sub_item, execution_trace)
                )
                observation_kind = "unknown"
                if isinstance(sub_item, str) and trace is not None:
                    observation_kind = trace.get("tool_name", "unknown")
                debug_items.append(
                    {
                        "observation_index": observation_index,
                        "grouped": False,
                        "kind": observation_kind,
                        "matched_trace": None if trace is None else _make_json_safe(trace),
                        "usage_debug": _build_usage_instruction_debug(
                            trace,
                            execution_trace,
                            observation_kind=observation_kind,
                        ),
                        "item": _make_json_safe(sub_item),
                    }
                )
                observation_index += 1
            continue

        if drop_original_input_observations and isinstance(item, str) and item in input_images:
            continue

        trace = (
            _find_trace_for_image(item, execution_trace)
            if isinstance(item, str)
            else _find_trace_for_object(item, execution_trace)
        )
        observation_kind = "original_input" if isinstance(item, str) and item in input_images else "unknown"
        if trace is not None:
            observation_kind = trace.get("tool_name", observation_kind)
        debug_items.append(
            {
                "observation_index": observation_index,
                "grouped": False,
                "kind": observation_kind,
                "matched_trace": None if trace is None else _make_json_safe(trace),
                "usage_debug": _build_usage_instruction_debug(
                    trace,
                    execution_trace,
                    observation_kind=observation_kind,
                ),
                "item": _make_json_safe(item),
            }
        )
        observation_index += 1

    return debug_items


def _extract_reasoner_message_stats(messages: List[Any]) -> Dict[str, Any]:
    image_paths: List[str] = []
    text_items: List[str] = []
    for msg in messages:
        content = _get_attr_or_key(msg, "content")
        if not isinstance(content, list):
            continue
        for item in content:
            image = _get_attr_or_key(item, "image")
            text = _get_attr_or_key(item, "text")
            if isinstance(image, str):
                image_paths.append(image)
            if isinstance(text, str):
                text_items.append(text)
    return {
        "image_paths": image_paths,
        "num_images": len(image_paths),
        "num_text_items": len(text_items),
        "text_preview": text_items,
    }


def build_reasoner_messages(
    original_messages: List[Message],
    useful_observation: Any,
    input_images: List[str],
    execution_trace: List[Dict[str, Any]],
    include_original_inputs: bool = True,
    drop_original_input_observations: bool = False,
    summary_context: str = None,
) -> tuple[List[Message], Dict[str, Any]]:
    if summary_context is None:
        summary_context = SUMMARY_CONTEXT
    if include_original_inputs:
        reasoner_messages = copy.deepcopy(original_messages)
    else:
        question_text = _extract_user_question(original_messages)
        reasoner_messages = [Message("user", [{"text": question_text}])]

    observation_content = _build_observation_content(
        useful_observation,
        input_images=input_images,
        execution_trace=execution_trace,
        drop_original_input_observations=drop_original_input_observations,
    )
    observation_debug = _build_observation_debug_info(
        useful_observation,
        input_images=input_images,
        execution_trace=execution_trace,
        drop_original_input_observations=drop_original_input_observations,
    )
    execution_trace_summary = _summarize_execution_trace(execution_trace)
    summary_content: List[Dict[str, str]] = [{"text": summary_context}]
    summary_content.extend(observation_content)

    reasoner_messages.append(Message("user", summary_content))
    serialized_messages = _make_json_safe(reasoner_messages)
    reasoner_input_package = {
        "include_original_inputs": include_original_inputs,
        "drop_original_input_observations": drop_original_input_observations,
        "summary_context": summary_context,
        "observation_content": _make_json_safe(observation_content),
        "observation_debug": _make_json_safe(observation_debug),
        "execution_trace_summary": execution_trace_summary,
        "summary_message_content": _make_json_safe(summary_content),
        "messages": serialized_messages,
        "message_stats": _extract_reasoner_message_stats(serialized_messages),
    }
    return reasoner_messages, reasoner_input_package
