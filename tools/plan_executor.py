import ast
import json
from typing import Any, Dict, List, Tuple

from qwen_agent.llm.schema import Message

from tools.agent_tools import (
    BuildDynamicSpatialMemory,
    BuildStaticSpatialMemory,
    Query3DObjectPosition,
    QueryCameraPose,
    RenderEgoRGB,
    RenderRGBBEV,
    RenderSemanticBEV,
    SafeSelect,
    SetViewpoint,
    StepCamera,
    TurnCamera,
)
from tools.plan_compiler import optimize_plan_code


class PlanExecutionError(Exception):
    pass


def _extract_input_images_from_messages(messages: List[Message]) -> List[str]:
    image_paths: List[str] = []
    for msg in messages:
        content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                image_paths.append(item["image"])
    return image_paths


def _extract_input_videos_from_messages(messages: List[Message]) -> List[str]:
    video_paths: List[str] = []
    for msg in messages:
        content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("video"):
                video_paths.append(item["video"])
    return video_paths


def _strip_runtime_only_annotations(code: str) -> str:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in list(node.args.args) + list(node.args.kwonlyargs):
                arg.annotation = None
            if node.args.vararg is not None:
                node.args.vararg.annotation = None
            if node.args.kwarg is not None:
                node.args.kwarg.annotation = None
            node.returns = None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _format_call_signature(name: str, arguments: Dict[str, Any]) -> str:
    parts = [f"{key}={repr(value)}" for key, value in arguments.items()]
    return f"{name}({', '.join(parts)})"


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _json_or_text(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _normalize_tool_output(name: str, raw_items: Any) -> Any:
    if raw_items is None:
        return None
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    texts: List[str] = []
    images: List[str] = []
    parsed_json_items: List[Any] = []

    for item in raw_items:
        text = _get_attr_or_key(item, "text")
        image = _get_attr_or_key(item, "image")
        if text:
            text = str(text)
            texts.append(text)
            parsed = _json_or_text(text)
            if not isinstance(parsed, str):
                parsed_json_items.append(parsed)
        if image:
            images.append(str(image))

    errors = [text for text in texts if text.strip().lower().startswith("error:")]
    if errors:
        raise PlanExecutionError(f"{name} failed: {errors[0]}")

    if name == "build_static_spatial_memory":
        if parsed_json_items:
            return parsed_json_items[0]
        raise PlanExecutionError("build_static_spatial_memory did not return session metadata.")

    if name == "build_dynamic_spatial_memory":
        raise PlanExecutionError("build_dynamic_spatial_memory is declared but not implemented in agent_tools.py.")

    if name in {"query_camera_pose", "query_3d_object_position"}:
        if parsed_json_items:
            result = parsed_json_items[-1]
            if name == "query_3d_object_position":
                if isinstance(result, dict) and "category_positions" in result:
                    category_positions = result["category_positions"]
                    if isinstance(category_positions, dict):
                        query_cache_id = result.get("__query_cache_id__")
                        if isinstance(query_cache_id, str):
                            category_positions = dict(category_positions)
                            category_positions["__query_cache_id__"] = query_cache_id
                        return category_positions
            return result
        raise PlanExecutionError(f"{name} did not return a structured JSON payload.")

    if name == "safe_select":
        if parsed_json_items:
            return parsed_json_items[-1]
        raise PlanExecutionError("safe_select did not return a selection.")

    if name in {"set_viewpoint", "step_camera", "turn_camera"}:
        return {"messages": texts}

    if name == "render_ego_rgb":
        if images:
            return images[0]
        raise PlanExecutionError("render_ego_rgb did not return an image.")

    if name == "render_rgb_bev":
        if images:
            return images
        raise PlanExecutionError("render_rgb_bev did not return any images.")

    if name == "render_semantic_bev":
        result: Dict[str, Any] = {}
        if images:
            result["image"] = images[0]
            result["images"] = images
        if parsed_json_items:
            meta = parsed_json_items[-1]
            if isinstance(meta, dict):
                result.update(meta)
            result["meta"] = meta
        if result:
            return result
        raise PlanExecutionError("render_semantic_bev did not return an image or metadata.")

    return {
        "texts": texts,
        "images": images,
        "json": parsed_json_items,
    }


class ToolRuntime:
    def __init__(
        self,
        messages: List[Message],
        precomputed_spatial_memory_path: str | None = None,
    ):
        self.messages = messages
        self.precomputed_spatial_memory_path = precomputed_spatial_memory_path
        self.execution_trace: List[Dict[str, Any]] = []
        self.current_view_actions: List[str] = []
        self.tools = {
            "build_static_spatial_memory": BuildStaticSpatialMemory(),
            "build_dynamic_spatial_memory": BuildDynamicSpatialMemory(),
            "query_camera_pose": QueryCameraPose(),
            "query_3d_object_position": Query3DObjectPosition(),
            "set_viewpoint": SetViewpoint(),
            "safe_select": SafeSelect(),
            "step_camera": StepCamera(),
            "turn_camera": TurnCamera(),
            "render_ego_rgb": RenderEgoRGB(),
            "render_rgb_bev": RenderRGBBEV(),
            "render_semantic_bev": RenderSemanticBEV(),
        }

    def _execute_tool(self, name: str, **kwargs):
        tool = self.tools[name]
        if getattr(tool.__class__, "call", None) is None:
            raise PlanExecutionError(f"{name} does not implement `call`.")
        raw_result = tool.call(
            params=json.dumps(kwargs, ensure_ascii=False),
            messages=self.messages,
            precomputed_spatial_memory_path=self.precomputed_spatial_memory_path,
        )
        return _normalize_tool_output(name, raw_result)

    def _record_trace(
        self,
        name: str,
        arguments: Dict[str, Any],
        result: Any,
        call_signature: str | None = None,
        synthetic: bool = False,
        executed_via: str | None = None,
    ) -> Any:
        if call_signature is None:
            call_signature = _format_call_signature(name, arguments)

        if name in {"build_static_spatial_memory", "build_dynamic_spatial_memory"}:
            self.current_view_actions = []
        elif name in {"set_viewpoint", "step_camera", "turn_camera"}:
            self.current_view_actions.append(call_signature)

        self.execution_trace.append(
            {
                "step_index": len(self.execution_trace) + 1,
                "tool_name": name,
                "call_signature": call_signature,
                "arguments": arguments,
                "view_context": list(self.current_view_actions),
                "result": result,
                "synthetic": synthetic,
                "executed_via": executed_via,
            }
        )
        return result

    def run_tool(self, name: str, **kwargs):
        normalized = self._execute_tool(name, **kwargs)
        return self._record_trace(name, kwargs, normalized)

    def compiler_shortcut_render_ego_rgb(
        self,
        frame_index: int,
        input_images: List[str],
        original_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        if frame_index < 1 or frame_index > len(input_images):
            raise PlanExecutionError(
                f"Compiler shortcut requested input image {frame_index}, but only {len(input_images)} inputs exist."
            )
        image_path = input_images[frame_index - 1]
        result = {
            "image": image_path,
            "compiler_shortcut": {
                "source": "input_image",
                "frame_index": frame_index,
            },
        }
        self._record_trace(
            "render_ego_rgb",
            original_call.get("arguments", {}),
            result,
            call_signature=original_call.get("call_signature"),
            synthetic=True,
            executed_via=f"compiler_shortcut_input_image[{frame_index}]",
        )
        return result

    def compiler_apply_turn_sequence(
        self,
        original_calls: List[Dict[str, Any]],
        merged_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        executed_via = "compiler_noop"
        if merged_kwargs:
            self._execute_tool("turn_camera", **merged_kwargs)
            executed_via = _format_call_signature("turn_camera", merged_kwargs)
        result = {"messages": ["compiler preserved original turn sequence"]}
        for original_call in original_calls:
            self._record_trace(
                "turn_camera",
                original_call.get("arguments", {}),
                result,
                call_signature=original_call.get("call_signature"),
                synthetic=True,
                executed_via=executed_via,
            )
        return result

    def compiler_apply_step_sequence(
        self,
        original_calls: List[Dict[str, Any]],
        reduced_kwargs_list: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        reduced_kwargs_list = reduced_kwargs_list or []
        executed_ops: List[str] = []
        for kwargs in reduced_kwargs_list:
            self._execute_tool("step_camera", **kwargs)
            executed_ops.append(_format_call_signature("step_camera", kwargs))
        result = {"messages": ["compiler preserved original step sequence"]}
        executed_via = " -> ".join(executed_ops) if executed_ops else "compiler_noop"
        for original_call in original_calls:
            self._record_trace(
                "step_camera",
                original_call.get("arguments", {}),
                result,
                call_signature=original_call.get("call_signature"),
                synthetic=True,
                executed_via=executed_via,
            )
        return result

    def build_exec_env(self) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "__builtins__": {},
            "List": List,
            "str": str,
        }
        for tool_name in self.tools:
            env[tool_name] = self._make_tool_wrapper(tool_name)
        env["_compiler_shortcut_render_ego_rgb"] = self.compiler_shortcut_render_ego_rgb
        env["_compiler_apply_turn_sequence"] = self.compiler_apply_turn_sequence
        env["_compiler_apply_step_sequence"] = self.compiler_apply_step_sequence
        return env

    def _make_tool_wrapper(self, tool_name: str):
        def _wrapper(**kwargs):
            return self.run_tool(tool_name, **kwargs)

        _wrapper.__name__ = tool_name
        return _wrapper


def execute_plan(
    code: str,
    input_images: List[str],
    messages: List[Message],
    video_input: bool = False,
    decomposition: Dict[str, Any] | None = None,
    precomputed_spatial_memory_path: str | None = None,
) -> Tuple[Any, List[Dict[str, Any]], str]:
    compiled_code = optimize_plan_code(code, decomposition=decomposition)
    runtime_code = _strip_runtime_only_annotations(compiled_code)
    runtime = ToolRuntime(
        messages,
        precomputed_spatial_memory_path=precomputed_spatial_memory_path,
    )
    exec_env = runtime.build_exec_env()
    local_env: Dict[str, Any] = {}
    exec(runtime_code, exec_env, local_env)
    plan_fn = local_env.get("plan_to_solve_problem") or exec_env.get("plan_to_solve_problem")
    if plan_fn is None:
        raise PlanExecutionError("Verified code did not define `plan_to_solve_problem`.")
    input_images = _extract_input_images_from_messages(messages) or input_images
    fn_arg_name = plan_fn.__code__.co_varnames[0] if getattr(plan_fn, "__code__", None) and plan_fn.__code__.co_argcount >= 1 else "input_images"
    if fn_arg_name == "input_video" or video_input:
        input_videos = _extract_input_videos_from_messages(messages)
        if not input_videos:
            raise PlanExecutionError(
                "The plan expects `input_video`, but no video item was found in the input messages."
            )
        useful_observation = plan_fn(input_videos[0])
    else:
        useful_observation = plan_fn(input_images)
    return useful_observation, runtime.execution_trace, compiled_code


def format_execution_feedback(previous_code: str, error_text: str) -> str:
    return (
        "Your previous Python plan passed verification but failed during execution.\n\n"
        f"Execution error:\n{error_text}\n\n"
        "Please regenerate the full function with a corrected plan.\n"
        "Rules:\n"
        "- Output exactly one Python code block.\n"
        "- Define exactly one function `plan_to_solve_problem(input_images: List[str])` or `plan_to_solve_problem(input_video: str)`.\n"
        "- Use only allowed tool functions.\n"
        "- End with `return useful_observation`.\n"
        "- Fix the execution issue instead of repeating the same failing code.\n\n"
        "Previous code:\n"
        "```python\n"
        f"{previous_code}\n"
        "```"
    )
