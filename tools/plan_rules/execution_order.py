import ast
from typing import List

from tools.plan_ast_utils import get_call_argument, get_stmt_call_name
from tools.plan_rules.catalog import TOOL_ARG_SCHEMAS


RENDER_TOOL_NAMES = {"render_ego_rgb", "render_rgb_bev", "render_semantic_bev"}
CAMERA_MOTION_TOOL_NAMES = {"turn_camera", "step_camera"}


def validate_no_dangling_camera_motion(tree: ast.Module) -> List[str]:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return []

    fn = tree.body[0]
    non_return_stmts = [stmt for stmt in fn.body if not isinstance(stmt, ast.Return)]
    stmt_call_names = [get_stmt_call_name(stmt) for stmt in non_return_stmts]

    errors: List[str] = []
    for idx, (stmt, call_name) in enumerate(zip(non_return_stmts, stmt_call_names)):
        if call_name not in CAMERA_MOTION_TOOL_NAMES:
            continue
        has_later_render = any(
            later_name in RENDER_TOOL_NAMES for later_name in stmt_call_names[idx + 1 :]
        )
        if not has_later_render:
            errors.append(
                f"`{call_name}` at line {stmt.lineno} is a dangling camera motion. "
                "Every camera motion must be followed by at least one later render call "
                "(`render_ego_rgb`, `render_rgb_bev`, or `render_semantic_bev`)."
            )
    return errors


def _iter_top_level_call_nodes(fn: ast.FunctionDef):
    for stmt in fn.body:
        if isinstance(stmt, ast.Return):
            continue
        call_node = None
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
        if call_node is not None and isinstance(call_node.func, ast.Name):
            yield stmt, call_node


def _is_camera_only_semantic_bev_call(call_node: ast.Call) -> bool:
    objects_arg = get_call_argument(
        call_node,
        "objects",
        TOOL_ARG_SCHEMAS["render_semantic_bev"]["params"],
    )
    queried_objects_arg = get_call_argument(
        call_node,
        "queried_objects",
        TOOL_ARG_SCHEMAS["render_semantic_bev"]["params"],
    )
    camera_indices_arg = get_call_argument(
        call_node,
        "camera_indices",
        TOOL_ARG_SCHEMAS["render_semantic_bev"]["params"],
    )
    if objects_arg is not None:
        if isinstance(objects_arg, ast.Constant) and objects_arg.value is None:
            pass
        else:
            return False
    if queried_objects_arg is not None:
        if isinstance(queried_objects_arg, ast.Constant) and queried_objects_arg.value is None:
            pass
        else:
            return False
    return camera_indices_arg is not None


def validate_no_zero_gain_object_related_semantic_bev(
    tree: ast.Module,
    inferred_problem_type: str | None,
) -> List[str]:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return []
    if inferred_problem_type != "object-related":
        return []

    fn = tree.body[0]
    render_calls: List[tuple[ast.stmt, ast.Call]] = []
    for stmt, call_node in _iter_top_level_call_nodes(fn):
        if call_node.func.id in RENDER_TOOL_NAMES:
            render_calls.append((stmt, call_node))

    if not render_calls:
        return []

    all_camera_only_semantic = True
    for _, call_node in render_calls:
        if call_node.func.id != "render_semantic_bev":
            all_camera_only_semantic = False
            break
        if not _is_camera_only_semantic_bev_call(call_node):
            all_camera_only_semantic = False
            break

    if not all_camera_only_semantic:
        return []

    first_stmt = render_calls[0][0]
    return [
        "This object-related plan has zero information gain: all rendered evidence is camera-only "
        f"`render_semantic_bev` without object annotations (first such render at line {first_stmt.lineno}). "
        "For `problem_type=\"object-related\"`, you must render object-relevant evidence such as "
        "`render_ego_rgb`, `render_rgb_bev`, `render_semantic_bev(..., objects=[...])`, or "
        "`render_semantic_bev(..., queried_objects=queried)`."
    ]
