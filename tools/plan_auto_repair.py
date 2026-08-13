import ast
import copy
from typing import Any, Dict, List, Optional, Tuple

from tools.plan_ast_utils import extract_plan_stages_from_tree, get_call_argument
from tools.plan_rules.catalog import TOOL_ARG_SCHEMAS
from tools.plan_rules.decomposition import (
    extract_expected_stage2_operations,
    extract_stage2_camera_operations_from_tree,
    is_global_view_decomposition,
)


def _get_plan_function(tree: ast.Module) -> Optional[ast.FunctionDef]:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    return tree.body[0]


def _get_stage2_stmt_indices(fn: ast.FunctionDef) -> List[int]:
    non_return_indices = [i for i, stmt in enumerate(fn.body) if not isinstance(stmt, ast.Return)]
    non_return_stmts = [fn.body[i] for i in non_return_indices]
    stage_info = extract_plan_stages_from_tree(ast.Module(body=[fn], type_ignores=[]))
    boundary_index = stage_info["boundary_statement_index"]
    if boundary_index is None:
        return non_return_indices
    stage2_non_return = non_return_indices[boundary_index + 1 :]
    return stage2_non_return


def _get_call_from_stmt(stmt: ast.stmt) -> Optional[ast.Call]:
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        return stmt.value
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    return None


def _get_stage2_camera_motion_stmt_indices(fn: ast.FunctionDef) -> List[int]:
    indices: List[int] = []
    for idx in _get_stage2_stmt_indices(fn):
        stmt = fn.body[idx]
        call = _get_call_from_stmt(stmt)
        if call is None or not isinstance(call.func, ast.Name):
            continue
        if call.func.id in {"turn_camera", "step_camera"}:
            indices.append(idx)
    return indices


def _build_motion_stmt(tool_name: str, direction: str) -> ast.stmt:
    call = ast.Call(
        func=ast.Name(id=tool_name, ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="session_id", value=ast.Name(id="session_id", ctx=ast.Load())),
            ast.keyword(arg="direction", value=ast.Constant(value=direction)),
        ],
    )
    if tool_name == "turn_camera" and direction != "back":
        call.keywords.append(ast.keyword(arg="angle", value=ast.Constant(value=90)))
    return ast.Expr(value=call)


def _append_renders_after_last_motion(fn: ast.FunctionDef) -> bool:
    motion_indices = _get_stage2_camera_motion_stmt_indices(fn)
    if not motion_indices:
        return False

    useful_idx = None
    for idx, stmt in enumerate(fn.body):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "useful_observation"
        ):
            useful_idx = idx
            break
    if useful_idx is None:
        return False

    insert_at = motion_indices[-1] + 1
    ego_assign = ast.parse(
        'auto_repair_ego_view = render_ego_rgb(session_id=session_id)'
    ).body[0]
    bev_assign = ast.parse(
        'auto_repair_bev = render_rgb_bev(session_id=session_id)'
    ).body[0]
    fn.body[insert_at:insert_at] = [ego_assign, bev_assign]

    if useful_idx >= insert_at:
        useful_idx += 2
    useful_stmt = fn.body[useful_idx]
    if not isinstance(useful_stmt, ast.Assign):
        return False
    if isinstance(useful_stmt.value, ast.List):
        useful_stmt.value.elts.extend(
            [
                ast.Name(id="auto_repair_ego_view", ctx=ast.Load()),
                ast.Name(id="auto_repair_bev", ctx=ast.Load()),
            ]
        )
    else:
        useful_stmt.value = ast.BinOp(
            left=copy.deepcopy(useful_stmt.value),
            op=ast.Add(),
            right=ast.List(
                elts=[
                    ast.Name(id="auto_repair_ego_view", ctx=ast.Load()),
                    ast.Name(id="auto_repair_bev", ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
        )
    return True


def _repair_stage2_motion_suffix(
    fn: ast.FunctionDef,
    expected_ops: List[Tuple[str, str]],
) -> bool:
    motion_indices = _get_stage2_camera_motion_stmt_indices(fn)
    actual_ops = []
    for idx in motion_indices:
        call = _get_call_from_stmt(fn.body[idx])
        if call is None or not isinstance(call.func, ast.Name):
            return False
        direction_expr = get_call_argument(
            call,
            "direction",
            TOOL_ARG_SCHEMAS[call.func.id]["params"],
        )
        if not (isinstance(direction_expr, ast.Constant) and isinstance(direction_expr.value, str)):
            return False
        actual_ops.append((call.func.id, direction_expr.value.lower()))

    actual_dirs = [direction for _, direction in actual_ops]
    expected_dirs = [direction for _, direction in expected_ops]
    if actual_dirs == expected_dirs:
        return False

    if not motion_indices and not actual_ops:
        return False

    if actual_dirs == expected_dirs[: len(actual_dirs)] and len(expected_dirs) > len(actual_dirs):
        insert_at = motion_indices[-1] + 1
        missing_ops = expected_ops[len(actual_ops) :]
        new_stmts = [_build_motion_stmt(tool_name, direction) for tool_name, direction in missing_ops]
        fn.body[insert_at:insert_at] = new_stmts
        return True

    if expected_dirs == actual_dirs[: len(expected_dirs)] and len(actual_dirs) > len(expected_dirs):
        excess = len(actual_dirs) - len(expected_dirs)
        del_indices = motion_indices[-excess:]
        for idx in reversed(del_indices):
            del fn.body[idx]
        return True

    if (
        len(actual_ops) == len(expected_ops)
        and len(actual_ops) >= 1
        and actual_dirs[:-1] == expected_dirs[:-1]
    ):
        last_idx = motion_indices[-1]
        fn.body[last_idx] = _build_motion_stmt(expected_ops[-1][0], expected_ops[-1][1])
        return True

    return False


def _repair_return_useful_observation(fn: ast.FunctionDef) -> bool:
    if not fn.body:
        return False
    last_stmt = fn.body[-1]
    if not isinstance(last_stmt, ast.Return) or last_stmt.value is None:
        return False
    if isinstance(last_stmt.value, ast.Name) and last_stmt.value.id == "useful_observation":
        return False

    assign_stmt = ast.Assign(
        targets=[ast.Name(id="useful_observation", ctx=ast.Store())],
        value=copy.deepcopy(last_stmt.value),
    )
    return_stmt = ast.Return(value=ast.Name(id="useful_observation", ctx=ast.Load()))
    fn.body[-1:] = [assign_stmt, return_stmt]
    return True


def _repair_global_view_missing_set_viewpoint(fn: ast.FunctionDef) -> bool:
    if any(
        isinstance(stmt, (ast.Assign, ast.Expr))
        and isinstance(_get_call_from_stmt(stmt), ast.Call)
        and isinstance(_get_call_from_stmt(stmt).func, ast.Name)
        and _get_call_from_stmt(stmt).func.id == "set_viewpoint"
        for stmt in fn.body
    ):
        return False

    insert_at = None
    for idx, stmt in enumerate(fn.body):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "session_id"
        ):
            insert_at = idx + 1
            break
    if insert_at is None:
        return False

    repair_stmts = ast.parse(
        'auto_repair_ref_pose = query_camera_pose(session_id=session_id, frame_indices=[1])[0]\n'
        'set_viewpoint(session_id=session_id, origin=auto_repair_ref_pose["position"], '
        'forward=auto_repair_ref_pose["forward"], up=auto_repair_ref_pose["up"])'
    ).body
    fn.body[insert_at:insert_at] = repair_stmts
    return True


def _literal_integer_value(expr: ast.AST) -> Optional[int]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        if isinstance(expr.operand, ast.Constant) and isinstance(expr.operand.value, int):
            return -expr.operand.value
    return None


def _literal_integer_list_value(expr: ast.AST) -> Optional[List[int]]:
    if isinstance(expr, (ast.List, ast.Tuple)):
        values: List[int] = []
        for elt in expr.elts:
            value = _literal_integer_value(elt)
            if value is None:
                return None
            values.append(value)
        return values

    if not isinstance(expr, ast.Call):
        return None
    if not isinstance(expr.func, ast.Name) or expr.func.id != "list":
        return None
    if len(expr.args) != 1 or expr.keywords:
        return None
    range_call = expr.args[0]
    if not isinstance(range_call, ast.Call):
        return None
    if not isinstance(range_call.func, ast.Name) or range_call.func.id != "range":
        return None
    if range_call.keywords or not 1 <= len(range_call.args) <= 3:
        return None
    args = [_literal_integer_value(arg) for arg in range_call.args]
    if any(arg is None for arg in args):
        return None
    try:
        return list(range(*args))
    except ValueError:
        return None


def _literal_integer_list_expr(values: List[int]) -> ast.List:
    return ast.List(
        elts=[ast.Constant(value=value) for value in values],
        ctx=ast.Load(),
    )


def _find_simple_assignment(fn: ast.FunctionDef, name: str) -> Optional[ast.Assign]:
    for stmt in fn.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        ):
            return stmt
    return None


def _name_used_only_as_render_semantic_bev_camera_indices(fn: ast.FunctionDef, name: str) -> bool:
    parent: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    seen_load = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Name) or node.id != name:
            continue
        if isinstance(node.ctx, ast.Store):
            continue
        seen_load = True
        kw = parent.get(node)
        call = parent.get(kw) if isinstance(kw, ast.keyword) else None
        if not (
            isinstance(kw, ast.keyword)
            and kw.arg == "camera_indices"
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "render_semantic_bev"
        ):
            return False
    return seen_load


def _truncate_indices(values: List[int], max_valid_index: int) -> List[int]:
    return [value for value in values if 1 <= value <= max_valid_index]


def _repair_out_of_range_render_semantic_bev_camera_indices(
    fn: ast.FunctionDef,
    input_image_count: Optional[int],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if input_image_count is None:
        return False, {"skipped": "missing_input_image_count"}

    repaired = False
    repairs: List[Dict[str, Any]] = []
    for stmt in fn.body:
        call = _get_call_from_stmt(stmt)
        if call is None or not isinstance(call.func, ast.Name) or call.func.id != "render_semantic_bev":
            continue
        camera_kw = next((kw for kw in call.keywords if kw.arg == "camera_indices"), None)
        if camera_kw is None:
            continue

        values = _literal_integer_list_value(camera_kw.value)
        repair_target = "inline"
        assignment_stmt = None
        if values is None and isinstance(camera_kw.value, ast.Name):
            repair_target = camera_kw.value.id
            if not _name_used_only_as_render_semantic_bev_camera_indices(fn, repair_target):
                continue
            assignment_stmt = _find_simple_assignment(fn, repair_target)
            if assignment_stmt is None:
                continue
            values = _literal_integer_list_value(assignment_stmt.value)

        if values is None or not values or max(values) <= input_image_count:
            continue

        truncated = _truncate_indices(values, input_image_count)
        if not truncated:
            continue

        if assignment_stmt is not None:
            assignment_stmt.value = _literal_integer_list_expr(truncated)
        else:
            camera_kw.value = _literal_integer_list_expr(truncated)
        repaired = True
        repairs.append(
            {
                "target": repair_target,
                "original_indices": values,
                "truncated_indices": truncated,
                "max_valid_index": input_image_count,
            }
        )

    if not repaired:
        return False, {"skipped": "no_repairable_render_semantic_bev_camera_indices"}
    return True, {
        "rule": "repair_out_of_range_render_semantic_bev_camera_indices",
        "repairs": repairs,
    }


def try_auto_repair_plan(
    code: str,
    verification_error: str,
    decomposition: Optional[Dict[str, Any]],
    input_image_count: Optional[int] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    stripped_error = verification_error.strip()
    if stripped_error.startswith("Syntax error at line"):
        return None, {"skipped": "input_not_parseable"}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, {"skipped": "input_not_parseable"}
    fn = _get_plan_function(tree)
    if fn is None:
        return None, {"skipped": "not_single_function"}

    if stripped_error == "The last statement must be exactly `return useful_observation`.":
        if not _repair_return_useful_observation(fn):
            return None, {"skipped": "return_repair_not_applicable"}
        ast.fix_missing_locations(tree)
        return (
            ast.unparse(tree),
            {
                "rule": "repair_return_useful_observation",
            },
        )

    if (
        "render_semantic_bev" in stripped_error
        and "uses `camera_indices` with max camera/frame index" in stripped_error
    ):
        repaired, repair_info = _repair_out_of_range_render_semantic_bev_camera_indices(
            fn,
            input_image_count,
        )
        if not repaired:
            return None, repair_info
        ast.fix_missing_locations(tree)
        return ast.unparse(tree), repair_info

    if (
        decomposition
        and is_global_view_decomposition(decomposition)
        and (
            "requires a prior `set_viewpoint(...)` call" in stripped_error
            or "initial viewpoint setup stage ending at a top-level `set_viewpoint(...)`" in stripped_error
        )
    ):
        if not _repair_global_view_missing_set_viewpoint(fn):
            return None, {"skipped": "global_view_set_viewpoint_repair_not_applicable"}
        ast.fix_missing_locations(tree)
        return (
            ast.unparse(tree),
            {
                "rule": "repair_global_view_missing_set_viewpoint",
                "camera_index": 1,
            },
        )

    if not decomposition:
        return None, {"skipped": "no_decomposition_for_motion_repair"}

    expected_ops = extract_expected_stage2_operations(decomposition)
    actual_ops = extract_stage2_camera_operations_from_tree(tree)
    expected_dirs = [direction for _, direction in expected_ops]
    actual_dirs = [direction for _, direction in actual_ops]

    if "dangling camera motion" in stripped_error:
        if actual_dirs != expected_dirs:
            return None, {
                "skipped": "dangling_motion_but_direction_mismatch",
                "expected_directions": expected_dirs,
                "actual_directions": actual_dirs,
            }
        if not _append_renders_after_last_motion(fn):
            return None, {"skipped": "failed_to_append_render_suffix"}
        ast.fix_missing_locations(tree)
        return (
            ast.unparse(tree),
            {
                "rule": "repair_dangling_motion_by_appending_renders",
                "expected_directions": expected_dirs,
                "actual_directions": actual_dirs,
            },
        )

    if stripped_error.startswith("Stage 2 camera directions do not align with the decomposition."):
        if not _repair_stage2_motion_suffix(fn, expected_ops):
            return None, {
                "skipped": "motion_suffix_repair_not_applicable",
                "expected_directions": expected_dirs,
                "actual_directions": actual_dirs,
            }
        ast.fix_missing_locations(tree)
        repaired_code = ast.unparse(tree)
        try:
            repaired_tree = ast.parse(repaired_code)
        except SyntaxError:
            return None, {"skipped": "repaired_code_not_parseable"}
        repaired_dirs = [
            direction for _, direction in extract_stage2_camera_operations_from_tree(repaired_tree)
        ]
        return (
            repaired_code,
            {
                "rule": "repair_stage2_motion_suffix",
                "expected_directions": expected_dirs,
                "original_directions": actual_dirs,
                "repaired_directions": repaired_dirs,
            },
        )

    return None, {"skipped": "no_matching_auto_repair_rule"}
