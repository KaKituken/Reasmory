import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from tools.plan_rules.decomposition import is_global_view_decomposition


def _stmt_call_info(stmt: ast.stmt) -> Tuple[ast.Call | None, ast.Name | None]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call):
        return stmt.value, stmt.targets[0]
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value, None
    return None, None


def _call_name(stmt: ast.stmt) -> str | None:
    call, _ = _stmt_call_info(stmt)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _collect_used_names(stmts: List[ast.stmt]) -> set[str]:
    used: set[str] = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
    return used


def _keyword_map(call: ast.Call) -> Dict[str, ast.AST]:
    kwargs: Dict[str, ast.AST] = {}
    for kw in call.keywords:
        if kw.arg is not None:
            kwargs[kw.arg] = kw.value
    return kwargs


def _literal_number(expr: ast.AST) -> float | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool):
        return float(expr.value)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        inner = _literal_number(expr.operand)
        if inner is not None:
            return -inner
    return None


def _literal_int(expr: ast.AST) -> int | None:
    value = _literal_number(expr)
    if value is None or int(value) != value:
        return None
    return int(value)


def _literal_str(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def _make_call_stmt(func_name: str, kwargs: Dict[str, ast.AST], assign_to: str | None = None) -> ast.stmt:
    call = ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[],
        keywords=[ast.keyword(arg=key, value=value) for key, value in kwargs.items()],
    )
    if assign_to is not None:
        return ast.Assign(targets=[ast.Name(id=assign_to, ctx=ast.Store())], value=call)
    return ast.Expr(value=call)


def _py_literal_ast(value: Any) -> ast.AST:
    return ast.parse(repr(value), mode="eval").body


def _call_metadata(call: ast.Call) -> Dict[str, Any]:
    arguments: Dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg is not None:
            arguments[kw.arg] = ast.unparse(kw.value)
    tool_name = call.func.id if isinstance(call.func, ast.Name) else ast.unparse(call.func)
    return {
        "tool_name": tool_name,
        "call_signature": ast.unparse(call),
        "arguments": arguments,
    }


def _kwargs_dict_ast(call: ast.Call) -> ast.Dict:
    keys: List[ast.AST] = []
    values: List[ast.AST] = []
    for kw in call.keywords:
        if kw.arg is not None:
            keys.append(ast.Constant(value=kw.arg))
            values.append(kw.value)
    return ast.Dict(keys=keys, values=values)


def _turn_axis(direction: str) -> str | None:
    if direction in {"left", "right", "back"}:
        return "yaw"
    if direction in {"up", "down"}:
        return "pitch"
    return None


def _turn_signed_degrees(direction: str, angle: float | None) -> float | None:
    if direction == "left":
        return float(90.0 if angle is None else angle)
    if direction == "right":
        return -float(90.0 if angle is None else angle)
    if direction == "back":
        return 180.0
    if direction == "up":
        return float(45.0 if angle is None else angle)
    if direction == "down":
        return -float(45.0 if angle is None else angle)
    return None


def _normalize_yaw_degrees(total: float) -> float:
    normalized = ((total + 180.0) % 360.0) - 180.0
    if normalized == -180.0:
        return 180.0
    return normalized


def _build_turn_stmt_from_total(
    session_expr: ast.AST,
    total_degrees: float,
    axis: str,
    assign_to: str | None = None,
) -> ast.stmt | None:
    if axis == "yaw":
        total_degrees = _normalize_yaw_degrees(total_degrees)
        if abs(total_degrees) < 1e-6:
            return None
        if abs(total_degrees) == 180.0:
            return _make_call_stmt(
                "turn_camera",
                {
                    "session_id": session_expr,
                    "direction": ast.Constant(value="back"),
                },
                assign_to=assign_to,
            )
        direction = "left" if total_degrees > 0 else "right"
        angle = abs(total_degrees)
        kwargs = {
            "session_id": session_expr,
            "direction": ast.Constant(value=direction),
        }
        if abs(angle - 90.0) > 1e-6:
            kwargs["angle"] = ast.Constant(value=angle)
        return _make_call_stmt("turn_camera", kwargs, assign_to=assign_to)

    if axis == "pitch":
        if abs(total_degrees) < 1e-6:
            return None
        direction = "up" if total_degrees > 0 else "down"
        angle = abs(total_degrees)
        kwargs = {
            "session_id": session_expr,
            "direction": ast.Constant(value=direction),
        }
        if abs(angle - 45.0) > 1e-6:
            kwargs["angle"] = ast.Constant(value=angle)
        return _make_call_stmt("turn_camera", kwargs, assign_to=assign_to)

    return None


def _step_axis_and_sign(direction: str) -> tuple[str, int] | None:
    mapping = {
        "forward": ("z", 1),
        "backward": ("z", -1),
        "right": ("x", 1),
        "left": ("x", -1),
        "down": ("y", 1),
        "up": ("y", -1),
    }
    return mapping.get(direction)


def _build_query_camera_pose_map(stmts: List[ast.stmt]) -> Dict[str, List[int]]:
    constant_ints: Dict[str, int] = {}
    query_frames: Dict[str, List[int]] = {}
    for stmt in stmts:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            literal = _literal_int(stmt.value)
            if literal is not None:
                constant_ints[target] = literal
                continue
        call, target_name = _stmt_call_info(stmt)
        if call is None or target_name is None or not isinstance(call.func, ast.Name) or call.func.id != "query_camera_pose":
            continue
        kwargs = _keyword_map(call)
        frame_indices = kwargs.get("frame_indices")
        if isinstance(frame_indices, ast.List):
            resolved_indices: List[int] = []
            for elt in frame_indices.elts:
                frame_idx = _literal_int(elt)
                if frame_idx is None and isinstance(elt, ast.Name):
                    frame_idx = constant_ints.get(elt.id)
                if frame_idx is None:
                    resolved_indices = []
                    break
                resolved_indices.append(frame_idx)
            if resolved_indices:
                query_frames[target_name.id] = resolved_indices
    return query_frames


def _extract_camera_index_from_set_viewpoint(stmt: ast.stmt, query_frames: Dict[str, List[int]]) -> int | None:
    call, _ = _stmt_call_info(stmt)
    if call is None or not isinstance(call.func, ast.Name) or call.func.id != "set_viewpoint":
        return None
    kwargs = _keyword_map(call)
    refs: Dict[str, tuple[str, int] | None] = {}
    for key in ("origin", "forward", "up"):
        expr = kwargs.get(key)
        if expr is None:
            continue
        if not isinstance(expr, ast.Subscript):
            return None
        outer = expr.value
        if not isinstance(outer, ast.Subscript):
            return None
        pose_list_index = _literal_int(outer.slice)
        if pose_list_index is None:
            return None
        if not isinstance(outer.value, ast.Name):
            return None
        field = _literal_str(expr.slice)
        if field is None:
            return None
        refs[key] = (outer.value.id, pose_list_index)
        if outer.value.id not in query_frames:
            return None
    source_names = {value[0] for value in refs.values()}
    if len(source_names) != 1:
        return None
    source_refs = set(refs.values())
    if len(source_refs) != 1:
        return None
    source_name, pose_list_index = next(iter(source_refs))
    frame_indices = query_frames.get(source_name)
    if frame_indices is None or pose_list_index < 0 or pose_list_index >= len(frame_indices):
        return None
    return frame_indices[pose_list_index]


def _build_shortcut_render_stmt(render_stmt: ast.stmt, frame_index: int, media_arg_name: str) -> ast.stmt | None:
    call, _ = _stmt_call_info(render_stmt)
    if call is None:
        return None
    if media_arg_name != "input_images":
        return None
    helper_call = ast.Call(
        func=ast.Name(id="_compiler_shortcut_render_ego_rgb", ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="frame_index", value=ast.Constant(value=frame_index)),
            ast.keyword(arg="input_images", value=ast.Name(id=media_arg_name, ctx=ast.Load())),
            ast.keyword(arg="original_call", value=_py_literal_ast(_call_metadata(call))),
        ],
    )
    if isinstance(render_stmt, ast.Assign):
        return ast.Assign(targets=render_stmt.targets, value=helper_call)
    if isinstance(render_stmt, ast.Expr):
        return ast.Expr(value=helper_call)
    return None


def _build_compiler_sequence_stmt(
    helper_name: str,
    block: List[ast.stmt],
    merged_kwargs: Dict[str, Any] | None = None,
    reduced_kwargs_list: List[Dict[str, Any]] | None = None,
) -> ast.stmt:
    original_calls: List[Dict[str, Any]] = []
    for block_stmt in block:
        call, _ = _stmt_call_info(block_stmt)
        if call is None:
            continue
        original_calls.append(_call_metadata(call))

    keywords = [ast.keyword(arg="original_calls", value=_py_literal_ast(original_calls))]
    if merged_kwargs is not None:
        keywords.append(ast.keyword(arg="merged_kwargs", value=_py_literal_ast(merged_kwargs)))
    if reduced_kwargs_list is not None:
        keywords.append(ast.keyword(arg="reduced_kwargs_list", value=_py_literal_ast(reduced_kwargs_list)))
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=[],
            keywords=keywords,
        )
    )


def _motion_pass(stmts: List[ast.stmt]) -> List[ast.stmt]:
    optimized: List[ast.stmt] = []
    i = 0

    while i < len(stmts):
        stmt = stmts[i]
        stmt_name = _call_name(stmt)

        if stmt_name == "turn_camera":
            j = i
            block: List[ast.stmt] = []
            while j < len(stmts) and _call_name(stmts[j]) == "turn_camera":
                block.append(stmts[j])
                j += 1
            later_used = _collect_used_names(stmts[j:])
            assigned_names = {
                target.id
                for block_stmt in block
                if isinstance(block_stmt, ast.Assign)
                for target in block_stmt.targets
                if isinstance(target, ast.Name)
            }
            if assigned_names & later_used:
                optimized.extend(block)
                i = j
                continue

            first_call, first_target = _stmt_call_info(block[0])
            first_kwargs = _keyword_map(first_call)
            session_expr = first_kwargs.get("session_id")
            first_dir = _literal_str(first_kwargs.get("direction"))
            axis = _turn_axis(first_dir) if first_dir is not None else None
            if session_expr is None or axis is None:
                optimized.extend(block)
                i = j
                continue

            total = 0.0
            mergeable = True
            for block_stmt in block:
                call, _ = _stmt_call_info(block_stmt)
                kwargs = _keyword_map(call)
                direction = _literal_str(kwargs.get("direction"))
                angle_expr = kwargs.get("angle")
                if direction is None or _turn_axis(direction) != axis:
                    mergeable = False
                    break
                signed = _turn_signed_degrees(direction, _literal_number(angle_expr) if angle_expr is not None else None)
                if signed is None:
                    mergeable = False
                    break
                total += signed
            if not mergeable:
                optimized.extend(block)
                i = j
                continue

            merged_stmt = _build_turn_stmt_from_total(
                session_expr=session_expr,
                total_degrees=total,
                axis=axis,
                assign_to=first_target.id if first_target is not None else None,
            )
            if merged_stmt is not None:
                merged_call, _ = _stmt_call_info(merged_stmt)
                if merged_call is not None:
                    optimized.append(
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id="_compiler_apply_turn_sequence", ctx=ast.Load()),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg="original_calls",
                                        value=_py_literal_ast([
                                            _call_metadata(_stmt_call_info(block_stmt)[0])
                                            for block_stmt in block
                                            if _stmt_call_info(block_stmt)[0] is not None
                                        ]),
                                    ),
                                    ast.keyword(arg="merged_kwargs", value=_kwargs_dict_ast(merged_call)),
                                ],
                            )
                        )
                    )
                else:
                    optimized.append(
                        _build_compiler_sequence_stmt(
                            "_compiler_apply_turn_sequence",
                            block=block,
                            merged_kwargs=None,
                        )
                    )
            else:
                optimized.append(
                    _build_compiler_sequence_stmt(
                        "_compiler_apply_turn_sequence",
                        block=block,
                        merged_kwargs=None,
                    )
                )
            i = j
            continue

        if stmt_name == "step_camera":
            j = i
            block: List[ast.stmt] = []
            while j < len(stmts) and _call_name(stmts[j]) == "step_camera":
                block.append(stmts[j])
                j += 1
            later_used = _collect_used_names(stmts[j:])
            assigned_names = {
                target.id
                for block_stmt in block
                if isinstance(block_stmt, ast.Assign)
                for target in block_stmt.targets
                if isinstance(target, ast.Name)
            }
            if assigned_names & later_used:
                optimized.extend(block)
                i = j
                continue

            reduced_stack: List[ast.stmt] = []
            for block_stmt in block:
                call, _ = _stmt_call_info(block_stmt)
                kwargs = _keyword_map(call)
                direction = _literal_str(kwargs.get("direction"))
                step_info = _step_axis_and_sign(direction) if direction is not None else None
                if step_info is None:
                    reduced_stack.append(block_stmt)
                    continue
                if reduced_stack:
                    prev_call, _ = _stmt_call_info(reduced_stack[-1])
                    prev_kwargs = _keyword_map(prev_call)
                    prev_direction = _literal_str(prev_kwargs.get("direction"))
                    prev_info = _step_axis_and_sign(prev_direction) if prev_direction is not None else None
                    if prev_info is not None and prev_info[0] == step_info[0] and prev_info[1] == -step_info[1]:
                        reduced_stack.pop()
                        continue
                reduced_stack.append(block_stmt)
            for reduced_stmt in reduced_stack:
                reduced_call, _ = _stmt_call_info(reduced_stmt)
                if reduced_call is None:
                    optimized.extend(block)
                    break
            else:
                optimized.append(
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="_compiler_apply_step_sequence", ctx=ast.Load()),
                            args=[],
                            keywords=[
                                ast.keyword(
                                    arg="original_calls",
                                    value=_py_literal_ast([
                                        _call_metadata(_stmt_call_info(block_stmt)[0])
                                        for block_stmt in block
                                        if _stmt_call_info(block_stmt)[0] is not None
                                    ]),
                                ),
                                ast.keyword(
                                    arg="reduced_kwargs_list",
                                    value=ast.List(
                                        elts=[
                                            _kwargs_dict_ast(_stmt_call_info(reduced_stmt)[0])
                                            for reduced_stmt in reduced_stack
                                            if _stmt_call_info(reduced_stmt)[0] is not None
                                        ],
                                        ctx=ast.Load(),
                                    ),
                                ),
                            ],
                        )
                    )
                )
            i = j
            continue

        optimized.append(stmt)
        i += 1

    return optimized


def _identity_render_shortcut_pass(stmts: List[ast.stmt], media_arg_name: str) -> List[ast.stmt]:
    optimized: List[ast.stmt] = []
    query_frames = _build_query_camera_pose_map(stmts)
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        stmt_name = _call_name(stmt)
        if (
            stmt_name == "set_viewpoint"
            and i + 1 < len(stmts)
            and _call_name(stmts[i + 1]) == "render_ego_rgb"
        ):
            frame_index = _extract_camera_index_from_set_viewpoint(stmt, query_frames)
            if frame_index is not None:
                optimized.append(stmt)
                shortcut_stmt = _build_shortcut_render_stmt(stmts[i + 1], frame_index, media_arg_name)
                if shortcut_stmt is not None:
                    optimized.append(shortcut_stmt)
                    i += 2
                    continue
        if (
            stmt_name == "set_viewpoint"
            and i + 2 < len(stmts)
            and _call_name(stmts[i + 1]) == "_compiler_apply_turn_sequence"
            and _call_name(stmts[i + 2]) == "render_ego_rgb"
        ):
            frame_index = _extract_camera_index_from_set_viewpoint(stmt, query_frames)
            turn_call, _ = _stmt_call_info(stmts[i + 1])
            is_identity_turn = (
                turn_call is not None
                and not any(kw.arg == "merged_kwargs" for kw in turn_call.keywords if kw.arg is not None)
            )
            if frame_index is not None and is_identity_turn:
                optimized.append(stmt)
                optimized.append(stmts[i + 1])
                shortcut_stmt = _build_shortcut_render_stmt(stmts[i + 2], frame_index, media_arg_name)
                if shortcut_stmt is not None:
                    optimized.append(shortcut_stmt)
                    i += 3
                    continue
        optimized.append(stmt)
        i += 1
    return optimized


def _is_global_room_size_question(decomposition: Optional[Dict[str, Any]]) -> bool:
    if not is_global_view_decomposition(decomposition):
        return False
    part3 = decomposition.get("Part3") if isinstance(decomposition, dict) else None
    if not isinstance(part3, dict):
        return False
    final_question = str(part3.get("final_question", "")).lower()
    return bool(
        re.search(r"\bsize\s+of\s+(?:the|this|that)\s+room\b", final_question)
        or re.search(r"\broom\s+size\b", final_question)
    )


def _drop_keyword(call: ast.Call, keyword_name: str) -> None:
    call.keywords = [
        keyword for keyword in call.keywords
        if keyword.arg != keyword_name
    ]


def _global_room_size_pass(
    stmts: List[ast.stmt],
    decomposition: Optional[Dict[str, Any]],
) -> List[ast.stmt]:
    if not _is_global_room_size_question(decomposition):
        return stmts

    optimized: List[ast.stmt] = []
    for stmt in stmts:
        stmt_name = _call_name(stmt)
        if stmt_name == "query_3d_object_position":
            continue

        call, _ = _stmt_call_info(stmt)
        if (
            call is not None
            and isinstance(call.func, ast.Name)
            and call.func.id == "render_semantic_bev"
        ):
            _drop_keyword(call, "queried_objects")
        optimized.append(stmt)
    return optimized


def optimize_plan_code(
    code: str,
    decomposition: Optional[Dict[str, Any]] = None,
) -> str:
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return code

    fn = tree.body[0]
    media_arg_name = fn.args.args[0].arg if fn.args.args else "input_images"
    fn.body = _global_room_size_pass(fn.body, decomposition)
    fn.body = _motion_pass(fn.body)
    fn.body = _identity_render_shortcut_pass(fn.body, media_arg_name)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
