import ast
import re
from typing import Dict, List, Optional


class StageExtractionError(Exception):
    pass


def normalize_python_plan_code(code: str) -> str:
    normalized = code
    normalized = re.sub(
        r"def\s+plan_to_solve_problem\s*\(\s*input_images\s*:\s*list\s*\[\s*str\s*\]\s*\)",
        "def plan_to_solve_problem(input_images: List[str])",
        normalized,
    )
    normalized = re.sub(
        r"def\s+plan_to_solve_problem\s*\(\s*input_video\s*:\s*list\s*\[\s*str\s*\]\s*\)",
        "def plan_to_solve_problem(input_video: str)",
        normalized,
    )
    return normalized


def extract_python_code(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = text.strip()
    sanitized_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("from typing import ") and "List" in stripped:
            continue
        if stripped == "import typing":
            continue
        sanitized_lines.append(line)
    return normalize_python_plan_code("\n".join(sanitized_lines).strip())


def get_stmt_call_name(stmt: ast.stmt) -> Optional[str]:
    call_node = None
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        call_node = stmt.value
    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call_node = stmt.value
    if call_node is None or not isinstance(call_node.func, ast.Name):
        return None
    return call_node.func.id


def get_call_argument(node: ast.Call, arg_name: str, param_order: List[str]) -> Optional[ast.AST]:
    for kw in node.keywords:
        if kw.arg == arg_name:
            return kw.value
    if arg_name in param_order:
        idx = param_order.index(arg_name)
        if idx < len(node.args):
            return node.args[idx]
    return None


def extract_plan_stages_from_tree(tree: ast.Module) -> Dict[str, object]:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise StageExtractionError(
            "Stage extraction requires exactly one top-level function definition."
        )

    fn = tree.body[0]
    non_return_stmts = [stmt for stmt in fn.body if not isinstance(stmt, ast.Return)]
    stmt_call_names = [get_stmt_call_name(stmt) for stmt in non_return_stmts]

    camera_op_indices = [
        idx for idx, name in enumerate(stmt_call_names) if name in {"turn_camera", "step_camera"}
    ]
    first_camera_op_index = camera_op_indices[0] if camera_op_indices else None

    if first_camera_op_index is not None:
        candidate_boundary_indices = [
            idx
            for idx, name in enumerate(stmt_call_names[:first_camera_op_index])
            if name == "set_viewpoint"
        ]
    else:
        candidate_boundary_indices = [
            idx for idx, name in enumerate(stmt_call_names) if name == "set_viewpoint"
        ]

    boundary_index = candidate_boundary_indices[-1] if candidate_boundary_indices else None

    if boundary_index is None:
        stage1_stmts = []
        stage2_stmts = non_return_stmts
    else:
        stage1_stmts = non_return_stmts[: boundary_index + 1]
        stage2_stmts = non_return_stmts[boundary_index + 1 :]

    return {
        "rule": "The last `set_viewpoint` before any `turn_camera`/`step_camera` marks the end of stage 1. "
        "If no camera operation exists, use the last `set_viewpoint` in the function. "
        "If no `set_viewpoint` exists, stage 1 is empty and all executable statements are assigned to stage 2.",
        "boundary_statement_index": boundary_index,
        "first_camera_operation_index": first_camera_op_index,
        "stage1": {
            "name": "initial_viewpoint_setup",
            "statements": [ast.unparse(stmt) for stmt in stage1_stmts],
            "tool_sequence": [get_stmt_call_name(stmt) for stmt in stage1_stmts],
        },
        "stage2": {
            "name": "explore_and_information_gathering",
            "statements": [ast.unparse(stmt) for stmt in stage2_stmts],
            "tool_sequence": [get_stmt_call_name(stmt) for stmt in stage2_stmts],
        },
    }


def extract_plan_stages(code: str) -> Dict[str, object]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise StageExtractionError(
            f"Syntax error at line {exc.lineno}, column {exc.offset}: {exc.msg}"
        ) from exc
    return extract_plan_stages_from_tree(tree)
