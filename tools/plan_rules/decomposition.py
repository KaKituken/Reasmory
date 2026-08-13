import ast
import re
from typing import Dict, List, Optional

from tools.plan_ast_utils import extract_plan_stages_from_tree, get_call_argument
from tools.plan_rules.catalog import TOOL_ARG_SCHEMAS


DIRECTION_WORDS = {"left", "right", "forward", "backward", "behind", "back", "up", "down", "front"}


def infer_problem_type_from_decomposition(decomposition: Dict[str, object]) -> str:
    part3 = decomposition.get("Part3")
    if not isinstance(part3, dict):
        return "object-related"

    reference_entity = part3.get("reference_entity")
    target_object = part3.get("target_object")

    entities: List[str] = []
    if isinstance(reference_entity, str) and reference_entity.strip():
        entities.append(reference_entity.strip().lower())

    if isinstance(target_object, str) and target_object.strip():
        entities.append(target_object.strip().lower())
    elif isinstance(target_object, list):
        entities.extend(
            item.strip().lower()
            for item in target_object
            if isinstance(item, str) and item.strip()
        )

    if entities and set(entities) == {"camera"}:
        return "camera-related"
    return "object-related"


def is_global_view_decomposition(decomposition: Optional[Dict[str, object]]) -> bool:
    return True # !!! Hard coded
    if not isinstance(decomposition, dict):
        return False
    part1 = decomposition.get("Part1") or {}
    part2 = decomposition.get("Part2") or []
    part3 = decomposition.get("Part3") or {}
    if not isinstance(part1, dict) or not isinstance(part3, dict):
        return False

    position = str(part1.get("position", "")).strip().lower()
    orientation = str(part1.get("orientation", "")).strip().lower()
    reference_entity = part3.get("reference_entity")
    final_question = part3.get("final_question")    # hack a little
    return (
        position == "arbitrary"
        and orientation == "arbitrary"
        and isinstance(part2, list)
        and len(part2) == 0
        and (reference_entity is None or "distance" in final_question.lower())
    )


def _canonicalize_direction(direction: str) -> str:
    mapping = {
        "behind": "back",
        "back": "back",
        "forward": "forward",
        "front": "front",
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "backward": "backward",
    }
    return mapping.get(direction, direction)


def _extract_direction_words(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    directions: List[str] = []

    phrase_rules = [
        (r"\bturn\s+180(?:\s+degrees?)?\s+around\b", "back"),
        (r"\bturn\s+around\b", "back"),
        (r"\brotate\s+180(?:\s+degrees?)?\b", "back"),
        (r"\bturn\s+180(?:\s+degrees?)?\b", "back"),
        (r"\bturn\s+back\b", "back"),
        (r"\bmove\s+forward\b", "forward"),
        (r"\bmove\s+backward\b", "backward"),
        (r"\bmove\s+back\b", "backward"),
        (r"\bmove\s+left\b", "left"),
        (r"\bmove\s+right\b", "right"),
        (r"\bturn\s+left\b", "left"),
        (r"\bturn\s+right\b", "right"),
        (r"\bturn\s+up\b", "up"),
        (r"\bturn\s+down\b", "down"),
        (r"\blook\s+left\b", "left"),
        (r"\blook\s+right\b", "right"),
        (r"\blook\s+up\b", "up"),
        (r"\blook\s+down\b", "down"),
    ]

    spans = []
    for pattern, normalized in phrase_rules:
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), normalized))

    spans.sort(key=lambda item: item[0])
    occupied: List[tuple] = []
    for start, end, normalized in spans:
        if any(not (end <= s or start >= e) for s, e in occupied):
            continue
        directions.append(normalized)
        occupied.append((start, end))

    scrubbed = text
    for start, end in sorted(occupied, reverse=True):
        scrubbed = scrubbed[:start] + " " * (end - start) + scrubbed[end:]

    for word in re.findall(r"[a-zA-Z]+", scrubbed):
        if word in DIRECTION_WORDS:
            directions.append(word)
    return [_canonicalize_direction(direction) for direction in directions]


def _normalize_contain_direction(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    directions = _extract_direction_words(value)
    unique = list(dict.fromkeys(directions))
    if len(unique) != 1:
        return None
    return unique[0]


def _extract_expected_stage2_directions(decomposition: Dict[str, object]) -> List[str]:
    expected: List[str] = []
    part2 = decomposition.get("Part2")
    if isinstance(part2, list):
        for item in part2:
            if isinstance(item, dict):
                expected.extend(_extract_direction_words(str(item.get("motion", ""))))

    part3 = decomposition.get("Part3")
    if isinstance(part3, dict):
        reference_entity = part3.get("reference_entity")
        contain_direction = _normalize_contain_direction(part3.get("contain_direction"))
        if (
            contain_direction == "back"
            and isinstance(reference_entity, str)
            and reference_entity.lower() != "camera"
        ):
            contain_direction = None
        if contain_direction is not None:
            expected.append(contain_direction)
    return expected


def extract_expected_stage2_operations(decomposition: Dict[str, object]) -> List[tuple[str, str]]:
    expected: List[tuple[str, str]] = []
    part2 = decomposition.get("Part2")
    if isinstance(part2, list):
        for item in part2:
            if not isinstance(item, dict):
                continue
            motion = str(item.get("motion", "")).strip().lower()
            directions = _extract_direction_words(motion)
            if len(directions) != 1:
                continue
            if motion.startswith("turn "):
                expected.append(("turn_camera", directions[0]))
            elif motion.startswith("step ") or motion.startswith("move "):
                expected.append(("step_camera", directions[0]))

    part3 = decomposition.get("Part3")
    if isinstance(part3, dict):
        reference_entity = part3.get("reference_entity")
        contain_direction = _normalize_contain_direction(part3.get("contain_direction"))
        if (
            contain_direction == "back"
            and isinstance(reference_entity, str)
            and reference_entity.lower() != "camera"
        ):
            contain_direction = None
        if contain_direction is not None:
            expected.append(("turn_camera", contain_direction))
    return expected


def _should_skip_direction_check(decomposition: Dict[str, object]) -> bool:
    part3 = decomposition.get("Part3")
    if not isinstance(part3, dict):
        return False
    contain_direction = _normalize_contain_direction(part3.get("contain_direction"))
    reference_entity = part3.get("reference_entity")
    return (
        contain_direction == "back"
        and isinstance(reference_entity, str)
        and reference_entity.lower() != "camera"
    )


def _part1_position_uses_existing_camera_anchor(decomposition: Dict[str, object]) -> bool:
    part1 = decomposition.get("Part1")
    if not isinstance(part1, dict):
        return True
    position = part1.get("position")
    if not isinstance(position, str):
        return True
    position_lower = position.lower()
    return any(keyword in position_lower for keyword in ["image", "view", "frame", "camera"])


def _part1_position_is_arbitrary(decomposition: Dict[str, object]) -> bool:
    part1 = decomposition.get("Part1")
    if not isinstance(part1, dict):
        return False
    position = part1.get("position")
    if not isinstance(position, str):
        return False
    return position.strip().lower() == "arbitrary"


def _extract_part1_camera_index(decomposition: Dict[str, object]) -> Optional[int]:
    part1 = decomposition.get("Part1")
    if not isinstance(part1, dict):
        return None
    position = part1.get("position")
    if not isinstance(position, str):
        return None

    text = position.lower()
    digit_match = re.search(r"\b(\d+)\b", text)
    if digit_match:
        return int(digit_match.group(1))

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    for token in re.findall(r"[a-zA-Z]+", text):
        if token in number_words:
            return number_words[token]
    return None


def _extract_stage1_query_camera_indices(tree: ast.Module) -> List[int]:
    stages = extract_plan_stages_from_tree(tree)
    fn = tree.body[0]
    non_return_stmts = [stmt for stmt in fn.body if not isinstance(stmt, ast.Return)]
    boundary_index = stages["boundary_statement_index"]
    if boundary_index is None:
        return []

    stage1_stmts = non_return_stmts[: boundary_index + 1]
    indices: List[int] = []
    for stmt in stage1_stmts:
        call_node = None
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
        if call_node is None or not isinstance(call_node.func, ast.Name):
            continue
        if call_node.func.id != "query_camera_pose":
            continue
        frame_expr = get_call_argument(
            call_node,
            "frame_indices",
            TOOL_ARG_SCHEMAS["query_camera_pose"]["params"],
        )
        if isinstance(frame_expr, (ast.List, ast.Tuple)):
            for elt in frame_expr.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    indices.append(int(elt.value))
    return indices


def _extract_stage2_camera_directions_from_tree(tree: ast.Module) -> List[str]:
    stages = extract_plan_stages_from_tree(tree)
    stage2_directions: List[str] = []
    for stmt in stages["stage2"]["statements"]:
        stmt_ast = ast.parse(stmt).body[0]
        call_node = None
        if isinstance(stmt_ast, ast.Assign) and isinstance(stmt_ast.value, ast.Call):
            call_node = stmt_ast.value
        elif isinstance(stmt_ast, ast.Expr) and isinstance(stmt_ast.value, ast.Call):
            call_node = stmt_ast.value
        if call_node is None or not isinstance(call_node.func, ast.Name):
            continue
        func_name = call_node.func.id
        if func_name not in {"turn_camera", "step_camera"}:
            continue
        direction_expr = get_call_argument(
            call_node,
            "direction",
            TOOL_ARG_SCHEMAS[func_name]["params"],
        )
        literal = None
        if isinstance(direction_expr, ast.Constant) and isinstance(direction_expr.value, str):
            literal = direction_expr.value.lower()
        if literal is not None:
            stage2_directions.append(literal)
    return stage2_directions


def extract_stage2_camera_operations_from_tree(tree: ast.Module) -> List[tuple[str, str]]:
    stages = extract_plan_stages_from_tree(tree)
    operations: List[tuple[str, str]] = []
    for stmt in stages["stage2"]["statements"]:
        stmt_ast = ast.parse(stmt).body[0]
        call_node = None
        if isinstance(stmt_ast, ast.Assign) and isinstance(stmt_ast.value, ast.Call):
            call_node = stmt_ast.value
        elif isinstance(stmt_ast, ast.Expr) and isinstance(stmt_ast.value, ast.Call):
            call_node = stmt_ast.value
        if call_node is None or not isinstance(call_node.func, ast.Name):
            continue
        func_name = call_node.func.id
        if func_name not in {"turn_camera", "step_camera"}:
            continue
        direction_expr = get_call_argument(
            call_node,
            "direction",
            TOOL_ARG_SCHEMAS[func_name]["params"],
        )
        if isinstance(direction_expr, ast.Constant) and isinstance(direction_expr.value, str):
            operations.append((func_name, direction_expr.value.lower()))
    return operations


def validate_plan_against_decomposition(tree: ast.Module, decomposition: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    stages = extract_plan_stages_from_tree(tree)
    if is_global_view_decomposition(decomposition):
        return errors

    if stages["boundary_statement_index"] is None:
        errors.append(
            "The plan must contain an initial viewpoint setup stage ending at a top-level "
            "`set_viewpoint(...)` before exploration begins."
        )
        return errors

    stage1_tool_sequence = [tool for tool in stages["stage1"]["tool_sequence"] if tool is not None]
    if (
        not _part1_position_is_arbitrary(decomposition)
        and
        not _part1_position_uses_existing_camera_anchor(decomposition)
        and "query_3d_object_position" not in stage1_tool_sequence
    ):
        errors.append(
            "The decomposition's Part1 position is not anchored to an existing camera/view/frame, "
            "so stage 1 must include `query_3d_object_position(...)` before setting the viewpoint."
        )
        return errors

    if _part1_position_uses_existing_camera_anchor(decomposition):
        expected_camera_index = _extract_part1_camera_index(decomposition)
        if expected_camera_index is not None:
            queried_indices = _extract_stage1_query_camera_indices(tree)
            if expected_camera_index not in queried_indices:
                errors.append(
                    "The decomposition's Part1 position refers to an existing camera/image/view, "
                    f"so stage 1 must query that camera via `query_camera_pose(...)`. "
                    f"Expected camera index {expected_camera_index}, got {queried_indices}."
                )
                return errors

    if _should_skip_direction_check(decomposition):
        return errors

    expected_directions = _extract_expected_stage2_directions(decomposition)
    actual_directions = _extract_stage2_camera_directions_from_tree(tree)

    if actual_directions and not expected_directions:
        errors.append(
            "Stage 2 contains camera operations with directions "
            f"{actual_directions}, but the decomposition does not specify any directional "
            "motion in Part2 or a single directional relation in Part3."
        )
        return errors

    if expected_directions != actual_directions:
        errors.append(
            "Stage 2 camera directions do not align with the decomposition. "
            f"The plan must realize exactly {expected_directions}, got {actual_directions}."
        )
    return errors
