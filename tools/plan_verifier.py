import ast
from typing import Dict, List, Optional, Set

from tools.plan_ast_utils import (
    extract_plan_stages,
    extract_plan_stages_from_tree,
    extract_python_code,
    normalize_python_plan_code,
)
from tools.plan_rules.catalog import ALLOWED_TOOL_NAMES, TOOL_ARG_SCHEMAS
from tools.plan_rules.decomposition import (
    infer_problem_type_from_decomposition,
    is_global_view_decomposition,
    validate_plan_against_decomposition,
)
from tools.plan_rules.execution_order import (
    validate_no_dangling_camera_motion,
    validate_no_zero_gain_object_related_semantic_bev,
)
from tools.plan_rules.execution_state import ExecutionOrderStateMixin
from tools.plan_rules.schema import SchemaValidationMixin
from tools.plan_rules.syntax import SyntaxValidationMixin


class PlanVerificationError(Exception):
    pass


class PythonPlanVerifier(
    SyntaxValidationMixin,
    SchemaValidationMixin,
    ExecutionOrderStateMixin,
    ast.NodeVisitor,
):
    def __init__(self, allowed_tool_names: Optional[Set[str]] = None):
        self.allowed_tool_names = allowed_tool_names or ALLOWED_TOOL_NAMES
        self.errors: List[str] = []
        self.defined_names: Set[str] = set()
        self.has_seen_set_viewpoint = False
        self.has_seen_reference_query = False
        self.name_types = {}
        self.input_image_count: Optional[int] = None
        self.allow_global_render_without_viewpoint = False

    def verify(
        self,
        code: str,
        decomposition: Optional[Dict[str, object]] = None,
        input_image_count: Optional[int] = None,
    ) -> ast.Module:
        code = normalize_python_plan_code(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise PlanVerificationError(
                f"Syntax error at line {exc.lineno}, column {exc.offset}: {exc.msg}"
            ) from exc

        self.errors = []
        self.input_image_count = input_image_count
        self.allow_global_render_without_viewpoint = is_global_view_decomposition(decomposition)
        self.visit(tree)
        if not self.errors:
            self.errors.extend(validate_no_dangling_camera_motion(tree))
        inferred_problem_type = (
            infer_problem_type_from_decomposition(decomposition)
            if decomposition is not None
            else None
        )
        if not self.errors:
            self.errors.extend(
                validate_no_zero_gain_object_related_semantic_bev(
                    tree, inferred_problem_type
                )
            )
        if not self.errors and decomposition is not None:
            self._validate_against_decomposition(tree, decomposition)
        if self.errors:
            raise PlanVerificationError("\n".join(self.errors))
        return tree

def format_verifier_feedback(previous_code: str, error_text: str) -> str:
    return (
        "Your previous Python plan failed verification.\n\n"
        f"Verifier error:\n{error_text}\n\n"
        "Please regenerate the full function.\n"
        "Rules:\n"
        "- Output exactly one Python code block.\n"
        "- Define exactly one function `plan_to_solve_problem(input_images: List[str])` or `plan_to_solve_problem(input_video: str)`.\n"
        "- Use only allowed tool functions.\n"
        "- Use valid Python placeholders such as \"__FRAME_WITH_SOME_PROPERTY__\".\n"
        "- End with `return useful_observation`.\n\n"
        "Previous code:\n"
        "```python\n"
        f"{previous_code}\n"
        "```"
    )

def _validate_against_decomposition(self, tree: ast.Module, decomposition: Dict[str, object]):
    self.errors.extend(validate_plan_against_decomposition(tree, decomposition))


PythonPlanVerifier._validate_against_decomposition = _validate_against_decomposition
