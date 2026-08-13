import ast
from typing import Set

from tools.plan_rules.catalog import TOOL_ARG_SCHEMAS


class SchemaValidationMixin:
    def _validate_statement(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                self._validate_assignment_target(target)
            self._validate_expr(stmt.value)
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
                inferred_type = self._infer_expr_type(stmt.value)
                if inferred_type is not None:
                    self.name_types[target_name] = inferred_type
                literal_integer_list = self._literal_integer_list_value(stmt.value)
                if literal_integer_list is not None:
                    self.literal_integer_lists[target_name] = literal_integer_list
                else:
                    self.literal_integer_lists.pop(target_name, None)
                self._update_query_result_lifetime_for_assignment(
                    target_name, stmt.value
                )
            return

        if isinstance(stmt, ast.Expr):
            if not isinstance(stmt.value, ast.Call):
                self.errors.append(
                    f"Only tool calls are allowed as standalone expressions (line {stmt.lineno})."
                )
                return
            self._validate_expr(stmt.value)
            return

        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self._validate_expr(stmt.value)
            return

        self.errors.append(
            f"Unsupported statement `{type(stmt).__name__}` at line {stmt.lineno}."
        )

    def _validate_assignment_target(self, target: ast.expr):
        if not isinstance(target, ast.Name):
            self.errors.append("Only simple variable assignment is allowed.")
            return
        if target.id in self.allowed_tool_names:
            self.errors.append(f"Assignment to reserved tool name `{target.id}` is not allowed.")
            return
        self.defined_names.add(target.id)

    def _validate_expr(self, expr: ast.AST):
        if self._literal_integer_list_value(expr) is not None:
            return

        if isinstance(expr, ast.Call):
            self._validate_call(expr)
            return

        if isinstance(expr, ast.Name):
            if expr.id in getattr(self, "stale_query_names", set()):
                self.errors.append(
                    f"Variable `{expr.id}` at line {getattr(expr, 'lineno', '?')} was derived from "
                    "a camera/object query before the active viewpoint changed, so it is no longer valid. "
                    "Re-run the relevant `query_camera_pose(...)` or `query_3d_object_position(...)` "
                    "under the current viewpoint."
                )
            if expr.id not in self.defined_names:
                self.errors.append(
                    f"Use of undefined variable `{expr.id}` at line {getattr(expr, 'lineno', '?')}."
                )
            return

        if isinstance(expr, ast.Constant):
            return

        if isinstance(expr, ast.List):
            for elt in expr.elts:
                self._validate_expr(elt)
            return

        if isinstance(expr, ast.Tuple):
            for elt in expr.elts:
                self._validate_expr(elt)
            return

        if isinstance(expr, ast.Dict):
            for key in expr.keys:
                if key is not None:
                    self._validate_expr(key)
            for value in expr.values:
                self._validate_expr(value)
            return

        if isinstance(expr, ast.Subscript):
            self._validate_subscript(expr)
            return

        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            self._validate_expr(expr.operand)
            return

        self.errors.append(
            f"Unsupported expression `{type(expr).__name__}` at line {getattr(expr, 'lineno', '?')}."
        )

    def _validate_call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            self.errors.append(
                f"Only direct tool calls are allowed (line {node.lineno})."
            )
            return

        func_name = node.func.id
        if func_name not in self.allowed_tool_names:
            self.errors.append(
                f"Disallowed function `{func_name}` at line {node.lineno}."
            )
            return

        self._validate_execution_order_for_call(func_name, node)
        self._validate_call_arguments(node, func_name)

        for arg in node.args:
            self._validate_expr(arg)
        for kw in node.keywords:
            if kw.arg is None:
                self.errors.append(
                    f"Starred keyword arguments are not allowed at line {node.lineno}."
                )
                continue
            self._validate_expr(kw.value)

        self._update_execution_order_after_call(func_name)

    def _expr_depends_on_query_result(self, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            return expr.func.id in {"query_camera_pose", "query_3d_object_position", "safe_select"}
        if isinstance(expr, ast.Name):
            return expr.id in getattr(self, "query_derived_names", set())
        if isinstance(expr, ast.Subscript):
            return self._expr_depends_on_query_result(expr.value)
        if isinstance(expr, ast.List):
            return any(self._expr_depends_on_query_result(elt) for elt in expr.elts)
        if isinstance(expr, ast.Tuple):
            return any(self._expr_depends_on_query_result(elt) for elt in expr.elts)
        if isinstance(expr, ast.Dict):
            return any(
                self._expr_depends_on_query_result(value)
                for value in expr.values
                if value is not None
            )
        if isinstance(expr, ast.UnaryOp):
            return self._expr_depends_on_query_result(expr.operand)
        return False

    def _update_query_result_lifetime_for_assignment(self, target_name: str, value_expr: ast.AST):
        if self._expr_depends_on_query_result(value_expr):
            self.query_derived_names.add(target_name)
            self.stale_query_names.discard(target_name)
            return
        self.query_derived_names.discard(target_name)
        self.stale_query_names.discard(target_name)

    def _validate_call_arguments(self, node: ast.Call, func_name: str):
        schema = TOOL_ARG_SCHEMAS.get(func_name)
        if schema is None:
            return

        allowed_params = schema["params"]
        required_params = set(schema["required"])

        if len(node.args) > len(allowed_params):
            self.errors.append(
                f"`{func_name}` at line {node.lineno} received too many positional arguments. "
                f"Expected at most {len(allowed_params)}."
            )
            return

        bound_args = {}
        for idx, arg in enumerate(node.args):
            param_name = allowed_params[idx]
            bound_args[param_name] = arg

        for kw in node.keywords:
            if kw.arg is None:
                continue
            if kw.arg not in allowed_params:
                self.errors.append(
                    f"`{func_name}` at line {node.lineno} got unexpected argument `{kw.arg}`. "
                    f"Allowed arguments are: {allowed_params}."
                )
                continue
            if kw.arg in bound_args:
                self.errors.append(
                    f"`{func_name}` at line {node.lineno} passed `{kw.arg}` multiple times."
                )
                continue
            bound_args[kw.arg] = kw.value

        missing = [name for name in allowed_params if name in required_params and name not in bound_args]
        if missing:
            self.errors.append(
                f"`{func_name}` at line {node.lineno} is missing required argument(s): {missing}."
            )

        if func_name == "build_static_spatial_memory":
            self._validate_build_static_spatial_memory_args(node, bound_args)
            return

        self._validate_session_id_arg(func_name, node, bound_args)

        if func_name == "query_camera_pose":
            self._expect_array(bound_args.get("frame_indices"), func_name, "frame_indices", node.lineno)
            self._validate_camera_index_bounds(
                bound_args.get("frame_indices"),
                func_name,
                "frame_indices",
                node.lineno,
            )
        elif func_name == "query_3d_object_position":
            self._expect_array(bound_args.get("category_names"), func_name, "category_names", node.lineno)
        elif func_name == "set_viewpoint":
            self._expect_vector3(bound_args.get("origin"), func_name, "origin", node.lineno)
            has_forward = "forward" in bound_args
            has_look_at = "look_at" in bound_args
            if not has_forward and not has_look_at:
                self.errors.append(
                    f"`{func_name}` at line {node.lineno} requires either `forward` or `look_at`."
                )
            if has_forward:
                self._expect_vector3(bound_args.get("forward"), func_name, "forward", node.lineno)
            if has_look_at:
                self._expect_vector3(bound_args.get("look_at"), func_name, "look_at", node.lineno)
            if "up" in bound_args:
                self._expect_vector3(bound_args.get("up"), func_name, "up", node.lineno)
        elif func_name == "safe_select":
            self._expect_object_position_map(
                bound_args.get("obj_queried"), func_name, "obj_queried", node.lineno
            )
            self._expect_string_like(bound_args.get("obj_name"), func_name, "obj_name", node.lineno)
            if "selection_criteria" in bound_args:
                self._expect_string_like(
                    bound_args.get("selection_criteria"), func_name, "selection_criteria", node.lineno
                )
        elif func_name == "step_camera":
            self._expect_string_enum(
                bound_args.get("direction"),
                func_name,
                "direction",
                {"forward", "backward", "left", "right", "up", "down"},
                node.lineno,
            )
        elif func_name == "turn_camera":
            self._expect_string_enum(
                bound_args.get("direction"),
                func_name,
                "direction",
                {"left", "right", "up", "down", "back"},
                node.lineno,
            )
            direction_literal = self._literal_string_value(bound_args.get("direction"))
            if "angle" in bound_args:
                self._expect_number(bound_args.get("angle"), func_name, "angle", node.lineno)
                angle_literal = self._literal_number_value(bound_args.get("angle"))
                if direction_literal == "back":
                    self.errors.append(
                        f"`{func_name}` at line {node.lineno} should not pass both `direction='back'` and `angle`. "
                        "`turn_camera(direction='back')` already means a 180-degree turn."
                    )
                if direction_literal != "back" and angle_literal == 180:
                    pass
        elif func_name == "render_rgb_bev":
            if "annotations" in bound_args:
                self._expect_array(bound_args.get("annotations"), func_name, "annotations", node.lineno)
            if "ego_marker_size" in bound_args:
                self._expect_integer(bound_args.get("ego_marker_size"), func_name, "ego_marker_size", node.lineno)
        elif func_name == "render_semantic_bev":
            if "camera_indices" in bound_args:
                self._expect_array(bound_args.get("camera_indices"), func_name, "camera_indices", node.lineno)
                self._validate_camera_index_bounds(
                    bound_args.get("camera_indices"),
                    func_name,
                    "camera_indices",
                    node.lineno,
                )
            if "objects" in bound_args:
                self._expect_object_list(bound_args.get("objects"), func_name, "objects", node.lineno)
            if "queried_objects" in bound_args:
                self._expect_object_position_map(
                    bound_args.get("queried_objects"),
                    func_name,
                    "queried_objects",
                    node.lineno,
                )

    def _validate_build_static_spatial_memory_args(self, node: ast.Call, bound_args):
        func_name = "build_static_spatial_memory"
        input_type_expr = bound_args.get("input_type")
        input_type_literal = self._literal_string_value(input_type_expr)
        if input_type_literal is not None and input_type_literal not in {"video", "images"}:
            self.errors.append(
                f"`{func_name}` at line {node.lineno} requires `input_type` to be either "
                "`\"video\"` or `\"images\"`."
            )

        if "fps" in bound_args:
            self._expect_number(bound_args.get("fps"), func_name, "fps", node.lineno)
        if "video_path" in bound_args:
            self._expect_string_like(bound_args.get("video_path"), func_name, "video_path", node.lineno)
        if "image_paths" in bound_args:
            self._expect_array(bound_args.get("image_paths"), func_name, "image_paths", node.lineno)

        if input_type_literal == "video" and "video_path" not in bound_args:
            self.errors.append(
                f"`{func_name}` at line {node.lineno} requires `video_path` when `input_type=\"video\"`."
            )
        if input_type_literal == "images" and "image_paths" not in bound_args:
            self.errors.append(
                f"`{func_name}` at line {node.lineno} requires `image_paths` when `input_type=\"images\"`."
            )

    def _validate_session_id_arg(self, func_name: str, node: ast.Call, bound_args):
        if "session_id" in bound_args:
            self._expect_string_like(bound_args.get("session_id"), func_name, "session_id", node.lineno)

    def _expect_array(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        if expr is None:
            return
        if isinstance(expr, (ast.List, ast.Tuple)):
            for elt in expr.elts:
                self._validate_expr(elt)
            return
        inferred = self._infer_expr_type(expr)
        if inferred is not None and inferred[0] == "list":
            return
        self.errors.append(
            f"`{func_name}` at line {lineno} expects `{arg_name}` to be a list or tuple."
        )

    def _validate_camera_index_bounds(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        max_valid_index = getattr(self, "input_image_count", None)
        if max_valid_index is None or expr is None:
            return
        literal_indices = self._literal_integer_list_value(expr)
        if literal_indices is None and isinstance(expr, ast.Name):
            literal_indices = getattr(self, "literal_integer_lists", {}).get(expr.id)

        if not literal_indices:
            return

        max_index = max(literal_indices)
        if max_index > max_valid_index:
            self.errors.append(
                f"`{func_name}` at line {lineno} uses `{arg_name}` with max camera/frame index "
                f"{max_index}, but only {max_valid_index} input image(s) are visible to the VLM. "
                "For video inputs, validate against the VLM preview frame count, not the "
                "reconstruction frame count."
            )

    def _expect_vector3(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        if expr is None:
            return
        if isinstance(expr, (ast.List, ast.Tuple)):
            if len(expr.elts) != 3:
                self.errors.append(
                    f"`{func_name}` at line {lineno} expects `{arg_name}` to contain exactly 3 values."
                )
            for elt in expr.elts:
                self._validate_expr(elt)
            return
        inferred = self._infer_expr_type(expr)
        if inferred is not None and inferred[0] == "list":
            return
        self.errors.append(
            f"`{func_name}` at line {lineno} expects `{arg_name}` to be a 3D vector."
        )

    def _expect_string_enum(self, expr: ast.AST, func_name: str, arg_name: str, allowed_values: Set[str], lineno: int):
        if expr is None:
            return
        literal = self._literal_string_value(expr)
        if literal is not None and literal not in allowed_values:
            self.errors.append(
                f"`{func_name}` at line {lineno} expects `{arg_name}` to be one of {sorted(allowed_values)}, "
                f"got `{literal}`."
            )
            return
        self._expect_string_like(expr, func_name, arg_name, lineno)

    def _expect_string_like(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        if expr is None:
            return
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return
        inferred = self._infer_expr_type(expr)
        if inferred is not None and inferred[0] == "scalar" and inferred[1] in {"string", "image_path"}:
            return
        self.errors.append(
            f"`{func_name}` at line {lineno} expects `{arg_name}` to be a string-like value."
        )

    def _expect_number(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        if expr is None:
            return
        if self._is_numeric_expr(expr):
            return
        inferred = self._infer_expr_type(expr)
        if inferred is not None and inferred == ("scalar", "number"):
            return
        self.errors.append(
            f"`{func_name}` at line {lineno} expects `{arg_name}` to be a numeric value."
        )

    def _expect_integer(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        if expr is None:
            return
        literal = self._literal_integer_value(expr)
        if literal is not None:
            return
        inferred = self._infer_expr_type(expr)
        if inferred is not None and inferred == ("scalar", "number"):
            return
        self.errors.append(
            f"`{func_name}` at line {lineno} expects `{arg_name}` to be an integer."
        )

    def _expect_object_list(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        if expr is None:
            return
        if isinstance(expr, ast.List):
            for elt in expr.elts:
                self._validate_bev_object_literal(elt, func_name, lineno)
            return
        inferred = self._infer_expr_type(expr)
        if inferred is not None and inferred[0] == "list":
            return
        self.errors.append(
            f"`{func_name}` at line {lineno} expects `{arg_name}` to be a list of object dictionaries."
        )

    def _validate_bev_object_literal(self, expr: ast.AST, func_name: str, lineno: int):
        if not isinstance(expr, ast.Dict):
            self.errors.append(
                f"`{func_name}` at line {lineno} expects each object in `objects` to be a dictionary."
            )
            return
        literal_keys = set()
        for key_node, value_node in zip(expr.keys, expr.values):
            key = self._literal_string_value(key_node)
            if key is None:
                self.errors.append(
                    f"`{func_name}` at line {lineno} expects object dictionary keys to be string literals."
                )
                continue
            literal_keys.add(key)
            if key == "name":
                self._expect_string_like(value_node, func_name, "objects[].name", lineno)
            elif key == "position":
                self._expect_vector3(value_node, func_name, "objects[].position", lineno)
            elif key == "orientation":
                self._expect_vector3(value_node, func_name, "objects[].orientation", lineno)
            else:
                self.errors.append(
                    f"`{func_name}` at line {lineno} got unsupported object field `{key}`. "
                    "Allowed fields are `name`, `position`, and `orientation`."
                )
        if "name" not in literal_keys or "position" not in literal_keys:
            self.errors.append(
                f"`{func_name}` at line {lineno} requires each object in `objects` to include "
                "`name` and `position`."
            )

    def _validate_subscript(self, expr: ast.Subscript):
        self._validate_expr(expr.value)
        self._validate_expr(expr.slice)

        base_type = self._infer_expr_type(expr.value)
        if base_type is None:
            return

        kind = base_type[0]
        index_value = self._literal_subscript_value(expr.slice)

        if kind == "dict":
            allowed_keys = base_type[1]
            if (
                isinstance(index_value, str)
                and isinstance(allowed_keys, set)
                and index_value not in allowed_keys
            ):
                self.errors.append(
                    f"Invalid key `{index_value}` at line {expr.lineno}. "
                    f"Allowed keys here are: {sorted(allowed_keys)}."
                )
            return

        if kind == "object_position_map":
            if not isinstance(index_value, str):
                self.errors.append(
                    f"Object-position query results must first be indexed by a string category name "
                    f"(line {expr.lineno})."
                )
            return

        if kind == "list":
            if not isinstance(index_value, int):
                self.errors.append(
                    f"List-like value at line {expr.lineno} must be indexed by an integer."
                )
            return

    def _expect_object_position_map(self, expr: ast.AST, func_name: str, arg_name: str, lineno: int):
        if expr is None:
            return
        inferred = self._infer_expr_type(expr)
        if inferred is None:
            self.errors.append(
                f"`{func_name}` at line {lineno} expects `{arg_name}` to be the dictionary returned by "
                "`query_3d_object_position(...)`."
            )
            return
        if inferred[0] not in {"object_position_map", "dict_any"}:
            self.errors.append(
                f"`{func_name}` at line {lineno} expects `{arg_name}` to be the dictionary returned by "
                "`query_3d_object_position(...)`."
            )
            return

    def _infer_expr_type(self, expr: ast.AST):
        if isinstance(expr, ast.Name):
            return self.name_types.get(expr.id)

        if self._literal_integer_list_value(expr) is not None:
            return ("list", "number")

        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            func_name = expr.func.id
            if func_name == "build_static_spatial_memory":
                return ("dict", {"session_id", "meta_info"})
            if func_name == "query_camera_pose":
                return ("list", "camera_pose")
            if func_name == "query_3d_object_position":
                return ("object_position_map", None)
            if func_name == "safe_select":
                return ("dict", {"category", "position", "instance_id"})
            if func_name in {"set_viewpoint", "step_camera", "turn_camera"}:
                return ("dict", {"messages"})
            if func_name == "render_ego_rgb":
                return ("scalar", "image_path")
            if func_name == "render_rgb_bev":
                return ("list", "image_path")
            if func_name == "render_semantic_bev":
                return ("dict", {"image", "images", "meta"})
            return None

        if isinstance(expr, ast.List):
            return ("list", "unknown")

        if isinstance(expr, ast.Subscript):
            base_type = self._infer_expr_type(expr.value)
            if base_type is None:
                return None
            kind = base_type[0]
            index_value = self._literal_subscript_value(expr.slice)

            if kind == "dict":
                if not isinstance(index_value, str):
                    return None
                key_type_map = {
                    "session_id": ("scalar", "string"),
                    "meta_info": ("dict_any", None),
                    "messages": ("list", "string"),
                    "image": ("scalar", "image_path"),
                    "images": ("list", "image_path"),
                    "meta": ("dict_any", None),
                    "camera_index": ("scalar", "number"),
                    "position": ("list", "number"),
                    "up": ("list", "number"),
                    "forward": ("list", "number"),
                    "right": ("list", "number"),
                }
                return key_type_map.get(index_value)

            if kind == "object_position_map":
                if isinstance(index_value, str):
                    return ("list", "vector")
                return None

            if kind == "list":
                if not isinstance(index_value, int):
                    return None
                element_kind = base_type[1]
                if element_kind == "camera_pose":
                    return ("dict", {"camera_index", "position", "up", "forward", "right"})
                if element_kind == "image_path":
                    return ("scalar", "image_path")
                if element_kind == "vector":
                    return ("list", "number")
                if element_kind == "number":
                    return ("scalar", "number")
                if element_kind == "string":
                    return ("scalar", "string")
            return None

        return None

    def _literal_subscript_value(self, slice_node: ast.AST):
        if isinstance(slice_node, ast.Constant):
            return slice_node.value
        if isinstance(slice_node, ast.UnaryOp) and isinstance(slice_node.op, ast.USub):
            if isinstance(slice_node.operand, ast.Constant) and isinstance(slice_node.operand.value, (int, float)):
                return -slice_node.operand.value
        return None

    def _literal_string_value(self, expr: ast.AST):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        return None

    def _literal_integer_value(self, expr: ast.AST):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return expr.value
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            if isinstance(expr.operand, ast.Constant) and isinstance(expr.operand.value, int):
                return -expr.operand.value
        return None

    def _literal_integer_list_value(self, expr: ast.AST):
        if not isinstance(expr, (ast.List, ast.Tuple)):
            return self._literal_integer_list_from_list_range(expr)
        values = []
        for elt in expr.elts:
            value = self._literal_integer_value(elt)
            if value is None:
                return None
            values.append(value)
        return values

    def _literal_integer_list_from_list_range(self, expr: ast.AST):
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

        values = [self._literal_integer_value(arg) for arg in range_call.args]
        if any(value is None for value in values):
            return None
        try:
            return list(range(*values))
        except ValueError:
            return None

    def _literal_number_value(self, expr: ast.AST):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool):
            return expr.value
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            if isinstance(expr.operand, ast.Constant) and isinstance(expr.operand.value, (int, float)):
                return -expr.operand.value
        return None

    def _is_numeric_expr(self, expr: ast.AST) -> bool:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool):
            return True
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            return self._is_numeric_expr(expr.operand)
        return False
