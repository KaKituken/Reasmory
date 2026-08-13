import ast


class SyntaxValidationMixin:
    def visit_Module(self, node: ast.Module):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.FunctionDef):
            self.errors.append(
                "The code must contain exactly one top-level function definition."
            )
            return
        self.visit(node.body[0])

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name != "plan_to_solve_problem":
            self.errors.append(
                f"The function name must be `plan_to_solve_problem`, got `{node.name}`."
            )
        if node.decorator_list:
            self.errors.append("Decorators are not allowed on the plan function.")
        if getattr(node, "type_params", None):
            self.errors.append("Generic type parameters are not allowed on the plan function.")
        if node.returns is not None:
            self.errors.append("A return annotation is not allowed on the plan function.")
        arg_names = [arg.arg for arg in node.args.args]
        allowed_arg_names = {"input_images", "input_video"}
        if len(arg_names) != 1 or arg_names[0] not in allowed_arg_names:
            self.errors.append(
                "The function must have exactly one argument named `input_images` or `input_video`."
            )
        if node.args.defaults or any(default is not None for default in node.args.kw_defaults):
            self.errors.append("Default argument values are not allowed.")
        if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            self.errors.append("The function may not use *args, **kwargs, or keyword-only args.")
        expected_annotations = {"input_images": "List[str]", "input_video": "str"}
        for arg in node.args.args:
            expected = expected_annotations.get(arg.arg)
            if expected is None:
                continue
            if arg.annotation is None or not self._is_allowed_arg_annotation(arg.arg, arg.annotation):
                self.errors.append(
                    f"The parameter `{arg.arg}` must be annotated as `{expected}`."
                )

        if not node.body or not isinstance(node.body[-1], ast.Return):
            self.errors.append("The function must end with `return useful_observation`.")
        elif not isinstance(node.body[-1].value, ast.Name) or node.body[-1].value.id != "useful_observation":
            self.errors.append("The last statement must be exactly `return useful_observation`.")

        input_arg_name = arg_names[0] if arg_names else "input_images"
        input_arg_kind = "video_path" if input_arg_name == "input_video" else "image_list"
        self._reset_function_validation_state(input_arg_name, input_arg_kind)
        for stmt in node.body:
            self._validate_statement(stmt)

    def _is_allowed_arg_annotation(self, arg_name: str, annotation: ast.AST) -> bool:
        if arg_name == "input_images":
            if not isinstance(annotation, ast.Subscript):
                return False
            if not isinstance(annotation.value, ast.Name) or annotation.value.id != "List":
                return False
            slice_node = annotation.slice
            return isinstance(slice_node, ast.Name) and slice_node.id == "str"
        if arg_name == "input_video":
            return isinstance(annotation, ast.Name) and annotation.id == "str"
        return False
