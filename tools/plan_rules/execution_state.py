class ExecutionOrderStateMixin:
    def _reset_function_validation_state(self, input_arg_name="input_images", input_arg_kind="image_list"):
        self.defined_names = {input_arg_name}
        self.has_seen_set_viewpoint = False
        self.has_seen_reference_query = False
        if input_arg_kind == "video_path":
            self.name_types = {input_arg_name: ("scalar", "string")}
        else:
            self.name_types = {input_arg_name: ("list", "image_path")}
        self.query_derived_names = set()
        self.stale_query_names = set()
        self.literal_integer_lists = {}

    def _validate_execution_order_for_call(self, func_name, node):
        if func_name == "set_viewpoint" and not self.has_seen_reference_query:
            self.errors.append(
                f"`set_viewpoint` at line {node.lineno} requires a prior reference query. "
                "Before choosing a viewpoint, first identify a meaningful reference by calling "
                "`query_camera_pose(...)` for a specific camera view or "
                "`query_3d_object_position(...)` for a specific point/object in space."
            )

        if (
            func_name.startswith("render_")
            and not self.has_seen_set_viewpoint
            and not getattr(self, "allow_global_render_without_viewpoint", False)
        ):
            self.errors.append(
                f"`{func_name}` at line {node.lineno} requires a prior `set_viewpoint(...)` call. "
                "World coordinates are not a meaningful final reference frame for rendering. "
                "Before rendering, select a concrete reference viewpoint by first calling "
                "`query_camera_pose(...)` for a relevant camera view or "
                "`query_3d_object_position(...)` for a relevant spatial point, then use "
                "`set_viewpoint(...)`."
            )

    def _update_execution_order_after_call(self, func_name):
        if func_name in {"query_camera_pose", "query_3d_object_position", "safe_select"}:
            self.has_seen_reference_query = True
        if func_name == "set_viewpoint":
            self.has_seen_set_viewpoint = True
        if func_name in {"set_viewpoint", "step_camera", "turn_camera"}:
            self.stale_query_names.update(self.query_derived_names)
