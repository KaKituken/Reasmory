ALLOWED_TOOL_NAMES = {
    "build_static_spatial_memory",
    "query_camera_pose",
    "query_3d_object_position",
    "set_viewpoint",
    "safe_select",
    "step_camera",
    "turn_camera",
    "render_ego_rgb",
    "render_rgb_bev",
    "render_semantic_bev",
}


TOOL_ARG_SCHEMAS = {
    "build_static_spatial_memory": {
        "params": ["input_type", "video_path", "fps", "image_paths"],
        "required": {"input_type"},
    },
    "query_camera_pose": {
        "params": ["session_id", "frame_indices"],
        "required": {"session_id", "frame_indices"},
    },
    "query_3d_object_position": {
        "params": ["session_id", "category_names"],
        "required": {"session_id", "category_names"},
    },
    "set_viewpoint": {
        "params": ["session_id", "origin", "forward", "look_at", "up"],
        "required": {"session_id", "origin"},
    },
    "safe_select": {
        "params": ["session_id", "obj_queried", "obj_name", "selection_criteria"],
        "required": {"session_id", "obj_queried", "obj_name"},
    },
    "step_camera": {
        "params": ["session_id", "direction"],
        "required": {"session_id", "direction"},
    },
    "turn_camera": {
        "params": ["session_id", "direction", "angle"],
        "required": {"session_id", "direction"},
    },
    "render_ego_rgb": {
        "params": ["session_id"],
        "required": {"session_id"},
    },
    "render_rgb_bev": {
        "params": ["session_id", "annotations", "ego_marker_size"],
        "required": {"session_id"},
    },
    "render_semantic_bev": {
        "params": ["session_id", "camera_indices", "objects", "queried_objects"],
        "required": {"session_id"},
    },
}
