import shutil
from time import time
import hashlib
import torch
import cv2
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool, BaseToolWithFileAccess
from qwen_agent.utils.output_beautify import typewriter_print
from qwen_agent.log import logger
from qwen_agent.llm.schema import ContentItem, Message
from qwen_agent.utils.utils import extract_images_from_messages
import os
import sys
import uuid
import gc
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from typing import Any, Dict, List
from pathlib import Path
import torchvision.transforms.functional as F
import numpy as np
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import json
import re

from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor

from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3 import build_sam3_image_model
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

from sam3.eval.postprocessors import PostProcessImage
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device

repo_root = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    from tools.vis_utils import (
        export_category_pos_to_html,
        save_pointcloud_with_vector_html,
        render_camera_trajectory,
        render_semantic_bev_entities,
    )
    from tools.run_time import Runtime
    from tools.spatial_memory import SpatialMemory, array_to_printable_list
    from evaluation.utils import extract_video_frames
except ImportError:
    from vis_utils import (
        export_category_pos_to_html,
        save_pointcloud_with_vector_html,
        render_camera_trajectory,
        render_semantic_bev_entities,
    )
    from run_time import Runtime
    from spatial_memory import SpatialMemory, array_to_printable_list
    from evaluation.utils import extract_video_frames

runtime = Runtime()


def get_managed_workspace_dir(*parts: str) -> str:
    """Scratch dir for rendered tool artifacts.

    Configured by REASMORY_WORKSPACE_ROOT (preferred) or the legacy DATA_DISK,
    falling back to /tmp/workspace so a fresh checkout still runs.
    """
    workspace_root = os.environ.get("REASMORY_WORKSPACE_ROOT", "").strip()
    if workspace_root:
        base_dir = workspace_root
    else:
        data_disk = os.environ.get("DATA_DISK")
        base_dir = (
            os.path.join(data_disk, "workspace")
            if data_disk
            else os.path.join("/tmp", "workspace")
        )
    output_dir = os.path.join(base_dir, *parts)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def extract_videos_from_messages(messages) -> List[str]:
    videos = []
    for msg in messages or []:
        content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if hasattr(item, 'video') and item["video"]:
                videos.append(item["video"])
    return videos


def _normalize_local_media_path(path: str) -> str:
    if path.startswith("file://"):
        return path[len("file://"):]
    return path


def _extract_video_frames_cached(video_path: str, fps: float = 1.0, max_frames: int = 64) -> List[str]:
    video_path = _normalize_local_media_path(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    stat = os.stat(video_path)
    cache_key = hashlib.md5(
        f"{os.path.abspath(video_path)}::{stat.st_mtime_ns}::{stat.st_size}::{fps}::{max_frames}".encode("utf-8")
    ).hexdigest()
    cache_root = Path(get_managed_workspace_dir("cache", "video_frames_1fps"))
    cache_dir = cache_root / cache_key
    manifest_path = cache_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            cached_frames = manifest.get("frame_paths", [])
            if cached_frames and all(os.path.exists(p) for p in cached_frames):
                return cached_frames
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid video with zero frames: {video_path}")
    if original_fps <= 0:
        original_fps = 30.0
    frame_step = max(int(round(original_fps / fps)), 1)
    sampled_1fps_count = max((total_frames + frame_step - 1) // frame_step, 1)
    cap.release()

    frame_paths = extract_video_frames(video_path, fps=fps, num_frames=max_frames)
    manifest_path.write_text(
        json.dumps(
            {
                "video_path": video_path,
                "fps": fps,
                "max_frames": max_frames,
                "original_fps": original_fps,
                "sampled_1fps_count": sampled_1fps_count,
                "sampling_strategy": "min(fps_budget,num_frames_budget)",
                "frame_paths": frame_paths,
            },
            indent=2,
        )
    )
    return frame_paths


# Step 1: Add a custom tool.
@register_tool('mask_propagation')
class MaskPropagation(BaseToolWithFileAccess):
    # The `description` tells the agent the functionality of this tool.
    description = 'Propagate a mask from one image to other images in the scene.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = [{
        'name': 'image_index',
        'type': 'integer',
        'description': 'The index of the image in the sequence to which the mask should be propagated.',
        'required': True
    },
    {
        'name': 'bbox_2d',
        'type': 'array',
        'items': {
            'type': 'integer'
        },
        'minItems': 4,
        'maxItems': 4,
        'description': 'The bounding box [x1, y1, x2, y2] defining the region of the mask in the source image.',
        'required': True
    }
    ]

    def call_test(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        # import ipdb; ipdb.set_trace()
        params = self._verify_json_format_args(params)

        img_idx = params['image_index'] - 1  # Convert to zero-based index
        bbox_param = params['bbox_2d']
        images = extract_images_from_messages(kwargs.get('messages', []))
        os.makedirs(self.work_dir, exist_ok=True)

        try:
            # open image, currently only support the first image
            image_arg = images[img_idx]
            if image_arg.startswith('file://'):
                image_arg = image_arg[len('file://'):]

            if image_arg.startswith('http'):
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                image = Image.open(image_arg)
            else:
                image = Image.open(os.path.join(self.work_dir, image_arg))
        except Exception as e:
            logger.warning(f'{e}')
            return [ContentItem(text=f'Error: Invalid input image {images}')]

        try:
            # Support a single bbox or a list of bboxes
            if not isinstance(bbox_param, (list, tuple)):
                raise ValueError('bbox_2d must be a list of coordinates or list of bboxes.')
            if len(bbox_param) == 4 and all(isinstance(v, (int, float)) for v in bbox_param):
                bboxes = [bbox_param]
            else:
                bboxes = bbox_param

            if not bboxes:
                raise ValueError('bbox_2d is empty.')

            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)

            img_width, img_height = image.size
            stroke_width = max(2, int(min(img_width, img_height) * 0.003))

            for bbox in bboxes:
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError('Each bbox must be a list of 4 numbers.')

                # rel_x1, rel_y1, rel_x2, rel_y2 = bbox
                rel_y1, rel_x1, rel_y2, rel_x2 = bbox
                abs_x1 = max(0, min(img_width, rel_x1 / 1000.0 * img_width))
                abs_y1 = max(0, min(img_height, rel_y1 / 1000.0 * img_height))
                abs_x2 = max(0, min(img_width, rel_x2 / 1000.0 * img_width))
                abs_y2 = max(0, min(img_height, rel_y2 / 1000.0 * img_height))

                if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                    raise ValueError(f'Invalid bbox coordinates: {bbox}')

                draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline='red', width=stroke_width)

            output_path = os.path.abspath(os.path.join(self.work_dir, f'{uuid.uuid4()}.png'))
            # annotated_image.save(output_path)
            # import ipdb; ipdb.set_trace()
            # return [ContentItem(image=output_path)]
            return [
                ContentItem(image='workspace/tools/mask_propagation/frame_0000.png'),
                ContentItem(image='workspace/tools/mask_propagation/frame_0030.png'),
                ContentItem(image='workspace/tools/mask_propagation/frame_0035.png'),
                ContentItem(image='workspace/tools/mask_propagation/frame_0180.png'),
            ]
        except Exception as e:
            obs = f'Tool Execution Error {str(e)}'
            return [ContentItem(text=obs)]

    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        global runtime
        params = self._verify_json_format_args(params)

        if not runtime.already_initialized:
            runtime.ensure_pi3()
            runtime.ensure_sam2()
            org_input_images = []
            org_input_images = extract_images_from_messages(kwargs.get('messages', [])[:2])
            fixed_org_input_images = []
            for img_path in org_input_images:
                if img_path.startswith('file://'):
                    fixed_org_input_images.append(img_path[len('file://'):])
                else:
                    fixed_org_input_images.append(img_path)
            runtime.ensure_spatial_memory(
                session_id='default',
                image_paths=fixed_org_input_images,
                construct_3d_spatial_memory_fn=construct_3d_spatial_memory,
            )

        sam2_model = runtime.sam2_model
        sam2_processor = runtime.sam2_processor
        spatial_memory = runtime.session_mem['default']
        img_idx = params['image_index'] - 1  # Convert to zero-based index
        bbox_param = params['bbox_2d']
        # import ipdb; ipdb.set_trace()
        
        messages = kwargs.get('messages', [])
        first_contain_img_index = 1
        # for i in range(len(messages)):
        #     if 'image' in messages[i].get('content', {}):
        #         first_contain_img_index = i
        #         break
        last_prop_index = first_contain_img_index
        for i in range(len(messages)-1, first_contain_img_index-1, -1):  # 从后往前找
            if 'tool_calls' in messages[i]:
                for call in messages[i]['tool_calls']:
                    if call['tool_name'] == 'mask_propagation':
                        last_prop_index = call['params']['image_index'] - 1
                        break
            if last_prop_index != first_contain_img_index:
                break
        # !BUG: we manually exclude the last round.
        images = extract_images_from_messages(messages[last_prop_index:last_prop_index+1])
        os.makedirs(self.work_dir, exist_ok=True)

        try:
            # open image, currently only support the first image
            image_arg = images[img_idx]
            if image_arg.startswith('file://'):
                image_arg = image_arg[len('file://'):]

            if image_arg.startswith('http'):
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                image = Image.open(image_arg)
            else:
                image = Image.open(os.path.join(self.work_dir, image_arg))
        except Exception as e:
            logger.warning(f'{e}')
            return [ContentItem(text=f'Error: Invalid input image {images}')]

        try:
            # Support a single bbox or a list of bboxes
            if not isinstance(bbox_param, (list, tuple)):
                raise ValueError('bbox_2d must be a list of coordinates or list of bboxes.')
            if len(bbox_param) == 4 and all(isinstance(v, (int, float)) for v in bbox_param):
                bboxes = [bbox_param]
            else:
                bboxes = bbox_param

            if not bboxes:
                raise ValueError('bbox_2d is empty.')

            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            annotated_image_path = os.path.abspath(os.path.join(self.work_dir, f'input_bbox_{uuid.uuid4()}.png'))

            img_width, img_height = image.size
            stroke_width = max(2, int(min(img_width, img_height) * 0.003))

            bbox_output_paths = []

            for tgt_idx in range(len(images)):
                # import ipdb; ipdb.set_trace()
                img_tgt_arg = images[tgt_idx]
                if img_tgt_arg.startswith('file://'):
                    img_tgt_arg = img_tgt_arg[len('file://'):]
                if img_tgt_arg.startswith('http'):
                    response = requests.get(img_tgt_arg)
                    response.raise_for_status()
                    img_tgt = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    img_tgt = Image.open(img_tgt_arg).convert("RGB")
                draw_bbox = ImageDraw.Draw(img_tgt)
                composite_mask = np.zeros((*img_tgt.size[::-1], 3), dtype=np.uint8)

                for obj_idx, bbox in enumerate(bboxes):
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        raise ValueError('Each bbox must be a list of 4 numbers.')

                    rel_x1, rel_y1, rel_x2, rel_y2 = bbox
                    abs_x1 = max(0, min(img_width, rel_x1 / 1000.0 * img_width))
                    abs_y1 = max(0, min(img_height, rel_y1 / 1000.0 * img_height))
                    abs_x2 = max(0, min(img_width, rel_x2 / 1000.0 * img_width))
                    abs_y2 = max(0, min(img_height, rel_y2 / 1000.0 * img_height))

                    if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                        raise ValueError(f'Invalid bbox coordinates: {bbox}')

                    if tgt_idx == img_idx:
                        draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline='red', width=stroke_width)
                        draw_bbox.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline='red', width=stroke_width)
                        continue

                    # propogate to other images
                    # get the mask in pixel space
                    input_boxes = [[[abs_x1, abs_y1, abs_x2, abs_y2]]]
                    inputs = sam2_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(sam2_model.device)
                    with torch.no_grad():
                        outputs = sam2_model(**inputs)
                    masks = sam2_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
                    mask = masks[0, -1]

                    memory_h, memory_w = spatial_memory['3d_positions'].shape[1:3]
                    # import ipdb; ipdb.set_trace()
                    # resize mask
                    mask_resized = F.resize(mask.unsqueeze(0).float(), size=(memory_h, memory_w), interpolation=F.InterpolationMode.NEAREST).squeeze(0).bool()
                    points_src = spatial_memory['3d_positions'][img_idx][mask_resized]
                    points_tgt = spatial_memory['3d_positions'][tgt_idx]
                    propagated_mask = self.propagate_mask_3d(points_src, points_tgt, eps=0.03)
                    propagated_mask = F.resize(torch.from_numpy(propagated_mask).unsqueeze(0).float(), size=img_tgt.size[::-1], interpolation=F.InterpolationMode.NEAREST).squeeze(0).bool().numpy()

                    # --- assign unique color for this object ---
                    color = self.get_color(obj_idx)
                    composite_mask[propagated_mask] = color

                    # --- draw bbox for propagated mask ---
                    coords = np.argwhere(propagated_mask)
                    if coords.size > 0:
                        min_y, min_x = coords.min(axis=0)
                        max_y, max_x = coords.max(axis=0)
                        draw_bbox.rectangle([min_x, min_y, max_x, max_y], outline=color, width=3)

                # --- save images ---
                mask_img = Image.fromarray(composite_mask)
                mask_output_path = os.path.abspath(os.path.join(self.work_dir, f'prop_mask_img_{img_idx}_{tgt_idx}_{uuid.uuid4()}.png'))
                mask_img.save(mask_output_path)

                bbox_output_path = os.path.abspath(os.path.join(self.work_dir, f'prop_bbox_img_{img_idx}_{tgt_idx}_{uuid.uuid4()}.png'))
                img_tgt.save(bbox_output_path)
                bbox_output_paths.append(bbox_output_path)

                annotated_image.save(annotated_image_path)

            # return [ContentItem(image=output_path)]
            content = []
            for idx, path in enumerate(bbox_output_paths):
                # content.append(ContentItem(text=f"Picture {idx+1}:"))
                content.append(ContentItem(image=path))
            return content
        except Exception as e:
            obs = f'Tool Execution Error {str(e)}'
            return [ContentItem(text=obs)]

    def get_color(self, index):
        cmap = plt.get_cmap("tab20")
        r, g, b, _ = cmap(index % 20)
        return tuple(int(255 * x) for x in (r, g, b))

    def propagate_mask_3d(self, points_src, points_tgt, eps=0.03):
        """
        points_src: [N, 3]
        points_tgt: [H, W, 3]
        """
        H, W, _ = points_tgt.shape
        pts_flat = points_tgt.reshape(-1, 3)

        tree = cKDTree(points_src)
        idxs = tree.query_ball_point(pts_flat, r=eps)

        mask_flat = np.array([len(x) > 0 for x in idxs])
        mask = mask_flat.reshape(H, W)
        return mask


@register_tool('frame_selection')
class FrameSelection(BaseToolWithFileAccess):
    # The `description` tells the agent the functionality of this tool.
    description = 'Select a frame from the video or multiple image inputs to inspect closely.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = [{
        'name': 'image_index',
        'type': 'integer',
        'description': 'The index of the image in the sequence to be selected. It is a 1-based index, which means the first image is indexed as 1, the second image is indexed as 2, and so on.',
        'required': True
    }
    ]

    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        # import ipdb; ipdb.set_trace()
        params = self._verify_json_format_args(params)

        img_idx = params['image_index'] - 1  # Convert to zero-based index
        images = extract_images_from_messages(kwargs.get('messages', []))
        os.makedirs(self.work_dir, exist_ok=True)

        try:
            # open image, currently only support the first image
            image_arg = images[img_idx]
            if image_arg.startswith('file://'):
                image_arg = image_arg[len('file://'):]

            if image_arg.startswith('http'):
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                image = Image.open(image_arg)
            else:
                image = Image.open(os.path.join(self.work_dir, image_arg))
            output_path = os.path.abspath(os.path.join(self.work_dir, f'{uuid.uuid4()}.png'))
            image.save(output_path)
            return [ContentItem(image=output_path)]
        except Exception as e:
            obs = f'Tool Execution Error {str(e)}'
            return [ContentItem(text=obs)]

# =================== Memory Entry Tools ===================
# @register_tool('build_static_spatial_memory_pi3')
# class BuildStaticSpatialMemoryPi3(BaseToolWithFileAccess):
@register_tool('build_static_spatial_memory')
class BuildStaticSpatialMemory(BaseToolWithFileAccess):
    # The `description` tells the agent the functionality of this tool.
    description = (
        "Build a reusable 3D spatial memory for a static scene from either a video or multiple images of the same scene.\n\n"
        "Use this tool at the beginning of a task when you need 3D spatial reasoning, camera poses, or view transformations. "
        "The returned spatial memory can be reused across later tool calls through the returned `session_id`, "
        "so you usually only need to build it once for the same scene.\n\n"
        "This tool is intended for static scenes with little or no object motion. "
        "If the scene contains significant motion or dynamic objects, use `build_dynamic_spatial_memory` instead.\n\n"
        "It returns a `session_id` and metadata about the constructed memory."
    )
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = [
        {
            'name': 'input_type',
            'type': 'string',
            'enum': ['video', 'images'],
            'description': (
                "Input format for building the spatial memory. "
                "Use `video` for a single video, or `images` for multiple images of the same scene."
            ),
            'required': True
        },
        {
            'name': 'video_path',
            'type': 'string',
            'description': (
                "Path to the input video. Required when `input_type` is `video`. "
                "Ignored when `input_type` is `images`."
            ),
            'required': False
        },
        {
            'name': 'image_paths',
            'type': 'array',
            'items': {'type': 'string'},
            'description': (
                "List of image paths from the same static scene. "
                "Required when `input_type` is `images`. Ignored when `input_type` is `video`."
            ),
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        global runtime
        # import ipdb; ipdb.set_trace()
        params = self._verify_json_format_args(params)
        precomputed_spatial_memory_path = kwargs.get("precomputed_spatial_memory_path")
        if precomputed_spatial_memory_path:
            print("Loading cached spatial memory...")
            precomputed_spatial_memory_path = _normalize_local_media_path(
                str(precomputed_spatial_memory_path)
            )
            if os.path.exists(precomputed_spatial_memory_path):
                cur_time = int(time()) % 10000
                session_id = f'session_{cur_time}'
                spatial_memory = runtime.load_spatial_memory_cache(
                    session_id=session_id,
                    cache_path=precomputed_spatial_memory_path,
                )
                result = {
                    "session_id": session_id,
                    "meta_info": spatial_memory.meta_info,
                    "loaded_from_cache": True,
                    "cache_path": precomputed_spatial_memory_path,
                }
                return [
                    ContentItem(text='Spatial memory loaded from cache successfully.'),
                    ContentItem(text=json.dumps(result)),
                    ContentItem(text="**Spatial memory cache hit: execution skipped reconstruction and reused a precomputed memory file.**"),
                ]
        messages = kwargs.get('messages', [])
        image_list = [
            _normalize_local_media_path(p)
            for p in extract_images_from_messages(messages)
        ]
        video_list = [
            _normalize_local_media_path(p)
            for p in extract_videos_from_messages(messages)
        ]

        if image_list:
            media_image_list = image_list
        elif video_list:
            media_image_list = _extract_video_frames_cached(video_list[0], fps=1.0, max_frames=64)
        elif params.get("input_type") == "video" and params.get("video_path"):
            media_image_list = _extract_video_frames_cached(params["video_path"], fps=1.0, max_frames=64)
        else:
            media_image_list = [
                _normalize_local_media_path(p)
                for p in params.get("image_paths", [])
            ]

        try:
            image_files = []
            for i in range(len(media_image_list)):
                image_path = media_image_list[i]
                if image_path.startswith('http'):
                    response = requests.get(image_path)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                elif os.path.exists(image_path):
                    image = Image.open(image_path)
                else:
                    image = Image.open(os.path.join(self.work_dir, image_path))
                image_files.append(image)
        except Exception as e:
            logger.warning(f'{e}')
            return [ContentItem(text=f'Error: Invalid input image/video-derived frames {media_image_list}')]
        # import ipdb; ipdb.set_trace()
        cur_time = int(time()) % 10000
        session_id = f'session_{cur_time}'
        runtime.ensure_spatial_memory(
            session_id=session_id,
            image_paths=media_image_list,
            construct_3d_spatial_memory_fn=construct_3d_spatial_memory,
        )
        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        # import ipdb; ipdb.set_trace()
        # save_pointcloud_with_vector_html(
        #     points=spatial_memory.position_3d.reshape(-1, 3),
        #     colors=spatial_memory.rgb_images.permute(0, 2, 3, 1).reshape(-1, 3),
        #     vector=spatial_memory.global_up,
        #     c2w_all=spatial_memory.camera_trajectory,
        #     filename='./debug_vis/02_24_spatial_memory_vis.html',
        #     downsample_ratio=0.01,
        #     point_size=1.0,
        # )
        result = {
            "session_id": session_id,
            "meta_info": spatial_memory.meta_info,
            "loaded_from_cache": False,
        }
        return [ContentItem(text=f'Spatial memory built successfully.'),
                ContentItem(text=json.dumps(result)),
                ContentItem(text="**Note that only the up direction of the coordinate system in the spatial memory is aligned with the global up direction of the scene by default. The forward and right directions are not determined and can be defined by the agent in subsequent tool calls, such as the `transform_spatial_memory` tool, to better suit the specific spatial reasoning tasks.**")]


@register_tool('build_dynamic_spatial_memory')
class BuildDynamicSpatialMemory(BaseToolWithFileAccess):
    description = (
        "Build a reusable spatio-temporal memory for a dynamic scene from a video. "
        "This memory represents both 3D spatial structure and temporal changes across frames.\n\n"
        "Use this tool when the scene contains moving objects, changing object positions, or when the question involves motion, "
        "temporal order, trajectory, or how a scene changes over time.\n\n"
        "Do not use this tool for purely static scenes unless temporal information is necessary, "
        "because dynamic memory is more complex than `build_static_spatial_memory`.\n\n"
        "The returned memory can be reused across later tool calls through the returned `session_id`."
    )

    parameters = [
        {
            'name': 'video_path',
            'type': 'string',
            'description': 'Path to the input video used to build the dynamic spatial memory.',
            'required': True
        },
        {
            'name': 'fps',
            'type': 'number',
            'description': (
                "Frame sampling rate used to process the video. "
                "A lower value is faster but may lose temporal detail."
            ),
            'required': False
        }
    ]


# =================== Memory Transformation Tools ===================
@register_tool('set_viewpoint')
class SetViewpoint(BaseTool):
    description = (
        "Set a new observation viewpoint for the current spatial memory by specifying an origin and orientation.\n\n"
        "Use this tool when you want to inspect the scene from a specific position and direction, "
        "for example from the viewpoint of a camera, an object, or any other 3D location.\n\n"
        "This tool changes how later rendering and reasoning tools observe the scene. "
        "It does not directly render an image. After setting a new viewpoint, you can use tools such as "
        "`render_ego_rgb`, `render_rgb_bev`, or `render_semantic_bev` to inspect the scene from that viewpoint.\n\n"
        "You should provide a 3D origin and a forward direction. "
        "The optional up direction does not need to be perfectly orthogonal to the forward direction; "
        "the tool will normalize and adjust the viewpoint internally."
        "This tool updates the active viewpoint of the spatial memory. "
        "After calling it, later query and render tools will use the new viewpoint."
    )
    parameters = [
        {
            'name': 'session_id',
            'type': 'string',
            'description': 'The session ID of the spatial memory to transform. This should be the same session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.',
            'required': True
        },
        {
            'name': 'origin',
            'type': 'array',
            'items': {
                'type': 'number'
            },
            'minItems': 3,
            'maxItems': 3,
            'description': 'The new origin point in the 3D space to which the spatial memory should be transformed. The format is [x, y, z].',
            'required': True
        },
        {
            'name': 'forward',
            'type': 'array',
            'items': {
                'type': 'number'
            },
            'minItems': 3,
            'maxItems': 3,
            'description': 'The new forward direction vector in the 3D space to which the spatial memory should be transformed. The format is [x, y, z]. Either `forward` or `look_at` must be provided.',
            'required': False
        },
        {
            'name': 'look_at',
            'type': 'array',
            'items': {
                'type': 'number'
            },
            'minItems': 3,
            'maxItems': 3,
            'description': 'A 3D target point [x, y, z] to look toward. If provided (and `forward` is omitted), the forward direction is computed automatically as the direction from `origin` to `look_at`. Use this to face an object whose position you obtained from `query_3d_object_position`.',
            'required': False
        },
        {
            'name': 'up',
            'type': 'array',
            'items': {
                'type': 'number'
            },
            'minItems': 3,
            'maxItems': 3,
            'description': 'The new up direction vector in the 3D space to which the spatial memory should be transformed. The format is [x, y, z]. By default, it is the same as the global up direction of the original spatial memory, which means only the forward direction is changed. You can specify a different up direction to achieve a more flexible transformation.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        global runtime
        params = self._verify_json_format_args(params)

        session_id = params['session_id']
        origin = params['origin']
        forward = params.get('forward', None)
        look_at = params.get('look_at', None)
        up = params.get('up', None)

        # `look_at` sugar: derive the forward direction from origin -> look_at.
        if forward is None and look_at is not None:
            direction = [float(look_at[i]) - float(origin[i]) for i in range(3)]
            norm = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
            if norm < 1e-8:
                return [ContentItem(text='Error: `look_at` target coincides with `origin`; cannot infer a forward direction.')]
            forward = [c / norm for c in direction]
        if forward is None:
            return [ContentItem(text='Error: `set_viewpoint` requires either `forward` or `look_at`.')]

        if session_id not in runtime.session_mem:
            return [ContentItem(text=f'Error: Session ID {session_id} not found. Please make sure to use the correct session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.')]

        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        result = spatial_memory.set_viewpoint(origin=origin, forward=forward, up=up)

        return [
            ContentItem(text=result['message']),
            ContentItem(text=f"Forward: {result['forward']}, right: {result['right']}."),
        ]


# Operation sugar for camera movement
@register_tool('step_camera')
class StepCamera(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    description = (
        "Move the current viewpoint by a small fixed distance while keeping the viewing direction unchanged.\n\n"
        "Use this tool when you want to inspect the scene from a nearby position without changing where the camera is facing, "
        "for example to step forward, backward, left, right, up, or down.\n\n"
        "This tool changes viewpoint position only. It does not rotate the camera. "
        "If you want to change the viewing direction while keeping the position fixed, use `turn_camera` instead.\n\n"
        "This is a convenient local-viewpoint adjustment tool. For arbitrary viewpoint control, use `set_viewpoint`."
        "This tool updates the active viewpoint. After calling it, later query and render tools will use the moved viewpoint."
    )
    parameters = [
        {
            'name': 'session_id',
            'type': 'string',
            'description': 'The session ID of the spatial memory.',
            'required': True
        },
        {
            'name': 'direction',
            'type': 'string',
            'enum': ['forward', 'backward', 'left', 'right', 'up', 'down'],
            'description': (
                "The direction in which to move the viewpoint. "
                "Directions are interpreted relative to the current viewpoint."
            ),
            'required': True
        }
    ]
    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        global runtime
        params = self._verify_json_format_args(params)

        session_id = params['session_id']
        direction = params['direction']

        if session_id not in runtime.session_mem:
            return [ContentItem(text=f'Error: Session ID {session_id} not found. Please make sure to use the correct session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.')]

        if direction not in ["forward", "backward", "left", "right", "up", "down"]:
            return [ContentItem(text=f'Error: Invalid direction {direction}. Valid values are ["forward", "backward", "left", "right", "up", "down"].')]
        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        result = spatial_memory.move_camera(direction=direction)

        return [ContentItem(text=result['message'])]


@register_tool('turn_camera')
class TurnCamera(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    description = (
        "Rotate the current viewpoint while keeping the position unchanged.\n\n"
        "Use this tool when you want to look in a different direction from the same location, "
        "for example to turn left, right, up, or down.\n\n"
        "This tool changes viewing direction only. It does not move the camera position. "
        "If you want to move to a new location while keeping the same orientation, use `step_camera` instead.\n\n"
        "This is a convenient local-viewpoint adjustment tool. For arbitrary viewpoint control, use `set_viewpoint`."
        "This tool updates the active viewpoint. After calling it, later query and render tools will use the rotated viewpoint."
    )
    parameters = [
        {
            'name': 'session_id',
            'type': 'string',
            'description': 'The session ID of the current spatial memory.',
            'required': True
        },
        {
            'name': 'direction',
            'type': 'string',
            'enum': ['left', 'right', 'up', 'down', 'back'],
            'description': (
                "The direction in which to rotate the viewpoint. "
                "Directions are interpreted relative to the current viewpoint."
            ),
            'required': True
        },
        {
            'name': 'angle',
            'type': 'number',
            'description': (
                "The angle in degrees to rotate the camera. "
                "The default value for left/right is 90 degrees, for up/down is 45 degrees."
            ),
            'required': False
        }
    ]
    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        global runtime
        params = self._verify_json_format_args(params)

        session_id = params['session_id']
        direction = params['direction']
        angle = params.get('angle', None)

        if session_id not in runtime.session_mem:
            return [ContentItem(text=f'Error: Session ID {session_id} not found. Please make sure to use the correct session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.')]

        if direction not in ["left", "right", "up", "down", "back"]:
            return [ContentItem(text=f'Error: Invalid direction {direction}. Valid values are ["left", "right", "up", "down", "back"].')]
        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        result = spatial_memory.rotate_camera(direction=direction, angle_deg=angle)

        return [ContentItem(text=result['message'])]


# ======================== Entity Query Tools =======================
@register_tool('query_camera_pose')
class QueryCameraPose(BaseToolWithFileAccess):
    # The `description` tells the agent the functionality of this tool.
    description = (
        "Query the camera poses of one or more frames from the current spatial memory.\n\n"
        "Use this tool when you need the camera position or viewing direction of specific frames, "
        "for example to reason about camera motion, compare viewpoints, choose a new viewpoint, "
        "or prepare later rendering steps such as BEV or ego-centric visualization.\n\n"
        "Important coordinate rule:\n"
        "- If no viewpoint-changing tool has been used, the returned poses are expressed in the original world coordinate system.\n"
        "- If the viewpoint has been changed by `set_viewpoint`, `step_camera`, or `turn_camera`, "
        "the returned poses are expressed in the current active viewpoint coordinate system.\n\n"
        "This tool returns, for each requested frame, the camera position, up vector, and forward vector "
        "under the current active viewpoint.\n\n"
        "When possible, query multiple frames at once, because comparing several camera poses together is often more useful "
        "than querying them one by one."
    )
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = [
        {
            'name': 'session_id',
            'type': 'string',
            'description': 'The session ID of the spatial memory to query. This should be the same session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.',
            'required': True
        },
        {
            'name': 'frame_indices',
            'type': 'array',
            'items': {
                'type': 'integer'
            },
            'description': 'A list of frame indices to query the camera pose for. **When you want to query multiple frames, you can provide a list of indices at once.** The indices should be one-based and should be within the range of the number of frames in the spatial memory.',
            'required': True
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        # import ipdb; ipdb.set_trace()
        global runtime
        params = self._verify_json_format_args(params)

        session_id = params['session_id']
        frame_index_list = params['frame_indices']
        # render_trajectory = params['render_trajectory']

        if session_id not in runtime.session_mem:
            return [ContentItem(text=f'Error: Session ID {session_id} not found. Please make sure to use the correct session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.')]

        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        if spatial_memory.query_map_scale is not None:
            frame_index_list = list(map(lambda x: int(x * spatial_memory.query_map_scale), frame_index_list))
        output_content = []
        camera_poses, errors = spatial_memory.query_camera_pose(frame_index_list)
        for error in errors:
            output_content.append(ContentItem(text=error))
        if camera_poses:
            output_content.append(ContentItem(text=json.dumps(camera_poses)))

        return output_content


class VoxelPropagator:
    # use a variable to track the time spent on voxelizetion for debugging
    voxelization_time = 0.0
    voxelization_count = 0
    hash_time = 0.0
    hash_count = 0
    intersection_time = 0.0
    intersection_count = 0

    @property
    def average_voxelization_time(self):
        if self.voxelization_count == 0:
            return 0.0
        return self.voxelization_time / self.voxelization_count

    @property
    def average_hash_time(self):
        if self.hash_count == 0:
            return 0.0
        return self.hash_time / self.hash_count
    
    @property
    def average_intersection_time(self):
        if self.intersection_count == 0:
            return 0.0
        return self.intersection_time / self.intersection_count

    def __init__(self, voxel_size: float, origin: np.ndarray, grid_dims: np.ndarray):
        """
        origin: (3,) world min bound
        grid_dims: (3,) voxel grid size
        """
        self.voxel_size = voxel_size
        self.origin = origin
        self.grid_dims = grid_dims

    def voxelize(self, pts: np.ndarray) -> np.ndarray:
        """
        pts: (N, 3)
        return: (N, 3) int voxel indices
        """
        vox = ((pts - self.origin) / self.voxel_size).astype(np.int32)
        return np.clip(vox, 0, self.grid_dims - 1)

    def hash_voxels(self, vox: np.ndarray) -> np.ndarray:
        """
        vox: (N, 3)
        return: (N,) int64 hash
        """
        result = (
            vox[:, 0]
            + vox[:, 1] * self.grid_dims[0]
            + vox[:, 2] * self.grid_dims[0] * self.grid_dims[1]
        )
        return result

    def build_src_keys(self, src_pts: np.ndarray) -> np.ndarray:
        vox = self.voxelize(src_pts)
        return np.unique(self.hash_voxels(vox))

    def propagate(self, src_keys: np.ndarray, tgt_pts: np.ndarray, need_voxelize: bool=True, need_hash: bool=True) -> np.ndarray:
        """
        tgt_pts: (Nt, 3)
        return: (Nt,) bool mask
        """
        if need_voxelize:
            tgt_vox = self.voxelize(tgt_pts)
            tgt_keys = self.hash_voxels(tgt_vox)
        elif need_hash:
            tgt_keys = self.hash_voxels(tgt_pts)
        else:
            tgt_keys = tgt_pts
        result = np.isin(tgt_keys, src_keys)
        return result

@register_tool('query_3d_object_position')
class Query3DObjectPosition(BaseToolWithFileAccess):
    # The `description` tells the agent the functionality of this tool.
    description = (
        "Query approximate 3D positions of object instances in the current spatial memory given their category names.\n\n"
        "This tool returns estimated object centers, not full object geometry. "
        "Use it when explicit 3D object locations are necessary for later reasoning or rendering, "
        "especially when visual checking alone is insufficient.\n\n"
        "Important coordinate rule:\n"
        "- If no viewpoint-changing tool has been used, the returned positions are expressed in the original world coordinate system.\n"
        "- If the viewpoint has been changed by `set_viewpoint`, `step_camera`, or `turn_camera`, "
        "the returned positions are expressed in the current active viewpoint coordinate system.\n\n"
        "Important reliability note:\n"
        "- The returned positions are approximate object centers.\n"
        "- This tool may be unreliable for ambiguous, heavily occluded, very small, or visually confusing objects.\n"
        "- Prefer direct visual checking when possible, and use this tool as a fallback when explicit 3D positions are required.\n\n"
        "The output is a dictionary mapping each queried category name to a list of estimated 3D positions."
    )

    parameters = [
        {
            'name': 'session_id',
            'type': 'string',
            'description': 'The session ID of the spatial memory to query.',
            'required': True
        },
        {
            'name': 'category_names',
            'type': 'array',
            'items': {'type': 'string'},
            'description': (
                "A list of object category names whose approximate 3D positions should be queried."
            ),
            'required': True
        }
    ]

    GLOBAL_COUNTER = 1
    SAM3_CHUNK_SIZE = int(os.environ.get("SAM3_CHUNK_SIZE", "8"))

    @staticmethod
    def create_empty_datapoint():
        """ A datapoint is a single image on which we can apply several queries at once. """
        return Datapoint(find_queries=[], images=[])
    
    @staticmethod
    def set_image(datapoint, pil_image):
        """ Add the image to be processed to the datapoint """
        w,h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h,w])]

    @staticmethod
    def add_text_prompt(datapoint, text_query):
        """ Add a text query to the datapoint """
        # in this function, we require that the image is already set.
        # that's because we'll get its size to figure out what dimension to resize masks and boxes
        # In practice you're free to set any size you want, just edit the rest of the function
        assert len(datapoint.images) == 1, "please set the image first"

        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[], # unused for inference
                is_exhaustive=True, # unused for inference
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=Query3DObjectPosition.GLOBAL_COUNTER,
                    original_image_id=Query3DObjectPosition.GLOBAL_COUNTER,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
        Query3DObjectPosition.GLOBAL_COUNTER += 1
        return Query3DObjectPosition.GLOBAL_COUNTER - 1
    
    @staticmethod
    def masked_crop_with_bg(
        img: Image.Image,
        mask: torch.Tensor,
        bbox,
        out_size=(224, 224),
        bg_color=(127, 127, 127),  # gray background, CLIP friendly
    ):
        """
        img: PIL.Image (RGB)
        mask: torch.Tensor (H, W) or (1, H, W), bool or {0,1}
        bbox: [x1, y1, x2, y2] in pixel coords
        """

        if mask.ndim == 3:
            mask = mask[0]
        mask = mask.bool().cpu().numpy()

        x1, y1, x2, y2 = map(int, bbox)

        w, h = img.size
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        img_np = np.array(img)  # (H, W, 3)
        bg = np.zeros_like(img_np)
        bg[:] = bg_color
        fg = np.where(mask[..., None], img_np, bg)
        fg_crop = fg[y1:y2, x1:x2]

        if fg_crop.size == 0:
            return None

        fg_crop = Image.fromarray(fg_crop)
        fg_crop = fg_crop.resize(out_size, Image.BICUBIC)

        return fg_crop

    @staticmethod
    def visualize_sam3_segmentations(
        img_pil: Image.Image,
        mask_list: List[np.ndarray],
        score_list: List[float] | None,
        *,
        category_name: str,
        image_idx: int,
    ) -> str | None:
        """Overlay post-NMS SAM3 masks on the source image and save a debug visualization."""
        base = np.array(img_pil.convert("RGB"), dtype=np.uint8)
        vis = base.copy()
        color_bank = [
            (255, 99, 71),
            (30, 144, 255),
            (50, 205, 50),
            (255, 215, 0),
            (186, 85, 211),
            (255, 140, 0),
        ]

        for rank, mask_mem in enumerate(mask_list):
            color = np.array(color_bank[rank % len(color_bank)], dtype=np.float32)
            mask_bool = np.asarray(mask_mem).astype(bool)
            if not mask_bool.any():
                continue

            mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
            mask_img = mask_img.resize(img_pil.size, Image.NEAREST)
            mask_bool_img = np.array(mask_img) > 0

            alpha = 0.42
            vis_region = vis[mask_bool_img].astype(np.float32)
            vis[mask_bool_img] = np.clip(
                vis_region * (1.0 - alpha) + color * alpha,
                0,
                255,
            ).astype(np.uint8)

        draw_img = Image.fromarray(vis)
        draw = ImageDraw.Draw(draw_img)
        if not mask_list:
            draw.text(
                (16, 16),
                f"{category_name}: no mask kept",
                fill=(255, 64, 64),
            )

        for rank, mask_mem in enumerate(mask_list):
            color = color_bank[rank % len(color_bank)]
            mask_bool = np.asarray(mask_mem).astype(bool)
            if not mask_bool.any():
                continue
            mask_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
            mask_img = mask_img.resize(img_pil.size, Image.NEAREST)
            coords = np.argwhere(np.array(mask_img) > 0)
            if coords.size == 0:
                continue
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            draw.rectangle([(int(x1), int(y1)), (int(x2), int(y2))], outline=color, width=3)
            score_text = "n/a" if score_list is None or rank >= len(score_list) else f"{float(score_list[rank]):.2f}"
            label = f"{category_name} {rank} ({score_text})"
            text_pos = (max(0, int(x1) + 2), max(0, int(y1) - 16))
            draw.text(text_pos, label, fill=color)

        debug_dir = get_managed_workspace_dir("query_3d_object_position_debug")
        safe_category = re.sub(r"[^a-zA-Z0-9_-]+", "_", category_name)
        save_path = os.path.join(
            debug_dir,
            f"sam3_{safe_category}_img{image_idx}.png",
        )
        draw_img.save(save_path)
        return save_path

    @staticmethod
    def supports_enable_thinking(processor: Any) -> bool:
        try:
            import inspect

            return "enable_thinking" in inspect.signature(processor.apply_chat_template).parameters
        except Exception:
            return False

    @staticmethod
    def extract_json_payload(text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch not in "[{":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                return obj
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Failed to extract JSON from model output: {text}")

    @staticmethod
    def normalize_qwen3_bbox_to_pixels(bbox: List[float], image_width: int, image_height: int) -> List[float]:
        if len(bbox) != 4:
            raise ValueError(f"Invalid Qwen bbox length: {bbox}")
        rel_x1, rel_y1, rel_x2, rel_y2 = bbox
        abs_x1 = max(0.0, min(float(image_width), float(rel_x1) / 1000.0 * image_width))
        abs_y1 = max(0.0, min(float(image_height), float(rel_y1) / 1000.0 * image_height))
        abs_x2 = max(0.0, min(float(image_width), float(rel_x2) / 1000.0 * image_width))
        abs_y2 = max(0.0, min(float(image_height), float(rel_y2) / 1000.0 * image_height))
        return [abs_x1, abs_y1, abs_x2, abs_y2]

    @staticmethod
    def coerce_qwen3_detections(parsed: Any, category: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
        if isinstance(parsed, dict):
            if "detections" in parsed and isinstance(parsed["detections"], list):
                raw_items = parsed["detections"]
            elif "bbox_2d" in parsed:
                raw_items = [parsed]
            else:
                raw_items = []
        elif isinstance(parsed, list):
            raw_items = parsed
        else:
            raw_items = []

        detections = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d")
            if bbox in (None, [], {}):
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            box = Query3DObjectPosition.normalize_qwen3_bbox_to_pixels(
                list(bbox),
                image_width=image_width,
                image_height=image_height,
            )
            score = item.get("score", item.get("confidence", 1.0))
            try:
                score = float(score)
            except Exception:
                score = 1.0
            detections.append({"label": category, "score": score, "box": box})
        return detections

    @staticmethod
    def normalize_xyxy_box_to_cxcywh(box_xyxy: List[float], image_width: int, image_height: int) -> List[float]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid absolute bbox: {box_xyxy}")
        return [
            ((x1 + x2) * 0.5) / float(image_width),
            ((y1 + y2) * 0.5) / float(image_height),
            (x2 - x1) / float(image_width),
            (y2 - y1) / float(image_height),
        ]

    @staticmethod
    def uniform_sample_indices(total: int, max_samples: int) -> List[int]:
        if total <= 0 or max_samples <= 0:
            return []
        if total <= max_samples:
            return list(range(total))
        return np.linspace(0, total - 1, num=max_samples, dtype=int).tolist()

    @staticmethod
    def create_empty_category_pos(category_names: List[str]) -> Dict[str, Dict[str, Any]]:
        return {
            category_name: {
                "pos_3d": [],
                "fg_images": [],
                "masks": [],
                "image_idxs": [],
                "scores": [],
                "debug_vis": [],
            }
            for category_name in category_names
        }

    @staticmethod
    def append_category_instances(
        category_entry: Dict[str, Any],
        *,
        image_idx: int,
        pos_list: List[np.ndarray],
        fg_list: List[Image.Image],
        mask_list: List[np.ndarray],
        score_list: List[float],
        debug_vis_path: str | None,
    ) -> None:
        if not pos_list:
            return
        category_entry["pos_3d"].append(pos_list)
        category_entry["fg_images"].append(fg_list)
        category_entry["masks"].append(mask_list)
        category_entry["image_idxs"].append(image_idx)
        category_entry["scores"].append(score_list)
        if debug_vis_path is not None:
            category_entry["debug_vis"].append({"image_idx": image_idx, "path": debug_vis_path})
    
    @staticmethod
    def mask_iou_bool(m1: np.ndarray, m2: np.ndarray, ratio_mode: bool = False) -> float:
        """m1/m2: bool array (H,W)"""
        inter = np.logical_and(m1, m2).sum()
        if inter == 0:
            return 0.0
        if ratio_mode:
            union = min(m1.sum(), m2.sum())
        else:
            union = np.logical_or(m1, m2).sum()
        return float(inter) / float(union + 1e-6)

    @staticmethod
    def mask_nms(
        masks: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor | None = None,
        *,
        iou_thresh: float = 0.8,
        max_keep: int | None = None,
    ):
        """
        masks: (N, 1, H, W) or (N, H, W) or list-like
        boxes: (N, 4)
        scores: (N,) optional. If None, keep original order as ranking.
        Return: keep_indices (List[int])
        """
        if masks is None or masks.numel() == 0:
            return []

        # Normalize shapes
        if masks.ndim == 4:        # (N,1,H,W)
            masks_ = masks[:, 0]
        elif masks.ndim == 3:      # (N,H,W)
            masks_ = masks
        else:
            raise ValueError(f"Unsupported masks shape: {masks.shape}")


        N = masks_.shape[0]
        if N == 0:
            return []
        elif N == 1:
            return {0: []}

        if scores is None:
            # use area as proxy scores
            areas = masks_.flatten(1).sum(dim=1)  # (N,)
            order = torch.argsort(areas, descending=True).tolist()
        else:
            order = torch.argsort(scores, descending=True).tolist()

        masks_np = masks_.detach().cpu().numpy().astype(bool)

        keep = {}
        for idx in order:
            suppressed = False
            for kept_idx in keep:
                iou = Query3DObjectPosition.mask_iou_bool(masks_np[idx], masks_np[kept_idx], ratio_mode=True)
                if iou >= iou_thresh:
                    suppressed = True
                    keep[kept_idx].append(idx)
                    break
            if not suppressed:
                keep[idx] = []
                if max_keep is not None and len(keep) >= max_keep:
                    break

        return keep

    @staticmethod
    def extract_category_instances_for_one_image(
        img_pil: Image.Image,
        masks: torch.Tensor,     # (N,1,H,W)
        boxes: torch.Tensor,     # (N,4) in pixel coords of img
        scores: torch.Tensor,    # (N,) confidence scores for each instance
        spatial_memory: SpatialMemory,    # has '3d_positions'
        image_idx: int,
        memory_h: int,
        memory_w: int,
        masked_crop_with_bg_fn,  # your self.masked_crop_with_bg
        out_size=(224, 224),
        nms_iou_thresh: float = 0.8,
    ):
        """
        Return:
        fg_list: List[PIL.Image]  length=K
        pos_list: List[np.ndarray] each (Ni,3) length=K
        """
        if masks is None or masks.numel() == 0:
            return [], []

        # 1) NMS to remove duplicates
        # import ipdb; ipdb.set_trace()
        keep_idx = Query3DObjectPosition.mask_nms(masks=masks, boxes=boxes, iou_thresh=nms_iou_thresh)
        if len(keep_idx) == 0:
            return [], []

        fg_list = []
        pos_list = []
        mask_list = []
        score_list = []
        frame_query_mask = getattr(spatial_memory, "visualization_mask_outlier_only", None)
        # import ipdb; ipdb.set_trace()
        if frame_query_mask is not None:
            frame_query_mask = np.asarray(frame_query_mask[image_idx]).astype(bool)
        for obj_idx in keep_idx.keys():
            mask = masks[obj_idx][0]          # (H,W) torch
            bbox = boxes[obj_idx]             # (4,) torch

            # 2) background-free crop -> 224x224
            fg = masked_crop_with_bg_fn(
                img=img_pil,
                mask=mask,
                bbox=bbox.detach().cpu().numpy(),
                out_size=out_size,
            )

            # 3) resize mask to memory resolution then fetch 3D points
            mask_mem = F.resize(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(memory_h, memory_w),
                interpolation=Image.NEAREST,
            ).squeeze(0).squeeze(0).bool().cpu().numpy()
            if frame_query_mask is not None:
                mask_mem = mask_mem & frame_query_mask
                if mask_mem.sum() == 0:
                    continue
            fg_list.append(fg)
            mask_list.append(mask_mem)
            pts3d = spatial_memory.position_3d[image_idx][mask_mem]  # (N,3)
            pos_list.append(pts3d)
            comp_index = [obj_idx] + keep_idx[obj_idx]  # current obj + all suppressed ones
            score_list.append(scores[comp_index].max().item())

        return fg_list, pos_list, mask_list, score_list
    
    @staticmethod
    def build_voxel_propagator(
        spatial_memory: SpatialMemory, 
        eps: float
    ) -> VoxelPropagator:
        all_pts_raw = np.concatenate(
            [p.reshape(-1, 3) for p in spatial_memory.position_3d], axis=0
        )
        all_pts = all_pts_raw
        query_mask = getattr(spatial_memory, "visualization_mask_outlier_only", None)
        if query_mask is not None:
            all_pts = all_pts[np.asarray(query_mask).reshape(-1).astype(bool)]
        valid = np.isfinite(all_pts[:, 0])
        all_pts = all_pts[valid]
        if len(all_pts) == 0:
            valid = np.isfinite(all_pts_raw[:, 0])
            all_pts = all_pts_raw[valid]

        origin = all_pts.min(axis=0) - eps
        max_bound = all_pts.max(axis=0) + eps
        voxel_size = eps
        grid_dims = np.ceil((max_bound - origin) / voxel_size).astype(int)

        return VoxelPropagator(voxel_size, origin, grid_dims)

    @staticmethod
    def propagate_mask_3d_with_voxelization(
        points_src: np.ndarray,    # (Ns, 3)
        points_tgt: np.ndarray,    # (H, W, 3)
        eps: float = 0.03,
    ):
        """
        A naive implementation of mask propagation by voxelization and occupancy check.

        points_src: (Ns, 3)
        points_tgt: (H, W, 3)
        return: (H, W) bool mask
        """
        H, W, _ = points_tgt.shape

        # Voxelization
        all_points = np.concatenate([points_src, points_tgt.reshape(-1, 3)], axis=0)
        min_bound = all_points.min(axis=0) - eps
        max_bound = all_points.max(axis=0) + eps
        grid_size = eps / np.sqrt(3)  # diagonal of voxel ~ eps
        grid_dims = np.ceil((max_bound - min_bound) / grid_size).astype(int)

        def point_to_voxel_idx(points):
            idxs = ((points - min_bound) / grid_size).astype(int)
            idxs = np.clip(idxs, a_min=0, a_max=grid_dims - 1)
            return idxs

        src_voxels = set(map(tuple, point_to_voxel_idx(points_src)))
        tgt_voxels = point_to_voxel_idx(points_tgt.reshape(-1, 3))

        mask_flat = np.array([tuple(voxel) in src_voxels for voxel in tgt_voxels], dtype=bool)
        return mask_flat.reshape(H, W)

    @staticmethod
    def agreement_on_frame(
        propagated_mask: np.ndarray,    # (H, W) bool
        tgt_category_masks: List[np.ndarray],    # [(H, W)] bool
        min_cov: float = 0.2,
    ):
        """
        Return:
            agreed: bool
            best_cov: float
        """
        best_cov = 0.0
        best_inst_idx = -1
        for k in range(len(tgt_category_masks)):
            mask_k = tgt_category_masks[k]  # (H, W) bool
            inter = (propagated_mask & mask_k).sum()
            cov = inter / (propagated_mask.sum() + 1e-6)
            if cov > best_cov:
                best_inst_idx = k
                best_cov = cov

        agreed = best_cov >= min_cov
        return agreed, best_cov, best_inst_idx
    
    @staticmethod
    def compute_multiview_agreement_voxel(
        src_frame_idx: int,
        src_inst_points: np.ndarray,        # (Ns, 3)
        pos_3d_all: np.ndarray,                   # [frame][H, W, 3]
        pos_3d_voxels_hash: np.ndarray,           # [frame][H, W, 3]
        category_masks: dict,               # frame_idx -> (K, H, W)
        voxel_propagator: VoxelPropagator,  # << 新增
        eps: float = 0.03,
        min_cov: float = 0.2,
        min_visible_ratio: float = 1e-3,
    ):
        """
        Return:
            agreement_score: float
            support: int
            details: list of dict
        """

        # ---- build src voxel keys once ----
        src_keys = voxel_propagator.build_src_keys(src_inst_points)

        k = 0
        m = 0
        details = []

        # import ipdb; ipdb.set_trace()
        # time_interval_1 = 0
        # time_interval_1_before_prop = 0
        # time_interval_1_prop = 0
        # time_interval_1_count = 0
        # time_interval_2 = 0
        # time_interval_2_count = 0
        for tgt_idx in range(len(pos_3d_voxels_hash)):
            # time_start = time()
            if tgt_idx == src_frame_idx:
                continue

            tgt_points = np.asarray(pos_3d_voxels_hash[tgt_idx])  # (H, W) float hash grid
            tgt_mask_shape = pos_3d_all[tgt_idx].shape[:2]  # (H, W)

            # ---- visibility check ----
            visible_mask = np.isfinite(tgt_points)
            if visible_mask.sum() == 0:
                continue
            tgt_points = tgt_points[visible_mask]
            tgt_points = tgt_points.astype(np.int64)    # ensure integer type for hashing
            # time_interval_1_before_prop += time() - time_start
            hit = voxel_propagator.propagate(src_keys, tgt_points, need_voxelize=False, need_hash=False)
            # time_interval_1_prop += time() - time_start
            propagated_mask = np.zeros(tgt_mask_shape, dtype=bool)
            propagated_mask[visible_mask] = hit.reshape(-1)

            vis_ratio = propagated_mask.sum() / np.prod(propagated_mask.shape)
            # time_interval_1 += time() - time_start
            # time_interval_1_count += 1

            if vis_ratio < min_visible_ratio:
                continue

            k += 1

            # ---- agreement check ----
            tgt_masks = category_masks.get(tgt_idx, None)
            if tgt_masks is None:
                details.append({"frame": tgt_idx, "visible": True, "agree": False})
                continue
            
            agree, cov, best_inst_idx = Query3DObjectPosition.agreement_on_frame(
                propagated_mask,
                tgt_masks,
                min_cov=min_cov,
            )

            if agree:
                m += 1

            details.append({
                "frame": tgt_idx,
                "best_inst_idx": best_inst_idx,
                "visible": True,
                "agree": agree,
                "coverage": cov,
            })
            # time_interval_2 += time() - time_start
            # time_interval_2_count += 1
        agreement_score = m / k if k > 0 else 1.0

        # import ipdb; ipdb.set_trace()
        return agreement_score, k, details

    def run_qwen3_on_image(
        self,
        image: Image.Image,
        categories: List[str],
    ) -> List[Dict[str, Any]]:
        global runtime
        runtime.ensure_qwen3_vl()
        qwen3_processor = runtime.qwen3_vl_processor
        qwen3_model = runtime.qwen3_vl_model
        detections: List[Dict[str, Any]] = []
        use_enable_thinking = self.supports_enable_thinking(qwen3_processor)

        for category in categories:
            prompt = (
                f"Detect all visible instances of '{category}' in this image. "
                "Return JSON only. Use schema "
                '{"detections":[{"bbox_2d":[x1,y1,x2,y2],"label":"object_name","score":0.0}]}. '
                "bbox_2d must be normalized integers in [0,1000] relative to the original image size. "
                "If the object is not present, return {\"detections\": []}. "
                "Do not output any extra text."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if use_enable_thinking:
                chat_kwargs["enable_thinking"] = False
            text_input = qwen3_processor.apply_chat_template(messages, **chat_kwargs)
            inputs = qwen3_processor(text=[text_input], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(qwen3_model.device)

            with torch.no_grad():
                generated_ids = qwen3_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = qwen3_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            try:
                parsed = self.extract_json_payload(output_text)
            except Exception as exc:
                logger.warning(f"Qwen3-VL recovery parsing failed for category '{category}': {exc}")
                continue
            detections.extend(
                self.coerce_qwen3_detections(
                    parsed,
                    category=category,
                    image_width=image.width,
                    image_height=image.height,
                )
            )
        return detections

    def run_sam3_box_prompt_on_image(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        global runtime
        runtime.ensure_sam3()
        processor = runtime.sam3_processor
        results: Dict[str, Dict[str, Any]] = {}

        for detection in detections:
            category = str(detection["label"])
            try:
                box_prompt = self.normalize_xyxy_box_to_cxcywh(
                    detection["box"],
                    image_width=image.width,
                    image_height=image.height,
                )
            except Exception as exc:
                logger.warning(f"Invalid recovery bbox for category '{category}': {exc}")
                continue
            state = processor.set_image(image)
            state = processor.add_geometric_prompt(box=box_prompt, label=True, state=state)
            masks = state.get("masks")
            boxes = state.get("boxes")
            scores = state.get("scores")
            if masks is None or boxes is None or scores is None or masks.numel() == 0:
                continue

            top_idx = int(torch.argmax(scores).item())
            category_result = results.setdefault(category, {"masks": [], "boxes": [], "scores": []})
            category_result["masks"].append(masks[top_idx].detach().cpu())
            category_result["boxes"].append(boxes[top_idx].detach().cpu())
            category_result["scores"].append(scores[top_idx].detach().cpu())
        
        # import ipdb; ipdb.set_trace()
        merged_results: Dict[str, Dict[str, Any]] = {}
        for category, category_result in results.items():
            if not category_result["masks"]:
                continue
            merged_results[category] = {
                "masks": torch.stack(category_result["masks"], dim=0),
                "boxes": torch.stack(category_result["boxes"], dim=0),
                "scores": torch.stack(category_result["scores"], dim=0),
            }
        return merged_results

    def collect_sam3_text_query_instances(
        self,
        category_pos: Dict[str, Dict[str, Any]],
        category_names: List[str],
        image_files: List[Image.Image],
        spatial_memory: SpatialMemory,
    ) -> None:
        global runtime
        runtime.ensure_sam3()
        transform = runtime.sam3_transform
        postprocessor = runtime.sam3_postprocessor
        sam3_model = runtime.sam3_model

        datapoints = []
        all_id_list = []
        for image in image_files:
            datapoint = self.create_empty_datapoint()
            self.set_image(datapoint, image)
            id_list = []
            for category_name in category_names:
                id_list.append(self.add_text_prompt(datapoint, category_name))
            all_id_list.append(id_list)
            datapoint = transform(datapoint)
            datapoints.append(datapoint)

        processed_results = {}
        chunk_size = max(1, self.SAM3_CHUNK_SIZE)
        print(
            f"Running SAM3 inference on {len(image_files)} images for categories: {category_names} "
            f"(chunk_size={chunk_size})"
        )
        for chunk_start in range(0, len(datapoints), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(datapoints))
            chunk_datapoints = datapoints[chunk_start:chunk_end]
            batch = collate(chunk_datapoints, dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device(sam3_model.device), non_blocking=True)
            with torch.no_grad():
                output = sam3_model(batch)
            chunk_results = postprocessor.process_results(output, batch.find_metadatas)
            processed_results.update(chunk_results)
            del batch, output, chunk_results
            gc.collect()
            torch.cuda.empty_cache()

        memory_h, memory_w = spatial_memory.memory_3d_map_size
        for category_idx, category_name in enumerate(category_names):
            for image_idx, image in enumerate(image_files):
                result = processed_results[all_id_list[image_idx][category_idx]]
                masks = result["masks"]
                if masks.shape[0] == 0:
                    self.visualize_sam3_segmentations(
                        img_pil=image,
                        mask_list=[],
                        score_list=[],
                        category_name=category_name,
                        image_idx=image_idx,
                    )
                    continue
                # if category_name == "dishwasher":
                #     import ipdb; ipdb.set_trace()
                fg_list, pos_list, mask_list, score_list = self.extract_category_instances_for_one_image(
                    img_pil=image,
                    masks=masks,
                    boxes=result["boxes"],
                    scores=result["scores"],
                    spatial_memory=spatial_memory,
                    image_idx=image_idx,
                    memory_h=memory_h,
                    memory_w=memory_w,
                    masked_crop_with_bg_fn=self.masked_crop_with_bg,
                    out_size=(224, 224),
                    nms_iou_thresh=0.3,
                )
                debug_vis_path = self.visualize_sam3_segmentations(
                    img_pil=image,
                    mask_list=mask_list,
                    score_list=score_list,
                    category_name=category_name,
                    image_idx=image_idx,
                )
                self.append_category_instances(
                    category_pos[category_name],
                    image_idx=image_idx,
                    pos_list=pos_list,
                    fg_list=fg_list,
                    mask_list=mask_list,
                    score_list=score_list,
                    debug_vis_path=debug_vis_path,
                )

    def recover_missing_categories_with_qwen3(
        self,
        category_pos: Dict[str, Dict[str, Any]],
        missing_categories: List[str],
        image_files: List[Image.Image],
        spatial_memory: SpatialMemory,
    ) -> Dict[str, Any]:
        if not missing_categories:
            return {"sampled_frame_indices": [], "detections_by_frame": {}}

        sampled_indices = self.uniform_sample_indices(len(image_files), max_samples=16)
        detections_by_frame: Dict[int, Dict[str, int]] = {}
        memory_h, memory_w = spatial_memory.memory_3d_map_size

        for image_idx in sampled_indices:
            # import ipdb; ipdb.set_trace()
            image = image_files[image_idx]
            image.save(f'./debug_vis/vis_img_{image_idx}.png')
            detections = self.run_qwen3_on_image(image, missing_categories)
            if not detections:
                continue
            sam3_results = self.run_sam3_box_prompt_on_image(image, detections)
            frame_detection_count: Dict[str, int] = {}
            for category_name in missing_categories:
                category_result = sam3_results.get(category_name)
                if category_result is None:
                    continue
                fg_list, pos_list, mask_list, score_list = self.extract_category_instances_for_one_image(
                    img_pil=image,
                    masks=category_result["masks"],
                    boxes=category_result["boxes"],
                    scores=category_result["scores"],
                    spatial_memory=spatial_memory,
                    image_idx=image_idx,
                    memory_h=memory_h,
                    memory_w=memory_w,
                    masked_crop_with_bg_fn=self.masked_crop_with_bg,
                    out_size=(224, 224),
                    nms_iou_thresh=0.3,
                )
                debug_vis_path = self.visualize_sam3_segmentations(
                    img_pil=image,
                    mask_list=mask_list,
                    score_list=score_list,
                    category_name=category_name,
                    image_idx=image_idx,
                )
                self.append_category_instances(
                    category_pos[category_name],
                    image_idx=image_idx,
                    pos_list=pos_list,
                    fg_list=fg_list,
                    mask_list=mask_list,
                    score_list=score_list,
                    debug_vis_path=debug_vis_path,
                )
                if pos_list:
                    frame_detection_count[category_name] = len(pos_list)
            if frame_detection_count:
                detections_by_frame[image_idx] = frame_detection_count
        return {
            "sampled_frame_indices": sampled_indices,
            "detections_by_frame": detections_by_frame,
        }

    def finalize_category_positions(
        self,
        category_pos: Dict[str, Dict[str, Any]],
        category_names: List[str],
        spatial_memory: SpatialMemory,
        num_images: int,
    ) -> None:
        voxel_propagator = Query3DObjectPosition.build_voxel_propagator(spatial_memory, eps=0.03)
        positions_3d_voxels_hash = []
        query_mask_all = getattr(spatial_memory, "visualization_mask_outlier_only", None)
        for frame_idx, pos in enumerate(tqdm(spatial_memory.position_3d, desc="Hashing filtered spatial memory")):
            flat_pos = pos.reshape(-1, 3)
            valid_mask = np.isfinite(flat_pos).all(axis=1)
            if query_mask_all is not None:
                valid_mask &= np.asarray(query_mask_all[frame_idx]).reshape(-1).astype(bool)
            flat_hash = np.full(flat_pos.shape[0], np.nan, dtype=np.float64)
            if valid_mask.any():
                vox = voxel_propagator.voxelize(flat_pos[valid_mask])
                flat_hash[valid_mask] = voxel_propagator.hash_voxels(vox).astype(np.float64)
            positions_3d_voxels_hash.append(flat_hash.reshape(pos.shape[:2]))

        for category_name in category_names:
            frame_agreement_scores = []
            frame_supports = []
            frame_details_all = []
            frame_segment_scores = []
            pos_3d = category_pos[category_name]["pos_3d"]
            masks = category_pos[category_name]["masks"]
            image_idxs = category_pos[category_name]["image_idxs"]
            scores = category_pos[category_name]["scores"]
            category_masks = {idx: mask_list for idx, mask_list in zip(image_idxs, masks)}
            for frame_idx, inst_list, inst_scores in zip(image_idxs, pos_3d, scores):
                if not isinstance(inst_list, (list, tuple)):
                    continue
                agreement_scores = []
                supports = []
                details_all = []
                for pts_np in tqdm(inst_list, desc=f"Computing agreement for category {category_name}"):
                    if pts_np is None:
                        continue
                    agreement_score, support, details = self.compute_multiview_agreement_voxel(
                        src_frame_idx=frame_idx,
                        src_inst_points=pts_np,
                        pos_3d_all=spatial_memory.position_3d,
                        pos_3d_voxels_hash=positions_3d_voxels_hash,
                        category_masks=category_masks,
                        voxel_propagator=voxel_propagator,
                        eps=1e-2,
                        min_cov=0.2,
                        min_visible_ratio=1e-3,
                    )
                    agreement_scores.append(agreement_score)
                    supports.append(support)
                    details_all.append(details)
                frame_agreement_scores.append(agreement_scores)
                frame_segment_scores.append(inst_scores)
                frame_supports.append(supports)
                frame_details_all.append(details_all)
            category_pos[category_name]["agreement_scores"] = frame_agreement_scores
            category_pos[category_name]["segment_scores"] = frame_segment_scores
            category_pos[category_name]["supports"] = frame_supports
            category_pos[category_name]["details"] = frame_details_all

        # self.aggregate_3d_positions(category_pos, agreement_thresh=0.3 if num_images <= 5 else 0.1)
        self.aggregate_3d_positions(category_pos, agreement_thresh=0.3 if num_images <= 5 else 0)
        for cat in category_pos:
            uncertain_idxs = category_pos[cat].get("uncertain_instances", [])
            if not uncertain_idxs:
                continue
            category_pos[cat]["merged_instances"] = [
                inst
                for idx, inst in enumerate(category_pos[cat]["merged_instances"])
                if idx not in uncertain_idxs
            ]

    
    def aggregate_3d_positions(self, category_pos, agreement_thresh=0.3):
        """
        Greedy merge view-level instances into multi-view 3D instances
        under strict per-frame mutual exclusion constraints.

        Results are written into:
            category_pos[cat]['merged_instances']
        """

        def edge_priority(e):
            # hard agree > soft visible > others
            if e["type"] == "hard":
                return (2, e["weight"])
            elif e["type"] == "soft":
                return (1, e["weight"])
            else:
                return (0, 0.0)

        for cat, cat_data in category_pos.items():
            if "details" not in cat_data or "pos_3d" not in cat_data:
                continue

            details = cat_data["details"]          # [src_inst][dict]
            pos_3d = cat_data["pos_3d"]            # [frame][inst] -> (N,3)
            agreement_scores = cat_data["agreement_scores"]
            segment_scores = cat_data['segment_scores']
            image_idxs = cat_data.get("image_idxs", None)

            # --------------------------------------------------
            # 1. Collect nodes
            # --------------------------------------------------
            nodes = set()
            nodes_segment_scores = {}
            edges = {}

            # We assume details is aligned with some src_frame
            # Convention: src_frame = index in pos_3d
            for src_frame, inst_details, agreement_list, segment_score_list in zip(image_idxs, details, agreement_scores, segment_scores):
                for src_inst_idx, (edge_list, agreement, segment_score) in enumerate(zip(inst_details, agreement_list, segment_score_list)):
                    if agreement < agreement_thresh:
                        continue
                    src_node = (src_frame, src_inst_idx)
                    nodes.add(src_node)
                    nodes_segment_scores[src_node] = segment_score
                    for e in edge_list:
                        if not e.get("visible", False):
                            continue
                        if e.get("coverage", 0.0) <= 0:
                            continue

                        tgt_frame = e["frame"]
                        tgt_inst = e["best_inst_idx"]
                        tgt_node = (tgt_frame, tgt_inst)
                        if tgt_node not in nodes:
                            # check tgt node agreement score
                            tht_node_agreement_score = category_pos[cat]["agreement_scores"][image_idxs.index(tgt_frame)][tgt_inst]
                            if tht_node_agreement_score < agreement_thresh:
                                continue
                            nodes.add(tgt_node)
                        if (src_node, tgt_node) not in edges and (tgt_node, src_node) not in edges:
                            # first time seeing this
                            edge_key = (src_node, tgt_node)
                            src_to_tgt_agreement = e['agree']
                            src_to_tgt_coverage = e['coverage']
                            tgt_details = category_pos[cat]['details'][image_idxs.index(tgt_frame)][tgt_inst]
                            tgt_to_src_agreement = False
                            tgt_to_src_coverage = 0.0
                            for possible_e in tgt_details:
                                if possible_e['frame'] == src_frame and possible_e['best_inst_idx'] == src_inst_idx:
                                    tgt_to_src_agreement = possible_e['agree']
                                    tgt_to_src_coverage = possible_e['coverage']
                                    break
                            
                            # edge type
                            if src_to_tgt_agreement and tgt_to_src_agreement:
                                dir_factor = 1.0
                                edge_type = "hard"
                            elif src_to_tgt_agreement or tgt_to_src_agreement:
                                dir_factor = 0.8
                                edge_type = "soft"
                            else:
                                dir_factor = src_to_tgt_coverage * tgt_to_src_coverage
                                edge_type = "weak"
                            edges[edge_key] = {
                                "weight": dir_factor,
                                "type": edge_type,
                            }
            # import ipdb; ipdb.set_trace()

            if not nodes:
                cat_data["merged_instances"] = []
                continue

            # --------------------------------------------------
            # 2. Init clusters (each node is its own cluster)
            # --------------------------------------------------
            clusters = {}
            node2cluster = {}
            cluster_frames = {}

            for cid, node in enumerate(nodes):
                clusters[cid] = {node}
                node2cluster[node] = cid
                cluster_frames[cid] = {node[0]}  # frame idx

            # --------------------------------------------------
            # 3. Sort edges by certainty
            # --------------------------------------------------
            # Convert edges dict to list
            edges = [
                {"src": src,
                 "tgt": tgt,
                 "weight": attr["weight"],
                 "type": attr["type"]}
                for (src, tgt), attr in edges.items()
            ]
            edges.sort(key=edge_priority, reverse=True)

            # --------------------------------------------------
            # 4. Greedy merge
            # --------------------------------------------------
            def can_merge(ca, cb):
                # strict mutex: no shared frame
                return len(cluster_frames[ca] & cluster_frames[cb]) == 0

            for e in edges:
                u = e["src"]
                v = e["tgt"]

                if u not in node2cluster or v not in node2cluster:
                    continue

                cu = node2cluster[u]
                cv = node2cluster[v]

                if cu == cv:
                    continue

                if not can_merge(cu, cv):
                    continue

                # merge cv -> cu
                for node in clusters[cv]:
                    clusters[cu].add(node)
                    node2cluster[node] = cu

                cluster_frames[cu].update(cluster_frames[cv])

                del clusters[cv]
                del cluster_frames[cv]

            # --------------------------------------------------
            # 5. Build merged instances (aggregate 3D points)
            # --------------------------------------------------
            merged_instances = []

            for cluster_nodes in clusters.values():
                pts_all = []

                for (frame_idx, inst_idx) in cluster_nodes:
                    try:
                        pts = pos_3d[image_idxs.index(frame_idx)][inst_idx]
                    except Exception:
                        continue

                    if pts is None or len(pts) == 0:
                        continue

                    pts_all.append(pts)

                if not pts_all:
                    continue

                merged_instances.append({
                    "nodes": sorted(list(cluster_nodes)),
                    "points_3d": np.concatenate(pts_all, axis=0),
                })

            cat_data["merged_instances"] = merged_instances

            # mark some groups with high uncertainty. One way is to check nodes count
            uncertain_instances = []
            for group_idx, inst in enumerate(merged_instances):
                if len(inst["nodes"]) <= 1:
                    # check whether it has an edge with others
                    has_edge = False
                    for e in edges:
                        if (e["src"] in inst["nodes"]) or (e["tgt"] in inst["nodes"]):
                            has_edge = True
                            break
                    if has_edge:
                        uncertain_instances.append(group_idx)
                    else:
                        # this instance is isolated, so we check its segment score
                        segment_scores = [nodes_segment_scores[node] for node in inst["nodes"]]
                        if max(segment_scores) < 0.65:  # set higher threshold for isolated instances since we have less confidence without multi-view agreement
                            uncertain_instances.append(group_idx)
            cat_data["uncertain_instances"] = uncertain_instances



    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        global runtime
        params = self._verify_json_format_args(params)

        session_id = params.get('session_id', 'default')
        category_names = params.get('category_names', [])
        if not category_names:
            return [ContentItem(text='Error: No category names provided')]

        try:
            spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        except KeyError:
            return [ContentItem(text=f'Error: Invalid session ID {session_id}')]

        runtime.ensure_sam3()
        image_files = spatial_memory.rgb_images_pil  # list of PIL.Image
        category_pos = self.create_empty_category_pos(category_names)
        self.collect_sam3_text_query_instances(category_pos, category_names, image_files, spatial_memory)
        # import ipdb; ipdb.set_trace()

        missing_categories = [
            category_name
            for category_name in category_names
            if len(category_pos[category_name].get("pos_3d", [])) == 0
        ]
        recovery_info = {
            "missing_categories_before_recovery": list(missing_categories),
            "sampled_frame_indices": [],
            "detections_by_frame": {},
        }
        if missing_categories:
            recovery_info.update(
                self.recover_missing_categories_with_qwen3(
                    category_pos,
                    missing_categories,
                    image_files,
                    spatial_memory,
                )
            )
        self.finalize_category_positions(category_pos, category_names, spatial_memory, num_images=len(image_files))
        still_missing_categories = [
            category_name
            for category_name in category_names
            if len(category_pos[category_name].get("merged_instances", [])) == 0
        ]
        # visualize final results
        # import ipdb; ipdb.set_trace()
        file_name = ''
        for cat in category_pos:
            file_name += f'{cat}_'
        file_name += 'voxel_filter.html'
        # import ipdb; ipdb.set_trace()
        export_category_pos_to_html(
            category_pos,
            output_html=f'./debug_vis/04_29_rel_dir_{session_id}_pi3_load_{file_name}',
            max_points_per_instance=1000,
            marker_size=2,
        )
        if still_missing_categories:
            # save to a txt file for easier debugging
            with open(f'./debug_vis/04_29_rel_dist_{session_id}_pi3_load_still_missing_categories.txt', 'w') as f:
                f.write(f"Still missing categories after recovery: {still_missing_categories}\n")
                f.write(f"Recovery info: {json.dumps(recovery_info, indent=2)}\n")
        # import ipdb; ipdb.set_trace()
        aggregated_pos_center = {}
        for key in category_pos:
            merged_instances = category_pos[key]['merged_instances']
            if not merged_instances:
                continue
            centers = []
            for inst in merged_instances:
                pts = inst['points_3d']
                center = np.median(pts, axis=0)
                center_list = array_to_printable_list(center)
                centers.append(center_list)
            aggregated_pos_center[key] = centers
        query_cache_id = f"query3d_{uuid.uuid4().hex}"
        spatial_memory.query_cache[query_cache_id] = {
            "session_id": session_id,
            "recovery_info": recovery_info,
            "instances": {
                key: [
                    {
                        "name": f"{key}_{idx + 1}",
                        "category": key,
                        "points_3d": np.asarray(inst["points_3d"], dtype=float),
                    }
                    for idx, inst in enumerate(category_pos[key].get("merged_instances", []))
                    if inst.get("points_3d") is not None and len(inst["points_3d"]) > 0
                ]
                for key in category_pos
            },
        }
        aggregated_pos_center["__query_cache_id__"] = query_cache_id
        result = {
            'category_positions': aggregated_pos_center,
            '__query_cache_id__': query_cache_id,
            'recovery_info': recovery_info,
        }

        return [ContentItem(text='3D positions extracted and aggregated successfully'), ContentItem(text=json.dumps(result, indent=2))]


@register_tool('safe_select')
class SafeSelect(BaseTool):
    description = (
        "Select ONE specific instance from the multiple candidates returned by `query_3d_object_position` "
        "for a given category. Use this when a category has several instances but the question refers to a "
        "single one (e.g. 'the chair near the window'). It renders a numbered bird's-eye-view of the candidate "
        "instances and uses one vision-model call to pick the instance that best matches `selection_criteria`.\n\n"
        "Returns a dictionary with `category`, `position` ([x, y, z]), and `instance_id`. The returned "
        "`position` can be fed directly to `set_viewpoint` (as `origin` or `look_at`)."
    )
    parameters = [
        {'name': 'session_id', 'type': 'string', 'description': 'The session ID of the spatial memory.', 'required': True},
        {'name': 'obj_queried', 'type': 'object', 'description': 'The raw dictionary returned by `query_3d_object_position`.', 'required': True},
        {'name': 'obj_name', 'type': 'string', 'description': 'The category name (a key of obj_queried) whose instance should be selected.', 'required': True},
        {'name': 'selection_criteria', 'type': 'string', 'description': 'A short natural-language description of which instance to pick (e.g. "the one closest to the window"). Optional.', 'required': False},
    ]

    def call(self, params: str, **kwargs) -> str:
        global runtime
        params = self._verify_json_format_args(params)
        session_id = params['session_id']
        obj_queried = params.get('obj_queried', {}) or {}
        obj_name = params['obj_name']
        criteria = params.get('selection_criteria', None)

        raw = obj_queried.get(obj_name, []) if isinstance(obj_queried, dict) else []
        candidates = []
        for it in raw:
            if isinstance(it, dict) and 'position' in it:
                candidates.append(list(it['position']))
            elif isinstance(it, (list, tuple)) and len(it) == 3:
                candidates.append(list(it))
        if len(candidates) == 0:
            return [ContentItem(text=f'Error: safe_select found no instances of "{obj_name}" in obj_queried.')]
        if len(candidates) == 1:
            return [ContentItem(text=json.dumps({"category": obj_name, "position": candidates[0], "instance_id": 0}))]

        # Render a numbered semantic BEV of the candidate instances.
        objects = [{"name": f"{obj_name} #{i}", "position": candidates[i]} for i in range(len(candidates))]
        bev_img = None
        try:
            bev_items = RenderSemanticBEV().call(
                json.dumps({"session_id": session_id, "objects": objects}), **kwargs
            )
            for it in (bev_items or []):
                img = getattr(it, 'image', None) if not isinstance(it, dict) else it.get('image')
                if img:
                    bev_img = img
        except Exception:
            bev_img = None

        idx = 0
        try:
            idx = self._vlm_pick(bev_img, obj_name, criteria, len(candidates))
        except Exception:
            idx = 0
        idx = max(0, min(int(idx), len(candidates) - 1))
        return [ContentItem(text=json.dumps({"category": obj_name, "position": candidates[idx], "instance_id": idx}))]

    def _vlm_pick(self, bev_img, obj_name, criteria, n):
        from tools.llm_cfg import build_llm_cfg
        from qwen_agent.agents import Assistant
        from qwen_agent.llm.schema import Message
        cfg = build_llm_cfg("gemini-3-flash-preview", 0.0, 1.0)
        assistant = Assistant(llm=cfg, function_list=[])
        crit = criteria or f"the most relevant {obj_name} for the question"
        prompt = (
            f"The bird's-eye-view diagram shows {n} candidate instances of '{obj_name}', "
            f"labeled '{obj_name} #0' through '{obj_name} #{n - 1}'. "
            f"Pick the single instance that best matches: {crit}. "
            f"Respond with ONLY the integer index (0 to {n - 1})."
        )
        content = []
        if bev_img:
            content.append({"image": bev_img})
        content.append({"text": prompt})
        resp = assistant.run_nonstream(messages=[Message('user', content)])
        text = ""
        for m in (resp or []):
            c = m['content'] if isinstance(m, dict) else getattr(m, 'content', None)
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):
                for it in c:
                    t = it.get('text') if isinstance(it, dict) else getattr(it, 'text', None)
                    if t:
                        text = t
        mm = re.search(r'\d+', text or "")
        return int(mm.group()) if mm else 0


# ======================== Render Tools =======================
@register_tool('render_semantic_bev')
class RenderSemanticBEV(BaseToolWithFileAccess):
    description = (
        "Render a symbolic bird's-eye-view (BEV) diagram of the scene based on the current spatial memory.\n\n"

        "This tool visualizes the spatial layout of selected entities (such as cameras or objects) "
        "in a top-down view. Each entity will be rendered as a labeled marker at its 3D position.\n\n"

        "Use this tool when you want to reason about spatial relationships such as:\n"
        "- left / right / front / behind\n"
        "- relative layout of objects\n"
        "- camera motion or viewpoint displacement\n\n"

        "The positions provided to this tool are interpreted under the current active viewpoint. "
        "If the viewpoint has been changed using `set_viewpoint`, `step_camera`, or `turn_camera`, "
        "the BEV diagram will reflect the updated coordinate system.\n\n"

        "Camera entities are provided explicitly through `camera_indices`, while objects can be provided either "
        "through `objects` with positions and optional orientations, or through `queried_objects` using the raw "
        "dictionary returned by `query_3d_object_position`. Positions are typically obtained from other tools "
        "such as `query_camera_pose` or `query_3d_object_position`."
    )
    parameters = [
        {
            "name": "session_id",
            "type": "string",
            "description": (
                "The session ID of the spatial memory used for rendering."
            ),
            "required": True,
        },
        {
            "name": "camera_indices",
            "type": "array",
            "description": (
                "A list of one-based camera/frame indices to visualize as cameras in the BEV."
            ),
            "items": {
                "type": "integer",
            },
            "required": False
        },
        {
            "name": "objects",
            "type": "array",
            "description": (
                "A list of objects to visualize in the BEV diagram. "
                "Each object must include a name and a 3D position. "
                "An optional orientation vector can be provided to render the object's facing direction."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Label shown in the BEV diagram."
                    },
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "3D position in the current coordinate system."
                    },
                    "orientation": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "Forward direction vector in the current coordinate system."
                    }
                }
            },
            "required": False
        },
        {
            "name": "queried_objects",
            "type": "object",
            "description": (
                "The raw dictionary returned by `query_3d_object_position`, mapping each category name "
                "to a list of approximate 3D positions. This is the recommended way to visualize queried "
                "object positions without manually indexing into variable-length lists."
            ),
            "required": False,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        # `params` are the arguments generated by the LLM agent.
        # import ipdb; ipdb.set_trace()
        global runtime
        params = self._verify_json_format_args(params)
        self.work_dir = get_managed_workspace_dir("tools", "render_semantic_bev")
        os.makedirs(self.work_dir, exist_ok=True)

        session_id = params['session_id']
        camera_indices = params.get('camera_indices', [1])  # default to the first camera/frame
        objects = params.get('objects', [])
        queried_objects = params.get('queried_objects', {})

        if session_id not in runtime.session_mem:
            return [ContentItem(text=f'Error: Session ID {session_id} not found. Please make sure to use the correct session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.')]

        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        output_content = []
        entities = []

        if spatial_memory.query_map_scale is not None:
            camera_indices_to_query = list(map(lambda x: int(x * spatial_memory.query_map_scale), camera_indices))
        else:
            camera_indices_to_query = camera_indices
        camera_poses, errors = spatial_memory.query_camera_pose(camera_indices_to_query)
        for error in errors:
            output_content.append(ContentItem(text=error))
        for cam_index, pose in zip(camera_indices, camera_poses):
            entities.append({
                "name": f"camera_{cam_index}",
                "type": "camera",
                "position": pose["position"],
                "orientation": pose["forward"],
            })

        for obj in objects:
            entities.append({
                "name": obj["name"],
                "type": "object",
                "position": obj["position"],
                "orientation": obj.get("orientation"),
            })

        if queried_objects is None:
            queried_objects = {}
        if not isinstance(queried_objects, dict):
            output_content.append(
                ContentItem(
                    text="Error: `queried_objects` must be a dictionary returned by `query_3d_object_position`."
                )
            )
            return output_content

        queried_object_positions = queried_objects
        if "category_positions" in queried_objects and isinstance(queried_objects.get("category_positions"), dict):
            queried_object_positions = queried_objects["category_positions"]

        for category_name, positions in queried_object_positions.items():
            if isinstance(category_name, str) and category_name.startswith("__"):
                continue
            if not isinstance(category_name, str):
                continue
            if not isinstance(positions, list):
                continue
            for idx, position in enumerate(positions, start=1):
                if not isinstance(position, (list, tuple)) or len(position) != 3:
                    continue
                entities.append({
                    "name": f"{category_name}_{idx}",
                    "type": "object",
                    "position": list(position),
                    "orientation": None,
                })

        if not entities:
            output_content.append(ContentItem(text='Error: No valid cameras or objects provided for BEV rendering.'))
            return output_content

        # file_id = uuid.uuid4()
        file_id = session_id
        save_path = os.path.join(self.work_dir, f'semantic_bev_{file_id}.png')
        result = spatial_memory.render_semantic_bev(entities=entities, save_path=save_path, title="Semantic BEV")
        # output_content.append(ContentItem(image=save_path))
        # import ipdb; ipdb.set_trace()

        if queried_objects:
            selected_fg_save_path = None
            query_cache_id = None
            if isinstance(queried_objects, dict):
                query_cache_id = queried_objects.get("__query_cache_id__")
            if not isinstance(query_cache_id, str) and isinstance(queried_object_positions, dict):
                query_cache_id = queried_object_positions.get("__query_cache_id__")
            if isinstance(query_cache_id, str):
                cached_query = spatial_memory.query_cache.get(query_cache_id)
                if isinstance(cached_query, dict) and cached_query.get("session_id") == session_id:
                    selected_instances = []
                    for category_name in queried_object_positions:
                        if not isinstance(category_name, str) or category_name.startswith("__"):
                            continue
                        selected_instances.extend(cached_query.get("instances", {}).get(category_name, []))
                    if selected_instances:
                        # Overlay the queried camera positions/orientations on the selected-fg BEV.
                        camera_bev_list = [
                            {
                                "name": e["name"],
                                "position": e["position"],
                                "forward": e.get("orientation"),
                            }
                            for e in entities
                            if e.get("type") == "camera"
                        ]
                        selected_fg_save_path = spatial_memory.render_selected_fg_bev(
                            selected_instances,
                            save_path=os.path.join(self.work_dir, f"semantic_bev_selected_fg_{file_id}.png"),
                            img_size=(960, 960),
                            ego_marker_size=0,
                            cameras=camera_bev_list,
                        )
            if selected_fg_save_path:
                print(f"[Done] Saved selected foreground BEV to: {selected_fg_save_path}")
                output_content.append(ContentItem(image=selected_fg_save_path))
        else:
            print(f"[Done] Saved semantic BEV PNG to: {save_path}")
            print(f"[Done] Saved semantic BEV with overlay to: {result.get('overlay_save_path', 'N/A')}")
            output_content.append(ContentItem(image=result['overlay_save_path']))

        
        # output_content.append(
        #     ContentItem(
        #         text=json.dumps(
        #             {
        #                 "entity_bev_list": result['entity_bev_list'],
        #                 "bev_meta_info": result['bev_meta_info'],
        #                 "overlay_save_path": result.get("overlay_save_path"),
        #                 "selected_fg": selected_fg_save_path,
        #             },
        #             indent=2,
        #         )
        #     )
        # )

        return output_content


@register_tool('render_rgb_bev')
class RenderRGBBEV(BaseToolWithFileAccess):
    description = (
        "Render an RGB bird's-eye-view (BEV) image of the current spatial memory.\n\n"
        "This tool generates a top-down visual rendering of the scene from the current active viewpoint. "
        "It is useful for visually checking spatial layout, camera displacement, object arrangement, "
        "and relative position from above.\n\n"
        "Use this tool when you want a realistic top-down visual view of the scene. "
        "If you need a simplified symbolic diagram with labeled entities and arrows, use `render_semantic_bev` instead.\n\n"
        "Important coordinate rule:\n"
        "- The rendering is based on the current active viewpoint.\n"
        "- If the viewpoint has been changed by `set_viewpoint`, `step_camera`, or `turn_camera`, "
        "the BEV rendering will reflect the updated viewpoint.\n\n"
        "This tool is mainly for visual inspection, not for directly returning object coordinates."
    )

    parameters = [
        {
            'name': 'session_id',
            'type': 'string',
            'description': (
                "The session ID of the spatial memory to render."
            ),
            'required': True
        },
        {
            'name': 'annotations',
            'type': 'array',
            'description': (
                "Optional visual annotations to overlay on the BEV rendering, "
                "such as points, vectors, or trajectories."
            ),
            'items': {
                'type': 'object'
            },
            'required': False
        },
        {
            'name': 'ego_marker_size',
            'type': 'integer',
            'description': (
                "Optional marker size for the ego viewpoint indicator drawn on the RGB BEV. "
                "Use a smaller value if the default marker looks too large."
            ),
            'required': False
        }
    ]
    def call(self, params: str, **kwargs) -> str:
        global runtime
        params = self._verify_json_format_args(params)
        self.work_dir = get_managed_workspace_dir("tools", "render_rgb_bev")
        os.makedirs(self.work_dir, exist_ok=True)

        session_id = params['session_id']
        ego_marker_size = params.get('ego_marker_size', 140)
        if session_id not in runtime.session_mem:
            return [ContentItem(text=f'Error: Session ID {session_id} not found. Please make sure to use the correct session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.')]
        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        save_paths = []
        img_id = session_id
        for i in range(3):
            # save_path = os.path.join(self.work_dir, f'ego_rgb_{uuid.uuid4()}.png')
            save_path = os.path.join(self.work_dir, f'ego_rgb_{session_id}_{i}.png')
            save_paths.append(save_path)
        spatial_memory.save_render_bev_view(
            save_paths=save_paths,
            img_size=(960, 960),
            ego_marker_size=ego_marker_size,
        )
        print(f"[Done] Saved RGB BEV PNG to: {save_paths}")
        return [ContentItem(image=save_path) for save_path in save_paths][-1:]


@register_tool('render_ego_rgb')
class RenderEgoRGB(BaseToolWithFileAccess):
    description = (
        "Render an RGB image from the current active viewpoint using the spatial memory.\n\n"
        "This tool generates a first-person or observer-centric visual view of the scene. "
        "Use it when you want to visually inspect what the scene looks like from the current viewpoint, "
        "for example to check what is visible in front, behind, left, or right.\n\n"
        "Important coordinate rule:\n"
        "- The rendering is based on the current active viewpoint.\n"
        "- If the viewpoint has been changed by `set_viewpoint`, `step_camera`, or `turn_camera`, "
        "the rendered image will reflect the updated viewpoint.\n\n"
        "This tool is useful for visual checking. "
        "If you instead need a top-down visual view, use `render_rgb_bev`. "
        "If you need a simplified symbolic diagram, use `render_semantic_bev`.\n\n"
        "Rendering note:\n"
        "- Regions with no reconstructed geometry are shown as a light gray background.\n"
        "- Treat that background as unknown / empty render area, not as a bright light source or scene object.\n"
        "- When the valid rendered region is too small, the renderer may automatically crop the image to focus on the visible content."
    )
    parameters = [
        {
            'name': 'session_id',
            'type': 'string',
            'description': (
                "The session ID of the spatial memory to render."
            ),
            'required': True
        },
    ]
    def call(self, params: str, **kwargs) -> str:
        global runtime
        params = self._verify_json_format_args(params)
        self.work_dir = get_managed_workspace_dir("tools", "render_ego_rgb")
        os.makedirs(self.work_dir, exist_ok=True)

        session_id = params['session_id']
        if session_id not in runtime.session_mem:
            return [ContentItem(text=f'Error: Session ID {session_id} not found. Please make sure to use the correct session ID that is returned when you build the spatial memory using the `build_static_spatial_memory` tool.')]
        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        save_path = os.path.join(self.work_dir, f'ego_rgb_{uuid.uuid4()}.png')
        # spatial_memory.save_render_perspective_view(save_path=save_path, img_size=(960, 540), FOV=80.0)
        spatial_memory.save_render_perspective_view(save_path=save_path, img_size=(960, 960))
        print(f"[Done] Saved semantic BEV PNG to: {save_path}")
        return [ContentItem(image=save_path)]


def extract_floor_plane(points, axis=2, quantile=0.05):
    """
    从点云中找到地板平面
    - points: (N, 3) numpy array
    - axis: int，作为高度的轴（默认 z=2）
    - quantile: float，取最底部多少比例的点
    """
    # 1. 取指定轴的最小 quantile 部分
    height_vals = points[:, axis]
    threshold = np.quantile(height_vals, quantile)
    floor_candidates = points[height_vals <= threshold]

    # 2. 用 RANSAC 拟合平面：z = ax + by + c
    X = floor_candidates[:, [i for i in range(3) if i != axis]]
    y = floor_candidates[:, axis]

    ransac = RANSACRegressor(residual_threshold=0.01, max_trials=1000)
    ransac.fit(X, y)

    coef = ransac.estimator_.coef_   # 平面参数 (a, b)
    intercept = ransac.estimator_.intercept_

    # 3. 构造平面法向量
    normal = np.zeros(3)
    normal[[i for i in range(3) if i != axis]] = coef
    normal[axis] = -1
    normal = normal / np.linalg.norm(normal)

    return {
        "plane_normal": normal,
        "plane_intercept": intercept,
        "inliers": floor_candidates[ransac.inlier_mask_]
    }


def align_spatial_memory_with_up(points_all, c2w_all, target_global_up):
    """
    用平均 up 向量对齐点云
    - points_all: (N, H, W, 3) 全部帧的 dense 3D positions
    - c2w_all: (M, 4, 4) 相机外参 (camera-to-world) 矩阵
    - target_axis: 'x', 'y', 'z' 表示要把 up 对齐到哪个全局轴
    
    return:
        rotated_points: 对齐后的点云 (N, H, W, 3)
        R: 旋转矩阵 (3, 3)
        mean_up: 平均的 up 向量
    """
    # 提取所有相机的 up 向量 (第二列是 down) [right, down, forward]
    # import ipdb; ipdb.set_trace()
    ups = -c2w_all[:, :3, 1]   # (M, 3)
    mean_up = np.mean(ups, axis=0)
    mean_up /= np.linalg.norm(mean_up)  # 单位化

    # 目标方向
    target = target_global_up

    # 计算旋转矩阵: 把 mean_up 旋转到 target
    v = np.cross(mean_up, target)
    c = np.dot(mean_up, target)
    if np.linalg.norm(v) < 1e-8:  # 已经对齐或反向
        if c > 0:  # 已经对齐
            R = np.eye(3)
        else:  # 反向
            # 找一个正交向量做 180° 旋转
            ortho = np.array([1, 0, 0]) if abs(mean_up[0]) < 0.9 else np.array([0, 1, 0])
            v = np.cross(mean_up, ortho)
            v /= np.linalg.norm(v)
            R = -np.eye(3) + 2 * np.outer(v, v)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

    # 应用旋转
    points_flat = points_all.reshape(-1, 3)
    rotated_flat = points_flat @ R.T
    rotated_points = rotated_flat.reshape(points_all.shape)

    return rotated_points, R, mean_up


def two_stage_up_estimation(points_all, points_color_all, c2w_all, target_axis="-y"):
    """
    先用 PCA 粗对齐，再用 RANSAC 精对齐
    - points_all: (N, H, W, 3) 全部帧的 dense 3D positions
    - c2w_all: (M, 4, 4) 相机外参 (camera-to-world) 矩阵
    - target_axis: 'x', 'y', 'z' 表示要把 up 对齐到哪个全局轴
    
    return:
        rotated_points: 对齐后的点云 (N, H, W, 3)
        R_total: 总的旋转矩阵 (3, 3)
        mean_up: 平均的 up 向量
    """
    target_global_up = {
        'x': np.array([1, 0, 0], dtype=float),
        'y': np.array([0, 1, 0], dtype=float),
        'z': np.array([0, 0, 1], dtype=float),
        '-x': np.array([-1, 0, 0], dtype=float),
        '-y': np.array([0, -1, 0], dtype=float),
        '-z': np.array([0, 0, -1], dtype=float),
    }[target_axis]
    # Step 1: Use mean up for coarse alignment
    rotated_points, R_mean, mean_up = align_spatial_memory_with_up(points_all, c2w_all, target_global_up=target_global_up)
    # import ipdb; ipdb.set_trace()

    # Step 2: RANSAC refinement using floor plane normal
    vertical_axis = {
        'x': 0,
        'y': 1,
        'z': 2,
    }[target_axis[-1]]  # 'x', 'y', or 'z'
    points_flat = rotated_points.reshape(-1, 3)
    floor = extract_floor_plane(points_flat, axis=vertical_axis, quantile=0.01)
    plane_normal = floor["plane_normal"]
    global_up = target_global_up
    if np.dot(plane_normal, global_up) < 0:
        plane_normal = -plane_normal

    v = np.cross(plane_normal, global_up)
    c = np.dot(plane_normal, global_up)
    if np.linalg.norm(v) < 1e-8:  # 已经对齐或反向
        if c > 0:  # 已经对齐
            R = np.eye(3)
        else:  # 反向
            # 找一个正交向量做 180° 旋转
            ortho = np.array([1, 0, 0]) if abs(plane_normal[0]) < 0.9 else np.array([0, 1, 0])
            v = np.cross(plane_normal, ortho)
            v /= np.linalg.norm(v)
            R = -np.eye(3) + 2 * np.outer(v, v)
    elif (np.dot(plane_normal, global_up) < 0.7 or c2w_all.shape[0] < 5):
        # RANSAC might give a noisy normal, so we check its consistency with the mean_up. If too different, we trust mean_up more and skip RANSAC correction
        return rotated_points, R_mean, global_up
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
    
    points_flat = points_flat @ R.T

    rotation = R @ R_mean
    # rotation = R_mean
    return points_flat.reshape(rotated_points.shape), rotation, global_up

def construct_3d_spatial_memory(images: torch.Tensor, pi3: Pi3):
    print("Intializing Pi3 model...")
    dtype = torch.bfloat16
    # imgs = images.to(dtype).to(pi3.device) # (N, 3, H, W)
    imgs = images.to(pi3.device) # (N, 3, H, W)
    print("Constructing 3D spatial memory with Pi3...")
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = pi3(imgs[None]) # Add batch dimension

    with torch.no_grad():
        predictions = pi3(imgs[None]) # Add batch dimension

    # import ipdb; ipdb.set_trace()
    predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions['conf'] = torch.sigmoid(predictions['conf'])
    edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
    predictions['conf'][edge] = 0.0
    del predictions['local_points']

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().float().numpy().squeeze(0)  # remove batch dimension

    normalized_intrinsics = predictions['intrinsics']  # (N, 3, 3)
    # normalized_intrinsics = []
    spatial_memory = predictions['points']  # (N, H, W, 3)
    confidence = predictions['conf'][..., 0] if predictions['conf'].ndim == 4 else predictions['conf']
    camera_trajectory = predictions['camera_poses']  # (N, 4, 4)

    # downsampling + two-stage alignment
    spatial_memory, R, global_up = two_stage_up_estimation(spatial_memory, 
                                                            predictions['images'],
                                                            predictions['camera_poses'], 
                                                            target_axis="-y"
                                                            )
    # rotate camera trajectory
    cams_rot = camera_trajectory[:, :3, :3]
    cams_transl = camera_trajectory[:, :3, 3]
    # camera_trajectory[:, :3, :3] = cams_rot @ R.T # buggy version
    camera_trajectory[:, :3, :3] = R @ cams_rot # !Attention: R @ rot, not rot @ R.T, because we want to rotate the points, which is like changing the coordinate system
    camera_trajectory[:, :3, 3] = cams_transl @ R.T

    # 保存对齐后的点云
    return spatial_memory, confidence, camera_trajectory, normalized_intrinsics, global_up


def construct_3d_spatial_memory_metric(images: torch.Tensor, pi3x: Pi3X):
    """
    Experimental Pi3X metric backend.

    This follows the existing Pi3 memory construction path, but uses Pi3X outputs.
    It is kept separate so the current Pi3 backend remains unchanged.
    """
    print("Initializing Pi3X metric backend...")
    dtype = torch.bfloat16
    imgs = images.to(pi3x.device)
    print("Constructing 3D spatial memory with Pi3X...")
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = pi3x(imgs[None])
    # with torch.no_grad():
    #     predictions = pi3x(imgs[None])

    predictions["images"] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions["conf"] = torch.sigmoid(predictions["conf"])
    edge = depth_edge(predictions["local_points"][..., 2], rtol=0.03)
    predictions["conf"][edge] = 0.0

    normalized_intrinsics = None
    if "rays" in predictions:
        try:
            from pi3.utils.geometry import recover_intrinsic_from_rays_d

            normalized_intrinsics = recover_intrinsic_from_rays_d(
                torch.nn.functional.normalize(predictions["local_points"], dim=-1),
                force_center_principal_point=True,
            )
        except Exception:
            normalized_intrinsics = None

    del predictions["local_points"]

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().float().numpy().squeeze(0)

    if normalized_intrinsics is not None:
        normalized_intrinsics = normalized_intrinsics.cpu().float().numpy().squeeze(0)

    spatial_memory = predictions["points"]
    confidence = predictions["conf"][..., 0] if predictions["conf"].ndim == 4 else predictions["conf"]
    camera_trajectory = predictions["camera_poses"]

    spatial_memory, R, global_up = two_stage_up_estimation(
        spatial_memory,
        predictions["images"],
        predictions["camera_poses"],
        target_axis="-y",
    )

    cams_rot = camera_trajectory[:, :3, :3]
    cams_transl = camera_trajectory[:, :3, 3]
    camera_trajectory[:, :3, :3] = R @ cams_rot
    camera_trajectory[:, :3, 3] = cams_transl @ R.T

    return spatial_memory, confidence, camera_trajectory, normalized_intrinsics, global_up


@register_tool('build_static_spatial_memory_metric')
# @register_tool('build_static_spatial_memory')
class BuildStaticSpatialMemoryMetric(BaseToolWithFileAccess):
# class BuildStaticSpatialMemory(BaseToolWithFileAccess):
    description = (
        "Build a reusable 3D spatial memory using the experimental Pi3X metric backend.\n\n"
        "This is a placeholder entry point for trying Pi3X as a new memory backend without "
        "changing the existing `build_static_spatial_memory` behavior. It uses the same static-scene "
        "assumption as the original tool, but targets approximate metric-scale reconstruction when "
        "Pi3X is available in the local third_party checkout.\n\n"
        "If the local `third_party/Pi3` repository does not yet include Pi3X, this tool will return "
        "a clear error instead of silently falling back to Pi3."
    )
    # parameters = BuildStaticSpatialMemoryPi3.parameters
    parameters = BuildStaticSpatialMemory.parameters

    def call(self, params: str, **kwargs) -> str:
        global runtime
        # import ipdb; ipdb.set_trace()
        params = self._verify_json_format_args(params)
        messages = kwargs.get('messages', [])
        image_list = [
            _normalize_local_media_path(p)
            for p in extract_images_from_messages(messages)
        ]
        video_list = [
            _normalize_local_media_path(p)
            for p in extract_videos_from_messages(messages)
        ]

        if image_list:
            media_image_list = image_list
        elif video_list:
            media_image_list = _extract_video_frames_cached(video_list[0], fps=1.0, max_frames=64)
        elif params.get("input_type") == "video" and params.get("video_path"):
            media_image_list = _extract_video_frames_cached(params["video_path"], fps=1.0, max_frames=64)
        else:
            media_image_list = [
                _normalize_local_media_path(p)
                for p in params.get("image_paths", [])
            ]

        try:
            for image_path in media_image_list:
                if image_path.startswith('http'):
                    response = requests.get(image_path)
                    response.raise_for_status()
                    Image.open(BytesIO(response.content))
                elif os.path.exists(image_path):
                    Image.open(image_path)
                else:
                    Image.open(os.path.join(self.work_dir, image_path))
        except Exception:
            return [ContentItem(text=f'Error: Invalid input image/video-derived frames {media_image_list}')]

        cur_time = int(time()) % 10000
        session_id = f'session_metric_{cur_time}'
        try:
            runtime.ensure_metric_spatial_memory(
                session_id=session_id,
                image_paths=media_image_list,
                construct_3d_spatial_memory_fn=construct_3d_spatial_memory_metric,
            )
        except Exception as exc:
            return [ContentItem(text=f"Error: {exc}")]

        spatial_memory: SpatialMemory = runtime.session_mem[session_id]
        result = {
            "session_id": session_id,
            "meta_info": spatial_memory.meta_info,
            "backend": "Pi3X_metric_placeholder",
        }
        return [
            ContentItem(text='Metric spatial memory built successfully.'),
            ContentItem(text=json.dumps(result)),
            ContentItem(
                text=(
                    "**This backend is experimental. It preserves the current alignment pipeline, "
                    "but swaps the reconstruction model to Pi3X. Intrinsics recovery is currently "
                    "best-effort from rays and may be unavailable depending on the local Pi3X checkout.**"
                )
            ),
        ]


if __name__ == "__main__":
    build_static_memory = BuildStaticSpatialMemory()
    # build_static_memory_metric = BuildStaticSpatialMemoryMetric()
    query_3d_object_position = Query3DObjectPosition()
    # render_camera_pose_egocentric_bev = RenderCameraPoseEgocentricBEV()
    render_semantic_bev = RenderSemanticBEV()
    # render_object_pose_egocentric_bev = RenderObjectPoseEgocentricBEV()
    render_rgb_bev = RenderRGBBEV()
    transform_spatial_memory = SetViewpoint()
    query_camera_pose = QueryCameraPose()
    render_ego_rgb = RenderEgoRGB()
    move_camera = StepCamera()
    rotate_camera = TurnCamera()
    message = Message(role="user", 
                      content=[
                        # {"image": "./data/MindCube/other_all_image/among/shoe_013/front_013.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/shoe_013/left_105.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/shoe_013/back_175.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/shoe_013/right_248.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/ball_543/front_036.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/ball_543/left_091.jpg"}
                        # {"image": "./data/MindCube/other_all_image/among/35122ec333dc84cf223ea366c3ad968dfece8ca02e57a8c241819837583f76f0/front_138.png"},
                        # {"image": "./data/MindCube/other_all_image/among/35122ec333dc84cf223ea366c3ad968dfece8ca02e57a8c241819837583f76f0/left_169.png"},
                        # {"image": "./data/MindCube/other_all_image/among/35122ec333dc84cf223ea366c3ad968dfece8ca02e57a8c241819837583f76f0/back_192.png"},
                        # {"image": "./data/MindCube/other_all_image/among/35122ec333dc84cf223ea366c3ad968dfece8ca02e57a8c241819837583f76f0/right_218.png"},
                        
                        # {"image": "./data/MindCube/other_all_image/among/shoe_216/front_007.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/shoe_216/left_084.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/shoe_216/back_157.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/shoe_216/right_246.jpg"}
                        
                        # sample No. 15
                        # {"image":"./data/MindCube/other_all_image/among/shoe_254/front_006.jpg"},
                        # {"image":"./data/MindCube/other_all_image/among/shoe_254/left_091.jpg"},
                        # {"image":"./data/MindCube/other_all_image/among/shoe_254/back_170.jpg"},
                        # {"image":"./data/MindCube/other_all_image/among/shoe_254/right_258.jpg"}
                        
                        # sample No. 115
                        # {"image": "./data/MindCube/other_all_image/among/train_037/front_020.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/train_037/left_087.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/train_037/back_147.jpg"},
                        # {"image": "./data/MindCube/other_all_image/among/train_037/right_202.jpg"}

                        # sample No. 856
                        # {"image": "./data/MindCube/other_all_image/around/b6e63674c6b7fe330c3fb7d04b204779993cdae5121b8b008d2460b372590fb0/1_frame_00260.png"},
                        # {"image": "./data/MindCube/other_all_image/around/b6e63674c6b7fe330c3fb7d04b204779993cdae5121b8b008d2460b372590fb0/2_frame_00095.png"},
                        # {"image": "./data/MindCube/other_all_image/around/b6e63674c6b7fe330c3fb7d04b204779993cdae5121b8b008d2460b372590fb0/3_frame_00197.png"},
                        # {"image": "./data/MindCube/other_all_image/around/b6e63674c6b7fe330c3fb7d04b204779993cdae5121b8b008d2460b372590fb0/4_frame_00137.png"}

                        # VSI
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_00.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_01.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_02.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_03.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_04.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_05.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_06.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_07.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_08.png"},
                        # {"image": "./data/vsi_sampled/1124_selected_frames/frame_09.png"}

                        {"video": "./data/VSIBench/arkitscenes/41069043.mp4"}
                    ])
    messages = [message]
    params = json.dumps({"input_type": "image"})
    _, info, _ = build_static_memory.call(params=params, messages=messages)
    info = json.loads(info.text)
    session_id = info['session_id']
    # query_3d_object_position.call(params=json.dumps({
    #     "session_id": session_id,
    #     "category_names": ["wall"],
    # }))
    # import ipdb; ipdb.set_trace()
    camera_pose = query_camera_pose.call(params=json.dumps({
        "session_id": session_id,
        "frame_indices": [1, 2, 3]
    }))[0]
    camera_pose = json.loads(camera_pose.text)
    camera_pos_1 = camera_pose[0]['position']
    camera_pos_2 = camera_pose[1]['position']
    camera_pos_3 = camera_pose[2]['position']
    camera_forward_1 = camera_pose[0]['forward']
    camera_forward_2 = camera_pose[1]['forward']
    camera_forward_3 = camera_pose[2]['forward']
    camera_backward_3 = [-c for c in camera_forward_3]
    camera_up_1 = camera_pose[0]['up']
    camera_up_2 = camera_pose[1]['up']
    camera_up_3 = camera_pose[2]['up']
    camera_right_2 = camera_pose[1]['right']
    camera_left_2 = [-c for c in camera_right_2]
    spatial_memory: SpatialMemory = runtime.session_mem[session_id]
    # spatial_memory.visualize_3d("./debug_vis/03_24_spatial_memory_uiuc_after.rrd")
    # spatial_memory.visualize_3d("./debug_vis/04_13_spatial_memory_counting.rrd")
    # import ipdb; ipdb.set_trace()
    # query_3d_object_position.call(params=json.dumps({
    #     "session_id": session_id,
    #     "category_names": ["shoe"],
    # }))
    # import ipdb; ipdb.set_trace()
    # render_semantic_bev.call(params=json.dumps({
    #     "session_id": session_id,
    #     "entities": [
    #         {
    #             "name": "camera_1",
    #             "type": "camera",
    #             "position": camera_pos_1,
    #             "orientation": camera_forward_1
    #         },
    #         {
    #             "name": "camera_2",
    #             "type": "camera",
    #             "position": camera_pos_2,
    #             "orientation": camera_forward_2
    #         },
    #         {
    #             "name": "camera_3",
    #             "type": "camera",
    #             "position": camera_pos_3,
    #             "orientation": camera_forward_3
    #         },
    #     ]
    # }))
    # import ipdb; ipdb.set_trace()
    transform_spatial_memory.call(params=json.dumps({
        "session_id": session_id,
        "origin": camera_pos_2,
        "forward": camera_forward_2,  # in meters
        "up": camera_up_2  # default up direction
    }))
    # import ipdb; ipdb.set_trace()
    # render_rgb_bev.call(params=json.dumps({
    #     "session_id": session_id,
    # }))
    # import ipdb; ipdb.set_trace()
    # render_ego_rgb.call(params=json.dumps({
    #     "session_id": session_id,
    # }))
    # rotate_camera.call(params=json.dumps({
    #     "session_id": session_id,
    #     "direction": "right"
    # }))
    # import ipdb; ipdb.set_trace()
    # render_ego_rgb.call(params=json.dumps({
    #     "session_id": session_id,
    # }))


    # import ipdb; ipdb.set_trace()
    # render_rgb_bev.call(params=json.dumps({
    #     "session_id": session_id,
    # }))

    # import ipdb; ipdb.set_trace()
    tables = query_object_pose.call(params=json.dumps({
        "session_id": session_id,
        "category_names": ["table"],
    }))
    # import ipdb; ipdb.set_trace()
    render_semantic_bev.call(params=json.dumps({
        "session_id": session_id,
        "queried_objects": tables
    }))
    # import ipdb; ipdb.set_trace()
    # render_ego_rgb.call(params=json.dumps({
    #     "session_id": session_id,
    # }))
    # rotate_camera.call(params=json.dumps({
    #     "session_id": session_id,
    #     "direction": "right"
    # }))
    # import ipdb; ipdb.set_trace()
    # render_ego_rgb.call(params=json.dumps({
    #     "session_id": session_id,
    # }))
    # move_camera.call(params=json.dumps({
    #     "session_id": session_id,
    #     "direction": "forward",
    # }))
    # import ipdb; ipdb.set_trace()
    # render_ego_rgb.call(params=json.dumps({
    #     "session_id": session_id,
    # }))
    # move_camera.call(params=json.dumps({
    #     "session_id": session_id,
    #     "direction": "right",
    # }))
    # import ipdb; ipdb.set_trace()
    # render_ego_rgb.call(params=json.dumps({
    #     "session_id": session_id,
    # }))
    # move_camera.call(params=json.dumps({
    #     "session_id": session_id,
    #     "direction": "left",
    # }))
    # import ipdb; ipdb.set_trace()
    # render_ego_rgb.call(params=json.dumps({
    #     "session_id": session_id,
    # }))

    # bev = render_camera_pose_egocentric_bev.call(params=json.dumps({
    #     "session_id": session_id,
    #     "camera_index_list": [1, 2],
    #     "origin": camera_pos_1,
    #     "forward_direction": camera_forward_1,
    # }))[0]
    # import ipdb; ipdb.set_trace()
    # bev_object = render_object_pose_egocentric_bev.call(params=json.dumps({
    #     "session_id": session_id,
    #     "origin": camera_pos_1,
    #     "forward_direction": camera_forward_1,
    #     "object_position_list": [{
    #         "object_name": "camera_2",
    #         "position": camera_pos_2,
    #     }]
    # }))[0]
    
    # import ipdb; ipdb.set_trace()
    # transform_spatial_memory.call(params=json.dumps({
    #     "session_id": session_id,
    #     "origin": camera_pos_1,
    #     "forward": camera_forward_1,  # in meters
    # }))
    # import ipdb; ipdb.set_trace()
    # camera_pos_2_trans = query_camera_pose.call(params=json.dumps({
    #     "session_id": session_id,
    #     "frame_index_list": [1, 2]
    # }))[0]
