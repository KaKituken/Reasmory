from typing import List, Optional, Tuple
import json
from io import BytesIO
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from sklearn.covariance import EmpiricalCovariance
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    from .vis_utils import (
        category_base_color,
        get_semantic_bev_style,
        render_perspective_view,
        render_semantic_bev_entities,
        save_pointcloud_with_vector_html,
    )
except:
    from vis_utils import (
        category_base_color,
        get_semantic_bev_style,
        render_perspective_view,
        render_semantic_bev_entities,
        save_pointcloud_with_vector_html,
    )

def array_to_printable_list(array):
    return list(map(lambda x: round(float(x), 2), array.tolist()))


def draw_bev_center_marker(rendered_img, marker_size: int = 220):
    """Draw the ego marker using the same style source as render_semantic_bev."""
    if rendered_img is None or not hasattr(rendered_img, "size"):
        return rendered_img
    if marker_size <= 0:
        return rendered_img

    width, height = rendered_img.size
    cx, cy = width // 2, height // 2
    style = get_semantic_bev_style(["ego"], marker_size=marker_size)
    marker_rgba = style["color_by_type"]["ego"]
    marker_fill = tuple(int(round(channel * 255)) for channel in marker_rgba[:3])
    marker_radius = max(5, int(np.sqrt(style["marker_size"]) / 1.4))
    arrow_len = max(20, min(width, height) // 7 // (220 / marker_size))  # arrow length scales with marker size, with a reasonable default
    shaft_width = max(3, marker_radius // 3)
    head_len = max(10, marker_radius + 4)
    head_half_width = max(7, marker_radius // 2 + 2)
    label_offset = max(12, marker_radius + 8)
    label_pad_x = 8
    label_pad_y = 5

    draw = ImageDraw.Draw(rendered_img)
    font = ImageFont.load_default()

    marker_outline = (0, 0, 0)
    arrow_color = (255, 0, 0)
    label_text = "ego"

    # Marker body
    draw.ellipse(
        [(cx - marker_radius, cy - marker_radius), (cx + marker_radius, cy + marker_radius)],
        fill=marker_fill,
        outline=marker_outline,
        width=2,
    )

    # Direction arrow
    draw.line(
        [(cx, cy), (cx, cy - arrow_len)],
        fill=arrow_color,
        width=shaft_width,
    )
    draw.polygon(
        [
            (cx, cy - arrow_len - head_len),
            (cx - head_half_width, cy - arrow_len + max(2, head_len // 5)),
            (cx + head_half_width, cy - arrow_len + max(2, head_len // 5)),
        ],
        fill=arrow_color,
    )

    # Label box
    label_x = cx + label_offset
    label_y = cy - label_offset
    text_bbox = draw.textbbox((label_x, label_y), label_text, font=font)
    bg_box = (
        text_bbox[0] - label_pad_x,
        text_bbox[1] - label_pad_y,
        text_bbox[2] + label_pad_x,
        text_bbox[3] + label_pad_y,
    )
    draw.rounded_rectangle(bg_box, radius=6, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.text((label_x, label_y), label_text, fill=(0, 0, 0), font=font)
    return rendered_img


def _hex_to_rgb_uint8(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def draw_category_legend(
    rendered_img: Image.Image,
    category_to_color: dict[str, Tuple[int, int, int]],
) -> Image.Image:
    if rendered_img is None or not category_to_color:
        return rendered_img

    draw = ImageDraw.Draw(rendered_img)
    font = ImageFont.load_default()
    title = "Categories"
    padding = 12
    swatch_size = 14
    row_gap = 8
    text_gap = 8
    categories = list(category_to_color.keys())

    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]

    max_text_width = 0
    max_text_height = 0
    for category in categories:
        text_bbox = draw.textbbox((0, 0), category, font=font)
        max_text_width = max(max_text_width, text_bbox[2] - text_bbox[0])
        max_text_height = max(max_text_height, text_bbox[3] - text_bbox[1])

    row_height = max(swatch_size, max_text_height)
    box_width = padding * 2 + max(title_width, swatch_size + text_gap + max_text_width)
    box_height = padding * 2 + title_height + 10
    box_height += len(categories) * row_height + max(0, len(categories) - 1) * row_gap

    img_width, _ = rendered_img.size
    x0 = max(10, img_width - box_width - 14)
    y0 = 14
    x1 = x0 + box_width
    y1 = y0 + box_height

    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(255, 255, 255), outline=(0, 0, 0), width=1)
    draw.text((x0 + padding, y0 + padding), title, fill=(0, 0, 0), font=font)

    cursor_y = y0 + padding + title_height + 10
    for category in categories:
        swatch_y = cursor_y + max(0, (row_height - swatch_size) // 2)
        draw.rectangle(
            (x0 + padding, swatch_y, x0 + padding + swatch_size, swatch_y + swatch_size),
            fill=category_to_color[category],
            outline=(0, 0, 0),
            width=1,
        )
        draw.text(
            (x0 + padding + swatch_size + text_gap, cursor_y),
            category,
            fill=(0, 0, 0),
            font=font,
        )
        cursor_y += row_height + row_gap

    return rendered_img


def render_xz_bbox_bev(
    instance_boxes: List[dict],
    category_to_rgb: dict[str, Tuple[int, int, int]],
    img_size: Tuple[int, int],
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
    cameras: Optional[List[dict]] = None,
) -> Image.Image:
    width, height = img_size
    cameras = cameras or []
    if not instance_boxes and not cameras:
        return Image.new("RGB", (width, height), background_rgb)

    all_corners = []
    for box in instance_boxes:
        all_corners.extend(
            [
                [box["min_x"], box["min_z"]],
                [box["min_x"], box["max_z"]],
                [box["max_x"], box["min_z"]],
                [box["max_x"], box["max_z"]],
            ]
        )
    # Include camera positions in the bounds so they are always visible.
    for cam in cameras:
        pos = cam.get("position")
        if pos is not None and len(pos) >= 3:
            all_corners.append([float(pos[0]), float(pos[2])])
    all_pos_2d = np.asarray(all_corners, dtype=float)
    x_min, x_max = all_pos_2d[:, 0].min(), all_pos_2d[:, 0].max()
    z_min, z_max = all_pos_2d[:, 1].min(), all_pos_2d[:, 1].max()
    dx = x_max - x_min
    dz = z_max - z_min
    max_range = max(dx, dz, 1.0)
    margin = 0.18 * max_range
    mid_x = 0.5 * (x_min + x_max)
    mid_z = 0.5 * (z_min + z_max)
    half_range = 0.5 * max_range

    dpi = 160
    fig_w = max(width / dpi, 1.0)
    fig_h = max(height / dpi, 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(np.asarray(background_rgb, dtype=float) / 255.0)
    ax.set_facecolor("white")
    ax.set_xlim(mid_x - half_range - margin, mid_x + half_range + margin)
    ax.set_ylim(mid_z - half_range - margin, mid_z + half_range + margin)

    for box in instance_boxes:
        color = np.asarray(category_to_rgb[box["category"]], dtype=float) / 255.0
        ax.add_patch(
            Rectangle(
                (box["min_x"], box["min_z"]),
                max(box["max_x"] - box["min_x"], 1e-6),
                max(box["max_z"] - box["min_z"], 1e-6),
                linewidth=2.4,
                edgecolor=color,
                facecolor=(*color, 0.10),
                zorder=3,
            )
        )

    # Draw camera entities (position marker + forward arrow + label) in the same XZ frame.
    def _cam_order_key(cam, fallback):
        name = str(cam.get("name", ""))
        tok = name.rsplit("_", 1)[-1]
        try:
            return int(tok)
        except Exception:
            return fallback
    ordered_cams = sorted(
        [c for c in cameras if c.get("position") is not None and len(c.get("position")) >= 3],
        key=lambda c: _cam_order_key(c, 0),
    )
    if len(ordered_cams) >= 2:
        traj = np.asarray([[float(c["position"][0]), float(c["position"][2])] for c in ordered_cams], dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", linestyle="--", linewidth=1.6, alpha=0.6, zorder=4)
    arrow_len = 0.12 * max_range
    for cam in ordered_cams:
        pos = cam["position"]
        cx, cz = float(pos[0]), float(pos[2])
        ax.scatter([cx], [cz], s=90, marker="^", c="black", edgecolors="white", linewidths=1.2, zorder=6)
        fwd = cam.get("forward")
        if fwd is not None and len(fwd) >= 3:
            fx, fz = float(fwd[0]), float(fwd[2])
            fn = (fx * fx + fz * fz) ** 0.5
            if fn > 1e-8:
                fx, fz = fx / fn, fz / fn
                ax.annotate(
                    "",
                    xy=(cx + fx * arrow_len, cz + fz * arrow_len),
                    xytext=(cx, cz),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
                    zorder=6,
                )
        ax.text(
            cx,
            cz,
            str(cam.get("name", "camera")),
            fontsize=9,
            color="black",
            zorder=7,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.15"),
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Z", fontsize=14)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_facecolor("white")

    buffer = BytesIO()
    plt.tight_layout()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

class SpatialMemory:
    CACHE_FORMAT_VERSION = 2
    # Previous background: muted slate-blue (92, 103, 118)
    # Current background matches Pi3's lightgray axes background.
    EGO_RENDER_BACKGROUND_RGB = (211, 211, 211)
    CONFIDENCE_THRESHOLD_STRICT = 0.08
    # CONFIDENCE_THRESHOLD_STRICT = 0.003
    # CONFIDENCE_THRESHOLD_STRICT = 0.10
    CONFIDENCE_THRESHOLD_LOOSE = 0.003
    # STD_THRESHOLD = 2.0
    STD_THRESHOLD = 3.0
    SCALE_FACTOR = 1.1  # emprical scale to adjust the metric scale of the spatial memory.

    def __init__(self,
                 rgb_images: List[np.ndarray],
                 position_3d: List[torch.Tensor],
                 confidence: Optional[np.ndarray] = None,
                 camera_trajectory: Optional[List] = None,
                 intrinsics: Optional[List] = None,
                 global_up: Optional[np.ndarray] = None,
                 align_xz_with_pca: bool = True,
                 query_map_scale: Optional[int] = None,
                 visualization_mask_outlier_only: Optional[np.ndarray] = None,
                 visualization_mask_remove_top_peak: Optional[np.ndarray] = None,
                 xz_pca_alignment_info: Optional[dict] = None):
        # import ipdb; ipdb.set_trace()
        self.rgb_images = rgb_images    # (N, 3, H, W)
        self.position_3d = position_3d  # (M, H, W, 3)
        self.confidence = confidence
        self.camera_trajectory = camera_trajectory
        self.intrinsics = intrinsics    # (3, 3). Normalized intrinsics
        self.CONFIDENCE_THRESHOLD = self.CONFIDENCE_THRESHOLD_STRICT if len(self.position_3d) > 8 else self.CONFIDENCE_THRESHOLD_LOOSE
        self.intrinsics_unnorm = self._unnorm_intrinsics(intrinsics, rgb_images[0].shape[1:]) if intrinsics is not None else None
        self.global_up = global_up
        self.query_cache = {}
        if visualization_mask_outlier_only is not None and visualization_mask_remove_top_peak is not None:
            self.visualization_mask_outlier_only = np.asarray(visualization_mask_outlier_only, dtype=bool)
            self.visualization_mask_remove_top_peak = np.asarray(visualization_mask_remove_top_peak, dtype=bool)
        else:
            (
                self.visualization_mask_outlier_only,
                self.visualization_mask_remove_top_peak,
            ) = self._build_visualization_masks(threshold_std=self.STD_THRESHOLD)
        # Keep the legacy attribute name pointing to the more aggressively filtered mask.
        self.visualization_mask = self.visualization_mask_remove_top_peak
        self.xz_pca_alignment_info = xz_pca_alignment_info
        if self.xz_pca_alignment_info is None and align_xz_with_pca:
            # self.xz_pca_alignment_info = self.align_xz_plane_with_pca()
            self.xz_pca_alignment_info = self.align_xz_plane_with_min_area_rect()
        self.query_map_scale = query_map_scale

    @staticmethod
    def _rgb_tensor_to_serializable(rgb_images: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(rgb_images, torch.Tensor):
            rgb_np = rgb_images.detach().cpu().numpy()
        else:
            rgb_np = np.asarray(rgb_images)
        if rgb_np.dtype == np.uint8:
            return rgb_np
        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        return np.round(rgb_np * 255.0).astype(np.uint8)

    @staticmethod
    def _rgb_tensor_from_serializable(rgb_images_uint8: np.ndarray) -> torch.Tensor:
        rgb_np = np.asarray(rgb_images_uint8, dtype=np.uint8)
        return torch.from_numpy(rgb_np.astype(np.float32) / 255.0)

    def _to_cache_arrays(self) -> dict:
        query_map_scale = getattr(self, "query_map_scale", None)
        xz_pca_alignment_json = (
            json.dumps(self.xz_pca_alignment_info, ensure_ascii=False)
            if self.xz_pca_alignment_info is not None
            else ""
        )
        return {
            "cache_format_version": np.asarray(self.CACHE_FORMAT_VERSION, dtype=np.int32),
            "rgb_images_uint8": self._rgb_tensor_to_serializable(self.rgb_images),
            "position_3d": np.asarray(self.position_3d, dtype=np.float32),
            "confidence": (
                np.asarray(self.confidence, dtype=np.float32)
                if self.confidence is not None
                else np.asarray([], dtype=np.float32)
            ),
            "camera_trajectory": (
                np.asarray(self.camera_trajectory, dtype=np.float32)
                if self.camera_trajectory is not None
                else np.asarray([], dtype=np.float32)
            ),
            "intrinsics": (
                np.asarray(self.intrinsics, dtype=np.float32)
                if self.intrinsics is not None
                else np.asarray([], dtype=np.float32)
            ),
            "global_up": (
                np.asarray(self.global_up, dtype=np.float32)
                if self.global_up is not None
                else np.asarray([], dtype=np.float32)
            ),
            "visualization_mask_outlier_only": np.asarray(self.visualization_mask_outlier_only, dtype=bool),
            "visualization_mask_remove_top_peak": np.asarray(self.visualization_mask_remove_top_peak, dtype=bool),
            "query_map_scale": (
                np.asarray([int(query_map_scale)], dtype=np.int32)
                if query_map_scale is not None
                else np.asarray([], dtype=np.int32)
            ),
            "xz_pca_alignment_info_json": np.asarray([xz_pca_alignment_json]),
        }

    def save(self, save_path: str, metadata: Optional[dict] = None) -> str:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._to_cache_arrays()
        np.savez_compressed(save_path, **arrays)
        if metadata is not None:
            metadata_path = save_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return str(save_path)

    @classmethod
    def load(cls, load_path: str, align_xz_with_pca: bool = True) -> "SpatialMemory":
        # import ipdb; ipdb.set_trace()
        load_path = Path(load_path)
        with np.load(load_path, allow_pickle=False) as data:
            version = int(np.asarray(data["cache_format_version"]).reshape(()))
            if version != cls.CACHE_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported spatial memory cache version {version}; "
                    f"expected {cls.CACHE_FORMAT_VERSION}."
                )
            rgb_images = cls._rgb_tensor_from_serializable(data["rgb_images_uint8"])
            position_3d = np.asarray(data["position_3d"], dtype=np.float32)
            confidence = np.asarray(data["confidence"], dtype=np.float32)
            camera_trajectory = np.asarray(data["camera_trajectory"], dtype=np.float32)
            intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
            global_up = np.asarray(data["global_up"], dtype=np.float32)
            visualization_mask_outlier_only = np.asarray(
                data["visualization_mask_outlier_only"],
                dtype=bool,
            )
            visualization_mask_remove_top_peak = np.asarray(
                data["visualization_mask_remove_top_peak"],
                dtype=bool,
            )
            query_map_scale = None
            query_map_scale_arr = np.asarray(data["query_map_scale"], dtype=np.int32)
            if query_map_scale_arr.size > 0:
                query_map_scale = int(query_map_scale_arr.reshape(-1)[0])
            query_map_scale = 4
            xz_pca_alignment_info = None
            xz_pca_alignment_json = str(np.asarray(data["xz_pca_alignment_info_json"]).reshape(-1)[0])
            if xz_pca_alignment_json:
                xz_pca_alignment_info = json.loads(xz_pca_alignment_json)

        return cls(
            rgb_images=rgb_images,
            position_3d=position_3d,
            confidence=confidence if confidence.size > 0 else None,
            camera_trajectory=camera_trajectory if camera_trajectory.size > 0 else None,
            intrinsics=intrinsics if intrinsics.size > 0 else None,
            global_up=global_up if global_up.size > 0 else None,
            align_xz_with_pca=False,
            query_map_scale=query_map_scale,
            visualization_mask_outlier_only=visualization_mask_outlier_only,
            visualization_mask_remove_top_peak=visualization_mask_remove_top_peak,
            xz_pca_alignment_info=xz_pca_alignment_info,
        )


    def __len__(self):
        return len(self.rgb_images)

    @property
    def meta_info(self):
        return {
            "num_images": len(self.rgb_images),
            "up_axis": array_to_printable_list(self.global_up) if self.global_up is not None else None,
            "coordinate_type": "right-handed", # default to right-handed coordinate system, can be updated based on the actual data
            "xz_pca_aligned": self.xz_pca_alignment_info is not None,
        }

    @property
    def rgb_images_pil(self):
        # Convert tensors to PIL images for better visualization and compatibility with some tools
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        return [to_pil(img.cpu()) for img in self.rgb_images]
    
    @property
    def memory_3d_map_size(self):
        if self.position_3d is None:
            return None
        mem_h, mem_w = self.position_3d.shape[1:3]
        return mem_h, mem_w

    def _unnorm_intrinsics(self, intrinsics_norm, img_size):
        img_h, img_w = img_size
        scale_x = img_w
        scale_y = img_h
        intrinsics_unnorm = intrinsics_norm.copy()
        intrinsics_unnorm[0, 0] *= scale_x
        intrinsics_unnorm[1, 1] *= scale_y
        intrinsics_unnorm[0, 2] *= scale_x
        intrinsics_unnorm[1, 2] *= scale_y
        return intrinsics_unnorm

    def _flatten_points_and_colors(self):
        points = self.position_3d.reshape(-1, 3)
        colors = self.rgb_images.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        return points, colors

    def _flatten_confidence(self):
        if self.confidence is None:
            return None
        return np.asarray(self.confidence, dtype=np.float32).reshape(-1)

    def _remove_outliers_mahalanobis(self, points: np.ndarray, threshold_std: float = 3.0):
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        if len(points) < 16:
            return np.ones(len(points), dtype=bool)

        cov = EmpiricalCovariance().fit(points)
        dist = cov.mahalanobis(points)
        mean_dist = np.mean(dist)
        std_dist = np.std(dist)
        if std_dist < 1e-8:
            return np.ones(len(points), dtype=bool)
        threshold = mean_dist + threshold_std * std_dist
        return dist < threshold

    def _compute_height_values(self, points: np.ndarray):
        if self.global_up is None:
            return None
        up = np.asarray(self.global_up, dtype=float)
        norm = np.linalg.norm(up)
        if norm < 1e-8:
            return None
        up = up / norm
        # Height should increase along the physical "up" direction.
        # In this codebase the common convention is y-down, e.g. global_up = [0, -1, 0],
        # so points with smaller y are physically higher. Projecting onto `up` preserves
        # that ordering: larger projected value means physically higher.
        return points @ up

    def _build_remove_top_peak_mask(self, points: np.ndarray):
        if len(points) < 512:
            return np.ones(len(points), dtype=bool)

        heights = self._compute_height_values(points)
        if heights is None:
            return np.ones(len(points), dtype=bool)

        valid_height_mask = np.isfinite(heights)
        if valid_height_mask.sum() < 512:
            return np.ones(len(points), dtype=bool)

        valid_heights = heights[valid_height_mask]
        low = np.percentile(valid_heights, 1.0)
        high = np.percentile(valid_heights, 99.5)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return np.ones(len(points), dtype=bool)

        num_bins = int(np.clip(np.sqrt(valid_heights.size), 48, 160))
        hist, bin_edges = np.histogram(valid_heights, bins=num_bins, range=(low, high))
        if hist.max() <= 0:
            return np.ones(len(points), dtype=bool)

        # Smooth the histogram slightly so the ceiling/sky peak is easier to isolate.
        smooth_kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        smooth_kernel /= smooth_kernel.sum()
        hist_smooth = np.convolve(hist.astype(float), smooth_kernel, mode="same")
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        candidate_bins = np.where(
            (bin_centers >= np.percentile(valid_heights, 65.0)) &
            (hist_smooth >= 0.08 * hist_smooth.max())
        )[0]
        if len(candidate_bins) == 0:
            return np.ones(len(points), dtype=bool)

        peak_idx = int(candidate_bins[-1])
        min_count_for_peak = max(16.0, 0.02 * hist_smooth.max())
        if hist_smooth[peak_idx] < min_count_for_peak:
            return np.ones(len(points), dtype=bool)

        valley_idx = None
        for idx in range(peak_idx - 1, -1, -1):
            if hist_smooth[idx] <= hist_smooth[idx + 1] and hist_smooth[idx] <= hist_smooth[max(idx - 1, 0)]:
                valley_idx = idx
                break

        if valley_idx is None:
            valley_idx = max(0, peak_idx - max(2, num_bins // 12))

        height_cutoff = bin_edges[valley_idx + 1]
        if not np.isfinite(height_cutoff):
            return np.ones(len(points), dtype=bool)

        keep_mask = np.ones(len(points), dtype=bool)
        remove_mask = valid_height_mask & (heights >= height_cutoff)

        # Only apply when the removed slice looks like a compact top layer instead of most of the scene.
        removed_ratio = float(remove_mask.mean())
        if removed_ratio < 0.005 or removed_ratio > 0.35:
            return keep_mask

        keep_mask[remove_mask] = False
        return keep_mask

    @staticmethod
    def _find_hist_peak(
        hist_smooth: np.ndarray,
        bin_centers: np.ndarray,
        search_mask: np.ndarray,
        min_peak_ratio: float = 0.04,
    ):
        candidate_bins = np.where(search_mask)[0]
        if len(candidate_bins) == 0 or hist_smooth.max() <= 0:
            return None

        best_idx = int(candidate_bins[np.argmax(hist_smooth[candidate_bins])])
        if hist_smooth[best_idx] < max(8.0, min_peak_ratio * hist_smooth.max()):
            return None
        return best_idx, float(bin_centers[best_idx]), float(hist_smooth[best_idx])

    def estimate_room_height_from_floor_ceiling(
        self,
        min_points: int = 512,
        lower_search_percentile: float = 35.0,
        upper_search_percentile: float = 65.0,
        min_height: float = 1.0,
        max_height: float = 8.0,
    ):
        """
        Estimate room height by detecting floor and ceiling peaks along global up.

        Height values are computed as point dot global_up, so larger values are
        physically higher. The floor is searched in the lower part of the height
        histogram, and the ceiling in the upper part.
        """
        if self.global_up is None:
            return {
                "status": "failed",
                "reason": "global_up is not available",
            }

        points, _ = self._get_visualization_points_and_colors(use_top_peak_filtered_mask=False)
        if len(points) < min_points:
            return {
                "status": "failed",
                "reason": f"not enough filtered points: {len(points)} < {min_points}",
            }

        heights = self._compute_height_values(points)
        if heights is None:
            return {
                "status": "failed",
                "reason": "could not compute height values",
            }

        valid_mask = np.isfinite(heights)
        heights = heights[valid_mask]
        if len(heights) < min_points:
            return {
                "status": "failed",
                "reason": f"not enough valid height values: {len(heights)} < {min_points}",
            }

        low = np.percentile(heights, 0.5)
        high = np.percentile(heights, 99.5)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return {
                "status": "failed",
                "reason": "invalid height range",
            }

        num_bins = int(np.clip(np.sqrt(heights.size), 64, 192))
        hist, bin_edges = np.histogram(heights, bins=num_bins, range=(low, high))
        if hist.max() <= 0:
            return {
                "status": "failed",
                "reason": "empty height histogram",
            }

        smooth_kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
        smooth_kernel /= smooth_kernel.sum()
        hist_smooth = np.convolve(hist.astype(float), smooth_kernel, mode="same")
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        lower_cut = np.percentile(heights, lower_search_percentile)
        upper_cut = np.percentile(heights, upper_search_percentile)
        floor_peak = self._find_hist_peak(
            hist_smooth,
            bin_centers,
            bin_centers <= lower_cut,
        )
        ceiling_peak = self._find_hist_peak(
            hist_smooth,
            bin_centers,
            bin_centers >= upper_cut,
        )

        if floor_peak is None or ceiling_peak is None:
            return {
                "status": "failed",
                "reason": "could not detect both floor and ceiling peaks",
                "height_range": array_to_printable_list(np.array([low, high])),
            }

        floor_idx, floor_height_value, floor_support = floor_peak
        ceiling_idx, ceiling_height_value, ceiling_support = ceiling_peak
        room_height = ceiling_height_value - floor_height_value
        if not np.isfinite(room_height) or room_height <= 0:
            return {
                "status": "failed",
                "reason": "invalid floor/ceiling ordering",
                "floor_height_value": round(float(floor_height_value), 3),
                "ceiling_height_value": round(float(ceiling_height_value), 3),
            }

        confidence = "ok"
        if room_height < min_height or room_height > max_height:
            confidence = "suspicious"

        return {
            "status": "ok",
            "room_height": round(float(room_height), 3),
            "height_unit": "same as spatial memory coordinates",
            "confidence": confidence,
            "floor_height_value": round(float(floor_height_value), 3),
            "ceiling_height_value": round(float(ceiling_height_value), 3),
            "floor_peak_bin": int(floor_idx),
            "ceiling_peak_bin": int(ceiling_idx),
            "floor_peak_support": round(float(floor_support), 1),
            "ceiling_peak_support": round(float(ceiling_support), 1),
            "height_range": array_to_printable_list(np.array([low, high])),
            "num_valid_points": int(len(heights)),
            "num_bins": int(num_bins),
        }

    def _build_visualization_masks(self, threshold_std: float = 3.0):
        points, _ = self._flatten_points_and_colors()
        valid_mask = np.isfinite(points).all(axis=1)
        confidence = self._flatten_confidence()
        if confidence is not None:
            valid_mask = valid_mask & np.isfinite(confidence) & (confidence > self.CONFIDENCE_THRESHOLD)
        if not valid_mask.any():
            empty_mask = valid_mask.reshape(self.position_3d.shape[:-1])
            return empty_mask, empty_mask

        filtered_valid_mask_outlier_only = self._remove_outliers_mahalanobis(
            points[valid_mask],
            threshold_std=threshold_std,
        )
        filtered_valid_mask_remove_top_peak = (
            filtered_valid_mask_outlier_only &
            self._build_remove_top_peak_mask(points[valid_mask])
        )

        visualization_mask_outlier_only = np.zeros(len(points), dtype=bool)
        visualization_mask_outlier_only[
            np.where(valid_mask)[0][filtered_valid_mask_outlier_only]
        ] = True

        visualization_mask_remove_top_peak = np.zeros(len(points), dtype=bool)
        visualization_mask_remove_top_peak[
            np.where(valid_mask)[0][filtered_valid_mask_remove_top_peak]
        ] = True

        return (
            visualization_mask_outlier_only.reshape(self.position_3d.shape[:-1]),
            visualization_mask_remove_top_peak.reshape(self.position_3d.shape[:-1]),
        )

    def _get_visualization_points_and_colors(self, use_top_peak_filtered_mask: bool = True):
        points, colors = self._flatten_points_and_colors()
        mask = (
            self.visualization_mask_remove_top_peak
            if use_top_peak_filtered_mask
            else self.visualization_mask_outlier_only
        )
        if mask is None:
            valid_mask = np.isfinite(points).all(axis=1)
            return points[valid_mask], colors[valid_mask]

        vis_mask = mask.reshape(-1)
        return points[vis_mask], colors[vis_mask]

    def align_xz_plane_with_pca(self):
        """
        Rotate the scene around the global up axis so that the dominant XZ-plane
        direction of the filtered point cloud aligns with +Z.

        This is intended for global-view initialization after gravity alignment.
        It uses the outlier-only visualization mask so floating points contribute
        less to the PCA estimate.
        """
        if self.global_up is None:
            return None

        points, _ = self._get_visualization_points_and_colors(use_top_peak_filtered_mask=False)
        if len(points) < 32:
            return None

        points_xz = np.asarray(points[:, [0, 2]], dtype=float)
        finite_mask = np.isfinite(points_xz).all(axis=1)
        points_xz = points_xz[finite_mask]
        if len(points_xz) < 32:
            return None

        centered_xz = points_xz - points_xz.mean(axis=0, keepdims=True)
        cov = centered_xz.T @ centered_xz / max(len(centered_xz) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_xz = eigvecs[:, int(np.argmax(eigvals))]
        principal_norm = np.linalg.norm(principal_xz)
        if not np.isfinite(principal_norm) or principal_norm < 1e-8:
            return None
        principal_xz = principal_xz / principal_norm

        if self.camera_trajectory is not None and len(self.camera_trajectory) > 0:
            camera_forward_xz = np.asarray(self.camera_trajectory[:, [0, 2], 2], dtype=float)
            valid_forward = camera_forward_xz[np.isfinite(camera_forward_xz).all(axis=1)]
            if len(valid_forward) > 0:
                mean_forward = valid_forward.mean(axis=0)
                if np.linalg.norm(mean_forward) > 1e-8 and float(np.dot(principal_xz, mean_forward)) < 0:
                    principal_xz = -principal_xz
        elif principal_xz[1] < 0:
            principal_xz = -principal_xz

        forward = np.array([principal_xz[0], 0.0, principal_xz[1]], dtype=float)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        self.transform_spatial_memory(
            origin=[0.0, 0.0, 0.0],
            forward=forward.tolist(),
            up=self.global_up.tolist() if isinstance(self.global_up, np.ndarray) else self.global_up,
        )
        return {
            "principal_direction_xz": array_to_printable_list(principal_xz),
            "applied_forward": array_to_printable_list(forward),
        }

    def align_xz_plane_with_min_area_rect(self):
        """
        Rotate the scene around the global up axis so that the dominant XZ-plane
        direction of the filtered point cloud aligns with +Z.

        This is an alternative to PCA-based alignment that may be more robust to
        certain scene geometries. It uses the same outlier-only visualization mask.
        """
        if self.global_up is None:
            return None

        points, _ = self._get_visualization_points_and_colors(use_top_peak_filtered_mask=False)
        if len(points) < 32:
            return None

        xz_projections = np.asarray(points[:, [0, 2]], dtype=np.float32)
        finite_mask = np.isfinite(xz_projections).all(axis=1)
        xz_projections = xz_projections[finite_mask]
        if len(xz_projections) < 32:
            return None

        rect = cv2.minAreaRect(xz_projections)
        box = cv2.boxPoints(rect)
        if box is None or len(box) != 4:
            return None

        edge_vectors = []
        for idx in range(4):
            edge = box[(idx + 1) % 4] - box[idx]
            edge_len = float(np.linalg.norm(edge))
            if np.isfinite(edge_len) and edge_len > 1e-6:
                edge_vectors.append((edge_len, edge))
        if not edge_vectors:
            return None

        _, principal_xz = max(edge_vectors, key=lambda item: item[0])
        principal_xz = np.asarray(principal_xz, dtype=float)
        principal_norm = np.linalg.norm(principal_xz)
        if not np.isfinite(principal_norm) or principal_norm < 1e-8:
            return None
        principal_xz = principal_xz / principal_norm

        if self.camera_trajectory is not None and len(self.camera_trajectory) > 0:
            camera_forward_xz = np.asarray(self.camera_trajectory[:, [0, 2], 2], dtype=float)
            valid_forward = camera_forward_xz[np.isfinite(camera_forward_xz).all(axis=1)]
            if len(valid_forward) > 0:
                mean_forward = valid_forward.mean(axis=0)
                if np.linalg.norm(mean_forward) > 1e-8 and float(np.dot(principal_xz, mean_forward)) < 0:
                    principal_xz = -principal_xz
        elif principal_xz[1] < 0:
            principal_xz = -principal_xz

        forward = np.array([principal_xz[0], 0.0, principal_xz[1]], dtype=float)
        forward_norm = np.linalg.norm(forward)
        if not np.isfinite(forward_norm) or forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        self.transform_spatial_memory(
            origin=[0.0, 0.0, 0.0],
            forward=forward.tolist(),
            up=self.global_up.tolist() if isinstance(self.global_up, np.ndarray) else self.global_up,
        )
        return {
            "principal_direction_alignment_method": "min_area_rect",
            "principal_direction_xz": array_to_printable_list(principal_xz),
            "applied_forward": array_to_printable_list(forward),
            "rect_center_xz": array_to_printable_list(np.asarray(rect[0], dtype=float)),
            "rect_size_xz": array_to_printable_list(np.asarray(rect[1], dtype=float)),
            "rect_angle_degrees": round(float(rect[2]), 2),
        }

    @staticmethod
    def _auto_crop_rendered_image(
        rendered_img,
        background_rgb: Tuple[int, int, int],
        min_valid_ratio_for_crop: float = 0.28,
        crop_padding_ratio: float = 0.08,
        color_tolerance: int = 10,
    ):
        img_np = np.asarray(rendered_img)
        if img_np.ndim != 3 or img_np.shape[2] < 3:
            return rendered_img

        bg = np.asarray(background_rgb, dtype=np.int16)
        rgb = img_np[..., :3].astype(np.int16)
        is_background = np.all(np.abs(rgb - bg[None, None, :]) <= color_tolerance, axis=-1)
        valid_mask = ~is_background
        valid_ratio = float(valid_mask.mean())
        if valid_ratio >= min_valid_ratio_for_crop or valid_mask.sum() == 0:
            return rendered_img

        height, width = valid_mask.shape
        row_counts = valid_mask.sum(axis=1)
        col_counts = valid_mask.sum(axis=0)
        min_row_support = max(6, int(0.015 * width))
        min_col_support = max(6, int(0.015 * height))
        kept_rows = np.where(row_counts >= min_row_support)[0]
        kept_cols = np.where(col_counts >= min_col_support)[0]

        if len(kept_rows) > 0 and len(kept_cols) > 0:
            y0, y1 = kept_rows.min(), kept_rows.max() + 1
            x0, x1 = kept_cols.min(), kept_cols.max() + 1
        else:
            ys, xs = np.where(valid_mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

        ys, xs = np.where(valid_mask)
        if len(xs) > 0 and len(ys) > 0:
            quantile = 5.0 if valid_ratio < 0.22 else 2.0
            q_x0 = int(np.floor(np.percentile(xs, quantile)))
            q_x1 = int(np.ceil(np.percentile(xs, 100.0 - quantile))) + 1
            q_y0 = int(np.floor(np.percentile(ys, quantile)))
            q_y1 = int(np.ceil(np.percentile(ys, 100.0 - quantile))) + 1
            if (q_x1 - q_x0) * (q_y1 - q_y0) < (x1 - x0) * (y1 - y0):
                x0, x1 = q_x0, q_x1
                y0, y1 = q_y0, q_y1

        pad_y = max(4, int((y1 - y0) * crop_padding_ratio))
        pad_x = max(4, int((x1 - x0) * crop_padding_ratio))
        y0 = max(0, y0 - pad_y)
        y1 = min(height, y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)

        cropped_area_ratio = ((y1 - y0) * (x1 - x0)) / float(height * width)
        if cropped_area_ratio > 0.9:
            return rendered_img

        cropped_valid_ratio = float(valid_mask[y0:y1, x0:x1].mean())
        if cropped_valid_ratio <= valid_ratio * 1.35:
            return rendered_img

        cropped = rendered_img.crop((x0, y0, x1, y1))
        return cropped.resize(rendered_img.size, resample=Image.BICUBIC)

    def get_camera_pose_at_frame(self, frame_idx: List[int]):
        """
        Return the camera pose (extrinsics) for the specified frame indices. 
        For model's better performance, the camera pose is formatted as a dictionary containing:
        [{
            'camera_index': idx,  # The index of the camera/frame
            'position': [x, y, z],  # Camera position in world coordinates
            'up': [x, y, z],        # Camera up vector
            'forward': [x, y, z],   # Camera forward vector
            'right': [x, y, z],     # Camera right vector (optional, can be computed from forward and up)
        },
        ...] 
        """
        if self.camera_trajectory is None:
            raise ValueError("Camera trajectory is not available in the spatial memory.")
        camera_poses = []
        for idx in frame_idx:
            if idx >= len(self.camera_trajectory):
                raise IndexError(f"Frame index {idx} is out of bounds for camera trajectory with length {len(self.camera_trajectory)}.")
            camera_pose = self.camera_trajectory[idx] # np.array of shape (4, 4)
            # keep .2 decimals for better readability
            position = array_to_printable_list(camera_pose[:3, 3])
            forward = array_to_printable_list(camera_pose[:3, 2]) # camera forward is the 3rd column of the extrinsic matrix
            right = array_to_printable_list(camera_pose[:3, 0])   # camera right is the 1st column of the extrinsic matrix
            if self.global_up is not None:
                up = array_to_printable_list(self.global_up)
            else:
                up = array_to_printable_list(-camera_pose[:3, 1]) # camera down is the 2nd column of the extrinsic matrix
            camera_poses.append({
                'camera_index': idx+1,  # 1-based index for better readability
                'position': position,
                'up': up,
                'forward': forward,
                'right': right,  # Camera right vector
            })
        return camera_poses

    def set_viewpoint(self, origin: List[float], forward: List[float], up: List[float] = None):
        self.transform_spatial_memory(origin=origin, forward=forward, up=up)
        return {
            "status": "ok",
            "message": "Spatial memory transformed successfully.",
            "forward": [0, 0, 1],
            "right": [1, 0, 0],
        }

    def query_camera_pose(self, frame_indices: List[int]):
        valid_frame_indices = []
        errors = []
        for frame_index in frame_indices:
            if frame_index < 1 or frame_index > len(self):
                errors.append(
                    f"Error: Frame index {frame_index} is out of range. "
                    f"Please make sure the index is between 1 and {len(self)}."
                )
            else:
                valid_frame_indices.append(frame_index - 1)

        if not valid_frame_indices:
            return [], errors
        return self.get_camera_pose_at_frame(valid_frame_indices), errors

    def get_intrinsics_at_frame(self, frame_idx: List[int]):
        if self.intrinsics is None:
            raise ValueError("Intrinsics are not available in the spatial memory.")
        return [self.intrinsics[idx] for idx in frame_idx]

    
    def transform_spatial_memory(self, origin: List[float], forward: List[float], up: List[float]=None):
        """
        Transform the spatial memory to a new coordinate system defined by the given origin, forward, and up vectors.
        This can be useful for aligning the spatial memory with a specific camera view or for normalizing the coordinate system.
        """
        # This is a placeholder implementation. The actual transformation would depend on the specific format of the spatial memory and the desired output format.
        # For example, you might want to apply a rotation and translation to the 3D positions based on the new origin, forward, and up vectors.
        if up is None:
            up = self.global_up if self.global_up is not None else [0, -1, 0]

        origin = np.array(origin, dtype=float)
        f = np.array(forward, dtype=float)
        # BUG: project forward to the xz plane to ensure it's perpendicular to up (which is important for right-handed coordinate system)
        # right = forward × up（右手系）
        up = np.array(up, dtype=float)  # 注意：up 在 y-down 体系里指向 y 负方向
        r = np.cross(f, up)
        r = r / np.linalg.norm(r)
        # orthogonalize forward to right and up
        f = np.cross(up, r)

        f = f / np.linalg.norm(f)
        up = up / np.linalg.norm(up)
        d = -up

        # R: 新坐标系，行向量是世界坐标下的基
        R = np.stack([r, d, f], axis=0)  # (3, 3)

        # SE3: 先平移后旋转
        pts = self.position_3d.reshape(-1, 3)
        self.position_3d = ((R @ (pts - origin).T).T).reshape(self.position_3d.shape)

        # 变换相机 c2w
        # c2w 把相机坐标变到世界坐标
        # 新世界 = R @ (旧世界 - origin)
        # 所以 c2w_new 的旋转部分: R @ c2w_old[:3, :3]
        #       平移部分: R @ (c2w_old[:3, 3] - origin)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ origin  # 先平移再旋转 = R(p - o) = Rp - Ro

        self.camera_trajectory = T @ self.camera_trajectory  # (M, 4, 4) 广播

        return R, origin
    
    def move_camera(self, direction:str, distance:float = 0.5):
        """
        Move the camera in the specified direction (e.g., 'forward', 'backward', 'left', 'right', 'up', 'down') by a certain distance. This can be useful for navigating through the spatial memory or for simulating camera movements.
        """
        next_origin = [0.0, 0.0, 0.0]
        if direction == 'forward':
            next_origin[2] += distance
        elif direction == 'backward':
            next_origin[2] -= distance
        elif direction == 'left':
            next_origin[0] -= distance
        elif direction == 'right':
            next_origin[0] += distance
        elif direction == 'up':
            next_origin[1] -= distance  # 注意 y-down 体系里 up 是 y 负方向
        elif direction == 'down':
            next_origin[1] += distance  # 注意 y-down 体系里 down 是 y 正方向
        else:
            raise ValueError(f"Invalid direction '{direction}'. Valid directions are: 'forward', 'backward', 'left', 'right', 'up', 'down'.")
        self.transform_spatial_memory(origin=next_origin, forward=[0, 0, 1], up=[0, -1, 0])  # 默认朝向 +z 方向，up 朝向 -y 方向
        return {
            "status": "ok",
            "message": f"Camera moved {direction} successfully.",
        }

    def rotate_camera(self, direction:str, angle_deg:Optional[float] = 90.0):
        """
        Rotate the camera to a new direction (e.g., 'left', 'right', 'up', 'down') by a certain angle (in degrees). This can be useful for changing the camera's orientation in the spatial memory or for simulating camera rotations.
        """
        if direction == 'left':
            # clockwise
            angle_rad = np.pi / 2 if angle_deg is None else np.deg2rad(angle_deg)
            R = np.array([[np.cos(angle_rad), 0, -np.sin(angle_rad)],
                          [0, 1, 0],
                          [np.sin(angle_rad), 0, np.cos(angle_rad)]])
        elif direction == 'right':
            # anti-clockwise
            angle_rad = np.pi / 2 if angle_deg is None else np.deg2rad(angle_deg)
            R = np.array([[np.cos(-angle_rad), 0, -np.sin(-angle_rad)],
                          [0, 1, 0],
                          [np.sin(-angle_rad), 0, np.cos(-angle_rad)]])
        elif direction == 'up':
            angle_rad = np.pi / 4 if angle_deg is None else np.deg2rad(angle_deg)
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle_rad), -np.sin(angle_rad)],
                          [0, np.sin(angle_rad), np.cos(angle_rad)]])
        elif direction == 'down':
            angle_rad = np.pi / 4 if angle_deg is None else np.deg2rad(angle_deg)
            R = np.array([[1, 0, 0],
                          [0, np.cos(-angle_rad), -np.sin(-angle_rad)],
                          [0, np.sin(-angle_rad), np.cos(-angle_rad)]])
        elif direction == 'back':
            R = np.array([[-1, 0, 0],
                          [0, 1, 0],
                          [0, 0, -1]])
        else:
            raise ValueError(f"Invalid direction '{direction}'. Valid directions are: 'left', 'right', 'up', 'down', 'back'.")
        new_forward = np.array([0, 0, 1], dtype=float)  # 默认朝向 +z 方向
        new_forward = new_forward @ R.T  # 旋转 forward 向量
        new_up = np.array([0, -1, 0], dtype=float)  # 默认 y-down 体系里 up 朝向 -y 方向
        new_up = new_up @ R.T  # 旋转 up 向量
        self.transform_spatial_memory(origin=[0, 0, 0], forward=new_forward, up=new_up)  # 默认以世界坐标系原点为观察点
        return {
            "status": "ok",
            "message": f"Camera rotated {direction} successfully.",
        }

    
    def render_perspective_view(
        self,
        img_size: Tuple[int, int]=(960, 540),
        FOV: Optional[float] = None,
        use_top_peak_filtered_mask: bool = False,
        auto_crop: bool = True,
    ) -> torch.Tensor:
        """
        Render a perspective view (e.g., RGB image) from the spatial memory observed from current origin and forward direction. This can be useful for visualizing the spatial memory or for providing a specific view to the model.
        """
        forward = np.array([0, 0, 1], dtype=float)  # 默认朝向 +z 方向
        up = np.array([0, -1, 0], dtype=float)       # 默认 y-down 体系里 up 朝向 -y 方向
        origin = np.array([0, 0, 0], dtype=float)   # 默认以世界坐标系原点为观察点

        points, colors = self._get_visualization_points_and_colors(
            use_top_peak_filtered_mask=use_top_peak_filtered_mask
        )
        rendered_img = render_perspective_view(
            points,
            colors,
            origin,
            forward,
            up,
            width=img_size[0],
            height=img_size[1],
            background_rgba=tuple(channel / 255.0 for channel in self.EGO_RENDER_BACKGROUND_RGB) + (1.0,),
            intrinsics=self.intrinsics_unnorm,
            FOV=FOV,
        )
        auto_crop = False
        if auto_crop:
            rendered_img = self._auto_crop_rendered_image(
                rendered_img,
                background_rgb=self.EGO_RENDER_BACKGROUND_RGB,
            )
        return rendered_img


    def render_bev_view(self, img_size: Tuple[int, int]=(960, 960), ego_marker_size: int = 140, height: int = 10) -> torch.Tensor:
        """
        Render a bird's-eye view (BEV) image from the spatial memory. This can be useful for visualizing the spatial layout of the scene or for providing a top-down view to the model.
        """
        # lift the camera up to a high position and look down
        forward = np.array([0, 1, 0], dtype=float)  # +y, looking down
        up = np.array([0, 0, 1], dtype=float)       # +z, forward
        origin = np.array([0, -height, 0], dtype=float)  # lift the camera up to y=10 and look down

        points, colors = self._get_visualization_points_and_colors(
            use_top_peak_filtered_mask=True
        )
        rendered_img = render_perspective_view(points, colors, origin, forward, up, width=img_size[0], height=img_size[1], FOV=100.0)
        rendered_img = draw_bev_center_marker(rendered_img, marker_size=ego_marker_size)
        return rendered_img

    def _estimate_bev_height_list(
        self,
        num_views: int,
        img_size: Tuple[int, int] = (960, 960),
        min_height: float = 2.0,
        margin_ratio: float = 1.12,
    ) -> List[float]:
        """
        Estimate BEV camera heights so that the largest view can roughly cover the
        whole visible scene under the current active coordinate frame.
        """
        points, _ = self._get_visualization_points_and_colors()
        if len(points) == 0 or num_views <= 0:
            return [float(min_height)] * max(1, num_views)
        if not isinstance(points, np.ndarray):
            points = np.asarray(points, dtype=float)

        # The BEV camera is centered at the current origin (0, 0, 0) and looks down.
        # To make the widest view cover the whole scene, estimate how high the camera
        # should be so that the x/z extents fit the current perspective frustum.
        max_abs_x = float(np.max(np.abs(points[:, 0])))
        max_abs_z = float(np.max(np.abs(points[:, 2])))

        width, height_px = img_size
        focal = 300.0  # matches render_perspective_view intrinsic
        half_fov_x = np.arctan((width / 2.0) / focal)
        half_fov_y = np.arctan((height_px / 2.0) / focal)

        fit_height_x = max_abs_x / max(np.tan(half_fov_x), 1e-6)
        fit_height_z = max_abs_z / max(np.tan(half_fov_y), 1e-6)
        max_height = max(min_height, margin_ratio * max(fit_height_x, fit_height_z))

        if num_views == 1:
            return [float(max_height)]

        min_adaptive_height = max(min_height, 0.38 * max_height)
        if min_adaptive_height >= max_height:
            return [float(max_height)] * num_views

        height_list = np.geomspace(min_adaptive_height, max_height, num=num_views)
        return [float(h) for h in height_list]

    def save_render_bev_view(
        self,
        save_paths: List[str],
        img_size: Tuple[int, int] = (480, 480),
        ego_marker_size: int = 140,
    ) -> List[str]:
        save_paths = [Path(save_path) for save_path in save_paths]
        height_list = self._estimate_bev_height_list(
            len(save_paths),
            img_size=img_size,
            min_height=1.0,
        )
        for save_path, height in zip(save_paths, height_list):
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.render_bev_view(img_size=img_size, ego_marker_size=ego_marker_size, height=height).save(save_path)
        return [str(save_path) for save_path in save_paths]

    def save_render_perspective_view(self, save_path: str, img_size: Tuple[int, int] = (960, 540), FOV: Optional[float] = None) -> str:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.render_perspective_view(img_size=img_size, FOV=FOV).save(save_path)
        return str(save_path)

    def render_selected_fg_bev(
        self,
        instances,
        save_path: str,
        img_size: Tuple[int, int] = (960, 960),
        ego_marker_size: int = 140,
        max_points: int = 80000,
        bbox_mode: bool = True,
        cameras: Optional[List[dict]] = None,
    ) -> str | None:
        """Render only the selected foreground point clouds in a perspective BEV view.

        `cameras` is an optional list of {"name", "position": [x,y,z], "forward": [x,y,z]}
        entities to overlay as camera markers + orientation arrows (bbox_mode only).
        """
        cameras = cameras or []
        if not instances:
            return None

        point_chunks = []
        color_chunks = []
        category_to_rgb = {}

        def normalize_category(instance: dict, fallback_idx: int) -> str:
            category = instance.get("category")
            if isinstance(category, str) and category:
                return category
            name = instance.get("name")
            if isinstance(name, str) and "_" in name:
                return name.rsplit("_", 1)[0]
            return f"instance_{fallback_idx + 1}"

        normalized_categories = [
            normalize_category(instance, idx)
            for idx, instance in enumerate(instances)
        ]
        unique_categories = []
        for category in normalized_categories:
            if category not in unique_categories:
                unique_categories.append(category)

        instance_boxes = []

        for idx, instance in enumerate(instances):
            points = instance.get("points_3d")
            if points is None:
                continue
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                continue
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            if len(points) == 0:
                continue

            category = normalized_categories[idx]
            if category not in category_to_rgb:
                cat_index = unique_categories.index(category)
                category_to_rgb[category] = _hex_to_rgb_uint8(
                    category_base_color(category, cat_index, max(1, len(unique_categories)))
                )

            points_xz = points[:, [0, 2]]
            center_xz = np.median(points_xz, axis=0)
            radial_dist = np.linalg.norm(points_xz - center_xz[None, :], axis=1)
            dist_thresh = np.percentile(radial_dist, 95.0) if len(radial_dist) > 1 else float(radial_dist[0])
            bbox_points = points[radial_dist <= dist_thresh + 1e-8]
            if len(bbox_points) == 0:
                bbox_points = points

            instance_boxes.append(
                {
                    "category": category,
                    "min_x": float(np.min(bbox_points[:, 0])),
                    "max_x": float(np.max(bbox_points[:, 0])),
                    "min_z": float(np.min(bbox_points[:, 2])),
                    "max_z": float(np.max(bbox_points[:, 2])),
                }
            )

            point_chunks.append(points)
            base_color = np.asarray(category_to_rgb[category], dtype=float) / 255.0
            color_chunks.append(np.repeat(base_color[None, :], len(points), axis=0))

        if not point_chunks:
            return None

        if bbox_mode:
            rendered_img = render_xz_bbox_bev(
                instance_boxes=instance_boxes,
                category_to_rgb=category_to_rgb,
                img_size=img_size,
                cameras=cameras,
            )
        else:
            points = np.concatenate(point_chunks, axis=0)
            colors = np.concatenate(color_chunks, axis=0)
            if len(points) > max_points:
                rng = np.random.default_rng(0)
                sample_idx = rng.choice(len(points), size=max_points, replace=False)
                points = points[sample_idx]
                colors = colors[sample_idx]

            max_abs_x = float(np.max(np.abs(points[:, 0])))
            max_abs_z = float(np.max(np.abs(points[:, 2])))
            width, height_px = img_size
            focal = 300.0
            half_fov_x = np.arctan((width / 2.0) / focal)
            half_fov_y = np.arctan((height_px / 2.0) / focal)
            fit_height_x = max_abs_x / max(np.tan(half_fov_x), 1e-6)
            fit_height_z = max_abs_z / max(np.tan(half_fov_y), 1e-6)
            height = max(1.0, 1.12 * max(fit_height_x, fit_height_z))

            forward = np.array([0, 1, 0], dtype=float)
            up = np.array([0, 0, 1], dtype=float)
            origin = np.array([0, -height, 0], dtype=float)
            rendered_img = render_perspective_view(
                points,
                colors,
                origin,
                forward,
                up,
                width=img_size[0],
                height=img_size[1],
                FOV=100.0,
                background_rgba=(1.0, 1.0, 1.0, 1.0),
            )
        rendered_img = draw_bev_center_marker(rendered_img, marker_size=ego_marker_size)
        rendered_img = draw_category_legend(rendered_img, category_to_rgb)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_img.save(save_path)
        return str(save_path)

    def render_semantic_bev(self, entities, save_path: str, title: str = "Semantic BEV"):
        if self.global_up is None or not np.array_equal(self.global_up, np.array([0, -1, 0])):
            raise AssertionError(
                "Currently we only support global up vector as [0, -1, 0], "
                "please build the spatial memory with this global up vector or use other tools "
                "to rotate the camera poses to make the global up vector as [0, -1, 0]."
            )

        entity_bev_list = []
        normalized_entities = []
        for item in entities:
            pos = np.array(item["position"], dtype=float)
            pos_2d = np.array([pos[0], pos[2]])
            orientation = item.get("orientation")
            entity_bev = {
                "name": item["name"],
                "type": item.get("type", "entity"),
                "position": array_to_printable_list(pos_2d),
            }
            normalized_entity = {
                "name": item["name"],
                "type": item.get("type", "entity"),
                "position": array_to_printable_list(pos),
            }
            if orientation is not None:
                orientation = np.array(orientation, dtype=float)
                orientation_2d = np.array([orientation[0], orientation[2]])
                entity_bev["orientation"] = array_to_printable_list(orientation_2d)
                normalized_entity["orientation"] = array_to_printable_list(orientation)

            entity_bev_list.append(entity_bev)
            normalized_entities.append(normalized_entity)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        render_semantic_bev_entities(
            normalized_entities,
            global_up=self.global_up,
            output_png=str(save_path),
            title=title,
            ego_pose={
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 1.0],
            },
        )

        overlay_save_path = save_path.with_name(f"{save_path.stem}_rgb_overlay{save_path.suffix}")
        bg_points, bg_colors = self._get_visualization_points_and_colors(
            use_top_peak_filtered_mask=True
        )
        if len(bg_points) > 0:
            if not isinstance(bg_points, np.ndarray):
                bg_points = np.asarray(bg_points)
            if not isinstance(bg_colors, np.ndarray):
                bg_colors = np.asarray(bg_colors)
            # max_bg_points = 120000
            max_bg_points = 200000
            if len(bg_points) > max_bg_points:
                rng = np.random.default_rng(0)
                sample_idx = rng.choice(len(bg_points), size=max_bg_points, replace=False)
                bg_points = bg_points[sample_idx]
                bg_colors = bg_colors[sample_idx]

            render_semantic_bev_entities(
                normalized_entities,
                global_up=self.global_up,
                output_png=str(overlay_save_path),
                title=f"{title} (RGB Overlay)",
                ego_pose={
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 1.0],
                },
                background_points=bg_points,
                background_colors=bg_colors,
                # background_point_size=2.0,
                background_point_size=3.0,
                background_alpha=0.95,
            )

        return {
            "save_path": str(save_path),
            "overlay_save_path": str(overlay_save_path) if len(bg_points) > 0 else None,
            "entity_bev_list": entity_bev_list,
            "bev_meta_info": {
                "forward_direction": "positive z on the plot",
                "right_direction": "positive x on the plot",
                "top_view_plane": "world x-z plane",
            },
        }


    def visualize_3d(
        self,
        save_path: str,
        max_points: int = 180000,
        point_size: float = 1.5,
        show_camera_trajectory: bool = True,
    ):
        """
        Save a cleaner interactive 3D visualization for the spatial memory.

        Prefer a ReRun `.rrd` recording when `rerun` is available. Fall back to the
        previous Plotly HTML export in environments where `rerun` is missing.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        points, colors = self._get_visualization_points_and_colors()
        if len(points) == 0:
            raise ValueError("No valid 3D points are available for visualization.")

        if not isinstance(points, np.ndarray):
            points = np.asarray(points)
        if colors is not None and not isinstance(colors, np.ndarray):
            colors = np.asarray(colors)

        # SpatialLM-style visualization works better with a capped number of points
        # than with a large dense cloud. Use deterministic random sampling for stability.
        if max_points is not None and len(points) > max_points:
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(len(points), size=max_points, replace=False)
            points = points[sample_idx]
            if colors is not None:
                colors = colors[sample_idx]

        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        scene_extent = float(np.linalg.norm(xyz_max - xyz_min))
        if scene_extent < 1e-6:
            scene_extent = 1.0
        scene_center = points.mean(axis=0)

        camera_trajectory = self.camera_trajectory
        if camera_trajectory is not None and not isinstance(camera_trajectory, np.ndarray):
            camera_trajectory = np.asarray(camera_trajectory)

        try:
            import rerun as rr
            import rerun.blueprint as rrb

            colors_uint8 = None
            if colors is not None:
                colors = np.clip(colors, 0.0, 1.0)
                colors_uint8 = (colors * 255).astype(np.uint8)

            rrd_path = save_path if save_path.suffix == ".rrd" else save_path.with_suffix(".rrd")
            blueprint = rrb.Blueprint(
                rrb.Spatial3DView(
                    name="3D",
                    origin="world",
                    background=[255, 255, 255],
                ),
                collapse_panels=True,
            )
            rr.init(
                "spatial_memory_visualization",
                spawn=False,
                default_blueprint=blueprint,
            )
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

            rr.log(
                "world/points",
                rr.Points3D(
                    positions=points,
                    colors=colors_uint8,
                    radii=np.full(len(points), point_size * 0.0015 * scene_extent, dtype=np.float32),
                ),
                static=True,
            )

            if self.global_up is not None:
                up = np.asarray(self.global_up, dtype=float)
                up_norm = np.linalg.norm(up)
                if up_norm > 1e-8:
                    up = up / up_norm
                    rr.log(
                        "world/global_up",
                        rr.Arrows3D(
                            origins=[scene_center],
                            vectors=[up * (0.18 * scene_extent)],
                            colors=[[255, 0, 0]],
                            labels=["global_up"],
                        ),
                        static=True,
                    )

            if show_camera_trajectory and camera_trajectory is not None and len(camera_trajectory) > 0:
                camera_positions = camera_trajectory[:, :3, 3]
                rr.log(
                    "world/camera_trajectory/path",
                    rr.LineStrips3D(
                        [camera_positions],
                        colors=[[0, 0, 0]],
                        radii=[0.0025 * scene_extent],
                        labels=["camera_trajectory"],
                    ),
                    static=True,
                )
                rr.log(
                    "world/camera_trajectory/points",
                    rr.Points3D(
                        positions=camera_positions,
                        colors=np.tile(np.array([[40, 40, 40]], dtype=np.uint8), (len(camera_positions), 1)),
                        radii=np.full(len(camera_positions), 0.004 * scene_extent, dtype=np.float32),
                    ),
                    static=True,
                )
                rr.log(
                    "world/camera_trajectory/start",
                    rr.Points3D(
                        positions=[camera_positions[0]],
                        colors=[[0, 180, 0]],
                        radii=[0.009 * scene_extent],
                        labels=["START"],
                    ),
                    static=True,
                )
                rr.log(
                    "world/camera_trajectory/end",
                    rr.Points3D(
                        positions=[camera_positions[-1]],
                        colors=[[220, 30, 30]],
                        radii=[0.009 * scene_extent],
                        labels=["END"],
                    ),
                    static=True,
                )

                stride = max(1, len(camera_trajectory) // 24)
                forward_origins = []
                forward_vectors = []
                for idx in range(0, len(camera_trajectory), stride):
                    pose = camera_trajectory[idx]
                    forward = pose[:3, 2]
                    forward_norm = np.linalg.norm(forward)
                    if forward_norm < 1e-8:
                        continue
                    forward_origins.append(pose[:3, 3])
                    forward_vectors.append(forward / forward_norm * (0.06 * scene_extent))

                if forward_origins:
                    rr.log(
                        "world/camera_trajectory/forward",
                        rr.Arrows3D(
                            origins=np.asarray(forward_origins),
                            vectors=np.asarray(forward_vectors),
                            colors=np.tile(np.array([[65, 105, 225]], dtype=np.uint8), (len(forward_origins), 1)),
                        ),
                        static=True,
                    )

            rr.save(rrd_path)
            return str(rrd_path)

        except ImportError:
            import plotly.graph_objects as go

            if colors is not None:
                colors = np.clip(colors, 0.0, 1.0)
                colors_uint8 = (colors * 255).astype(np.uint8)
                marker_color = [f"rgb({r},{g},{b})" for r, g, b in colors_uint8]
            else:
                marker_color = "rgba(80, 80, 80, 0.9)"

            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(size=point_size, color=marker_color, opacity=0.9),
                    name="Point Cloud",
                )
            )

            if self.global_up is not None:
                up = np.asarray(self.global_up, dtype=float)
                up_norm = np.linalg.norm(up)
                if up_norm > 1e-8:
                    up = up / up_norm
                    up_end = scene_center + up * (0.18 * scene_extent)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[scene_center[0], up_end[0]],
                            y=[scene_center[1], up_end[1]],
                            z=[scene_center[2], up_end[2]],
                            mode="lines+markers+text",
                            line=dict(color="red", width=8),
                            marker=dict(size=[2, 5], color="red"),
                            text=["", "global_up"],
                            textposition="top center",
                            name="Global Up",
                        )
                    )

            if show_camera_trajectory and camera_trajectory is not None and len(camera_trajectory) > 0:
                camera_positions = camera_trajectory[:, :3, 3]
                fig.add_trace(
                    go.Scatter3d(
                        x=camera_positions[:, 0],
                        y=camera_positions[:, 1],
                        z=camera_positions[:, 2],
                        mode="lines+markers",
                        line=dict(color="black", width=5),
                        marker=dict(size=3, color="black"),
                        name="Camera Trajectory",
                    )
                )

            fig.update_layout(
                title="Spatial Memory Visualization",
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(aspectmode="data"),
            )
            fig.write_html(str(save_path), include_plotlyjs="cdn")
            return str(save_path)
