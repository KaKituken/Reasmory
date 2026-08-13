from typing import Dict, Optional, List
from pathlib import Path
import colorsys
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
import numpy as np
import open3d as o3d
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
from PIL import Image
import copy

# -----------------------
# Color helpers
# -----------------------
def hsv_to_rgb_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def perturb_color(base_hex: str, k: int) -> str:
    """
    Slightly perturb a hex color in HSV space to differentiate instances.
    """
    base_hex = base_hex.lstrip("#")
    r = int(base_hex[0:2], 16) / 255.0
    g = int(base_hex[2:4], 16) / 255.0
    b = int(base_hex[4:6], 16) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # deterministic tiny perturbation by instance index
    # keep within same hue band
    h = (h + (k * 0.07)) % 1.0
    s = min(1.0, max(0.45, s - 0.05 * (k % 3)))
    v = min(1.0, max(0.60, v - 0.04 * ((k + 1) % 4)))
    return hsv_to_rgb_hex(h, s, v)

def category_base_color(cat: str, cat_index: int, total: int) -> str:
    """
    Deterministic base color per category.
    """
    # evenly spaced hues
    h = (cat_index / max(1, total)) % 1.0
    return hsv_to_rgb_hex(h, 0.75, 0.95)

def build_merged_node_set(merged_instances):
    node_to_group = {}
    for gid, inst in enumerate(merged_instances):
        for node in inst.get("nodes", []):
            node_to_group[tuple(node)] = gid
    return node_to_group


def get_semantic_bev_style(unique_types, marker_size: int = 220):
    cmap = plt.get_cmap("tab10")
    color_by_type = {
        entity_type: cmap(i % 10)
        for i, entity_type in enumerate(sorted(set(unique_types)))
    }
    return {
        "color_by_type": color_by_type,
        "marker_size": marker_size,
        "marker_edgecolor": "black",
        "marker_linewidth": 1.5,
        "arrow_color": "red",
        "arrow_linewidth": 2,
        "label_fontsize": 12,
        "label_fontweight": "bold",
        "label_bbox": dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.3",
        ),
        "legend_markersize": 10,
    }

def export_category_pos_to_html(
    category_pos: Dict,
    output_html: str,
    max_points_per_instance: int = 8000,
    marker_size: int = 2,
    show_legend: bool = True,
    title: str = "3D Point Cloud Viewer (category_pos)",
):
    categories = sorted(list(category_pos.keys()))
    num_cats = len(categories)

    fig = go.Figure()

    pre_merge_trace_ids = []
    post_merge_trace_ids = []

    total_points = 0

    for ci, cat in enumerate(categories):
        if cat not in category_pos:
            continue
        if "pos_3d" not in category_pos[cat]:
            continue

        pos_3d = category_pos[cat]["pos_3d"]
        image_idxs = category_pos[cat]["image_idxs"]
        agreement_scores = category_pos[cat].get("agreement_scores", None)

        merged_instances = category_pos[cat].get("merged_instances", None)
        merged_node_map = (
            build_merged_node_set(merged_instances)
            if merged_instances is not None
            else {}
        )

        base_color = category_base_color(cat, ci, num_cats)

        # -------------------------------------------------
        # 1️⃣ PRE-MERGE VISUALIZATION
        # -------------------------------------------------
        for frame_i, inst_list, agreement_list in zip(image_idxs, pos_3d, agreement_scores):
            if inst_list is None:
                continue
            for inst_j, (pts_np, agreement) in enumerate(zip(inst_list, agreement_list)):
                if pts_np is None:
                    continue

                pts_np = np.asarray(pts_np)
                if pts_np.shape[0] > max_points_per_instance:
                    idx = np.random.choice(pts_np.shape[0], max_points_per_instance, replace=False)
                    pts_np = pts_np[idx]

                color = perturb_color(base_color, inst_j)
                opacity = 0.25 + 0.75 * float(agreement)

                trace = go.Scatter3d(
                    x=pts_np[:, 0],
                    y=pts_np[:, 1],
                    z=pts_np[:, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=color, opacity=opacity),
                    name=f"[pre] {cat} | f{frame_i} | i{inst_j}",
                    hovertemplate=(
                        f"<b>{cat}</b><br>"
                        f"frame={frame_i}, inst={inst_j}<br>"
                        f"agreement={agreement:.2f}<extra></extra>"
                    ),
                    showlegend=show_legend,
                    visible=True,
                )
                fig.add_trace(trace)
                pre_merge_trace_ids.append(len(fig.data) - 1)
                total_points += pts_np.shape[0]

        # -------------------------------------------------
        # 2️⃣ POST-MERGE VISUALIZATION
        # -------------------------------------------------
        if merged_instances is not None:
            for gid, inst in enumerate(merged_instances):
                nodes = inst.get("nodes", [])
                if not nodes:
                    continue

                merged_pts = []
                for (img_idx, inst_idx) in nodes:
                    try:
                        fpos = pos_3d[image_idxs.index(img_idx)][inst_idx]
                        if fpos is not None:
                            merged_pts.append(np.asarray(fpos))
                    except Exception:
                        continue

                if not merged_pts:
                    continue

                merged_pts = np.concatenate(merged_pts, axis=0)
                if merged_pts.shape[0] > max_points_per_instance:
                    idx = np.random.choice(merged_pts.shape[0], max_points_per_instance, replace=False)
                    merged_pts = merged_pts[idx]

                color = perturb_color(base_color, gid)
                trace = go.Scatter3d(
                    x=merged_pts[:, 0],
                    y=merged_pts[:, 1],
                    z=merged_pts[:, 2],
                    mode="markers",
                    marker=dict(size=marker_size + 1, color=color, opacity=0.9),
                    name=f"[merged] {cat} | group {gid} | n={len(nodes)}",
                    showlegend=show_legend,
                    visible=False,  # default hide
                )
                fig.add_trace(trace)
                post_merge_trace_ids.append(len(fig.data) - 1)
                total_points += merged_pts.shape[0]

    # -------------------------------------------------
    # UI BUTTONS
    # -------------------------------------------------
    n_traces = len(fig.data)

    def vis_mask(active_ids):
        return [i in active_ids for i in range(n_traces)]

    fig.update_layout(
        title=f"{title} | categories={num_cats} | points≈{total_points}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Before Merge",
                        method="update",
                        args=[{"visible": vis_mask(pre_merge_trace_ids)}],
                    ),
                    dict(
                        label="After Merge",
                        method="update",
                        args=[{"visible": vis_mask(post_merge_trace_ids)}],
                    ),
                ],
                x=0.0,
                y=1.1,
            )
        ],
        margin=dict(l=0, r=0, t=80, b=0),
    )

    output_html = str(output_html)
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")
    print(f"[Done] Saved interactive viewer to: {output_html}")

def compute_node_centroids(category_pos, cat, image_idxs):
    """
    Return dict: node (frame, inst_idx) -> centroid (3,)
    """
    centroids = {}

    pos_3d = category_pos[cat]["pos_3d"]

    for frame_i, inst_list in enumerate(pos_3d):
        if inst_list is None:
            continue
        frame = image_idxs[frame_i]

        for inst_idx, pts in enumerate(inst_list):
            if pts is None or len(pts) == 0:
                continue
            pts = np.asarray(pts)
            if pts.ndim != 2 or pts.shape[1] != 3:
                continue
            centroids[(frame, inst_idx)] = pts.mean(axis=0)

    return centroids

def visualize_instance_graph(
    category_pos,
    cat,
    image_idxs,
    nodes,
    edges,
    output_html,
    title="Instance Agreement Graph",
):
    """
    nodes: set of (frame, inst_idx)
    edges: dict[(src_node, tgt_node)] -> {weight, type}
    """

    # ---- node positions ----
    centroids = compute_node_centroids(category_pos, cat, image_idxs)

    fig = go.Figure()

    # ---- draw nodes ----
    xs, ys, zs, texts = [], [], [], []
    for node in nodes:
        if node not in centroids:
            continue
        x, y, z = centroids[node]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        texts.append(f"frame={node[0]}<br>inst={node[1]}")

    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            text=[f"{i}" for i in range(len(xs))],
            textposition="top center",
            marker=dict(
                size=6,
                color="blue",
                opacity=0.9,
            ),
            name="nodes",
            hovertext=texts,
            hoverinfo="text",
        )
    )

    # ---- edge styles ----
    EDGE_STYLE = {
        "hard": dict(color="red", width=4, opacity=0.9),
        "soft": dict(color="orange", width=2, opacity=0.6),
        "weak": dict(color="gray", width=1, opacity=0.3),
    }

    # ---- draw edges ----
    for (src, tgt), attr in edges.items():
        if src not in centroids or tgt not in centroids:
            continue

        x0, y0, z0 = centroids[src]
        x1, y1, z1 = centroids[tgt]

        style = EDGE_STYLE[attr["type"]]

        fig.add_trace(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(
                    color=style["color"],
                    width=style["width"],
                ),
                opacity=style["opacity"],
                hoverinfo="text",
                text=(
                    f"{src} → {tgt}<br>"
                    f"type={attr['type']}<br>"
                    f"weight={attr['weight']:.3f}"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"{title} | category={cat} | nodes={len(nodes)} | edges={len(edges)}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")

    print(f"[OK] Graph visualization saved to {output_html}")


def save_pointcloud_with_vector_html(points, vector=None, filename="cloud.html", 
                                     colors=None, c2w_all=None, point_size=2, downsample_ratio=1.0):
    """
    保存交互式 HTML (点云 + 向量箭头)
    - points: (N, 3) numpy array
    - vector: (3,) numpy array
    - filename: 输出的 HTML 文件
    - colors: (N, 3) numpy array, RGB in [0,1]
    - c2w_all: (M, 4, 4) numpy array, 相机外参矩阵 (可选)
    - point_size: 点大小
    - downsample_ratio: 下采样比例 (0~1)，比如 0.1 表示取 10%
    """
    # if arrays are tensors, convert to numpy
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if colors is not None and not isinstance(colors, np.ndarray):
        colors = colors.cpu().numpy()
    if c2w_all is not None and not isinstance(c2w_all, np.ndarray):
        c2w_all = c2w_all.cpu().numpy()
    if vector is not None and not isinstance(vector, np.ndarray):
        vector = vector.cpu().numpy()
    # 去掉无效点
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]

    # 下采样
    if 0 < downsample_ratio < 1.0:
        n = int(len(points) * downsample_ratio)
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    # 颜色处理：RGB [0,1] → hex
    # import ipdb; ipdb.set_trace()
    if colors is not None:
        colors = (colors * 255).astype(np.uint8)
        colors_hex = [f"rgb({r},{g},{b})" for r, g, b in colors]
    else:
        colors_hex = "gray"

    # 点云
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=point_size, color=colors_hex, opacity=0.8),
        name="Point Cloud"
    ))

    # 箭头（平均 up 向量）
    if vector is not None:
        center = points.mean(0)
        v = vector / np.linalg.norm(vector)
        fig.add_trace(go.Cone(
            x=[center[0]], y=[center[1]], z=[center[2]],
            u=[v[0]], v=[v[1]], w=[v[2]],
            sizemode="absolute", sizeref=2,
            colorscale=[[0, 'red'], [1, 'red']],
            name="Mean Up"
        ))

    if c2w_all is not None:
        axis_scale = np.linalg.norm(points.max(0) - points.min(0)) * 0.05

        for i, c2w in enumerate(c2w_all):
            R = c2w[:3, :3]
            t = c2w[:3, 3]

            right = R[:, 0]
            up = R[:, 1]
            forward = R[:, 2]

            # Right (Red)
            fig.add_trace(go.Cone(
                x=[t[0]], y=[t[1]], z=[t[2]],
                u=[right[0]], v=[right[1]], w=[right[2]],
                sizemode="absolute",
                sizeref=axis_scale,
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                name=f"Cam{i}_Right"
            ))

            # Down (Green)
            fig.add_trace(go.Cone(
                x=[t[0]], y=[t[1]], z=[t[2]],
                u=[up[0]], v=[up[1]], w=[up[2]],
                sizemode="absolute",
                sizeref=axis_scale,
                colorscale=[[0, 'green'], [1, 'green']],
                showscale=False,
                name=f"Cam{i}_Down"
            ))

            # Forward (Blue)
            fig.add_trace(go.Cone(
                x=[t[0]], y=[t[1]], z=[t[2]],
                u=[forward[0]], v=[forward[1]], w=[forward[2]],
                sizemode="absolute",
                sizeref=axis_scale,
                colorscale=[[0, 'blue'], [1, 'blue']],
                showscale=False,
                name=f"Cam{i}_Forward"
            ))

        # also keep trajectory
        cams = np.array([c[:3, 3] for c in c2w_all])
        fig.add_trace(go.Scatter3d(
            x=cams[:, 0], y=cams[:, 1], z=cams[:, 2],
            mode="lines+markers",
            marker=dict(size=4, color="black"),
            line=dict(width=2, color="black"),
            name="Camera Trajectory"
        ))

    # 坐标轴设置
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        title="Point Cloud with Mean Up Vector"
    )

    # 保存
    fig.write_html(filename)
    print(f"Saved interactive visualization to {filename}")


def render_camera_trajectory(
    c2w_all,
    global_up,
    output_png="camera_trajectory.png",
):
    """
    Render camera trajectory (top view) optimized for VLM reasoning.

    - Y is up
    - Top view = (X, Z)
    - Strong visual encoding for direction & time
    """

    poses = c2w_all

    # ----------------------------------
    # 1. Extract positions
    # ----------------------------------
    positions = np.array([p["position"] for p in poses])
    forwards = np.array([p["forward"] for p in poses])

    pos_2d = positions[:, [0, 2]]
    fwd_2d = forwards[:, [0, 2]]

    fwd_norm = np.linalg.norm(fwd_2d, axis=1, keepdims=True) + 1e-8
    fwd_2d = fwd_2d / fwd_norm

    # ----------------------------------
    # 2. Smooth trajectory
    # ----------------------------------
    t = np.arange(len(pos_2d))
    cs_x = CubicSpline(t, pos_2d[:, 0])
    cs_y = CubicSpline(t, pos_2d[:, 1])

    t_smooth = np.linspace(0, len(pos_2d) - 1, 400)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)

    # ----------------------------------
    # 3. Plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(10, 9))

    # ---- Time-gradient trajectory ----
    points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments,
        cmap="plasma",
        norm=plt.Normalize(0, 1),
        linewidth=6,
    )

    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)

    # ----------------------------------
    # 4. Expand boundary to avoid clipping
    # ----------------------------------
    margin_ratio = 0.15
    x_min, x_max = pos_2d[:, 0].min(), pos_2d[:, 0].max()
    y_min, y_max = pos_2d[:, 1].min(), pos_2d[:, 1].max()

    dx = (x_max - x_min)
    dy = (y_max - y_min)

    max_range = max(dx, dy)
    # expand to square
    if dx < max_range:
        mid_x = (x_min + x_max) / 2
        x_min = mid_x - max_range / 2
        x_max = mid_x + max_range / 2
    if dy < max_range:
        mid_y = (y_min + y_max) / 2
        y_min = mid_y - max_range / 2
        y_max = mid_y + max_range / 2

    ax.set_xlim(x_min - dx * margin_ratio, x_max + dx * margin_ratio)
    ax.set_ylim(y_min - dy * margin_ratio, y_max + dy * margin_ratio)

    # ---- Direction arrows along trajectory ----
    arrow_step = 30
    for i in range(0, len(x_smooth) - 1, arrow_step):
        ax.arrow(
            x_smooth[i],
            y_smooth[i],
            x_smooth[i + 1] - x_smooth[i],
            y_smooth[i + 1] - y_smooth[i],
            shape="full",
            lw=0,
            length_includes_head=True,
            head_width=0.03,
            head_length=0.05,
            color="black",
        )

    # ---- Camera points ----
    ax.scatter(
        pos_2d[:, 0],
        pos_2d[:, 1],
        s=120,
        c="black",
        zorder=3,
    )

    # ---- Start / End markers ----
    ax.scatter(
        pos_2d[0, 0],
        pos_2d[0, 1],
        s=300,
        c="green",
        marker="o",
        edgecolors="black",
        linewidths=2,
        zorder=4,
    )
    ax.text(
        pos_2d[0, 0],
        pos_2d[0, 1]+0.05,
        "START",
        fontsize=14,
        fontweight="bold",
        color="green",
        clip_on=False,
    )

    ax.scatter(
        pos_2d[-1, 0],
        pos_2d[-1, 1],
        s=300,
        c="red",
        marker="o",
        edgecolors="black",
        linewidths=2,
        zorder=4,
    )
    ax.text(
        pos_2d[-1, 0],
        pos_2d[-1, 1]+0.05,
        "END",
        fontsize=14,
        fontweight="bold",
        color="red",
        clip_on=False,
    )

    # ---- Forward arrows ----
    scene_range = np.linalg.norm(pos_2d.max(axis=0) - pos_2d.min(axis=0))
    arrow_scale = 0.08 * scene_range
    label_offset = 0.06 * scene_range

    for i in range(len(pos_2d)):
        px, py = pos_2d[i]
        fx, fy = fwd_2d[i]

        ax.arrow(
            px,
            py,
            fx * arrow_scale,
            fy * arrow_scale,
            head_width=0.05,
            head_length=0.07,
            fc="red",
            ec="red",
            linewidth=2,
            zorder=3,
        )

        # offset label perpendicular to forward
        right = np.array([-fy, fx])
        lx = px + right[0] * label_offset
        ly = py + right[1] * label_offset

        ax.text(
            lx,
            ly,
            f"{poses[i].get('camera_index', i)}",
            fontsize=12,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.3",
            ),
            zorder=5,
        )

    # ----------------------------------
    # 4. Clean layout
    # ----------------------------------
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Z", fontsize=14)
    ax.set_title("Camera Trajectory (Top View)", fontsize=16)

    ax.grid(False)
    ax.set_facecolor("white")

    plt.tight_layout()

    output_png = str(output_png)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"[Done] Saved trajectory PNG to: {output_png}")


def render_semantic_bev_entities(
    entities,
    global_up,
    output_png="semantic_bev.png",
    title="Semantic BEV",
    marker_size: int = 220,
    ego_pose=None,
    background_points: Optional[np.ndarray] = None,
    background_colors: Optional[np.ndarray] = None,
    background_point_size: float = 2.0,
    background_alpha: float = 0.9,
):
    """
    Render a symbolic BEV diagram for entities with optional forward arrows.

    - Y is up
    - Top view = (X, Z)
    - Each entity is shown as a labeled marker
    - Optional orientation is shown as a direction arrow
    """
    if not entities:
        raise ValueError("`entities` is empty.")

    positions = []
    orientations = []
    names = []
    types = []

    for idx, entity in enumerate(entities):
        if "position" not in entity:
            raise ValueError(f"Entity at index {idx} is missing `position`.")
        pos = np.asarray(entity["position"], dtype=float)
        if pos.shape != (3,):
            raise ValueError(f"Entity at index {idx} must have a 3D `position`.")
        positions.append(pos)
        names.append(str(entity.get("name", f"entity_{idx}")))
        types.append(str(entity.get("type", "entity")))

        orientation = entity.get("orientation")
        if orientation is None:
            orientations.append(None)
            continue
        orientation = np.asarray(orientation, dtype=float)
        if orientation.shape != (3,):
            orientations.append(None)
            continue
        norm = np.linalg.norm(orientation[[0, 2]])
        if norm < 1e-8:
            orientations.append(None)
            continue
        orientations.append(orientation / (np.linalg.norm(orientation) + 1e-8))

    positions = np.asarray(positions, dtype=float)
    pos_2d = positions[:, [0, 2]]

    unique_types = sorted(set(types))
    style = get_semantic_bev_style(unique_types, marker_size=marker_size)
    color_by_type = style["color_by_type"]

    all_pos_2d = [pos_2d]
    bg_pos_2d = None
    bg_colors = None
    if background_points is not None and background_colors is not None:
        bg_points = np.asarray(background_points, dtype=float)
        bg_colors = np.asarray(background_colors, dtype=float)
        if bg_points.ndim == 2 and bg_points.shape[1] == 3 and len(bg_points) == len(bg_colors):
            bg_pos_2d = bg_points[:, [0, 2]]
            all_pos_2d.append(bg_pos_2d)

    all_pos_2d = np.concatenate(all_pos_2d, axis=0)
    x_min, x_max = all_pos_2d[:, 0].min(), all_pos_2d[:, 0].max()
    y_min, y_max = all_pos_2d[:, 1].min(), all_pos_2d[:, 1].max()
    dx = x_max - x_min
    dy = y_max - y_min
    max_range = max(dx, dy, 1.0)
    margin = 0.18 * max_range

    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    half_range = 0.5 * max_range

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(mid_x - half_range - margin, mid_x + half_range + margin)
    ax.set_ylim(mid_y - half_range - margin, mid_y + half_range + margin)

    scene_range = max_range + 1e-8
    marker_size = style["marker_size"]
    arrow_scale = 0.12 * scene_range
    label_offset = 0.06 * scene_range

    label_positions = []

    def choose_label_position(
        anchor_pos: np.ndarray,
        candidate_offsets: List[np.ndarray],
        occupied_positions: List[np.ndarray],
        avoid_points: List[np.ndarray],
        min_label_dist: float | None = None,
        min_point_dist: float | None = None,
    ) -> np.ndarray:
        min_label_dist = min_label_dist if min_label_dist is not None else label_offset * 0.95
        min_point_dist = min_point_dist if min_point_dist is not None else label_offset * 0.75
        best_score = None
        best_pos = anchor_pos + candidate_offsets[0]
        for offset in candidate_offsets:
            candidate = anchor_pos + offset
            label_clearance = min(
                [np.linalg.norm(candidate - pos) for pos in occupied_positions] or [float("inf")]
            )
            point_clearance = min(
                [np.linalg.norm(candidate - pos) for pos in avoid_points] or [float("inf")]
            )
            penalty = 0.0
            if label_clearance < min_label_dist:
                penalty += (min_label_dist - label_clearance) * 10.0
            if point_clearance < min_point_dist:
                penalty += (min_point_dist - point_clearance) * 8.0
            score = penalty - 0.05 * (candidate[1] - anchor_pos[1])
            if best_score is None or score < best_score:
                best_score = score
                best_pos = candidate
        return best_pos

    if bg_pos_2d is not None and bg_colors is not None and len(bg_pos_2d) > 0:
        ax.scatter(
            bg_pos_2d[:, 0],
            bg_pos_2d[:, 1],
            s=background_point_size,
            c=np.clip(bg_colors, 0.0, 1.0),
            alpha=background_alpha,
            linewidths=0,
            zorder=1,
            rasterized=True,
        )

    # Ordered camera trajectory: connect camera entities in temporal (frame-index) order
    # so the exploration path over time is visible (useful for appearance-order reasoning).
    _cam_seq = []
    for _i, (_nm, _ty) in enumerate(zip(names, types)):
        if _ty != "camera":
            continue
        _tok = str(_nm).rsplit("_", 1)[-1]
        try:
            _num = int(_tok)
        except Exception:
            _num = _i
        _cam_seq.append((_num, _i))
    _cam_seq.sort(key=lambda t: t[0])
    if len(_cam_seq) >= 2:
        _traj = np.array([pos_2d[i] for _, i in _cam_seq])
        ax.plot(_traj[:, 0], _traj[:, 1], color="#1f77b4", linestyle="--",
                linewidth=1.8, alpha=0.65, zorder=2)
        for _k in range(len(_traj) - 1):
            _p0, _p1 = _traj[_k], _traj[_k + 1]
            if np.linalg.norm(_p1 - _p0) < 1e-6:
                continue
            ax.annotate("", xy=(_p1[0], _p1[1]), xytext=(_p0[0], _p0[1]),
                        arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.65, lw=1.3),
                        zorder=2)
        ax.scatter(_traj[0, 0], _traj[0, 1], s=marker_size * 1.4, facecolors="none",
                   edgecolors="green", linewidths=2.4, zorder=6)
        ax.scatter(_traj[-1, 0], _traj[-1, 1], s=marker_size * 1.4, facecolors="none",
                   edgecolors="red", linewidths=2.4, zorder=6)

    for idx, (name, entity_type, pos, orient) in enumerate(zip(names, types, pos_2d, orientations)):
        color = color_by_type[entity_type]
        px, py = pos

        ax.scatter(
            px,
            py,
            s=marker_size,
            c=[color],
            edgecolors=style["marker_edgecolor"],
            linewidths=style["marker_linewidth"],
            zorder=3,
        )

        if orient is not None:
            fwd_2d = orient[[0, 2]]
            fwd_norm = np.linalg.norm(fwd_2d) + 1e-8
            fwd_2d = fwd_2d / fwd_norm
            ax.arrow(
                px,
                py,
                fwd_2d[0] * arrow_scale,
                fwd_2d[1] * arrow_scale,
                head_width=0.05 * scene_range,
                head_length=0.07 * scene_range,
                fc=style["arrow_color"],
                ec=style["arrow_color"],
                linewidth=style["arrow_linewidth"],
                length_includes_head=True,
                zorder=4,
            )
            right = np.array([-fwd_2d[1], fwd_2d[0]])
            label_pos = pos + right * label_offset
        else:
            label_pos = pos + np.array([label_offset, label_offset])
        label_positions.append(label_pos)

        ax.text(
            label_pos[0],
            label_pos[1],
            name,
            fontsize=style["label_fontsize"],
            fontweight=style["label_fontweight"],
            bbox=style["label_bbox"],
            zorder=5,
        )

    if ego_pose is not None:
        ego_pos = np.asarray(ego_pose.get("position", [0.0, 0.0, 0.0]), dtype=float)
        ego_orient = np.asarray(ego_pose.get("orientation", [0.0, 0.0, 1.0]), dtype=float)
        ego_pos_2d = ego_pos[[0, 2]]
        ego_fwd_2d = ego_orient[[0, 2]]
        ego_fwd_norm = np.linalg.norm(ego_fwd_2d)
        if ego_fwd_norm < 1e-8:
            ego_fwd_2d = np.array([0.0, 1.0], dtype=float)
        else:
            ego_fwd_2d = ego_fwd_2d / ego_fwd_norm

        same_pos_tol = max(1e-4, scene_range * 0.03)
        same_dir_cos = 0.92
        ego_arrow_color = "limegreen"
        ego_label_bbox = dict(
            facecolor="#f3fff3",
            edgecolor="green",
            boxstyle="round,pad=0.25",
        )

        overlapping_camera_idx = None
        occupied_positions = [np.asarray(pos, dtype=float) for pos in label_positions]
        avoid_points = [np.asarray(pos, dtype=float) for pos in pos_2d]
        for idx, (entity_type, pos, orient) in enumerate(zip(types, pos_2d, orientations)):
            if entity_type != "camera":
                continue
            if np.linalg.norm(pos - ego_pos_2d) > same_pos_tol:
                continue
            if orient is None:
                overlapping_camera_idx = idx
                break
            cam_fwd_2d = orient[[0, 2]]
            cam_fwd_norm = np.linalg.norm(cam_fwd_2d)
            if cam_fwd_norm < 1e-8:
                overlapping_camera_idx = idx
                break
            cam_fwd_2d = cam_fwd_2d / cam_fwd_norm
            if float(np.dot(cam_fwd_2d, ego_fwd_2d)) > same_dir_cos:
                overlapping_camera_idx = idx
                break

        if overlapping_camera_idx is not None:
            base_label_pos = label_positions[overlapping_camera_idx]
            ego_label_pos = choose_label_position(
                base_label_pos,
                candidate_offsets=[
                    np.array([label_offset * 1.25, -label_offset * 0.55]),
                    np.array([label_offset * 1.15, label_offset * 0.35]),
                    np.array([-label_offset * 1.15, -label_offset * 0.55]),
                    np.array([0.0, -label_offset * 1.1]),
                ],
                occupied_positions=occupied_positions,
                avoid_points=avoid_points,
                min_label_dist=label_offset * 0.9,
                min_point_dist=label_offset * 0.7,
            )
            ax.text(
                ego_label_pos[0],
                ego_label_pos[1],
                "ego",
                fontsize=max(10, style["label_fontsize"] - 1),
                fontweight=style["label_fontweight"],
                color="green",
                bbox=ego_label_bbox,
                zorder=6,
            )
        else:
            same_origin_camera_idx = None
            for idx, (entity_type, pos) in enumerate(zip(types, pos_2d)):
                if entity_type == "camera" and np.linalg.norm(pos - ego_pos_2d) <= same_pos_tol:
                    same_origin_camera_idx = idx
                    break

            if same_origin_camera_idx is not None:
                ego_right = np.array([-ego_fwd_2d[1], ego_fwd_2d[0]])
                ax.arrow(
                    ego_pos_2d[0],
                    ego_pos_2d[1],
                    ego_fwd_2d[0] * arrow_scale,
                    ego_fwd_2d[1] * arrow_scale,
                    head_width=0.05 * scene_range,
                    head_length=0.07 * scene_range,
                    fc=ego_arrow_color,
                    ec=ego_arrow_color,
                    linewidth=style["arrow_linewidth"],
                    length_includes_head=True,
                    zorder=4,
                )
                ego_label_pos = choose_label_position(
                    ego_pos_2d,
                    candidate_offsets=[
                        -ego_right * label_offset * 1.25 + np.array([0.0, label_offset * 0.45]),
                        ego_right * label_offset * 1.25 + np.array([0.0, label_offset * 0.45]),
                        np.array([0.0, -label_offset * 1.2]),
                        -ego_right * label_offset * 1.45,
                    ],
                    occupied_positions=occupied_positions,
                    avoid_points=avoid_points + [ego_pos_2d],
                    min_label_dist=label_offset,
                    min_point_dist=label_offset * 0.8,
                )
                ax.text(
                    ego_label_pos[0],
                    ego_label_pos[1],
                    "ego",
                    fontsize=max(10, style["label_fontsize"] - 1),
                    fontweight=style["label_fontweight"],
                    color="green",
                    bbox=ego_label_bbox,
                    zorder=6,
                )
            else:
                camera_color = color_by_type.get("camera", style["color_by_type"][unique_types[0]])
                ax.scatter(
                    ego_pos_2d[0],
                    ego_pos_2d[1],
                    s=marker_size,
                    c=[camera_color],
                    edgecolors=style["marker_edgecolor"],
                    linewidths=style["marker_linewidth"],
                    zorder=3,
                )
                ax.arrow(
                    ego_pos_2d[0],
                    ego_pos_2d[1],
                    ego_fwd_2d[0] * arrow_scale,
                    ego_fwd_2d[1] * arrow_scale,
                    head_width=0.05 * scene_range,
                    head_length=0.07 * scene_range,
                    fc=ego_arrow_color,
                    ec=ego_arrow_color,
                    linewidth=style["arrow_linewidth"],
                    length_includes_head=True,
                    zorder=4,
                )
                ego_right = np.array([-ego_fwd_2d[1], ego_fwd_2d[0]])
                ego_label_pos = choose_label_position(
                    ego_pos_2d,
                    candidate_offsets=[
                        -ego_right * label_offset * 1.1,
                        ego_right * label_offset * 1.1,
                        np.array([0.0, label_offset * 1.15]),
                        np.array([0.0, -label_offset * 1.15]),
                    ],
                    occupied_positions=occupied_positions,
                    avoid_points=avoid_points + [ego_pos_2d],
                )
                ax.text(
                    ego_label_pos[0],
                    ego_label_pos[1],
                    "ego",
                    fontsize=style["label_fontsize"],
                    fontweight=style["label_fontweight"],
                    color="green",
                    bbox=ego_label_bbox,
                    zorder=6,
                )

    if unique_types:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_by_type[entity_type],
                markeredgecolor=style["marker_edgecolor"],
                markersize=style["legend_markersize"],
                label=entity_type,
            )
            for entity_type in unique_types
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True)

    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Z", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_facecolor("white")

    plt.tight_layout()

    output_png = str(output_png)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()


def get_camera_look_at(
        eye: np.ndarray, 
        forward_direction: np.ndarray,
        forward_distance=1.0):
    forward_norm = np.linalg.norm(forward_direction) + 1e-8
    forward_dir_normalized = forward_direction / forward_norm
    look_at = eye + forward_dir_normalized * forward_distance
    return look_at


def render_perspective_view(
        points: np.ndarray, 
        colors: np.ndarray, 
        camera_pose: np.ndarray, 
        cam_forward: np.ndarray,
        cam_up: np.ndarray,
        width: int = 512,
        height: int = 512,
        background_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        intrinsics: Optional[np.ndarray] = None,
        FOV: Optional[float] = None,    # if None, will use the original camera intrinsics to determine FOV. Otherwise, use this fixed FOV for all views.
    ) -> Image.Image:
    """
    Render a perspective view (e.g., RGB image) from the spatial memory observed from current origin and forward direction. This can be useful for visualizing the spatial memory or for providing a specific view to the model.
    Placeholder implementation: just save a blank image for now.
    """
    # import ipdb; ipdb.set_trace()
    # ---- Coordinate conversion: memory → Open3D ----
    convert = np.diag([1, -1, -1])
    points_cvt = copy.deepcopy(convert @ points.T).T
    cam_forward_cvt = copy.deepcopy(convert @ cam_forward)
    cam_up_cvt = copy.deepcopy(convert @ cam_up)
    camera_pose_cvt = copy.deepcopy(convert @ camera_pose)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cvt)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    look_at = get_camera_look_at(camera_pose_cvt, cam_forward_cvt)

    if FOV is None:
        # import ipdb; ipdb.set_trace()
        assert intrinsics is not None, "If FOV is not provided, intrinsics must be provided to determine FOV."
        org_w, org_h = intrinsics[0, 2] * 2, intrinsics[1, 2] * 2
        resize_ratio = min(width / org_w, height / org_h)
        intrinsic_matrix = resize_ratio * intrinsics
        intrinsic_matrix[-1, -1] = 1.0
        intrinsic_matrix[0, 2] = width / 2
        intrinsic_matrix[1, 2] = height / 2
    else:
        # import ipdb; ipdb.set_trace()
        fx = 0.5 * width / np.tan(np.deg2rad(FOV / 2))
        fy = fx  # 通常假设像素是方形
        # import ipdb; ipdb.set_trace()

        intrinsic_matrix = np.array([
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1]
        ])
    renderer = OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.clear_geometry()
    scene.set_background(np.asarray(background_rgba, dtype=np.float32))
    material = MaterialRecord()
    material.shader = "defaultUnlit"
    scene.add_geometry("pc", pcd, material)
    scene.camera.look_at(look_at, camera_pose_cvt, cam_up_cvt)

    cam_to_points = np.linalg.norm(points_cvt - camera_pose_cvt[None, :], axis=1)
    finite_dist = cam_to_points[np.isfinite(cam_to_points)]
    if len(finite_dist) == 0:
        near_plane, far_plane = 0.1, 10.0
    else:
        far_plane = max(10.0, float(np.percentile(finite_dist, 99.5)) * 1.25)
        near_plane = max(0.05, min(0.2, far_plane / 200.0))

    scene.camera.set_projection(
        intrinsic_matrix.astype(np.float64),
        near_plane, far_plane, width, height
    )
    img = renderer.render_to_image()
    img = np.asarray(img)
    img = Image.fromarray(img)
    # import ipdb; ipdb.set_trace()
    return img

    
    


if __name__ == "__main__":
    # 示例用法
    np.random.seed(42)
    up_vector = np.array([0, 1, 0])
    c2w_all = [
        {'position': [0, 0, 0.5], 'forward': [0, 0, -1], 'camera_index': 1},
        {'position': [1, 0, 0], 'forward': [-1, 0, 0], 'camera_index': 2},
    ]
    render_camera_trajectory(c2w_all, up_vector, output_png='./debug_vis/camera_trajectory.png')
