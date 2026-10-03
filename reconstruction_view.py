"""Shared reconstruction geometry and appearance for both map viewers.

Keep the CUDA reprojection and multi-view depth filter identical to the original
view_reconstruction.py. Feature-selection masks are unsuitable for displaying a
map: unconfirmed far points distort its bounds and the initial viewing scale.
"""

from pathlib import Path

import numpy as np


RENDER_OPTIONS = Path(__file__).resolve().parent / "misc" / "renderoption.json"


def build_reconstruction(filename, filter_threshold=.005, filter_count=3, stride=2):
    """Return the colored map cloud and camera-to-world pose matrices.

    Defaults reproduce view_reconstruction.py's command-line behavior. This
    visualization path uses CUDA, independently of the CPU-only localizer.
    """
    import torch

    if stride < 1 or int(stride) != stride:
        raise ValueError("Cloud stride must be a positive integer")
    if not np.isfinite(filter_threshold) or filter_threshold <= 0:
        raise ValueError("Depth filter threshold must be finite and positive")
    if filter_count < 1 or int(filter_count) != filter_count:
        raise ValueError("Depth filter count must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError("Map rendering requires CUDA, like view_reconstruction.py; localize.py runs on CPU")

    import droid_backends
    import open3d as o3d
    from lietorch import SE3

    blob = torch.load(filename, map_location="cpu", weights_only=True)
    images = blob["images"].cuda()[..., ::stride, ::stride]
    disps = blob["disps"].cuda()[..., ::stride, ::stride].contiguous()
    poses = blob["poses"].cuda()
    intrinsics = (8.0 / stride) * blob["intrinsics"].cuda()

    index = torch.arange(len(images), device="cuda")
    threshold = filter_threshold * torch.ones_like(disps.mean(dim=[1, 2]))
    with torch.no_grad():
        camera_poses = SE3(poses).inv()
        points = droid_backends.iproj(camera_poses.data, disps, intrinsics[0])
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, threshold)
        mask = (counts >= filter_count) & (disps > .25 * disps.mean())
        colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0
        points_np = points[mask].cpu().numpy()
        colors_np = colors[mask].cpu().numpy()
        pose_matrices = camera_poses.matrix().cpu().numpy()
    if not len(points_np):
        raise ValueError("No map points pass the multi-view filter; check --filter-count/--filter-threshold")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_np)
    cloud.colors = o3d.utility.Vector3dVector(colors_np)
    return cloud, pose_matrices


def configure_rendering(visualizer):
    if not RENDER_OPTIONS.is_file():
        raise FileNotFoundError(RENDER_OPTIONS)
    # Open3D returns None on success, not a boolean.
    visualizer.get_render_option().load_from_json(str(RENDER_OPTIONS))


def create_camera_actor(pose=None, scale=.05, color=(0., .5, .9)):
    """Same frustum geometry as the original reconstruction viewer."""
    import open3d as o3d

    points = np.array([[0, 0, 0], [-1, -1, 1.5], [1, -1, 1.5], [1, 1, 1.5],
                       [-1, 1, 1.5], [-.5, 1, 1.5], [.5, 1, 1.5], [0, 1.2, 1.5]])
    lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2],
                      [3, 0], [0, 4], [5, 7], [7, 6]])
    actor = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points * scale),
                                lines=o3d.utility.Vector2iVector(lines))
    actor.paint_uniform_color(color)
    if pose is not None:
        actor.transform(pose)
    return actor
