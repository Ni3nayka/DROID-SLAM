#!/usr/bin/env python3
"""View a localization CSV over its reference map; supports headless PLY export."""

import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from localization.map import ReferenceMap
from reconstruction_view import build_reconstruction, configure_rendering, create_camera_actor


def trajectory_segments(rows):
    """Only connect adjacent localized rows; never bridge a lost interval."""
    points, edges = [], []
    previous = None
    for row in rows:
        point = np.array([float(row[k]) for k in ("x", "y", "z")])
        if row["status"] == "lost" or not np.isfinite(point).all():
            previous = None
            continue
        index = len(points)
        points.append(point)
        if previous is not None:
            edges.append([previous, index])
        previous = index
    return np.array(points).reshape(-1, 3), np.array(edges, dtype=int).reshape(-1, 2)


def build_geometry(reference_map, rows, scale=1., stride=2, filter_threshold=.005, filter_count=3):
    import open3d as o3d
    cloud, _ = build_reconstruction(reference_map.path, filter_threshold, filter_count, stride)
    cloud.scale(scale, center=(0, 0, 0))

    def lines(points, edges, color):
        geometry = o3d.geometry.LineSet()
        geometry.points = o3d.utility.Vector3dVector(points)
        geometry.lines = o3d.utility.Vector2iVector(edges)
        geometry.paint_uniform_color(color)
        return geometry

    centers = reference_map.centers * scale
    old = lines(centers, np.array([[i, i + 1] for i in range(len(centers) - 1)]).reshape(-1, 2), [.2, .5, 1.])
    points, edges = trajectory_segments(rows)
    new = lines(points, edges, [1., .25, .1])
    return cloud, old, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--map", type=Path, help="Defaults to map path in metadata.json")
    parser.add_argument("--frame", type=int, help="Initial original video frame index")
    parser.add_argument("--step", type=int, default=1, help="Rows to advance with N/P")
    parser.add_argument("--cloud-stride", type=int, default=2, help="Pixel stride; 2 matches view_reconstruction")
    parser.add_argument("--filter-threshold", "--filter_threshold", type=float, default=.005,
                        help="Multi-view depth threshold, as in view_reconstruction")
    parser.add_argument("--filter-count", "--filter_count", type=int, default=3,
                        help="Required agreeing views, as in view_reconstruction")
    parser.add_argument("--show-video", action="store_true", help="Show source image at selected PTS")
    parser.add_argument("--export", type=Path, help="Export map and trajectory geometries as PLY")
    parser.add_argument("--no-window", action="store_true", help="Only load/check/export geometry")
    parser.add_argument("--screenshot", type=Path, help="Render a PNG in a hidden window and exit (requires a display)")
    args = parser.parse_args()
    if args.step < 1 or args.cloud_stride < 1:
        parser.error("--step and --cloud-stride must be positive")
    if args.screenshot and args.no_window:
        parser.error("--screenshot requires rendering; omit --no-window")
    with args.trajectory.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("Trajectory contains no rows")
    metadata = json.loads(args.trajectory.with_name("metadata.json").read_text())
    reference_map = ReferenceMap.load(args.map or metadata["map"])
    if reference_map.digest != metadata["map_sha256"]:
        parser.error("Map content does not match the map used for localization")
    scale = metadata.get("position_scale", 1.)
    cloud, old, new = build_geometry(reference_map, rows, scale, args.cloud_stride,
                                     args.filter_threshold, args.filter_count)
    import open3d as o3d
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(args.export / "map.ply"), cloud):
            raise OSError("Could not export map")
        for name, geometry in [("map_trajectory", old), ("video_trajectory", new)]:
            if len(geometry.lines) and not o3d.io.write_line_set(str(args.export / f"{name}.ply"), geometry):
                raise OSError(f"Could not export {name}")
    print(f"Map points: {len(cloud.points)}; trajectory rows: {len(rows)}; valid positions: {len(new.points)}")
    if args.no_window:
        return
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name="Localization: N/P frame, Space play/pause", width=960, height=960,
                             visible=not bool(args.screenshot)):
        raise RuntimeError("Cannot open visualization window; use --no-window --export for headless export")
    configure_rendering(vis)
    for geometry in [cloud, old, new]:
        if len(geometry.points):
            vis.add_geometry(geometry)
    # Show the same reference-camera frustums as view_reconstruction. These also
    # give both viewers the same scene bounds and initial view scale.
    for pose in reference_map.poses:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix().T
        matrix[:3, 3] = -matrix[:3, :3] @ pose[:3] * scale
        vis.add_geometry(create_camera_actor(matrix, scale=.05 * scale))
    centers = reference_map.centers
    span = np.linalg.norm(np.ptp(centers, axis=0))
    camera_size = max(span * .02, .01) * scale
    camera = o3d.geometry.LineSet()
    state = {"row": 0, "playing": False, "last_time": time.monotonic(), "camera_added": False}
    if args.frame is not None:
        state["row"] = min(range(len(rows)), key=lambda i: abs(int(rows[i]["frame_index"]) - args.frame))
    video = None
    if args.show_video and not args.screenshot and "video" in metadata:
        import av
        video = av.open(metadata["input"])

    def show():
        row = rows[state["row"]]
        print(f"Frame {row['frame_index']}, {row['timestamp_sec']} s: {row['status']}  "
              f"xyz=({row['x']}, {row['y']}, {row['z']})")
        if state["camera_added"]:
            vis.remove_geometry(camera, reset_bounding_box=False)
            state["camera_added"] = False
        if row["status"] != "lost":
            center = np.array([float(row[k]) for k in ("x", "y", "z")])
            rotation = Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
            base = np.array([[0, 0, 0], [-1, -.6, 1.5], [1, -.6, 1.5], [1, .6, 1.5], [-1, .6, 1.5]])
            camera.points = o3d.utility.Vector3dVector((base * camera_size) @ rotation.T + center)
            camera.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
            camera.paint_uniform_color([1, .8, 0])
            vis.add_geometry(camera, reset_bounding_box=False)
            state["camera_added"] = True
        if video is not None:
            stream = video.streams.video[0]
            absolute = float(row["timestamp_sec"]) + metadata["video"].get("first_pts_seconds", 0.)
            video.seek(int(absolute / stream.time_base), stream=stream, backward=True)
            for frame in video.decode(stream):
                if frame.time is not None and frame.time + 1e-6 >= absolute:
                    image = frame.to_ndarray(format="bgr24")
                    rotation = metadata["video"].get("display_rotation", 0)
                    image = np.ascontiguousarray(np.rot90(image, (rotation // 90) % 4))
                    factor = min(1., 960 / image.shape[1])
                    cv2.imshow("Source video", cv2.resize(image, None, fx=factor, fy=factor))
                    cv2.waitKey(1)
                    break

    def move(delta):
        state["row"] = max(0, min(len(rows) - 1, state["row"] + delta))
        state["last_time"] = time.monotonic()
        show()
        return False

    def toggle(_):
        state["playing"] = not state["playing"]
        state["last_time"] = time.monotonic()
        return False

    def animate(_):
        index = state["row"]
        if state["playing"] and index < len(rows) - 1:
            delay = max(.001, float(rows[index + 1]["timestamp_sec"]) - float(rows[index]["timestamp_sec"]))
            if time.monotonic() - state["last_time"] >= delay:
                move(1)
        return False

    vis.register_key_callback(ord("N"), lambda _: move(args.step))
    vis.register_key_callback(ord("P"), lambda _: move(-args.step))
    vis.register_key_callback(32, toggle)
    vis.register_animation_callback(animate)
    try:
        show()
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(str(args.screenshot), do_render=True)
            if not args.screenshot.is_file():
                raise OSError(f"Could not save screenshot: {args.screenshot}")
            print(f"Screenshot: {args.screenshot}")
        else:
            vis.run()
    finally:
        vis.destroy_window()
        if video is not None:
            video.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
