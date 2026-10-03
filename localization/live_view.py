"""Optional GUI process. Dense map geometry is shared with the old viewers."""

from collections import deque
import multiprocessing as mp
import queue
import time

import numpy as np


def _viewer(messages, ready, stop, map_path, show_map, preview, screenshot):
    vis = None
    try:
        import cv2
        cv2.setNumThreads(1)
        if show_map:
            import open3d as o3d
            from scipy.spatial.transform import Rotation
            from reconstruction_view import build_reconstruction, configure_rendering, create_camera_actor
            cloud, cameras = build_reconstruction(map_path)
            vis = o3d.visualization.Visualizer()
            if not vis.create_window(window_name="Live map localization", width=960, height=960,
                                     visible=not bool(screenshot)):
                raise RuntimeError("Cannot create Open3D window")
            configure_rendering(vis)
            vis.add_geometry(cloud)
            old = o3d.geometry.LineSet()
            old.points = o3d.utility.Vector3dVector(cameras[:, :3, 3])
            old.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(cameras) - 1)])
            old.paint_uniform_color([.2, .5, 1.])
            vis.add_geometry(old)
            for pose in cameras:
                vis.add_geometry(create_camera_actor(pose))
            camera = None
            trail = o3d.geometry.LineSet()
            trail_added = False
            # Bound GUI memory and redraw cost even on day-long streams.
            points = deque(maxlen=2000)
            segments = deque(maxlen=2000)
            segment = 0
        ready.put({"ready": True, "map_points": len(cloud.points) if show_map else 0})
        last = None
        screen_saved = False
        while not stop.is_set():
            item = None
            for _ in range(2):
                try:
                    item = messages.get_nowait()
                except queue.Empty:
                    break
            now = time.monotonic()
            if item is not None:
                last, image = item
                valid = last["valid"] and now <= last["valid_until_monotonic"]
                if show_map:
                    if camera is not None:
                        vis.remove_geometry(camera, reset_bounding_box=False)
                        camera = None
                    if valid:
                        matrix = np.eye(4)
                        matrix[:3, :3] = Rotation.from_quat(last["quaternion_xyzw"]).as_matrix()
                        matrix[:3, 3] = last["position"]
                        camera = create_camera_actor(matrix, scale=.025, color=(1., .8, 0.))
                        vis.add_geometry(camera, reset_bounding_box=False)
                        points.append(last["position"])
                        segments.append((segment, last.get("track_segment", 0)))
                        trail.points = o3d.utility.Vector3dVector(list(points))
                        ids = list(segments)
                        trail.lines = o3d.utility.Vector2iVector(np.array(
                            [[i, i + 1] for i in range(len(ids) - 1) if ids[i] == ids[i + 1]],
                            dtype=np.int32).reshape(-1, 2))
                        trail.paint_uniform_color([1., .25, .1])
                        if len(trail.lines):
                            if not trail_added:
                                vis.add_geometry(trail, reset_bounding_box=False)
                                trail_added = True
                            else:
                                vis.update_geometry(trail)
                    else:
                        segment += 1
                if preview and image is not None:
                    image = image.copy()
                    label = (f"{last['state']}  frame={last['frame_index']}  "
                             f"age={last['frame_age_ms']:.1f}ms  n={last['inliers']}")
                    cv2.rectangle(image, (0, 0), (image.shape[1], 26), (0, 0, 0), -1)
                    cv2.putText(image, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .45,
                                (0, 220, 0) if valid else (0, 0, 255), 1)
                    cv2.imshow("Live frame (Esc to stop)", image)
            if last is not None and now > last["valid_until_monotonic"]:
                if show_map and camera is not None:
                    vis.remove_geometry(camera, reset_bounding_box=False)
                    camera = None
                    segment += 1
                if preview:
                    panel = np.zeros((64, 584, 3), np.uint8)
                    cv2.putText(panel, "STALE: no fresh pose", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1)
                    cv2.imshow("Live frame (Esc to stop)", panel)
            if vis is not None:
                if not vis.poll_events():
                    stop.set()
                    break
                vis.update_renderer()
                if screenshot and last is not None and last["valid"] and not screen_saved:
                    vis.capture_screen_image(str(screenshot), do_render=True)
                    screen_saved = True
            if preview and cv2.waitKey(1) & 255 == 27:
                stop.set()
            stop.wait(.01)
    except BaseException as exc:
        ready.put({"error": f"{type(exc).__name__}: {exc}"})
        # Reserve stop for an intentional user close. Setting it before the
        # queue feeder delivers this error could look like a successful close.
    finally:
        if vis is not None:
            vis.destroy_window()
        if preview:
            import cv2
            cv2.destroyAllWindows()


class LiveViewer:
    def __init__(self, map_path, show_map=False, preview=False, screenshot=None):
        ctx = mp.get_context("spawn")
        self.messages = ctx.Queue(maxsize=2)
        self.ready = ctx.Queue()
        self.stop = ctx.Event()
        self.process = ctx.Process(target=_viewer, args=(self.messages, self.ready, self.stop,
                                   map_path, show_map, preview, screenshot), daemon=True)
        self.dropped = 0

    def start(self):
        self.process.start()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                result = self.ready.get(timeout=.1)
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError("Viewer exited during startup")
        raise RuntimeError("Viewer startup timed out")

    def check(self):
        try:
            result = self.ready.get_nowait()
        except queue.Empty:
            result = None
        if result is not None and "error" in result:
            raise RuntimeError(result["error"])
        if not self.process.is_alive() and not self.stop.is_set():
            raise RuntimeError("Viewer exited unexpectedly")

    def publish(self, message, image=None):
        self.check()
        try:
            self.messages.put_nowait((message, image))
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.stop.set()
        if self.process.pid is not None:
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        for channel in (self.messages, self.ready):
            channel.cancel_join_thread()
            channel.close()
