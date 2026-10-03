"""Real-time contracts: no historical poses, bounded buffers, loss and expiry."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from localization.live_matcher import MatchRequest, MatchResult
from localization.live_output import LiveWriter
from localization.live_source import LatestSlot, LiveSource
from localization.map import intrinsic_matrix
from localization.pose import PoseConfig, estimate_pose
from localization.realtime import LiveConfig, LiveTracker, pose_message
from localization.video import VideoReader, VideoSample


class FakeMatcher:
    info = {"reference_ids": [0]}

    def __init__(self):
        self.busy = False
        self.result = None
        self.requests = []

    def submit(self, request):
        if self.busy:
            return False
        self.requests.append(request)
        self.busy = True
        return True

    def poll(self):
        result, self.result = self.result, None
        if result is not None:
            self.busy = False
        return result


class RealtimeTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        cv2.setRNGSeed(42)
        rng = np.random.default_rng(73)
        self.gray = cv2.GaussianBlur(rng.integers(0, 256, (240, 320), np.uint8), (3, 3), .6)
        uv = cv2.goodFeaturesToTrack(self.gray, 150, .01, 12).reshape(-1, 2)
        self.K = intrinsic_matrix([280., 280., 160., 120.])
        xyz = np.c_[(uv - [160., 120.]) / 280. * 3., np.full(len(uv), 3.)]
        self.pose = estimate_pose(xyz, uv, self.K, self.gray.shape, PoseConfig(),
                                  reference_ids=np.zeros(len(uv), dtype=int))
        self.assertIsNotNone(self.pose)
        self.matcher = FakeMatcher()
        self.now = 10.
        self.tracker = LiveTracker(self.matcher, PoseConfig(), clock=lambda: self.now)

    def step(self, index, image=None):
        self.now = 10 + index / 30
        return self.tracker.process_frame(index, self.gray if image is None else image, self.K, self.now)

    def match(self, pose=None, reason="ok"):
        self.matcher.result = MatchResult(self.matcher.requests[-1], pose or self.pose, reason, 200.)

    def test_latest_slot_drops_history_instead_of_backlog(self):
        slot = LatestSlot()
        for i in range(100):
            slot.put(i)
        self.assertEqual(slot.take(), 99)
        self.assertIsNone(slot.take())
        self.assertEqual(slot.dropped, 99)

    def test_delayed_pose_is_propagated_to_current_frame(self):
        self.assertIsNone(self.step(0)[0])
        self.assertIsNone(self.step(1)[0])
        self.match()
        moved = cv2.warpAffine(self.gray, np.float32([[1, 0, 5], [0, 1, 0]]), (320, 240))
        pose, status, _ = self.step(2, moved)
        self.assertEqual(status, "localized")
        np.testing.assert_allclose(pose.center, [-5 * 3 / 280, 0, 0], atol=.008)
        self.assertEqual(self.tracker.state, "TRACKING")
        self.assertEqual(len(self.matcher.requests), 1)

    def test_result_from_before_stall_is_rejected(self):
        self.step(0)
        self.match()
        self.tracker.invalidate()
        pose, status, _ = self.step(1)
        self.assertIsNone(pose)
        self.assertEqual(status, "lost")
        self.assertEqual(self.tracker.stats["expired_results"], 1)
        self.assertEqual(len(self.matcher.requests), 2)

    def test_result_with_missing_anchor_is_rejected(self):
        self.step(0)
        self.match()
        self.matcher.result.request = replace(self.matcher.result.request, frame_id=-7)
        self.assertIsNone(self.step(1)[0])
        self.assertEqual(self.tracker.stats["expired_results"], 1)

    def test_lost_frame_then_global_recovery(self):
        self.step(0)
        self.match()
        self.assertIsNotNone(self.step(1)[0])
        self.assertIsNone(self.step(2, np.zeros_like(self.gray))[0])
        self.assertIsNone(self.matcher.requests[-1].reference_ids)
        # This black-frame job fails; a fresh query is submitted on frame 3.
        self.matcher.result = MatchResult(self.matcher.requests[-1], None, "no_features", 10)
        self.assertIsNone(self.step(3)[0])
        self.match()
        self.assertEqual(self.step(4)[1], "relocalized")

    def test_expired_frame_and_packet_never_contain_current_pose(self):
        self.step(0)
        self.match()
        self.step(1)
        self.now += 1
        pose, _, reason = self.tracker.process_frame(2, self.gray, self.K, self.now - .5)
        self.assertIsNone(pose)
        self.assertEqual(reason, "frame_too_old")
        sample = VideoSample(2, .1, None, "", "host_receive", None, 0)
        message = pose_message(sample, self.pose, "TRACKING", "ok", 10, 11, 1, .15)
        self.assertFalse(message["valid"])
        self.assertIsNone(message["position"])
        json.dumps(message, allow_nan=False)

    def test_out_of_order_frames_are_rejected(self):
        self.step(1)
        with self.assertRaisesRegex(ValueError, "Frame IDs"):
            self.step(0)

    def test_history_and_pending_matching_are_bounded(self):
        for i in range(100):
            self.step(i)
        self.assertLessEqual(len(self.tracker.history), self.tracker.live_config.history_frames)
        self.assertLessEqual(self.tracker.history[-1].received - self.tracker.history[0].received, 1.)
        self.assertEqual(len(self.matcher.requests), 1)  # slow worker never builds a queue

    def test_catchup_over_budget_is_not_published(self):
        self.step(0)
        self.match()
        self.now += .033
        original = self.tracker._flow
        def slow_flow(*args):
            self.now += .03
            return self.pose
        # Initial tracker flow is empty; the catch-up LK call costs 30 ms.
        with patch("localization.realtime.MapTracker._flow", side_effect=slow_flow):
            with patch.object(self.tracker, "_flow", wraps=original):
                pose, _, _ = self.tracker.process_frame(1, self.gray, self.K, self.now)
        self.assertIsNone(pose)
        self.assertEqual(self.tracker.stats["catchup_failures"], 1)

    def test_conflicting_map_result_invalidates_instead_of_snapping(self):
        self.step(0)
        self.match()
        self.step(1)
        for index in range(2, 11):
            self.step(index)  # local refresh, with uninterrupted tracking
        conflicting = replace(self.pose, xyz=self.pose.xyz + [4., 0, 0],
                              tvec=self.pose.tvec - np.array([[4.], [0.], [0.]]))
        self.match(conflicting)
        pose, _, reason = self.step(11)
        self.assertIsNone(pose)
        self.assertEqual(reason, "map_pose_conflict")
        self.assertIsNone(self.matcher.requests[-1].reference_ids)

    def test_tracking_requires_periodic_map_verification(self):
        self.tracker.live_config = replace(LiveConfig(), verification_seconds=.05)
        self.step(0)
        self.match()
        self.step(1)
        self.step(2)
        pose, _, reason = self.step(3)
        self.assertIsNone(pose)
        self.assertEqual(reason, "map_verification_expired")

    def test_capture_camera_failure_is_visible_and_released(self):
        class Camera:
            released = False
            def isOpened(self): return True
            def set(self, *args): return False
            def get(self, *args): return 30
            def getBackendName(self): return "test"
            def read(self): return False, None
            def release(self): self.released = True
        camera = Camera()
        with patch("localization.live_source.cv2.VideoCapture", return_value=camera):
            source = LiveSource(camera="0")
            source.start()
            self.assertTrue(source.done.wait(2))
            source.close()
        self.assertIn("disconnected", source.error)
        self.assertTrue(camera.released)

    def test_paced_capture_waits_for_source_pts(self):
        gray = self.gray
        class Reader:
            metadata = {}
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                for i, t in enumerate([0., .08, .17]):
                    yield VideoSample(i, t, i, "1/30", "pts", gray, 0)
        with patch("localization.live_source.VideoReader", Reader):
            source = LiveSource(video="fake.mp4")
            started = time.monotonic()
            source.start()
            self.assertTrue(source.done.wait(2))
            duration = time.monotonic() - started
            source.close()
        self.assertGreaterEqual(duration, .16)
        self.assertEqual(source.slot.take().sample.index, 2)
        self.assertEqual(source.slot.dropped, 2)

    def test_h264_encoder_side_data_does_not_break_video_reader(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "side_data.mp4"
            with av.open(str(path), "w") as output:
                stream = output.add_stream("libx264", rate=30)
                stream.width, stream.height = 320, 240
                stream.pix_fmt = "yuv420p"
                frame = av.VideoFrame.from_ndarray(cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR), format="bgr24")
                for packet in stream.encode(frame):
                    output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            with VideoReader(path) as reader:
                frames = list(reader)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].rotation, 0)
            self.assertEqual(frames[0].image.shape, (240, 320, 3))

    def test_async_logging_preserves_csv_and_terminal_invalidity(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = LiveWriter(directory, {})
            sample = VideoSample(4, .1, 3, "1/30", "pts", self.gray, 0)
            packet = pose_message(sample, self.pose, "TRACKING", "ok", 10, 10.01, 10, .15)
            writer.append(sample, self.pose, "localized", "ok", packet)
            writer.append(None, None, "lost", "eof", dict(packet, valid=False, state="STOPPED", position=None))
            report = writer.close(True, None, {"dropped_frames": 3})
            self.assertEqual(report["processed_frames"], 1)
            latest = json.loads((Path(directory) / "latest_pose.json").read_text())
            self.assertFalse(latest["valid"])
            self.assertEqual(len((Path(directory) / "poses.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((Path(directory) / "trajectory.csv").read_text().splitlines()), 2)

    def test_runner_stalled_camera_expires_pose_and_records_error(self):
        import live_localize
        pose, gray = self.pose, self.gray
        class Matcher(FakeMatcher):
            def __init__(self, *args):
                super().__init__()
                self.process = SimpleNamespace(is_alive=lambda: True)
            def start(self):
                return dict(self.info, map_sha256="fixture", keyframes=1)
            def poll(self):
                if not self.busy:
                    return None
                self.busy = False
                return MatchResult(self.requests[-1], pose, "ok", 1.)
            def close(self): pass
        source = LiveSource(camera="0")
        def capture_then_stall():
            origin = time.monotonic()
            for i in range(3):
                source.stop_event.wait(max(0, origin + i * .03 - time.monotonic()))
                received = time.monotonic()
                sample = VideoSample(i, received - origin, None, "", "host_receive",
                                     cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0)
                source._publish(sample, received)
            source.stop_event.wait(2)
        source._camera = capture_then_stall
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calib = root / "calib.txt"
            calib.write_text("280 280 160 120\n")
            args = live_localize.parse_args(["--map", "fixture.pth", "--camera", "0",
                "--calib", str(calib), "--output", str(root / "out"), "--pixels", "76800",
                "--max-frame-age", ".08", "--source-timeout", ".2"])
            with patch("live_localize.BackgroundMatcher", Matcher), patch("live_localize.LiveSource", return_value=source):
                with self.assertRaisesRegex(RuntimeError, "source timeout"):
                    live_localize.run(args)
            packets = [json.loads(line) for line in (root / "out" / "poses.jsonl").read_text().splitlines()]
            self.assertTrue(any(p["valid"] for p in packets))
            self.assertTrue(any(p["state"] == "STALE" for p in packets))
            self.assertEqual(packets[-1]["state"], "ERROR")
            self.assertFalse(packets[-1]["valid"])
            self.assertIsNone(packets[-1]["position"])
            self.assertFalse(json.loads((root / "out" / "report.json").read_text())["completed"])
            self.assertFalse(source.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
