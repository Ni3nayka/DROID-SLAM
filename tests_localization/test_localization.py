"""Run with: python -m unittest discover -s tests_localization -v"""

import csv
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from localization.export import ResultWriter
from localization.features import FeatureConfig, FeatureDatabase
from localization.map import ReferenceMap, intrinsic_matrix
from localization.pose import PoseConfig, estimate_pose
from localization.tracker import MapTracker, TrackerConfig
from localization.video import Preprocessor, VideoReader, VideoSample
from view_localization import trajectory_segments


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        cv2.setNumThreads(2)
        cv2.setRNGSeed(42)
        self.rng = np.random.default_rng(17)

    def make_map(self):
        image = self.rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), .6)
        K = np.array([280., 285., 160., 120.])
        rotation = Rotation.from_euler("xyz", [.12, -.2, .08])
        pose = np.r_[.3, -.1, .2, rotation.as_quat()]
        blob = {"images": torch.from_numpy(image.transpose(2, 0, 1)[None].copy()),
                "disps": torch.full((1, 240, 320), .5),
                "poses": torch.from_numpy(pose[None]),
                "intrinsics": torch.from_numpy(K[None] / 8), "tstamps": torch.tensor([0.])}
        path = self.path / "map.pth"
        torch.save(blob, path)
        return ReferenceMap.load(path)

    def test_droid_geometry_round_trip_and_center(self):
        m = self.make_map()
        uv = np.array([[30., 40.], [160., 120.], [280., 180.]])
        xyz, valid = m.lift(0, uv)
        rotation = Rotation.from_quat(m.poses[0, 3:])
        projected, _ = cv2.projectPoints(xyz, rotation.as_rotvec(), m.poses[0, :3],
                                        intrinsic_matrix(m.intrinsics[0]), None)
        np.testing.assert_allclose(projected.reshape(-1, 2), uv, atol=1e-8)
        np.testing.assert_allclose(rotation.apply(m.centers[0]) + m.poses[0, :3], 0, atol=1e-10)
        self.assertTrue(valid.all())
        m.disps[0, 40, 30] = np.nan
        _, valid = m.lift(0, np.array([[30., 40.], [-1., 0.], [160., 120.]]))
        self.assertEqual(valid.tolist(), [False, False, True])

    def test_pnp_with_outliers_and_camera_to_world_orientation(self):
        K = intrinsic_matrix([430, 435, 320, 240])
        center = np.array([1.2, -.3, 2.])
        Rwc = Rotation.from_euler("xyz", [.1, -.3, .2]).as_matrix()
        camera_points = self.rng.uniform([-2, -1.5, 4], [2, 1.5, 9], (150, 3))
        xyz = camera_points @ Rwc.T + center
        pixels = camera_points[:, :2] / camera_points[:, 2:] * [430, 435] + [320, 240]
        pixels += self.rng.normal(0, .2, pixels.shape)
        pixels[:35] = self.rng.uniform([0, 0], [640, 480], (35, 2))
        pose = estimate_pose(xyz, pixels, K, (480, 640), PoseConfig())
        self.assertIsNotNone(pose)
        self.assertGreater(len(pose.uv), 100)
        np.testing.assert_allclose(pose.center, center, atol=.015)
        np.testing.assert_allclose(Rotation.from_quat(pose.quaternion).as_matrix(), Rwc, atol=.003)
        self.assertIsNone(estimate_pose(xyz[:8], pixels[:8], K, (480, 640), PoseConfig()))

    def test_reject_degenerate_image_support(self):
        xyz = np.column_stack((np.linspace(-1, 1, 50), np.zeros(50), np.full(50, 5.)))
        uv = xyz[:, :2] / xyz[:, 2:] * 300 + [160, 120]
        self.assertIsNone(estimate_pose(xyz, uv, intrinsic_matrix([300, 300, 160, 120]),
                                       (240, 320), PoseConfig()))

    def test_feature_cache_loss_and_reacquisition(self):
        m = self.make_map()
        config = FeatureConfig(max_features=800)
        database = FeatureDatabase.build(m, config, self.path / "cache")
        cached = FeatureDatabase.build(m, config, self.path / "cache")
        np.testing.assert_array_equal(database.references[0].xyz, cached.references[0].xyz)
        tracker = MapTracker(cached, PoseConfig(), TrackerConfig())
        gray = cv2.cvtColor(m.images[0], cv2.COLOR_BGR2GRAY)
        K = intrinsic_matrix(m.intrinsics[0])
        first, status, _ = tracker.process(gray, K)
        self.assertEqual(status, "localized")
        np.testing.assert_allclose(first.center, m.centers[0], atol=.01)
        lost, status, _ = tracker.process(np.zeros_like(gray), K)
        self.assertIsNone(lost)
        self.assertEqual(status, "lost")
        recovered, status, _ = tracker.process(gray, K)
        self.assertIsNotNone(recovered)
        self.assertEqual(status, "relocalized")

    def test_repeated_appearance_at_different_places_is_rejected(self):
        from localization.features import ReferenceFeatures
        m = self.make_map()
        original = FeatureDatabase.build(m, FeatureConfig(max_features=800))
        first = original.references[0]
        second = ReferenceFeatures(1, first.uv.copy(), first.xyz + [8., 0., 0.], first.descriptors.copy())
        ambiguous = FeatureDatabase([first, second], original.config)
        pose, reason = ambiguous.localize(cv2.cvtColor(m.images[0], cv2.COLOR_BGR2GRAY),
                                         intrinsic_matrix(m.intrinsics[0]), PoseConfig())
        self.assertIsNone(pose)
        self.assertEqual(reason, "ambiguous_place")

    def test_map_rejects_low_resolution_depth_export(self):
        m = self.make_map()
        blob = torch.load(m.path, weights_only=True)
        blob["disps"] = blob["disps"][:, ::8, ::8]
        torch.save(blob, m.path)
        with self.assertRaisesRegex(ValueError, "disps"):
            ReferenceMap.load(m.path)

    def test_flow_follows_known_image_translation(self):
        m = self.make_map()
        database = FeatureDatabase.build(m, FeatureConfig(max_features=800))
        tracker = MapTracker(database, PoseConfig(), TrackerConfig())
        gray = cv2.cvtColor(m.images[0], cv2.COLOR_BGR2GRAY)
        K = intrinsic_matrix(m.intrinsics[0])
        first, status, _ = tracker.process(gray, K)
        self.assertEqual(status, "localized")
        shifted = cv2.warpAffine(gray, np.array([[1., 0., 3.], [0., 1., 0.]]), gray.shape[::-1])
        second, status, _ = tracker.process(shifted, K)
        self.assertEqual(status, "localized")
        self.assertEqual(second.method, "optical_flow")
        expected = first.center + first.rotation.T @ np.array([-3 * 2 / 280, 0, 0])
        np.testing.assert_allclose(second.center, expected, atol=.005)

    def test_preprocessing_matches_demo_calibration_and_crop(self):
        calib = self.path / "calib.txt"
        np.savetxt(calib, [1018., 911., 641., 372., .13, -.43, .008, .005, .49])
        image = self.rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        pre = Preprocessor(calib)
        processed, K = pre.process(image)
        full_K = intrinsic_matrix([1018, 911, 641, 372])
        expected = cv2.undistort(image, full_K, pre.values[4:])
        expected = cv2.resize(expected, (591, 332))[:328, :584]
        self.assertLess(np.abs(processed.astype(float) - expected).mean(), .1)
        np.testing.assert_allclose(K[0], full_K[0] * 591 / 1280)
        np.testing.assert_allclose(K[1], full_K[1] * 332 / 720)
        self.assertEqual(processed.shape, (328, 584, 3))

    def make_video(self):
        import av
        path = self.path / "variable_rate.mkv"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("ffv1", rate=30)
            stream.width, stream.height = 80, 64
            stream.pix_fmt = "bgr0"
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            for i, pts in enumerate([100, 133, 200, 267]):
                frame = av.VideoFrame.from_ndarray(np.full((64, 80, 3), i * 40, np.uint8), format="bgr24")
                frame.pts = pts
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        return path

    def test_video_preserves_variable_pts(self):
        path = self.make_video()
        with VideoReader(path) as reader:
            frames = list(reader)
            self.assertTrue(reader.metadata["reached_end"])
        self.assertEqual([f.index for f in frames], [0, 1, 2, 3])
        np.testing.assert_allclose([f.timestamp for f in frames], [0, .033, .1, .167], atol=1e-6)
        self.assertEqual({f.timestamp_source for f in frames}, {"pts"})
        with VideoReader(path, rotation=90) as reader:
            rotated = next(iter(reader))
        self.assertEqual(rotated.image.shape, (80, 64, 3))

    def test_export_preserves_lost_rows_and_trajectory_gaps(self):
        writer = ResultWriter(self.path / "result", {})
        for i in range(3):
            writer.append(VideoSample(i, i * .04, i, "1/25", "pts", None, 0), None, "lost", "no_features")
        report = writer.close(True)
        with (self.path / "result/trajectory.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3)
        self.assertTrue(np.isnan(float(rows[0]["x"])))
        self.assertEqual(report["lost_intervals"][0]["end_frame"], 2)
        self.assertTrue(json.loads((self.path / "result/metadata.json").read_text())["completed"])
        point = dict(x="0", y="0", z="0", status="localized")
        points, edges = trajectory_segments([point, point, rows[0], point])
        self.assertEqual(len(points), 3)
        self.assertEqual(edges.tolist(), [[0, 1]])


if __name__ == "__main__":
    unittest.main()
