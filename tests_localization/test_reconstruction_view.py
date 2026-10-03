"""Regression checks against the original reconstruction visualization pipeline."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from localization.map import ReferenceMap
from reconstruction_view import build_reconstruction, configure_rendering
from view_localization import build_geometry


@unittest.skipUnless(torch.cuda.is_available(), "Original reconstruction rendering requires CUDA")
class ReconstructionViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "map.pth"
        n, h, w = 9, 96, 128
        rng = np.random.default_rng(5)
        poses = torch.zeros(n, 7)
        poses[:, 0] = torch.arange(n) * .025
        poses[:, 6] = 1
        disps = torch.full((n, h, w), .5)
        # This unconfirmed distant patch passes a single-frame depth-range mask.
        # A viewer must reject it by comparing its geometry to other views.
        disps[4, 25:65, 35:85] = .13
        self.blob = {"images": torch.tensor(rng.integers(0, 256, (n, 3, h, w), dtype=np.uint8)),
                     "disps": disps, "poses": poses,
                     "intrinsics": torch.tensor([100., 100., 64., 48.]).repeat(n, 1) / 8,
                     "tstamps": torch.arange(n).float()}
        torch.save(self.blob, self.path)

    def original_geometry(self):
        """Frozen behavior of view_reconstruction.py before the shared builder."""
        import droid_backends
        from lietorch import SE3

        images = self.blob["images"].cuda()[..., ::2, ::2]
        disps = self.blob["disps"].cuda()[..., ::2, ::2].contiguous()
        poses = self.blob["poses"].cuda()
        intrinsics = 4 * self.blob["intrinsics"].cuda()
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0],
            torch.arange(len(images), device="cuda"), .005 * torch.ones_like(disps.mean(dim=[1, 2])))
        mask = (counts >= 3) & (disps > .25 * disps.mean())
        colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.
        return points[mask].cpu().numpy(), colors[mask].cpu().numpy(), SE3(poses).inv().matrix().cpu().numpy()

    def test_both_viewers_match_original_points_and_colors(self):
        expected_points, expected_colors, expected_poses = self.original_geometry()
        self.assertGreater(len(expected_points), 1000)
        cloud, poses = build_reconstruction(self.path)
        np.testing.assert_array_equal(np.asarray(cloud.points), expected_points)
        np.testing.assert_array_equal(np.asarray(cloud.colors), expected_colors)
        np.testing.assert_array_equal(poses, expected_poses)
        localized_cloud, _, _ = build_geometry(ReferenceMap.load(self.path), [])
        np.testing.assert_array_equal(np.asarray(localized_cloud.points), expected_points)
        np.testing.assert_array_equal(np.asarray(localized_cloud.colors), expected_colors)

    def test_unconfirmed_far_points_do_not_expand_scene_bounds(self):
        reference_map = ReferenceMap.load(self.path)
        pixels = np.array([[60., 40.]])
        self.assertTrue(reference_map.feature_depth_mask(4, pixels)[0])
        raw, valid = reference_map.lift(4, pixels)
        self.assertTrue(valid[0])
        self.assertGreater(raw[0, 2], 7.)
        cloud, _, _ = build_geometry(reference_map, [])
        np.testing.assert_allclose(np.asarray(cloud.points)[:, 2], 2., atol=1e-6)

    def test_export_scale_applies_to_cloud_and_reference_trajectory(self):
        reference_map = ReferenceMap.load(self.path)
        cloud, _, _ = build_geometry(reference_map, [])
        scaled, old, new = build_geometry(reference_map, [], scale=2.5)
        np.testing.assert_allclose(np.asarray(scaled.points), np.asarray(cloud.points) * 2.5)
        np.testing.assert_allclose(np.asarray(old.points), reference_map.centers * 2.5)
        self.assertEqual(len(new.points), 0)

    def test_original_render_options_load_successfully(self):
        from types import SimpleNamespace
        import open3d as o3d

        options = o3d.visualization.RenderOption()
        options.background_color = [0, 0, 0]
        options.point_size = 8
        configure_rendering(SimpleNamespace(get_render_option=lambda: options))
        np.testing.assert_array_equal(options.background_color, [1, 1, 1])
        self.assertEqual(options.point_size, 2)


if __name__ == "__main__":
    unittest.main()
