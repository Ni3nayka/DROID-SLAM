"""Run with the repository venv; ROS message types, no camera/GUI required."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ros/droid_slam'))
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from droid_slam.calibration import from_camera_info, from_file, save_camera_info
from droid_slam.image_source import RosFrame, decode_image, frame_deadline
from localization.video import Preprocessor


class AdapterTests(unittest.TestCase):
    def info(self):
        info = CameraInfo(width=640, height=480, distortion_model='plumb_bob')
        info.header.frame_id = 'rgb_optical'
        info.k = [610., 0., 320., 0., 611., 240., 0., 0., 1.]
        info.d = [.1, -.05, 0., 0., 0.]
        info.r = np.eye(3).ravel().tolist()
        info.p = [600., 0., 315., 0., 0., 602., 236., 0., 0., 0., 1., 0.]
        return info

    def test_rgb_padding_is_not_interpreted_as_pixels(self):
        message = Image(height=2, width=2, encoding='rgb8', step=8,
                        data=bytes([255, 0, 0, 0, 255, 0, 91, 92] * 2))
        image = decode_image(message)
        self.assertEqual(image.tolist(), [[[0, 0, 255], [0, 255, 0]]] * 2)
        message.data[0] = 0
        self.assertEqual(image[0, 0, 2], 255)

    def test_mono_bgra_and_rgba(self):
        for encoding, channels, expected in [('mono8', [70], [70, 70, 70]),
            ('bgra8', [10, 20, 30, 255], [10, 20, 30]), ('rgba8', [10, 20, 30, 255], [30, 20, 10])]:
            message = Image(height=1, width=1, encoding=encoding, step=len(channels), data=bytes(channels))
            self.assertEqual(decode_image(message)[0, 0].tolist(), expected)

    def test_invalid_buffer_and_depth_rejected(self):
        for message in [Image(height=1, width=2, encoding='bgr8', step=5, data=bytes(5)),
                        Image(height=1, width=2, encoding='16UC1', step=4, data=bytes(4))]:
            with self.assertRaises(ValueError): decode_image(message)
        with self.assertRaises(ValueError):
            decode_image(CompressedImage(format='16UC1; compressedDepth png', data=b'x'), 'compressed')

    def test_compressed_color(self):
        image = np.full((10, 20, 3), [20, 90, 150], np.uint8)
        ok, data = cv2.imencode('.png', image)
        self.assertTrue(ok)
        np.testing.assert_array_equal(decode_image(CompressedImage(format='png', data=data.tobytes()), 'compressed'), image)
        with self.assertRaises(ValueError): decode_image(CompressedImage(data=b'bad'), 'compressed')

    def test_raw_info_and_resolution_frame_guards(self):
        calibration = from_camera_info(self.info())
        self.assertEqual(calibration.values[:4], (610, 611, 320, 240))
        calibration.check_image(np.zeros((480, 640, 3)), 'rgb_optical')
        with self.assertRaises(ValueError): calibration.check_image(np.zeros((720, 1280, 3)), 'rgb_optical')
        with self.assertRaises(ValueError): calibration.check_image(np.zeros((480, 640, 3)), 'depth_optical')

    def test_rectified_uses_projection_matrix_without_distortion(self):
        calibration = from_camera_info(self.info(), rectified=True)
        self.assertEqual(calibration.values, (600, 602, 315, 236))

    def test_uncalibrated_fisheye_and_binning_are_rejected(self):
        for change in ('uncalibrated', 'fisheye', 'binning', 'roi'):
            info = self.info()
            if change == 'uncalibrated': info.k[0] = 0
            if change == 'fisheye': info.distortion_model = 'equidistant'
            if change == 'binning': info.binning_x = 2
            if change == 'roi': info.roi.x_offset = 50
            with self.assertRaises(ValueError): from_camera_info(info)

    def test_yaml_roundtrip_and_profile_specific_factory_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera.yaml'
            save_camera_info(path, self.info())
            original, loaded = from_camera_info(self.info()), from_file(path)
            self.assertEqual(original.values, loaded.values)
            self.assertEqual((loaded.width, loaded.height), (640, 480))
        low = from_file(ROOT / 'calib/realsense_d435_948122071094_color_640x480.yaml')
        high = from_file(ROOT / 'calib/realsense_d435_948122071094_color_1280x720.yaml')
        self.assertNotAlmostEqual(high.values[0] / low.values[0], 2.)  # different sensor crop/aspect

    def test_memory_calibration_matches_old_file_preprocessing(self):
        path = ROOT / 'calib/sasung_cam_calibrated.txt'
        old = Preprocessor(path, [1280, 720])
        new = Preprocessor.from_values(np.loadtxt(path), [1280, 720])
        image = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), np.uint8)
        a, ka = old.process(image)
        b, kb = new.process(image)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ka, kb)

    def test_ros_clock_is_not_compared_with_monotonic_clock(self):
        message = Image()
        message.header.stamp.sec = 1000000000
        frame = RosFrame(0, message, 20., 1000000000 * 1000000000 + 30000000)
        self.assertAlmostEqual(frame_deadline(frame, .15), 20.12)
        with self.assertRaises(ValueError): frame_deadline(RosFrame(0, message, 20., frame.ros_received_ns + 1000000000), .15)
        with self.assertRaises(ValueError): frame_deadline(RosFrame(0, message, 20., frame.ros_received_ns - 1000000000), .15)

    def test_zero_stamp_requires_explicit_sim_time(self):
        frame = RosFrame(0, Image(), 20., 0)
        with self.assertRaises(ValueError): frame_deadline(frame, .15)
        self.assertEqual(frame_deadline(frame, .15, allow_zero_stamp=True), 20.15)


if __name__ == '__main__':
    unittest.main()
