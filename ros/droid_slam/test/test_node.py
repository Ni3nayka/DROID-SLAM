"""Real ROS graph + real map matcher: pose, loss, stale stream, recovery and TF."""
from array import array
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ros/droid_slam'))
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from droid_slam.node import LocalizationNode, qos


class RosNodeTests(unittest.TestCase):
    def test_real_graph_stamps_metric_pose_loss_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rng = np.random.default_rng(88)
            image = cv2.GaussianBlur(rng.integers(0, 256, (240, 320, 3), np.uint8), (3, 3), .6)
            K = [280., 0., 160., 0., 285., 120., 0., 0., 1.]
            blob = {'images': torch.from_numpy(image.transpose(2, 0, 1)[None].copy()),
                    'disps': torch.full((1, 240, 320), .5),
                    # A nonzero camera centre verifies inversion AND metric scaling.
                    'poses': torch.tensor([[-.25, .1, -.4, 0., 0., 0., 1.]]),
                    'intrinsics': torch.tensor([[280., 285., 160., 120.]]) / 8,
                    'tstamps': torch.tensor([0.])}
            torch.save(blob, path / 'map.pth')
            arguments = ['--ros-args', '-p', f'map_path:={path / "map.pth"}',
                '-p', 'show_gui:=false', '-p', 'show_preview:=false', '-p', 'pixels:=76800',
                '-p', 'image_topic:=/fixture/image', '-p', 'camera_info_topic:=/fixture/info',
                '-p', 'qos_reliability:=best_effort',
                '-p', 'scale_to_meters:=2.0', '-p', 'publish_tf:=true', '-p', f'cache_dir:={path / "cache"}']
            rclpy.init(args=arguments, domain_id=79)
            localizer = producer = executor = None
            packets, poses, transforms = [], [], []
            sent = set()
            try:
                localizer = LocalizationNode()
                producer = Node('fixture')
                images = producer.create_publisher(Image, '/fixture/image', qos())
                infos = producer.create_publisher(CameraInfo, '/fixture/info', qos())
                producer.create_subscription(String, '/droid_slam/status', lambda m: packets.append(json.loads(m.data)), 10)
                producer.create_subscription(PoseStamped, '/droid_slam/pose', poses.append, 10)
                producer.create_subscription(TFMessage, '/tf', transforms.append, 10)
                executor = SingleThreadedExecutor()
                executor.add_node(localizer)
                executor.add_node(producer)
                info = CameraInfo(width=320, height=240, distortion_model='plumb_bob')
                info.header.frame_id = 'fixture_optical'
                info.k = K
                info.d = [0.] * 5
                info.r = np.eye(3).ravel().tolist()
                info.p = np.c_[np.asarray(K).reshape(3, 3), np.zeros(3)].ravel().tolist()

                def pump(seconds, color=None, until=None):
                    end = time.monotonic() + seconds
                    next_frame = 0.
                    while time.monotonic() < end:
                        now = time.monotonic()
                        if color is not None and now >= next_frame:
                            next_frame = now + 1 / 30
                            stamp = producer.get_clock().now().to_msg()
                            sent.add((stamp.sec, stamp.nanosec))
                            info.header.stamp = stamp
                            msg = Image(height=240, width=320, encoding='bgr8', step=960,
                                        data=array('B', color.tobytes()))
                            msg.header = info.header
                            infos.publish(info)
                            images.publish(msg)
                        executor.spin_once(timeout_sec=.003)
                        if until and until():
                            return
                    if until:
                        self.fail(f'Condition timed out; error={localizer.error}; latest={packets[-1:]!r}')

                pump(12, image, until=lambda: bool(poses and transforms))
                self.assertTrue(any(p['valid'] for p in packets))
                valid_packet = next(p for p in reversed(packets) if p['valid'])
                first = poses[-1]
                self.assertIn((first.header.stamp.sec, first.header.stamp.nanosec), sent)
                self.assertEqual(first.header.frame_id, 'droid_map')
                np.testing.assert_allclose([first.pose.position.x, first.pose.position.y, first.pose.position.z],
                                           [.5, -.2, .8], atol=.02)
                transform = transforms[-1].transforms[0]
                self.assertEqual(transform.child_frame_id, 'droid_camera_optical_frame')
                self.assertEqual(transform.header, first.header)
                np.testing.assert_allclose([transform.transform.translation.x, transform.transform.translation.y,
                                           transform.transform.translation.z], [.5, -.2, .8], atol=.02)
                packets.clear()
                pump(.4, np.zeros_like(image))
                self.assertTrue(any(p['state'] == 'LOST' and not p['valid'] for p in packets))
                packets.clear()
                pump(4, image, until=lambda: any(p['valid'] for p in packets))
                packets.clear()
                pump(.5)
                self.assertTrue(any(p['state'] == 'STALE' and not p['valid'] for p in packets))
                pose_count = len(poses)
                pump(.2)
                self.assertEqual(len(poses), pose_count)
                # A computation that was valid before queuing must not become
                # a fresh ROS pose when publication happens after its deadline.
                localizer._send(dict(valid_packet, valid_until_monotonic=time.monotonic() - 1))
                self.assertEqual(localizer.last_packet['reason'], 'publisher_deadline')
                pump(.1)
                self.assertEqual(len(poses), pose_count)
                packets.clear()
                pump(4, image, until=lambda: any(p['valid'] for p in packets))
                self.assertIsNone(localizer.error)
            finally:
                if localizer:
                    localizer.close()
                if executor:
                    executor.shutdown()
                if localizer:
                    localizer.destroy_node()
                if producer:
                    producer.destroy_node()
                rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
