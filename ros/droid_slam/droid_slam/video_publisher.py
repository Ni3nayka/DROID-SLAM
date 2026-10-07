"""Publish an existing video through ROS at its original cadence, for integration tests."""
from .bootstrap import prepare
prepare()

import argparse
from array import array
from pathlib import Path
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from localization.video import VideoReader
from .calibration import from_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--calib-size', nargs=2, type=int, default=[1280, 720])
    p.add_argument('--image-topic', default='/droid_test/image_raw')
    p.add_argument('--camera-info-topic', default='/droid_test/camera_info')
    p.add_argument('--frame-id', default='test_camera_optical_frame')
    p.add_argument('--compressed', action='store_true')
    p.add_argument('--best-effort', action='store_true', help='Opt in to lossy sensor QoS for transport tests')
    p.add_argument('--wait-seconds', type=float, default=15)
    p.add_argument('--start-delay', type=float, default=4., help='Allow map and GUI initialization before replay')
    args = p.parse_args(argv)
    calibration = from_file(args.calib, *args.calib_size)
    rclpy.init(args=[])
    node = Node('droid_video_publisher')
    profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT if args.best_effort else ReliabilityPolicy.RELIABLE)
    images = node.create_publisher(CompressedImage if args.compressed else Image, args.image_topic, profile)
    infos = node.create_publisher(CameraInfo, args.camera_info_topic, profile)
    info = CameraInfo()
    info.header.frame_id = args.frame_id
    info.width, info.height = calibration.width, calibration.height
    fx, fy, cx, cy = calibration.values[:4]
    info.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
    info.d = list(calibration.values[4:])
    info.distortion_model = 'rational_polynomial' if len(info.d) > 5 else 'plumb_bob'
    info.r = np.eye(3).ravel().tolist()
    info.p = np.c_[np.asarray(info.k).reshape(3, 3), np.zeros(3)].ravel().tolist()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while images.get_subscription_count() == 0:
            if time.monotonic() > deadline:
                raise RuntimeError('No image subscriber discovered before timeout')
            rclpy.spin_once(node, timeout_sec=.05)
        time.sleep(args.start_delay)
        with VideoReader(Path(args.video)) as reader:
            origin = None
            ros_origin = None
            for sample in reader:
                if origin is None:
                    origin = time.monotonic()
                    ros_origin = node.get_clock().now().nanoseconds
                target = origin + sample.timestamp
                time.sleep(max(0., target - time.monotonic()))
                image = sample.image
                if image.shape[:2] != (info.height, info.width):
                    raise ValueError('Video dimensions do not match calibration')
                stamp = ros_origin + round(sample.timestamp * 1e9)
                info.header.stamp.sec, info.header.stamp.nanosec = divmod(stamp, 1000000000)
                if args.compressed:
                    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise RuntimeError('JPEG encoding failed')
                    message = CompressedImage(format='bgr8; jpeg compressed bgr8', data=array('B', data.tobytes()))
                else:
                    message = Image(height=info.height, width=info.width, encoding='bgr8', step=info.width * 3,
                                    data=array('B', image.tobytes()))
                message.header = info.header
                infos.publish(info)
                images.publish(message)
                rclpy.spin_once(node, timeout_sec=0)
        print(f'Published {reader.metadata["decoded_frames"]} frames', flush=True)
        if not args.best_effort:
            from rclpy.duration import Duration
            if not images.wait_for_all_acked(Duration(seconds=2)):
                raise RuntimeError('Final image was not acknowledged by the subscriber')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
