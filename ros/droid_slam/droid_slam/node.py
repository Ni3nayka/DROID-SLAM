"""ROS 2 node: callbacks receive data; one worker owns tracking and GUI processes."""

from .bootstrap import prepare
prepare()

from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from localization.features import FeatureConfig
from localization.pose import PoseConfig
from localization.live_matcher import BackgroundMatcher
from localization.live_source import LatestSlot
from localization.live_view import LiveViewer
from localization.live_output import LiveWriter
from localization.realtime import LiveConfig, LiveTracker, pose_message
from localization.video import VideoSample
from .calibration import from_camera_info, from_file, save_camera_info
from .image_source import RosFrame, decode_image, frame_deadline, stamp_ns


DEFAULTS = dict(map_path='', calib_path='', calib_width=0, calib_height=0,
    image_topic='/camera/color/image_raw', camera_info_topic='/camera/color/camera_info',
    image_transport='raw', input_rectified=False, save_calibration_path='',
    show_gui=True, show_preview=True, screenshot_path='', output_dir='', overwrite=False,
    qos_reliability='reliable', check_header_age=True, pixels=384 * 512,
    features=1800, min_inliers=20, threads=2, matcher_threads=2,
    cache_dir='', map_frame='droid_map', camera_frame='droid_camera_optical_frame',
    scale_to_meters=0., publish_tf=False, max_frames=0, duration=0.)
DEFAULTS.update(asdict(LiveConfig()))


def qos(reliability='best_effort'):
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.BEST_EFFORT if reliability == 'best_effort' else ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)


class LocalizationNode(Node):
    def __init__(self):
        super().__init__('droid_slam')
        self.options = {name: self.declare_parameter(name, default,
            ParameterDescriptor(read_only=True)).value for name, default in DEFAULTS.items()}
        p = self.options
        if not p['map_path'] or not Path(p['map_path']).expanduser().is_file():
            raise ValueError('map_path must point to an existing DROID reconstruction .pth')
        if p['image_transport'] not in ('raw', 'compressed') or p['qos_reliability'] not in ('best_effort', 'reliable'):
            raise ValueError('Invalid image_transport or qos_reliability')
        for name in ('pixels', 'features', 'min_inliers', 'threads', 'matcher_threads', *LiveConfig.__dataclass_fields__):
            if not np.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if p['features'] < p['min_inliers'] or p['min_inliers'] < 6:
            raise ValueError('Require features >= min_inliers >= 6')
        for name in ('duration', 'max_frames', 'scale_to_meters'):
            if not np.isfinite(p[name]) or p[name] < 0:
                raise ValueError(f'{name} must be nonnegative')
        if p['publish_tf'] and p['scale_to_meters'] <= 0:
            raise ValueError('TF requires a known positive scale_to_meters; monocular map units are not meters')
        self.fixed_calibration = from_file(p['calib_path'], p['calib_width'], p['calib_height'],
                                          p['input_rectified']) if p['calib_path'] else None
        if self.fixed_calibration is None and not p['camera_info_topic']:
            raise ValueError('Provide calib_path or camera_info_topic')
        if p['output_dir'] and not p['overwrite'] and any((Path(p['output_dir']).expanduser() / name).exists()
                for name in ('trajectory.csv', 'metadata.json', 'poses.jsonl')):
            raise ValueError('Output exists; select another output_dir or set overwrite:=true')
        self.frames, self.results = LatestSlot(), LatestSlot()
        self.info_lock = threading.Lock()
        self.camera_info = None
        self.received = 0
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.report = None
        self.state = 'STARTING'
        self.last_packet = None
        self.last_log = 0.
        self.publish_ages, self.image_publish_ages = deque(maxlen=10000), deque(maxlen=10000)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.diagnostic_pub = self.create_publisher(DiagnosticArray, '~/diagnostics', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '~/pose', 10) if p['scale_to_meters'] > 0 else None
        self.tf = TransformBroadcaster(self) if p['publish_tf'] else None
        msg_type = Image if p['image_transport'] == 'raw' else CompressedImage
        self.image_sub = self.create_subscription(msg_type, p['image_topic'], self._image, qos(p['qos_reliability']))
        self.info_sub = None if self.fixed_calibration else self.create_subscription(
            CameraInfo, p['camera_info_topic'], self._info, qos(p['qos_reliability']))
        # Watchdogs must continue while /clock is paused in a rosbag.
        self.timer = self.create_timer(.01, self._drain, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.worker = threading.Thread(target=self._work, name='ros-localizer', daemon=True)
        self.worker.start()
        self.get_logger().info(f"Images: {p['image_topic']}; calibration: {p['calib_path'] or p['camera_info_topic']}")

    def _image(self, message):
        if self.stop.is_set():
            return
        frame = RosFrame(self.received, message, time.monotonic(), self.get_clock().now().nanoseconds)
        self.received += 1
        self.frames.put(frame)

    def _info(self, message):
        with self.info_lock:
            self.camera_info = message

    def _send(self, packet):
        now = time.monotonic()
        packet = dict(packet)
        if packet['valid'] and now > packet['valid_until_monotonic']:
            packet.update(valid=False, position=None, quaternion_xyzw=None, state='STALE', reason='publisher_deadline')
        if packet['valid']:
            self.publish_ages.append((now - packet['received_monotonic']) * 1000)
            if packet.get('image_to_pose_ms') is not None:
                packet['image_to_publish_ms'] = packet['image_to_pose_ms'] + (now - packet['published_monotonic']) * 1000
                self.image_publish_ages.append(packet['image_to_publish_ms'])
        packet.update(published_monotonic=now, capture_drops=self.frames.dropped,
                      publication_drops=self.results.dropped)
        self.last_packet = packet
        self.state = packet['state']
        self.status_pub.publish(String(data=json.dumps(packet, allow_nan=False)))
        if packet['valid'] and self.pose_pub is not None:
            stamped = PoseStamped()
            stamped.header.frame_id = self.options['map_frame']
            stamped.header.stamp.sec, stamped.header.stamp.nanosec = divmod(packet['ros_stamp_ns'], 1000000000)
            scaled = np.asarray(packet['position']) * self.options['scale_to_meters']
            stamped.pose.position.x, stamped.pose.position.y, stamped.pose.position.z = map(float, scaled)
            q = stamped.pose.orientation
            q.x, q.y, q.z, q.w = packet['quaternion_xyzw']
            self.pose_pub.publish(stamped)
            if self.tf is not None:
                transform = TransformStamped(header=stamped.header, child_frame_id=self.options['camera_frame'])
                transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, scaled)
                transform.transform.rotation = q
                self.tf.sendTransform(transform)
        if now - self.last_log >= 1 or packet['state'] == 'ERROR':
            d = DiagnosticStatus(name='droid_slam/localization', hardware_id='camera')
            d.level = DiagnosticStatus.OK if packet['valid'] else (DiagnosticStatus.ERROR if packet['state'] == 'ERROR' else DiagnosticStatus.WARN)
            d.message = packet['state'] + ': ' + packet['reason']
            d.values = [KeyValue(key=k, value=str(packet.get(k))) for k in
                        ('frame_index', 'inliers', 'frame_age_ms', 'capture_drops', 'calibration_source', 'image_frame')]
            diagnostics = DiagnosticArray(status=[d])
            diagnostics.header.stamp = self.get_clock().now().to_msg()
            self.diagnostic_pub.publish(diagnostics)
            self.get_logger().info(f"{d.message}; frame={packet.get('frame_index')}; drops={self.frames.dropped}")
            self.last_log = now

    def _drain(self):
        packet = self.results.take()
        if packet is not None:
            self._send(packet)
        elif self.last_packet and self.last_packet['valid'] and time.monotonic() > self.last_packet['valid_until_monotonic']:
            self._send(dict(self.last_packet, valid=False, position=None, quaternion_xyzw=None,
                            state='STALE', reason='no_fresh_pose'))

    def _work(self):
        p = self.options
        matcher = viewer = writer = None
        tracker = None
        processed = valid_count = rejected = 0
        started = time.monotonic()
        streaming_started = None
        startup_drops = 0
        compute, ages = deque(maxlen=10000), deque(maxlen=10000)
        last_packet = None
        last_stamp = None
        last_image_frame = None
        segment = 0
        metadata = None

        def emit(packet, image=None, sample=None, pose=None, status='lost'):
            nonlocal last_packet
            last_packet = packet
            self.results.put(packet)
            if viewer and not self.stop.is_set():
                viewer.publish(packet, image if p['show_preview'] else None)
            if writer:
                writer.append(sample, pose, status, packet['reason'], packet)

        def invalid(state, reason, frame=None):
            nonlocal segment
            segment += 1
            packet = dict(last_packet or {}, schema_version=1, state=state, reason=reason, valid=False,
                          position=None, quaternion_xyzw=None, inliers=0, method='none',
                          valid_until_monotonic=time.monotonic(), published_monotonic=time.monotonic(),
                          units='map_units', track_segment=segment)
            if frame:
                packet.update(frame_index=frame.index, ros_stamp_ns=stamp_ns(frame.message.header),
                              image_frame=frame.message.header.frame_id)
            emit(packet)

        try:
            cv2.setNumThreads(p['threads'])
            cv2.setRNGSeed(42)
            config = LiveConfig(**{name: p[name] for name in LiveConfig.__dataclass_fields__})
            matcher = BackgroundMatcher(Path(p['map_path']).expanduser(), FeatureConfig(max_features=p['features']),
                PoseConfig(min_inliers=p['min_inliers']), Path(p['cache_dir']).expanduser() if p['cache_dir'] else prepare() / '.cache/localization',
                p['matcher_threads'])
            info = matcher.start()
            tracker = LiveTracker(matcher, PoseConfig(min_inliers=p['min_inliers']), config)
            if p['show_gui'] or p['show_preview']:
                screenshot = Path(p['screenshot_path']).expanduser() if p['screenshot_path'] else None
                if screenshot:
                    screenshot.parent.mkdir(parents=True, exist_ok=True)
                viewer = LiveViewer(Path(p['map_path']).expanduser(), p['show_gui'], p['show_preview'], screenshot)
                viewer.start()
            metadata = {'mode': 'ros2', 'map': str(Path(p['map_path']).expanduser().resolve()),
                        'map_sha256': info['map_sha256'], 'input': p['image_topic'],
                        'source': {'kind': 'camera', 'transport': 'ros2'}, 'ros_parameters': p,
                        'map_timestamps_are_video_times': False}
            if p['output_dir']:
                writer = LiveWriter(Path(p['output_dir']).expanduser(), metadata)
            calibration = preprocessor = None
            streaming_started = time.monotonic()
            startup_drops = self.frames.dropped
            invalid('WAITING_IMAGE', 'waiting_for_camera')
            while not self.stop.is_set():
                if viewer:
                    viewer.check()
                    if viewer.stop.is_set():
                        break
                if writer:
                    writer.check()
                if p['duration'] and time.monotonic() - streaming_started >= p['duration']:
                    break
                if not matcher.process.is_alive():
                    raise RuntimeError('Map matcher exited')
                frame = self.frames.take(.01)
                if frame is None:
                    if last_packet and last_packet['state'] not in ('STALE', 'WAITING_IMAGE') and time.monotonic() > last_packet['valid_until_monotonic']:
                        tracker.invalidate()
                        invalid('STALE', 'no_fresh_image')
                    continue
                begin = time.monotonic()
                try:
                    deadline = frame_deadline(frame, config.max_frame_age, p['check_header_age'],
                                              self.get_parameter('use_sim_time').value)
                    if begin > deadline:
                        raise ValueError('Frame expired in input queue')
                    image = decode_image(frame.message, p['image_transport'])
                    with self.info_lock:
                        ci = self.camera_info
                    current = self.fixed_calibration or (from_camera_info(ci, p['input_rectified'], p['camera_info_topic']) if ci else None)
                    if current is None:
                        tracker.invalidate()
                        invalid('WAITING_CALIBRATION', 'waiting_for_color_CameraInfo', frame)
                        continue
                    current.check_image(image, frame.message.header.frame_id)
                    if current != calibration:
                        tracker.invalidate('INITIALIZING')
                        calibration = current
                        preprocessor = calibration.preprocessor(p['pixels'])
                        metadata['calibration'] = asdict(calibration)
                        if ci and not self.fixed_calibration and p['save_calibration_path']:
                            save_camera_info(p['save_calibration_path'], ci)
                    stamp = stamp_ns(frame.message.header)
                    if ((last_stamp is not None and stamp <= last_stamp) or
                            (last_image_frame is not None and last_image_frame != frame.message.header.frame_id)):
                        tracker.invalidate()
                        segment += 1
                    last_stamp, last_image_frame = stamp, frame.message.header.frame_id
                    working, K = preprocessor.process(image)
                except (ValueError, cv2.error) as exc:
                    rejected += 1
                    tracker.invalidate()
                    invalid('INVALID_INPUT', str(exc), frame)
                    continue
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
                pose, status, reason = tracker.process_frame(frame.index, gray, K, frame.received)
                now = time.monotonic()
                sample = VideoSample(frame.index, stamp / 1e9, stamp, '1/1000000000', 'ros_header', None, 0)
                if now > deadline:
                    tracker.invalidate()
                    pose, status, reason = None, 'lost', 'frame_deadline_exceeded'
                packet = pose_message(sample, pose, tracker.state, reason, frame.received, now,
                                      (now - begin) * 1000, config.max_frame_age)
                if not packet['valid'] or not last_packet or not last_packet['valid']:
                    segment += 1
                packet.update(ros_stamp_ns=stamp, image_frame=frame.message.header.frame_id,
                              calibration_source=calibration.source, track_segment=segment,
                              valid_until_monotonic=deadline,
                              image_to_pose_ms=max(0., (frame.ros_received_ns - stamp) / 1e6) +
                              (now - frame.received) * 1000 if p['check_header_age'] else None)
                emit(packet, working, sample, pose, status)
                processed += 1
                valid_count += int(packet['valid'])
                compute.append(packet['processing_ms'])
                ages.append(packet['frame_age_ms'])
                if p['max_frames'] and processed >= p['max_frames']:
                    break
            invalid('STOPPED', 'shutdown')
        except BaseException as exc:
            # Ctrl-C can reach the child processes before this thread observes
            # the stop event. Their intentional exit is not a tracking failure.
            self.error = None if self.stop.is_set() else f'{type(exc).__name__}: {exc}'
            try:
                invalid('ERROR' if self.error else 'STOPPED', self.error or 'shutdown')
            except Exception:
                self.results.put(dict(schema_version=1, state='ERROR', reason=self.error, valid=False,
                    position=None, quaternion_xyzw=None, valid_until_monotonic=time.monotonic()))
        finally:
            self.stop.set()
            elapsed = time.monotonic() - (streaming_started or started)
            for resource in (viewer, matcher):
                if resource:
                    try:
                        resource.close()
                    except Exception as exc:
                        self.error = self.error or f'Resource shutdown: {exc}'
            self.report = {'processed_frames': processed, 'valid_frames': valid_count, 'rejected_frames': rejected,
                'received_frames': self.received, 'capture_drops': self.frames.dropped,
                'startup_dropped_frames': startup_drops,
                'stream_capture_drops': self.frames.dropped - startup_drops,
                'streaming_seconds': elapsed, 'tracking': tracker.stats if tracker else {},
                'frame_age_p95_ms': float(np.percentile(ages, 95)) if ages else None,
                'compute_p95_ms': float(np.percentile(compute, 95)) if compute else None,
                'error': self.error}
            if writer:
                metadata['time_origin'] = 'ROS image header.stamp epoch (use_sim_time when enabled)'
                try:
                    writer.close(self.error is None, self.error, self.report)
                except Exception as exc:
                    self.error = f'Logger shutdown: {exc}'
            self.report['error'] = self.error
            self.done.set()

    def close(self):
        self.stop.set()
        self.worker.join(timeout=15)
        if self.worker.is_alive():
            self.error = self.error or 'Localization worker did not stop within 15 seconds'
        if self.context.ok():
            self._send(dict(schema_version=1, state='ERROR' if self.error else 'STOPPED',
                            reason=self.error or 'shutdown', valid=False, position=None, quaternion_xyzw=None,
                            valid_until_monotonic=time.monotonic()))
        if self.report is not None:
            self.report['error'] = self.error
            self.report['receive_to_publish_p95_ms'] = float(np.percentile(self.publish_ages, 95)) if self.publish_ages else None
            self.report['image_to_publish_p95_ms'] = float(np.percentile(self.image_publish_ages, 95)) if self.image_publish_ages else None
            if self.options['output_dir']:
                from localization.export import write_json
                write_json(Path(self.options['output_dir']).expanduser() / 'ros_report.json', self.report)


def main(args=None):
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = LocalizationNode()
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=.05)
        node._drain()
        code = 1 if node.error else 0
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception as exc:
        print(f'droid_slam: {type(exc).__name__}: {exc}', flush=True)
        code = 1
    finally:
        # A terminal SIGINT and the one forwarded by ros2 launch can arrive
        # twice. Let bounded cleanup finish; SIGTERM still remains available.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node:
            node.close()
            code = 1 if node.error else code
            if node.report:
                print(json.dumps(node.report, indent=2), flush=True)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
