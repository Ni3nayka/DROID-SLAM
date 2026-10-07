#!/usr/bin/env python3
"""Physical D435 integration: factory CameraInfo, GUI, driver outage and recovery.

Source both workspaces first. This owns and stops only the two processes it
launches. No other driver should be holding the camera during this test.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from std_msgs.msg import String


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--serial', default='948122071094')
    p.add_argument('--calib', type=Path, help='Also exercise explicit calibration-file mode instead of CameraInfo')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--domain-id', type=int, default=58)
    p.add_argument('--gui', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain_id)
    rclpy.init(args=[])
    monitor = rclpy.create_node('droid_camera_validation')
    packets = []
    monitor.create_subscription(String, '/droid_slam/status', lambda m: packets.append(json.loads(m.data)), 10)
    driver = localizer = None
    driver_command = ['ros2', 'launch', 'realsense', 'driver.launch.py', 'model:=d435', f'serial_no:={args.serial}']
    driver_log = (args.output / 'driver.log').open('w')
    node_log = (args.output / 'node.log').open('w')
    result = {'completed': False}

    def launch_driver():
        return subprocess.Popen(driver_command, stdout=driver_log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_for(predicate, timeout=35):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(monitor, timeout_sec=.02)
            if localizer.poll() is not None:
                raise RuntimeError(f'Localizer exited with {localizer.returncode}; see node.log')
            if predicate():
                return
        raise RuntimeError(f'Test condition timed out. Last state: {packets[-1:]!r}')

    def frame_stamps():
        return {p['ros_stamp_ns'] for p in packets if 'ros_stamp_ns' in p and 'received_monotonic' in p}

    try:
        driver = launch_driver()
        command = ['ros2', 'launch', 'droid_slam', 'localization.launch.py', f'map_path:={args.map.resolve()}',
                   f'output_dir:={args.output.resolve() / "localization"}',
                   f'save_calibration_path:={args.output.resolve() / "camera_info.yaml"}',
                   f'show_gui:={str(args.gui).lower()}', f'show_preview:={str(args.gui).lower()}']
        if args.calib:
            command.append(f'calib_path:={args.calib.resolve()}')
        localizer = subprocess.Popen(command, stdout=node_log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_for(lambda: len(frame_stamps()) >= 90)
        result['initial_distinct_frames'] = len(frame_stamps())
        result['states_before_outage'] = sorted({p['state'] for p in packets})
        assert not any(p['state'] in ('INVALID_INPUT', 'ERROR') for p in packets)
        packets.clear()
        stop(driver)
        driver = None
        wait_for(lambda: any(p['state'] == 'STALE' and not p['valid'] and p['position'] is None for p in packets), 5)
        result['stale_on_driver_stop'] = True
        packets.clear()
        driver = launch_driver()
        wait_for(lambda: len(frame_stamps()) >= 90)
        result['recovered_distinct_frames'] = len(frame_stamps())
        assert not any(p['state'] in ('INVALID_INPUT', 'ERROR') for p in packets)
        result['valid_poses_after_reconnect'] = sum(p['valid'] for p in packets)
        # Send a process-group SIGINT as a real terminal does, including the
        # additional SIGINT that launch forwards to the localization process.
        stop(localizer)
        report = json.loads((args.output / 'localization/ros_report.json').read_text())
        assert report['error'] is None, report
        metadata = json.loads((args.output / 'localization/metadata.json').read_text())
        result['calibration_source'] = metadata['calibration']['source']
        if args.calib:
            assert result['calibration_source'] == str(args.calib.resolve())
        else:
            assert (args.output / 'camera_info.yaml').is_file()
        node_log.flush()
        assert 'Traceback' not in (args.output / 'node.log').read_text()
        result['clean_shutdown'] = True
        result['processed_frames'] = report['processed_frames']
        result['stream_capture_drops'] = report['stream_capture_drops']
        result['completed'] = True
    finally:
        stop(localizer)
        stop(driver)
        driver_log.close()
        node_log.close()
        monitor.destroy_node()
        rclpy.shutdown()
        (args.output / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
