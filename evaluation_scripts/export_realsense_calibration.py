#!/usr/bin/env python3
"""Read (never modify) factory RGB intrinsics of a connected RealSense."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', required=True, help='Device serial or ASIC serial; both are recorded')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--output', type=Path, required=True, help='ROS camera-calibration YAML (JSON syntax is valid YAML)')
    args = p.parse_args()
    import pyrealsense2 as rs
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        asic = device.get_info(rs.camera_info.asic_serial_number) if device.supports(rs.camera_info.asic_serial_number) else ''
        if args.serial not in (serial, asic):
            continue
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color or profile.format() != rs.format.rgb8:
                    continue
                video = profile.as_video_stream_profile()
                if (video.width(), video.height(), video.fps()) != (args.width, args.height, args.fps):
                    continue
                i = video.get_intrinsics()
                if i.model not in (rs.distortion.none, rs.distortion.brown_conrady) and any(i.coeffs):
                    raise ValueError(f'Cannot label nonzero {i.model} as OpenCV plumb_bob distortion')
                data = {'image_width': i.width, 'image_height': i.height, 'camera_name': f'd435_{serial}_color',
                    'distortion_model': 'plumb_bob',
                    'camera_matrix': {'rows': 3, 'cols': 3, 'data': [i.fx, 0., i.ppx, 0., i.fy, i.ppy, 0., 0., 1.]},
                    'distortion_coefficients': {'rows': 1, 'cols': 5, 'data': list(i.coeffs)},
                    'rectification_matrix': {'rows': 3, 'cols': 3, 'data': [1., 0., 0., 0., 1., 0., 0., 0., 1.]},
                    'projection_matrix': {'rows': 3, 'cols': 4, 'data': [i.fx, 0., i.ppx, 0., 0., i.fy, i.ppy, 0., 0., 0., 1., 0.]},
                    'device_serial': serial, 'asic_serial': asic, 'sdk_distortion_model': str(i.model),
                    'firmware_version': device.get_info(rs.camera_info.firmware_version),
                    'fps': args.fps, 'source': 'Factory calibration read from connected device via librealsense',
                    'captured_utc': datetime.now(timezone.utc).isoformat()}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(data, indent=2) + '\n')
                print(f'Saved {args.output}; use serial_no:={serial} in the ROS driver')
                return
        raise ValueError('Requested RGB resolution/FPS is not available')
    raise ValueError(f'Camera {args.serial} is not connected')


if __name__ == '__main__':
    main()
