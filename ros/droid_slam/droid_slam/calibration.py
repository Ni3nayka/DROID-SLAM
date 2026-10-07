"""Colour-camera calibration from CameraInfo, ROS YAML, or the legacy text file."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml

from localization.video import Preprocessor


@dataclass(frozen=True)
class Calibration:
    values: tuple
    width: int
    height: int
    frame_id: str = ''
    source: str = ''

    def preprocessor(self, pixels):
        return Preprocessor.from_values(self.values, (self.width, self.height), pixels)

    def check_image(self, image, frame_id):
        if image.shape[:2] != (self.height, self.width):
            raise ValueError(f'Calibration is {self.width}x{self.height}, image is '
                             f'{image.shape[1]}x{image.shape[0]}; use CameraInfo for this stream profile')
        if self.frame_id and self.frame_id != frame_id:
            raise ValueError(f'CameraInfo frame {self.frame_id!r} differs from image frame {frame_id!r}')


def _calibration(K, D, width, height, model, frame_id, source):
    K = np.asarray(K, dtype=float).reshape(3, 3)
    D = np.asarray(D, dtype=float).reshape(-1)
    if width <= 0 or height <= 0 or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError('Invalid calibration dimensions or coefficients')
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose([K[0, 1], K[1, 0]], 0)):
        raise ValueError('Camera is uncalibrated or has unsupported skew')
    if model not in ('plumb_bob', 'rational_polynomial', 'none', ''):
        raise ValueError(f'Unsupported distortion model {model!r}; use a rectified colour stream')
    if len(D) not in (0, 4, 5, 8, 12, 14):
        raise ValueError('Unsupported distortion coefficient count')
    values = tuple([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D])
    return Calibration(values, int(width), int(height), frame_id, source)


def from_camera_info(info, rectified=False, source='CameraInfo'):
    if info.binning_x not in (0, 1) or info.binning_y not in (0, 1):
        raise ValueError('CameraInfo binning is unsupported; publish calibration for the actual image')
    roi = info.roi
    if roi.x_offset or roi.y_offset or roi.width not in (0, info.width) or roi.height not in (0, info.height):
        raise ValueError('CameraInfo crop/ROI needs adjusted calibration')
    if rectified:
        P = np.asarray(info.p).reshape(3, 4)
        R = np.asarray(info.r).reshape(3, 3)
        if not np.allclose(R, np.eye(3)) or not np.allclose(P[:, 3], 0):
            raise ValueError('Nonidentity stereo rectification is unsupported for this colour adapter')
        return _calibration(P[:, :3], [], info.width, info.height, 'none', info.header.frame_id, source)
    return _calibration(info.k, info.d, info.width, info.height, info.distortion_model,
                        info.header.frame_id, source)


def from_file(path, width=0, height=0, rectified=False):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in ('.yaml', '.yml'):
        data = yaml.safe_load(path.read_text())
        from sensor_msgs.msg import CameraInfo
        info = CameraInfo()
        info.width, info.height = int(data['image_width']), int(data['image_height'])
        info.distortion_model = data.get('distortion_model', 'plumb_bob')
        info.k = list(map(float, data['camera_matrix']['data']))
        info.d = list(map(float, data['distortion_coefficients']['data']))
        info.r = list(map(float, data.get('rectification_matrix', {'data': np.eye(3).ravel()})['data']))
        default_p = np.c_[np.asarray(info.k).reshape(3, 3), np.zeros(3)].ravel()
        info.p = list(map(float, data.get('projection_matrix', {'data': default_p})['data']))
        return from_camera_info(info, rectified, str(path))
    if width <= 0 or height <= 0:
        raise ValueError('Text calibration needs calib_width and calib_height')
    values = np.loadtxt(path).reshape(-1)
    Preprocessor.from_values(values, (width, height))  # same validation as offline mode
    if rectified and len(values) > 4 and np.any(values[4:] != 0):
        raise ValueError('Rectified images need their rectified K, not a raw-camera text calibration')
    return Calibration(tuple(values), width, height, source=str(path))


def save_camera_info(path, info):
    """Snapshot received factory calibration, including its optical frame and timestamp."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'image_width': info.width, 'image_height': info.height, 'camera_name': info.header.frame_id,
            'distortion_model': info.distortion_model,
            'camera_matrix': {'rows': 3, 'cols': 3, 'data': list(map(float, info.k))},
            'distortion_coefficients': {'rows': 1, 'cols': len(info.d), 'data': list(map(float, info.d))},
            'rectification_matrix': {'rows': 3, 'cols': 3, 'data': list(map(float, info.r))},
            'projection_matrix': {'rows': 3, 'cols': 4, 'data': list(map(float, info.p))},
            'source': 'ROS CameraInfo', 'frame_id': info.header.frame_id,
            'ros_stamp': {'sec': info.header.stamp.sec, 'nanosec': info.header.stamp.nanosec}}
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(yaml.safe_dump(data, sort_keys=False))
    tmp.replace(path)
