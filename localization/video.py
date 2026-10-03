"""Video decoding with presentation timestamps and calibrated preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .map import intrinsic_matrix


@dataclass
class VideoSample:
    index: int
    timestamp: float
    pts: int | None
    time_base: str
    timestamp_source: str
    image: np.ndarray
    rotation: int


class VideoReader:
    def __init__(self, path, rotation=None, allow_fps_fallback=False):
        import av
        self.path = Path(path)
        self.rotation = rotation
        self.allow_fps_fallback = allow_fps_fallback
        self.container = av.open(str(path))
        if not self.container.streams.video:
            self.container.close()
            raise ValueError("Input has no video stream")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.metadata = {
            "path": str(self.path.resolve()), "width": self.stream.width,
            "height": self.stream.height, "declared_frames": self.stream.frames,
            "average_rate": str(self.stream.average_rate),
            "time_base": str(self.stream.time_base), "decoded_frames": 0,
            "fallback_timestamps": 0, "rotation_override": rotation,
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.container.close()

    def __iter__(self):
        import av
        origin = None
        previous = None
        rate = float(self.stream.average_rate) if self.stream.average_rate else 0
        try:
            for index, frame in enumerate(self.container.decode(self.stream)):
                if frame.is_corrupt:
                    raise ValueError(f"Corrupt decoded video frame {index}")
                if frame.pts is not None and frame.time_base:
                    time = float(frame.pts * frame.time_base)
                    if origin is None:
                        origin = time - index / rate if index and rate > 0 else time
                        self.metadata["first_pts_seconds"] = origin
                    timestamp = time - origin
                    source = "pts"
                elif self.allow_fps_fallback and rate > 0:
                    timestamp = index / rate
                    source = "fps_fallback"
                    self.metadata["fallback_timestamps"] += 1
                else:
                    raise ValueError(f"Frame {index} has no PTS; explicit --allow-fps-fallback is required")
                if previous is not None and timestamp < previous:
                    raise ValueError(f"Non-monotonic timestamp at frame {index}: {timestamp} < {previous}")
                previous = timestamp
                rotation = self.rotation if self.rotation is not None else frame.rotation
                if rotation % 90:
                    raise ValueError("Only quarter-turn display rotations are supported")
                if self.rotation is None:
                    for side in frame.side_data:
                        if "DISPLAYMATRIX" in str(side.type):
                            matrix = np.frombuffer(bytes(side), dtype=np.int32).reshape(3, 3)
                            if np.linalg.det(matrix[:2, :2].astype(float)) < 0:
                                raise ValueError("Mirrored display matrix is unsupported; normalize the video first")
                image = frame.to_ndarray(format="bgr24")
                if rotation % 360:
                    image = np.ascontiguousarray(np.rot90(image, (rotation // 90) % 4))
                self.metadata["decoded_frames"] = index + 1
                self.metadata["display_rotation"] = rotation
                self.metadata["display_size"] = [image.shape[1], image.shape[0]]
                yield VideoSample(index, timestamp, frame.pts, str(frame.time_base), source, image, rotation)
        except av.error.FFmpegError as exc:
            raise RuntimeError(f"Video decoding failed after {self.metadata['decoded_frames']} frames: {exc}") from exc
        declared = self.metadata["declared_frames"]
        if declared and declared != self.metadata["decoded_frames"]:
            raise ValueError(f"Decoded {self.metadata['decoded_frames']} frames, container declares {declared}")
        self.metadata["reached_end"] = True


class Preprocessor:
    def __init__(self, calib_path, calibration_size=None, pixels=384 * 512):
        self.values = np.loadtxt(calib_path).reshape(-1).astype(np.float64)
        if (len(self.values) not in (4, 8, 9, 12, 16, 18)
                or not np.isfinite(self.values).all() or np.any(self.values[:2] <= 0)):
            raise ValueError("Calibration must contain fx fy cx cy and 0/4/5/8/12/14 distortion coefficients")
        self.calibration_size = calibration_size
        self.pixels = pixels
        self._shape = None
        self.metadata = {"values": self.values.tolist(), "calibration_size": calibration_size,
                         "target_pixels": pixels, "calibration_orientation": "displayed image"}

    def process(self, image):
        h, w = image.shape[:2]
        if self._shape != (h, w):
            K = intrinsic_matrix(self.values[:4])
            if self.calibration_size is not None:
                cw, ch = self.calibration_size
                K[0] *= w / cw
                K[1] *= h / ch
            self.remap = None
            if len(self.values) > 4:
                self.remap = cv2.initUndistortRectifyMap(K, self.values[4:], None, K, (w, h), cv2.CV_32FC1)
            scale = np.sqrt(self.pixels / (h * w))
            self.resized = (int(w * scale), int(h * scale))
            if min(self.resized) < 8:
                raise ValueError("Working image is too small")
            self.K = K.copy()
            self.K[0] *= self.resized[0] / w
            self.K[1] *= self.resized[1] / h
            self._shape = (h, w)
            self.metadata.update(input_size=[w, h], resized_size=list(self.resized),
                                 working_size=[v - v % 8 for v in self.resized], K=self.K.tolist())
        if self.remap is not None:
            image = cv2.remap(image, *self.remap, interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, self.resized, interpolation=cv2.INTER_LINEAR)
        rh, rw = image.shape[:2]
        image = np.ascontiguousarray(image[:rh - rh % 8, :rw - rw % 8])
        return image, self.K.copy()
