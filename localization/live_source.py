"""Paced video / continuous camera capture with a single replaceable frame slot."""

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np

from .video import VideoReader, VideoSample


class LatestSlot:
    """One producer, one consumer. Replacing an unread item counts as a drop."""

    def __init__(self):
        self.condition = threading.Condition()
        self.item = None
        self.dropped = 0

    def put(self, item):
        with self.condition:
            if self.item is not None:
                self.dropped += 1
            self.item = item
            self.condition.notify()

    def take(self, timeout=0.):
        with self.condition:
            if self.item is None:
                self.condition.wait(timeout)
            item, self.item = self.item, None
            return item


@dataclass
class CapturedFrame:
    sample: VideoSample
    received_monotonic: float


class LiveSource:
    def __init__(self, *, video=None, camera=None, rotation=None, size=None, fps=None):
        self.video, self.camera = video, camera
        self.rotation, self.size, self.fps = rotation, size, fps
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.metadata = {"kind": "paced_video" if video else "camera",
                         "clock": "scheduled_pts" if video else "host_receive_monotonic",
                         "captured_frames": 0}
        self.thread = threading.Thread(target=self._run, name="frame-capture", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.)
        self.metadata["capture_thread_stopped"] = not self.thread.is_alive()

    def _publish(self, sample, received):
        self.metadata["captured_frames"] += 1
        self.slot.put(CapturedFrame(sample, received))

    def _run(self):
        try:
            if self.video:
                self._replay()
            else:
                self._camera()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.done.set()

    def _replay(self):
        with VideoReader(self.video, self.rotation) as reader:
            self.metadata["video"] = reader.metadata
            origin = None
            for sample in reader:
                if origin is None:
                    origin = time.monotonic() - sample.timestamp
                due = origin + sample.timestamp
                if self.stop_event.wait(max(0., due - time.monotonic())):
                    break
                # Scheduled presentation, not decoding completion, is the replay
                # capture clock. Decoder delays therefore cannot hide frame age.
                self._publish(sample, due)

    def _camera(self):
        identifier = int(self.camera) if str(self.camera).isdigit() else str(self.camera)
        if isinstance(identifier, int) or identifier.startswith("/dev/video"):
            capture = cv2.VideoCapture(identifier)
        else:
            capture = cv2.VideoCapture(identifier, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000])
        try:
            if not capture.isOpened():
                raise RuntimeError("Cannot open camera / stream")
            # Backend may ignore buffer size; continuous reads plus LatestSlot
            # bound our application queue, not a device's internal latency.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            if self.size:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            if self.fps:
                capture.set(cv2.CAP_PROP_FPS, self.fps)
            self.metadata.update(backend=capture.getBackendName(),
                                 width=capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                 height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                 fps=capture.get(cv2.CAP_PROP_FPS))
            origin = None
            index = 0
            while not self.stop_event.is_set():
                ok, image = capture.read()
                received = time.monotonic()
                if not ok or image is None:
                    raise RuntimeError("Camera disconnected or stream ended")
                if origin is None:
                    origin = received
                rotation = self.rotation or 0
                if rotation:
                    image = np.ascontiguousarray(np.rot90(image, rotation // 90))
                sample = VideoSample(index, received - origin, None, "", "host_receive",
                                     image, rotation)
                self._publish(sample, received)
                index += 1
        finally:
            capture.release()
