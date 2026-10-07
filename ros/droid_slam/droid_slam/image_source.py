"""Decode ROS colour images without cv_bridge's NumPy 1.x binary dependency."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class RosFrame:
    index: int
    message: object
    received: float
    ros_received_ns: int


def stamp_ns(header):
    return int(header.stamp.sec) * 1000000000 + int(header.stamp.nanosec)


def decode_image(message, transport='raw'):
    if transport == 'compressed':
        if 'compressedDepth' in message.format:
            raise ValueError('Depth transport is not a colour camera image')
        image = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid compressed image')
        return image
    encodings = {'bgr8': (3, None), 'rgb8': (3, cv2.COLOR_RGB2BGR),
                 'bgra8': (4, cv2.COLOR_BGRA2BGR), 'rgba8': (4, cv2.COLOR_RGBA2BGR),
                 'mono8': (1, cv2.COLOR_GRAY2BGR)}
    if message.encoding not in encodings:
        raise ValueError(f'Unsupported colour encoding {message.encoding!r}; publish bgr8/rgb8/mono8')
    channels, conversion = encodings[message.encoding]
    h, w, step = message.height, message.width, message.step
    if h <= 0 or w <= 0 or step < w * channels or len(message.data) != h * step:
        raise ValueError('Invalid Image dimensions, row stride or buffer length')
    # All supported components are bytes: is_bigendian has no effect. Explicit
    # strides preserve row padding; the result owns its buffer after return.
    shape = (h, w) if channels == 1 else (h, w, channels)
    strides = (step, 1) if channels == 1 else (step, channels, 1)
    image = np.ndarray(shape, np.uint8, buffer=message.data, strides=strides)
    return image.copy() if conversion is None else cv2.cvtColor(image, conversion)


def frame_deadline(frame, max_age, check_header_age=True, allow_zero_stamp=False):
    """Keep ROS time separate from the monotonic watchdog; return expiry on host clock."""
    stamp = stamp_ns(frame.message.header)
    if stamp == 0 and not allow_zero_stamp:
        raise ValueError('Image header.stamp is zero; camera driver must timestamp images')
    deadline = frame.received + max_age
    if check_header_age:
        age = (frame.ros_received_ns - stamp) / 1e9
        if age < -.05:
            raise ValueError('Image timestamp is in the future; check ROS clocks/use_sim_time')
        if age > max_age:
            raise ValueError('Image arrived with an expired ROS timestamp')
        deadline -= max(0., age)
    return deadline
