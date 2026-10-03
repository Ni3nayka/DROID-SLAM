"""DROID map geometry, without importing the CUDA SLAM backend."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def file_digest(path):
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intrinsic_matrix(values):
    fx, fy, cx, cy = values
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class ReferenceMap:
    images: np.ndarray  # N,H,W,3, BGR
    disps: np.ndarray  # inverse depth at image resolution
    poses: np.ndarray  # world -> camera; tx,ty,tz,qx,qy,qz,qw
    intrinsics: np.ndarray  # full image resolution
    timestamps: np.ndarray
    path: Path
    digest: str

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        blob = torch.load(path, map_location="cpu", weights_only=True)
        required = ("images", "disps", "poses", "intrinsics", "tstamps")
        if not isinstance(blob, dict) or any(k not in blob for k in required):
            raise ValueError("Expected a DROID reconstruction, not network weights")
        arrays = {}
        for key in required:
            if not isinstance(blob[key], torch.Tensor):
                raise ValueError(f"Map field {key} must be a tensor")
            arrays[key] = blob[key].numpy()
        images = arrays["images"]
        if images.ndim != 4 or images.shape[1] != 3 or images.dtype != np.uint8:
            raise ValueError("Map images must be uint8 N x 3 x H x W")
        n, _, h, w = images.shape
        if n == 0 or min(h, w) < 8:
            raise ValueError("Map has no usable images")
        shapes = {"disps": (n, h, w), "poses": (n, 7),
                  "intrinsics": (n, 4), "tstamps": (n,)}
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"Map {key}: expected {shape}, got {arrays[key].shape}")
        poses = arrays["poses"].astype(np.float64)
        intrinsics = arrays["intrinsics"].astype(np.float64) * 8.0
        if not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-8):
            raise ValueError("Invalid map poses")
        if not np.isfinite(intrinsics).all() or np.any(intrinsics[:, :2] <= 0):
            raise ValueError("Invalid map intrinsics")
        if not np.isfinite(arrays["tstamps"]).all():
            raise ValueError("Invalid map timestamps")
        poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
        return cls(images.transpose(0, 2, 3, 1), arrays["disps"], poses,
                   intrinsics, arrays["tstamps"], path, file_digest(path))

    @property
    def centers(self):
        rotations = Rotation.from_quat(self.poses[:, 3:]).as_matrix()
        return -np.einsum("nji,nj->ni", rotations, self.poses[:, :3])

    def lift(self, index, pixels):
        """Return world points and a validity mask, retaining input indexing."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        h, w = self.disps.shape[1:]
        finite = np.isfinite(pixels).all(axis=1)
        safe = np.where(np.isfinite(pixels), pixels, 0)
        ix = np.rint(safe[:, 0]).astype(int).clip(0, w - 1)
        iy = np.rint(safe[:, 1]).astype(int).clip(0, h - 1)
        d = self.disps[index, iy, ix]
        valid = (finite & (pixels[:, 0] >= 0) & (pixels[:, 0] <= w - 1)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] <= h - 1)
                 & np.isfinite(d) & (d > 0))
        fx, fy, cx, cy = self.intrinsics[index]
        rays = np.column_stack(((safe[:, 0] - cx) / fx,
                                (safe[:, 1] - cy) / fy, np.ones(len(pixels))))
        camera = rays / np.where(valid, d, 1)[:, None]
        rotation = Rotation.from_quat(self.poses[index, 3:]).as_matrix()
        world = (camera - self.poses[index, :3]) @ rotation
        world[~valid] = np.nan
        return world, valid

    def feature_depth_mask(self, index, pixels):
        """Reject far-depth floors and depth discontinuities, relative to map scale."""
        disp = self.disps[index]
        finite = np.isfinite(disp) & (disp > 0)
        if not finite.any():
            return np.zeros(len(pixels), dtype=bool)
        median = np.median(disp[finite])
        safe = np.where(finite, disp, 0).astype(np.float32)
        low = cv2.erode(safe, np.ones((3, 3), np.uint8))
        high = cv2.dilate(safe, np.ones((3, 3), np.uint8))
        x = np.rint(pixels[:, 0]).astype(int).clip(0, disp.shape[1] - 1)
        y = np.rint(pixels[:, 1]).astype(int).clip(0, disp.shape[0] - 1)
        d = safe[y, x]
        return ((d > median * .03) & (d < median * 30)
                & ((high[y, x] - low[y, x]) < .5 * d))
