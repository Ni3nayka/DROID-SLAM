"""Robust pose estimates and explicit world/camera transform conventions."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseConfig:
    min_inliers: int = 20
    min_inlier_ratio: float = .3
    ransac_px: float = 3.0
    max_median_px: float = 2.0
    min_coverage: float = .015
    iterations: int = 2000


@dataclass
class PoseEstimate:
    rvec: np.ndarray
    tvec: np.ndarray
    xyz: np.ndarray
    uv: np.ndarray
    query_ids: np.ndarray
    reference_ids: np.ndarray
    matches: int
    median_error: float
    coverage: float
    method: str = "matching"

    @property
    def rotation(self):
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self):
        return (-self.rotation.T @ self.tvec).reshape(3)

    @property
    def quaternion(self):
        # Output orientation maps camera vectors to the map/world frame.
        return Rotation.from_matrix(self.rotation.T).as_quat()


def project(xyz, rvec, tvec, K):
    uv = cv2.projectPoints(np.asarray(xyz, np.float64), rvec, tvec, K, None)[0]
    rotation = cv2.Rodrigues(rvec)[0]
    depths = (xyz @ rotation.T + tvec.reshape(3))[:, 2]
    return uv.reshape(-1, 2), depths


def estimate_pose(xyz, uv, K, image_shape, config, query_ids=None, reference_ids=None):
    xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
    uv = np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 2)
    n = len(xyz)
    if n < max(6, config.min_inliers) or len(uv) != n:
        return None
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        return None
    query_ids = np.arange(n) if query_ids is None else np.asarray(query_ids)
    reference_ids = np.zeros(n, dtype=int) if reference_ids is None else np.asarray(reference_ids)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            xyz, uv, K, None, iterationsCount=config.iterations,
            reprojectionError=config.ransac_px, confidence=.999,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < config.min_inliers:
            return None
        keep = inliers.ravel()
        for _ in range(2):
            rvec, tvec = cv2.solvePnPRefineLM(xyz[keep], uv[keep], K, None, rvec, tvec)
            projected, depths = project(xyz, rvec, tvec, K)
            errors = np.linalg.norm(projected - uv, axis=1)
            keep = np.flatnonzero((errors <= config.ransac_px) & (depths > 0))
            if len(keep) < config.min_inliers:
                return None
    except cv2.error:
        return None
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    if len(keep) / n < config.min_inlier_ratio:
        return None
    median = float(np.median(errors[keep]))
    area = cv2.contourArea(cv2.convexHull(uv[keep].astype(np.float32)))
    coverage = area / (image_shape[0] * image_shape[1])
    if median > config.max_median_px or coverage < config.min_coverage:
        return None
    return PoseEstimate(rvec, tvec, xyz[keep], uv[keep], query_ids[keep],
                        reference_ids[keep], n, median, coverage)


def poses_agree(first, second, translation_fraction=.12, angle_degrees=15):
    """Scale-independent check; translation tolerance is relative to scene depth."""
    angle = Rotation.from_matrix(first.rotation @ second.rotation.T).magnitude()
    depth = np.median(np.linalg.norm(first.xyz - first.center, axis=1))
    return (angle < np.deg2rad(angle_degrees)
            and np.linalg.norm(first.center - second.center) < translation_fraction * depth)
