"""Frame-by-frame map localization with optical flow and global reacquisition."""

from dataclasses import dataclass

import cv2
import numpy as np

from .pose import estimate_pose, poses_agree


@dataclass(frozen=True)
class TrackerConfig:
    independent: bool = False
    refresh_interval: int = 10
    global_interval: int = 60
    flow_fb_px: float = 1.0


class MapTracker:
    def __init__(self, database, pose_config, config):
        self.database = database
        self.pose_config = pose_config
        self.config = config
        self.previous_gray = None
        self.previous_pose = None
        self.previous_K = None
        self.ever_localized = False
        self.frames = 0

    def _flow(self, gray, K):
        if (self.previous_pose is None or self.previous_gray.shape != gray.shape
                or not np.allclose(K, self.previous_K)):
            return None
        previous = self.previous_pose
        points = previous.uv.astype(np.float32).reshape(-1, 1, 2)
        params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        forward, ok1, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None, **params)
        if forward is None:
            return None
        backward, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, forward, None, **params)
        if backward is None:
            return None
        pixels = forward.reshape(-1, 2)
        error = np.linalg.norm(backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
        h, w = gray.shape
        keep = ((ok1.ravel() != 0) & (ok2.ravel() != 0) & (error < self.config.flow_fb_px)
                & np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
        pose = estimate_pose(previous.xyz[keep], pixels[keep], K, gray.shape, self.pose_config,
                             reference_ids=previous.reference_ids[keep])
        if pose is not None:
            pose.method = "optical_flow"
        return pose

    def process(self, gray, K):
        was_lost = self.previous_pose is None
        pose = None
        reason = "ok"
        if not self.config.independent:
            pose = self._flow(gray, K)
        global_due = self.frames % self.config.global_interval == 0
        refresh_due = self.frames % self.config.refresh_interval == 0
        if self.config.independent or was_lost or global_due:
            # Global verification is authoritative: an ambiguous place must not
            # silently inherit a potentially incorrect optical-flow pose.
            pose, reason = self.database.localize(gray, K, self.pose_config)
        elif pose is None or refresh_due:
            nearby = self.database.nearby(self.previous_pose)
            matched, reason = self.database.localize(gray, K, self.pose_config, nearby)
            if matched is not None and (pose is None or poses_agree(pose, matched)):
                pose = matched
                reason = "ok"
            elif matched is not None or pose is None or reason == "ambiguous_place":
                pose, reason = self.database.localize(gray, K, self.pose_config)
            else:
                reason = "ok"  # a valid geometric flow estimate remains available
        self.frames += 1
        if pose is None:
            status = "lost"
        else:
            status = "relocalized" if was_lost and self.ever_localized else "localized"
            self.ever_localized = True
        self.previous_gray = gray.copy()
        self.previous_pose = pose
        self.previous_K = K.copy()
        return pose, status, reason
