"""Causal map localization. Only the calling thread owns tracking state."""

from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from .live_matcher import MatchRequest
from .pose import poses_agree
from .tracker import MapTracker, TrackerConfig


@dataclass(frozen=True)
class LiveConfig:
    refresh_seconds: float = .3
    global_seconds: float = 2.
    history_seconds: float = 1.
    history_frames: int = 45
    max_frame_age: float = .15
    max_tracking_gap: float = .25
    verification_seconds: float = 3.
    catchup_budget_ms: float = 25.


@dataclass
class TrackingFrame:
    frame_id: int
    received: float
    gray: object
    K: object


class LiveTracker(MapTracker):
    """Reuse the tested LK/PnP path; offline MapTracker.process is unchanged."""

    def __init__(self, matcher, pose_config, config=LiveConfig(), clock=time.monotonic):
        super().__init__(None, pose_config, TrackerConfig())
        self.matcher, self.live_config, self.clock = matcher, config, clock
        self.history = deque(maxlen=config.history_frames)
        self.generation = 0
        self.state = "INITIALIZING"
        self.last_received = None
        self.last_frame_id = None
        self.last_request = -float("inf")
        self.last_global = -float("inf")
        self.verified_at = None
        self.force_global = True
        self.available_ids = set(matcher.info["reference_ids"])
        self.stats = {name: 0 for name in ("matching_results", "accepted_matches", "expired_results",
                                         "catchup_failures", "conflicts", "resets")}
        self.match_times = deque(maxlen=10000)

    def invalidate(self, state="STALE"):
        self.previous_pose = None
        self.previous_gray = None
        self.previous_K = None
        self.history.clear()
        self.generation += 1
        self.force_global = True
        self.verified_at = None
        self.state = state
        self.stats["resets"] += 1

    def _propagate(self, result):
        request = result.request
        anchor = next((i for i, f in enumerate(self.history) if f.frame_id == request.frame_id), None)
        if request.generation != self.generation or anchor is None:
            self.stats["expired_results"] += 1
            return None
        frames = list(self.history)[anchor:]
        if self.clock() - frames[0].received > self.live_config.history_seconds:
            self.stats["expired_results"] += 1
            return None
        pose = result.pose
        replay = MapTracker(None, self.pose_config, self.config)
        started = self.clock()
        if len(frames) > 1:
            # Pyramidal LK can usually cover the matcher delay in one step.
            # PnP, positive depth, support area and forward/backward flow checks
            # validate the CURRENT observation; the old pose is never emitted.
            replay.previous_gray, replay.previous_K = frames[0].gray, frames[0].K
            replay.previous_pose = pose
            direct = replay._flow(frames[-1].gray, frames[-1].K)
            if direct is not None:
                direct.method = "map_global" if request.reference_ids is None else "map_local"
                if (self.clock() - started) * 1000 <= self.live_config.catchup_budget_ms:
                    return direct
                self.stats["catchup_failures"] += 1
                return None
        for previous, current in zip(frames, frames[1:]):
            if (self.clock() - started) * 1000 > self.live_config.catchup_budget_ms:
                self.stats["catchup_failures"] += 1
                return None
            replay.previous_gray, replay.previous_K = previous.gray, previous.K
            replay.previous_pose = pose
            pose = replay._flow(current.gray, current.K)
            if pose is None or (self.clock() - started) * 1000 > self.live_config.catchup_budget_ms:
                self.stats["catchup_failures"] += 1
                return None
        pose.method = "map_global" if request.reference_ids is None else "map_local"
        return pose

    def process_frame(self, frame_id, gray, K, received):
        config = self.live_config
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            raise ValueError("Frame IDs must increase monotonically")
        if self.last_received is not None and received < self.last_received:
            raise ValueError("Capture timestamps must not move backwards")
        self.last_frame_id = frame_id
        if self.clock() - received > config.max_frame_age:
            self.invalidate()
            return None, "lost", "frame_too_old"
        if (self.last_received is not None and received - self.last_received > config.max_tracking_gap):
            self.invalidate("LOST")
        self.last_received = received
        was_lost = self.previous_pose is None
        pose = self._flow(gray, K)
        self.history.append(TrackingFrame(frame_id, received, gray, K))
        while self.history and received - self.history[0].received > config.history_seconds:
            self.history.popleft()
        reason = "ok" if pose is not None else "awaiting_map_match"
        result = self.matcher.poll()
        if result is not None:
            self.stats["matching_results"] += 1
            self.match_times.append(result.duration_ms)
            if result.request.generation != self.generation:
                self.stats["expired_results"] += 1
            elif result.pose is not None:
                corrected = self._propagate(result)
                if corrected is not None:
                    if pose is not None and not poses_agree(pose, corrected):
                        # Never silently snap a tracked camera to a conflicting
                        # place. Invalidate and require a fresh GLOBAL query.
                        self.stats["conflicts"] += 1
                        self.invalidate("LOST")
                        self.history.append(TrackingFrame(frame_id, received, gray, K))
                        pose, reason = None, "map_pose_conflict"
                    else:
                        pose, reason = corrected, "ok"
                        self.verified_at = received
                        self.force_global = False
                        self.stats["accepted_matches"] += 1
                else:
                    self.force_global = True
            else:
                self.force_global = True
                reason = result.reason if pose is None else "tracking_pending_verification"
                if result.reason == "ambiguous_place":
                    self.invalidate("LOST")
                    self.history.append(TrackingFrame(frame_id, received, gray, K))
                    pose, reason = None, "ambiguous_place"
        if pose is not None and (self.verified_at is None or received - self.verified_at > config.verification_seconds):
            pose, reason = None, "map_verification_expired"
            self.force_global = True
        if self.clock() - received > config.max_frame_age:
            self.invalidate()
            return None, "lost", "processing_deadline_exceeded"
        if pose is None:
            self.state = "LOST" if self.ever_localized else "INITIALIZING"
            status = "lost"
        else:
            self.state = "TRACKING"
            status = "relocalized" if was_lost and self.ever_localized else "localized"
            self.ever_localized = True
        self.previous_gray, self.previous_K, self.previous_pose = gray, K, pose
        now = self.clock()
        global_due = pose is None or self.force_global or now - self.last_global >= config.global_seconds
        if not self.matcher.busy and (global_due or now - self.last_request >= config.refresh_seconds):
            ids = None
            if not global_due:
                ids = sorted({j for i in np.unique(pose.reference_ids)
                              for j in range(int(i) - 3, int(i) + 4)} & self.available_ids)
            if self.matcher.submit(MatchRequest(frame_id, self.generation, gray, K, ids)):
                self.last_request = now
                if global_due:
                    self.last_global = now
        return pose, status, reason


def pose_message(sample, pose, state, reason, received, published, processing_ms, max_age):
    """JSON-safe current pose. Consumers MUST also check valid_until_monotonic."""
    valid = pose is not None and state == "TRACKING" and published - received <= max_age
    return {"schema_version": 1, "frame_index": sample.index, "timestamp_sec": sample.timestamp,
            "timestamp_source": sample.timestamp_source, "state": state, "valid": valid,
            "reason": reason, "received_monotonic": received, "published_monotonic": published,
            "valid_until_monotonic": received + max_age,
            "frame_age_ms": max(0., published - received) * 1000, "processing_ms": processing_ms,
            "position": pose.center.tolist() if valid else None,
            "quaternion_xyzw": pose.quaternion.tolist() if valid else None,
            "inliers": len(pose.uv) if valid else 0,
            "reprojection_error_px": pose.median_error if valid else None,
            "method": pose.method if valid else "none", "units": "map_units",
            "orientation": "camera_to_world"}
