#!/usr/bin/env python3
"""Real-time camera localization in a fixed map, or a video replay paced by PTS."""

import argparse
from collections import deque
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from localization.features import FeatureConfig
from localization.live_matcher import BackgroundMatcher
from localization.live_output import LiveWriter
from localization.live_source import LiveSource
from localization.pose import PoseConfig
from localization.realtime import LiveConfig, LiveTracker, pose_message
from localization.video import Preprocessor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--map", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Replay at original presentation times, not maximum decoding speed")
    source.add_argument("--camera", help="Device index (0), /dev/videoN, or RTSP/HTTP stream URL")
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--calib-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--capture-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--capture-fps", type=float)
    p.add_argument("--rotation", type=int, choices=[0, 90, 180, 270])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--view", action="store_true", help="Show the original dense 3D map in a separate process (CUDA)")
    p.add_argument("--preview", action="store_true", help="Show incoming frames and tracking state in the GUI process")
    p.add_argument("--view-screenshot", type=Path, help="Hidden 3D window; save the first valid camera pose (display required)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--duration", type=float, help="Stop after this many seconds of streaming")
    p.add_argument("--source-timeout", type=float, default=5., help="Fail if no frame arrives for this long")
    p.add_argument("--pixels", type=int, default=384 * 512)
    p.add_argument("--features", type=int, default=1800)
    p.add_argument("--min-inliers", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--matcher-threads", type=int, default=2)
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/localization"))
    p.add_argument("--refresh-seconds", type=float, default=.3)
    p.add_argument("--global-seconds", type=float, default=2.)
    p.add_argument("--max-frame-age", type=float, default=.15, help="Pose expiry measured from host receipt (seconds)")
    p.add_argument("--history-seconds", type=float, default=1.)
    p.add_argument("--history-frames", type=int, default=45)
    p.add_argument("--max-tracking-gap", type=float, default=.25)
    p.add_argument("--verification-seconds", type=float, default=3.)
    p.add_argument("--catchup-budget-ms", type=float, default=25.)
    args = p.parse_args(argv)
    for name in ("pixels", "features", "min_inliers", "threads", "matcher_threads", "max_frames",
                 "duration", "source_timeout", "capture_fps", "refresh_seconds", "global_seconds",
                 "max_frame_age", "history_seconds", "history_frames", "max_tracking_gap",
                 "verification_seconds", "catchup_budget_ms"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.features < args.min_inliers or args.min_inliers < 6:
        p.error("Require features >= min-inliers >= 6")
    for size in (args.calib_size, args.capture_size):
        if size is not None and min(size) <= 0:
            p.error("Image dimensions must be positive")
    if args.view_screenshot:
        args.view = True
    return args


def timing_summary(values):
    if not values:
        return None
    return dict(zip(("median_ms", "p95_ms", "p99_ms", "max_ms"),
                    map(float, np.percentile(values, [50, 95, 99, 100]))))


def run(args):
    outputs = ("trajectory.csv", "metadata.json", "report.json", "realtime_report.json",
               "poses.jsonl", "latest_pose.json")
    if not args.overwrite and any((args.output / name).exists() for name in outputs):
        raise ValueError("Output already contains a run; choose another directory or use --overwrite")
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(42)
    preprocessor = Preprocessor(args.calib, args.calib_size, args.pixels)
    feature_config, pose_config = FeatureConfig(max_features=args.features), PoseConfig(min_inliers=args.min_inliers)
    config = LiveConfig(**{name: getattr(args, name) for name in LiveConfig.__dataclass_fields__})
    matcher = BackgroundMatcher(args.map, feature_config, pose_config, args.cache_dir, args.matcher_threads)
    source = LiveSource(video=args.video, camera=args.camera, rotation=args.rotation,
                        size=args.capture_size, fps=args.capture_fps)
    viewer = writer = tracker = None
    error = None
    completed = False
    end_reason = "error"
    last_message = None
    track_segment = 0
    processed = 0
    times, ages = deque(maxlen=10000), deque(maxlen=10000)
    source_started = False
    streaming_started = None
    started = time.monotonic()
    summary = {}
    try:
        logging.info("Preparing map matcher before starting capture...")
        info = matcher.start()
        tracker = LiveTracker(matcher, pose_config, config)
        if args.view or args.preview:
            from localization.live_view import LiveViewer
            if args.view_screenshot:
                args.view_screenshot.parent.mkdir(parents=True, exist_ok=True)
            viewer = LiveViewer(args.map, args.view, args.preview, args.view_screenshot)
            logging.info("Preparing viewer...")
            summary["viewer"] = viewer.start()
        from localization.map import file_digest
        metadata = {"mode": "realtime", "map": str(args.map.resolve()), "map_sha256": info["map_sha256"],
                    "map_keyframes": info["keyframes"],
                    "input": str(args.video.resolve()) if args.video else "camera_stream",
                    "calib": str(args.calib.resolve()), "calib_sha256": file_digest(args.calib),
                    "preprocessing": preprocessor.metadata, "features": asdict(feature_config),
                    "pose": asdict(pose_config), "live": asdict(config), "source": source.metadata,
                    "opencv": cv2.__version__, "seed": 42,
                    "latency_clock": "scheduled_pts" if args.video else "host_receive_monotonic",
                    "capture_to_host_latency_measured": False,
                    "map_timestamps_are_video_times": False, "recording": "processed_frames_only"}
        writer = LiveWriter(args.output, metadata)
        source.start()
        source_started = True
        streaming_started = time.monotonic()
        summary["startup_seconds"] = streaming_started - started
        logging.info("Capture started; waiting for first localization")
        last_arrival = streaming_started
        last_log = streaming_started - 1
        while True:
            now = time.monotonic()
            writer.check()
            if viewer:
                viewer.check()
                if viewer.stop.is_set():
                    end_reason = "viewer_closed"
                    break
            if args.duration and now - streaming_started >= args.duration:
                end_reason = "duration_limit"
                break
            frame = source.slot.take(timeout=.005)
            if frame is None:
                if last_message is not None and now > last_message["valid_until_monotonic"] and tracker.state != "STALE":
                    tracker.invalidate()
                    message = dict(last_message, state="STALE", valid=False, position=None,
                                   quaternion_xyzw=None, reason="no_fresh_frame", published_monotonic=now,
                                   inliers=0, method="none", reprojection_error_px=None,
                                   frame_age_ms=(now - last_message["received_monotonic"]) * 1000)
                    writer.append(None, None, "lost", message["reason"], message)
                    if viewer:
                        viewer.publish(message)
                if source.done.is_set():
                    if source.error:
                        raise RuntimeError(source.error)
                    end_reason = "end_of_video"
                    break
                if now - last_arrival > args.source_timeout:
                    raise RuntimeError("No frames received before source timeout")
                # Also detect a dead matcher while the source is stalled.
                if not matcher.process.is_alive():
                    raise RuntimeError("Map matcher process exited")
                continue
            last_arrival = time.monotonic()
            sample = frame.sample
            frame_started = time.monotonic()
            image, K = preprocessor.process(sample.image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pose, status, reason = tracker.process_frame(sample.index, gray, K, frame.received_monotonic)
            published = time.monotonic()
            elapsed_ms = (published - frame_started) * 1000
            message = pose_message(sample, pose, tracker.state, reason, frame.received_monotonic,
                                   published, elapsed_ms, config.max_frame_age)
            if pose is not None and not message["valid"]:
                tracker.invalidate()
                pose, status, reason = None, "lost", "publishing_deadline_exceeded"
                message.update(state="STALE", reason=reason)
            if not message["valid"] or last_message is None or not last_message["valid"]:
                track_segment += 1
            message["track_segment"] = track_segment
            writer.append(sample, pose, status, reason, message)
            if viewer:
                viewer.publish(message, image if args.preview else None)
            processed += 1
            times.append(elapsed_ms)
            ages.append(message["frame_age_ms"])
            last_message = message
            if published - last_log >= 1:
                logging.info("Frame %d: %s; age %.1f ms; inliers %d; capture drops %d", sample.index,
                             message["state"], message["frame_age_ms"], message["inliers"], source.slot.dropped)
                last_log = published
            if args.max_frames and processed >= args.max_frames:
                end_reason = "frame_limit"
                break
        if processed == 0:
            raise RuntimeError("No frames processed")
        completed = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt):
            end_reason = "interrupted"
        raise
    finally:
        stopped = time.monotonic()
        if source_started:
            source.close()
        # Publish invalidity even on EOF, Ctrl-C or errors. A crashed publisher is
        # covered by each packet's deadline, which clients must check themselves.
        if writer is not None:
            terminal = dict(last_message or {}, schema_version=1, valid=False, state="STOPPED" if completed else "ERROR",
                            position=None, quaternion_xyzw=None, reason=end_reason if completed else error,
                            inliers=0, method="none", reprojection_error_px=None,
                            published_monotonic=stopped, valid_until_monotonic=stopped)
            try:
                writer.append(None, None, "lost", terminal["reason"], terminal)
            except RuntimeError:
                logging.exception("Could not log final stream state")
        if viewer:
            viewer.close()
        matcher.close()
        summary.update(end_reason=end_reason, streaming_seconds=stopped - streaming_started if streaming_started else 0,
                       captured_frames=source.metadata["captured_frames"], dropped_frames=source.slot.dropped,
                       unprocessed_at_stop=max(0, source.metadata["captured_frames"] - processed - source.slot.dropped),
                       timing_sample_count=len(times), timing_window_frames=10000,
                       frame_compute=timing_summary(times), frame_age=timing_summary(ages),
                       matcher_time=timing_summary(tracker.match_times) if tracker else None,
                       tracking=tracker.stats if tracker else {},
                       viewer_dropped_updates=viewer.dropped if viewer else 0)
        summary["stream_processing_fps"] = processed / summary["streaming_seconds"] if summary["streaming_seconds"] else 0
        if writer is not None:
            if "video" in source.metadata:
                metadata["video"] = source.metadata["video"]
            return_report = writer.close(completed, error, summary)
    return return_report


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        logging.error("%s", exc)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
