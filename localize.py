#!/usr/bin/env python3
"""Localize a video or still image in an existing DROID-SLAM map."""

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

from localization.export import ResultWriter, diagnostic_image
from localization.features import FeatureConfig, FeatureDatabase
from localization.map import ReferenceMap, file_digest
from localization.pose import PoseConfig
from localization.tracker import MapTracker, TrackerConfig
from localization.video import Preprocessor, VideoReader, VideoSample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--map", required=True, type=Path, help="Saved reconstruction .pth")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--image", type=Path, help="Localize one image; time is recorded as 0")
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Directory for CSV and reports")
    parser.add_argument("--calib-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Resolution for which calibration was measured, in displayed orientation")
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270],
                        help="Override video display rotation, counterclockwise degrees")
    parser.add_argument("--allow-fps-fallback", action="store_true",
                        help="Allow explicitly labelled approximate times when PTS is absent")
    parser.add_argument("--pixels", type=int, default=384 * 512, help="Working image area before cropping")
    parser.add_argument("--features", type=int, default=1800)
    parser.add_argument("--ratio", type=float, default=.7, help="SIFT descriptor ratio test")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--min-inlier-ratio", type=float, default=.3)
    parser.add_argument("--ransac-px", type=float, default=3.)
    parser.add_argument("--max-median-px", type=float, default=2.)
    parser.add_argument("--min-coverage", type=float, default=.015)
    parser.add_argument("--independent", action="store_true", help="Global image matching for every frame")
    parser.add_argument("--refresh-interval", type=int, default=10, help="Refresh map features every N frames")
    parser.add_argument("--global-interval", type=int, default=60, help="Global verification every N frames")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/localization"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--start", type=float, default=0., help="Start time in original video, seconds")
    parser.add_argument("--end", type=float, help="Exclusive end time in original video, seconds")
    parser.add_argument("--max-frames", type=int, help="Limit processed frames for a short check")
    parser.add_argument("--scale", type=float, default=1., help="Multiplier for exported positions")
    parser.add_argument("--diagnostic-every", type=int, default=0, help="Save annotated JPEG every N frames; 0 disables")
    parser.add_argument("--preview", action="store_true", help="Show current frame and geometric inliers")
    parser.add_argument("--overwrite", action="store_true", help="Replace an earlier run in the output directory")
    parser.add_argument("--threads", type=int, default=4, help="OpenCV worker threads")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in ["pixels", "features", "candidates", "min_inliers", "refresh_interval", "global_interval", "threads"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ["scale", "ransac_px", "max_median_px"]:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ["ratio", "min_inlier_ratio", "min_coverage"]:
        if not 0 < getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.min_inliers < 6 or args.features < args.min_inliers:
        parser.error("Require features >= min-inliers >= 6")
    if not np.isfinite(args.start) or args.start < 0:
        parser.error("--start must be finite and nonnegative")
    if args.end is not None and (not np.isfinite(args.end) or args.end <= args.start):
        parser.error("--end must be finite and greater than --start")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.diagnostic_every < 0:
        parser.error("--diagnostic-every must be nonnegative")
    if args.calib_size is not None and min(args.calib_size) <= 0:
        parser.error("--calib-size values must be positive")
    return args


def run(args):
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(args.seed)
    if not args.overwrite and any((args.output / name).exists() for name in
                                  ["trajectory.csv", "metadata.json", "report.json"]):
        raise ValueError("Output already contains a run; choose another directory or use --overwrite")
    reference_map = ReferenceMap.load(args.map)
    logging.info("Map: %d keyframes, %d x %d", len(reference_map.images),
                 reference_map.images.shape[2], reference_map.images.shape[1])
    feature_config = FeatureConfig(args.features, args.ratio, args.candidates)
    pose_config = PoseConfig(args.min_inliers, args.min_inlier_ratio, args.ransac_px,
                             args.max_median_px, args.min_coverage)
    tracker_config = TrackerConfig(args.independent, args.refresh_interval, args.global_interval)
    database = FeatureDatabase.build(reference_map, feature_config, None if args.no_cache else args.cache_dir)
    tracker = MapTracker(database, pose_config, tracker_config)
    preprocessor = Preprocessor(args.calib, args.calib_size, args.pixels)
    source_path = args.video or args.image
    metadata = {"map": str(reference_map.path), "map_sha256": reference_map.digest,
                "map_keyframes": len(reference_map.images), "input": str(source_path.resolve()),
                "input_sha256": file_digest(source_path), "calib": str(args.calib.resolve()),
                "calib_sha256": file_digest(args.calib), "features": asdict(feature_config),
                "pose": asdict(pose_config), "tracker": asdict(tracker_config),
                "preprocessing": preprocessor.metadata, "opencv": cv2.__version__,
                "seed": args.seed, "selection": {"start": args.start, "end": args.end,
                                                  "max_frames": args.max_frames},
                "map_timestamps_are_video_times": False}
    writer = ResultWriter(args.output, metadata, args.scale)
    error = None
    completed = False

    def process(sample):
        image, K = preprocessor.process(sample.image)
        pose, status, reason = tracker.process(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), K)
        writer.append(sample, pose, status, reason)
        save_diagnostic = args.diagnostic_every and (writer.rows - 1) % args.diagnostic_every == 0
        if args.preview or save_diagnostic:
            diagnostic = diagnostic_image(image, K, pose, status, sample)
            if save_diagnostic:
                directory = args.output / "diagnostics"
                directory.mkdir(exist_ok=True)
                if not cv2.imwrite(str(directory / f"{sample.index:08d}.jpg"), diagnostic):
                    raise OSError("Could not write diagnostic image")
            if args.preview:
                cv2.imshow("Map localization (Esc to stop)", diagnostic)
                if cv2.waitKey(1) & 255 == 27:
                    raise KeyboardInterrupt
        if writer.rows == 1 or writer.rows % 100 == 0:
            logging.info("Frame %d, %.3fs: %s; localized %d/%d", sample.index, sample.timestamp,
                         status, writer.rows - writer.counts["lost"], writer.rows)

    try:
        if args.image:
            image = cv2.imread(str(args.image))
            if image is None:
                raise ValueError(f"Cannot read image: {args.image}")
            if args.rotation:
                image = np.ascontiguousarray(np.rot90(image, args.rotation // 90))
            process(VideoSample(0, 0., None, "", "still_image", image, args.rotation or 0))
        else:
            with VideoReader(args.video, args.rotation, args.allow_fps_fallback) as reader:
                import av
                metadata["pyav"] = av.__version__
                metadata["video"] = reader.metadata
                for sample in reader:
                    if sample.timestamp < args.start:
                        continue
                    if args.end is not None and sample.timestamp >= args.end:
                        break
                    process(sample)
                    if args.max_frames is not None and writer.rows >= args.max_frames:
                        break
                if not writer.rows:
                    raise ValueError("No frames in the selected interval")
        completed = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = writer.close(completed, error)
        if args.preview:
            cv2.destroyAllWindows()
        logging.info("Saved %s (%d frames, %.1f%% localized)", args.output / "trajectory.csv",
                     report["processed_frames"], 100 * report["localized_fraction"])
    return report


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = run(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        logging.error("%s", exc)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
