#!/usr/bin/env python3
"""Test independent localization on frames excluded by the mapping FPS filter.

The FFmpeg fps filter is replayed through PyAV. Hashes of decoded YUV pixels
exclude every selected mapping image (including duplicate images), not just
the sparse keyframes retained by DROID. This is same-session validation, not
an independent measurement of absolute pose accuracy.
"""

import argparse
from hashlib import sha256
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import cv2
import numpy as np

from localization.export import ResultWriter, diagnostic_image
from localization.features import FeatureConfig, FeatureDatabase
from localization.map import ReferenceMap, file_digest
from localization.pose import PoseConfig
from localization.video import Preprocessor, VideoReader


def heldout_indices(video, mapping_fps):
    original_hashes, mapping_hashes = [], set()
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        graph = None

        def collect():
            while True:
                try:
                    frame = graph.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    break
                mapping_hashes.add(sha256(frame.to_ndarray(format="yuv420p").tobytes()).digest())

        for frame in container.decode(stream):
            original_hashes.append(sha256(frame.to_ndarray(format="yuv420p").tobytes()).digest())
            if graph is None:
                graph = av.filter.Graph()
                source = graph.add_buffer(template=frame)
                fps = graph.add("fps", str(mapping_fps))
                sink = graph.add("buffersink")
                source.link_to(fps)
                fps.link_to(sink)
                graph.configure()
            graph.push(frame)
            collect()
        if graph is None:
            raise ValueError("Video has no frames")
        graph.push(None)
        collect()
    heldout = [i for i, digest in enumerate(original_hashes) if digest not in mapping_hashes]
    return heldout, len(original_hashes), len(mapping_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--map-fps", type=float, default=10.)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--calib-size", nargs=2, type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or not np.isfinite(args.map_fps) or args.map_fps <= 0:
        parser.error("--samples and --map-fps must be positive")
    if any((args.output / n).exists() for n in ["trajectory.csv", "metadata.json", "report.json"]):
        parser.error("Use a fresh output directory")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cv2.setNumThreads(4)
    cv2.setRNGSeed(42)
    heldout, decoded, selected = heldout_indices(args.video, args.map_fps)
    if not heldout:
        parser.error("FPS filtering left no held-out frames")
    sample_positions = np.unique(np.linspace(0, len(heldout) - 1, min(args.samples, len(heldout))).astype(int))
    indices = {heldout[i] for i in sample_positions}
    logging.info("%d source frames; %d unique mapping images; %d excluded from mapping; testing %d",
                 decoded, selected, len(heldout), len(indices))
    reference_map = ReferenceMap.load(args.map)
    database = FeatureDatabase.build(reference_map, FeatureConfig(), ".cache/localization")
    preprocessor = Preprocessor(args.calib, args.calib_size)
    metadata = {"map": str(reference_map.path), "map_sha256": reference_map.digest,
                "input": str(args.video.resolve()), "input_sha256": file_digest(args.video),
                "calib": str(args.calib.resolve()), "calib_sha256": file_digest(args.calib),
                "preprocessing": preprocessor.metadata,
                "validation": {"kind": "same_session_heldout", "mapping_fps": args.map_fps,
                    "exclusion": "decoded YUV hashes selected by FFmpeg fps filter",
                    "source_frames": decoded, "unique_mapping_images": selected,
                    "heldout_frames": len(heldout), "sample_indices": sorted(indices),
                    "independent_localization_per_sample": True, "ground_truth_available": False}}
    writer = ResultWriter(args.output, metadata)
    completed, error = False, None
    try:
        with VideoReader(args.video) as reader:
            metadata["video"] = reader.metadata
            for sample in reader:
                if sample.index not in indices:
                    continue
                image, K = preprocessor.process(sample.image)
                pose, reason = database.localize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), K, PoseConfig())
                status = "localized" if pose is not None else "lost"
                writer.append(sample, pose, status, reason)
                diagnostic = diagnostic_image(image, K, pose, status, sample)
                cv2.imwrite(str(args.output / f"frame_{sample.index:06d}.jpg"), diagnostic)
                logging.info("Held-out frame %d: %s, inliers=%d", sample.index, status, len(pose.uv) if pose else 0)
        if writer.rows != len(indices):
            raise ValueError("Not all selected held-out frames were decoded")
        completed = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = writer.close(completed, error)
    print(f"Localized {report['processed_frames'] - report['status_counts']['lost']}/{writer.rows} held-out frames")


if __name__ == "__main__":
    main()
