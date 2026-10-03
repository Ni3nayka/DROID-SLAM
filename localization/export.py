"""Incremental CSV output, provenance and machine-readable run statistics."""

import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np


FIELDS = ["frame_index", "timestamp_sec", "pts", "time_base", "timestamp_source",
          "x", "y", "z", "qx", "qy", "qz", "qw", "status", "method", "inliers",
          "matches", "inlier_ratio", "reprojection_error_px", "coverage", "reference_ids", "reason"]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class ResultWriter:
    def __init__(self, directory, metadata, scale=1.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.metadata.update(completed=False, units="map_units" if scale == 1 else "scaled_map_units",
                             position_scale=scale, quaternion_order="qx qy qz qw",
                             orientation="camera_to_world", position="camera optical center in map frame",
                             time_origin="first decoded video frame", interpolation=False)
        self.scale = scale
        self.stream = (self.directory / "trajectory.csv").open("w", newline="")
        self.csv = csv.DictWriter(self.stream, fieldnames=FIELDS)
        self.csv.writeheader()
        self.started = time.perf_counter()
        self.counts = {"localized": 0, "relocalized": 0, "lost": 0}
        self.errors = []
        self.inlier_counts = []
        self.lost_intervals = []
        self.open_lost = None
        self.rows = 0
        write_json(self.directory / "metadata.json", self.metadata)

    def append(self, sample, pose, status, reason):
        row = dict(frame_index=sample.index, timestamp_sec=f"{sample.timestamp:.9f}",
                   pts=sample.pts, time_base=sample.time_base, timestamp_source=sample.timestamp_source,
                   status=status, reason=reason, method="none", inliers=0, matches=0,
                   inlier_ratio=0, coverage=0, reference_ids="", reprojection_error_px="nan")
        row.update(dict.fromkeys(["x", "y", "z", "qx", "qy", "qz", "qw"], "nan"))
        if pose is not None:
            values = np.r_[pose.center * self.scale, pose.quaternion]
            row.update(zip(["x", "y", "z", "qx", "qy", "qz", "qw"], [f"{v:.10g}" for v in values]))
            row.update(method=pose.method, inliers=len(pose.uv), matches=pose.matches,
                       inlier_ratio=len(pose.uv) / pose.matches, coverage=pose.coverage,
                       reprojection_error_px=pose.median_error,
                       reference_ids=";".join(map(str, np.unique(pose.reference_ids))))
            self.errors.append(pose.median_error)
            self.inlier_counts.append(len(pose.uv))
        if status == "lost":
            if self.open_lost is None:
                self.open_lost = {"start_frame": sample.index, "start_sec": sample.timestamp}
            self.open_lost.update(end_frame=sample.index, end_sec=sample.timestamp)
        elif self.open_lost is not None:
            self.lost_intervals.append(self.open_lost)
            self.open_lost = None
        self.csv.writerow(row)
        self.stream.flush()
        self.counts[status] += 1
        self.rows += 1

    def close(self, completed, error=None):
        self.stream.close()
        if self.open_lost is not None:
            self.lost_intervals.append(self.open_lost)
            self.open_lost = None
        elapsed = time.perf_counter() - self.started
        report = {"completed": completed, "error": error, "processed_frames": self.rows,
                  "status_counts": self.counts,
                  "localized_fraction": (self.rows - self.counts["lost"]) / self.rows if self.rows else 0,
                  "lost_intervals": self.lost_intervals, "elapsed_seconds": elapsed,
                  "processing_fps": self.rows / elapsed if elapsed else 0,
                  "median_reprojection_error_px": float(np.median(self.errors)) if self.errors else None,
                  "median_inliers": float(np.median(self.inlier_counts)) if self.inlier_counts else None}
        self.metadata.update(completed=completed, error=error)
        write_json(self.directory / "metadata.json", self.metadata)
        write_json(self.directory / "report.json", report)
        return report


def diagnostic_image(image, K, pose, status, sample):
    from .pose import project
    image = image.copy()
    if pose is not None:
        projected, _ = project(pose.xyz, pose.rvec, pose.tvec, K)
        for observed, predicted in zip(pose.uv, projected):
            a = tuple(np.rint(observed).astype(int))
            b = tuple(np.rint(predicted).astype(int))
            cv2.circle(image, a, 2, (0, 220, 0), -1)
            cv2.line(image, a, b, (0, 0, 255), 1)
    label = f"{sample.index}  {sample.timestamp:.3f}s  {status}"
    if pose is not None:
        label += f"  n={len(pose.uv)} err={pose.median_error:.2f}px"
    cv2.rectangle(image, (0, 0), (image.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(image, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    return image
