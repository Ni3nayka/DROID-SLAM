"""Bounded asynchronous logging, compatible with the existing CSV viewer."""

from dataclasses import replace
import json
from pathlib import Path
import queue
import threading

from .export import ResultWriter, write_json


class LiveWriter:
    def __init__(self, directory, metadata):
        self.directory, self.metadata = Path(directory), metadata
        self.queue = queue.Queue(maxsize=128)
        self.ready = threading.Event()
        self.error = None
        self.report = None
        self.completed = False
        self.run_error = None
        self.summary = {}
        self.thread = threading.Thread(target=self._run, name="pose-log", daemon=True)
        self.thread.start()
        if not self.ready.wait(10):
            raise RuntimeError("Logger startup timed out")
        self.check()

    def check(self):
        if self.error:
            raise RuntimeError(f"Pose logging failed: {self.error}")

    def append(self, sample, pose, status, reason, message):
        self.check()
        # No frame images in the logging queue; pose arrays are immutable after
        # submission. Failing explicitly is preferable to silently losing rows.
        sample = replace(sample, image=None) if sample is not None else None
        try:
            self.queue.put_nowait((sample, pose, status, reason, message))
        except queue.Full as exc:
            raise RuntimeError("Pose logger cannot keep up (128 pending records)") from exc

    def close(self, completed, error, summary):
        self.completed, self.run_error, self.summary = completed, error, summary
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=5)
            except queue.Full as exc:
                raise RuntimeError("Pose logger shutdown timed out") from exc
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Pose logger did not stop")
        self.check()
        return self.report

    def _run(self):
        writer = None
        try:
            writer = ResultWriter(self.directory, self.metadata)
            writer.metadata["time_origin"] = ("first received camera frame" if
                self.metadata.get("source", {}).get("kind") == "camera" else "first decoded video frame")
            # Long-running streams keep bounded diagnostic samples in RAM.
            from collections import deque
            writer.errors = deque(maxlen=10000)
            writer.inlier_counts = deque(maxlen=10000)
            self.ready.set()
            with (self.directory / "poses.jsonl").open("w") as stream:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    sample, pose, status, reason, message = item
                    if sample is not None:
                        writer.append(sample, pose, status, reason)
                    stream.write(json.dumps(message, allow_nan=False) + "\n")
                    stream.flush()
                    write_json(self.directory / "latest_pose.json", message)
            self.report = writer.close(self.completed, self.run_error)
            self.report.update(self.summary)
            write_json(self.directory / "realtime_report.json", self.report)
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if writer is not None and not writer.stream.closed:
                writer.close(False, self.error)
        finally:
            self.ready.set()
