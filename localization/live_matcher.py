"""One asynchronous map-matching job at a time; no queued historical queries."""

from dataclasses import dataclass
import multiprocessing as mp
import queue
import time


@dataclass
class MatchRequest:
    frame_id: int
    generation: int
    gray: object
    K: object
    reference_ids: object = None


@dataclass
class MatchResult:
    request: MatchRequest
    pose: object
    reason: str
    duration_ms: float


def _worker(requests, results, map_path, feature_config, pose_config, cache_dir, threads):
    try:
        import cv2
        from .features import FeatureDatabase
        from .map import ReferenceMap
        cv2.setNumThreads(threads)
        cv2.setRNGSeed(42)
        reference = ReferenceMap.load(map_path)
        database = FeatureDatabase.build(reference, feature_config, cache_dir)
        results.put({"ready": True, "map_sha256": reference.digest,
                     "keyframes": len(reference.images),
                     "reference_ids": sorted(database.by_index)})
        while True:
            request = requests.get()
            if request is None:
                return
            started = time.monotonic()
            pose, reason = database.localize(request.gray, request.K, pose_config, request.reference_ids)
            # History in the fast process owns the image; do not send it back.
            request.gray = None
            results.put(MatchResult(request, pose, reason, (time.monotonic() - started) * 1000))
    except BaseException as exc:
        results.put({"error": f"{type(exc).__name__}: {exc}"})


class BackgroundMatcher:
    def __init__(self, map_path, feature_config, pose_config, cache_dir, threads=2):
        context = mp.get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.results = context.Queue(maxsize=1)
        self.process = context.Process(target=_worker, args=(self.requests, self.results, map_path,
                                      feature_config, pose_config, cache_dir, threads), daemon=True)
        self.busy = False
        self.submitted_at = None
        self.info = None

    def start(self, timeout=120.):
        self.process.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = self.results.get(timeout=.1)
                if "error" in message:
                    raise RuntimeError(message["error"])
                self.info = message
                return message
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError("Map matcher exited during startup")
        raise RuntimeError("Map matcher startup timed out")

    def submit(self, request):
        if self.busy:
            return False
        self.requests.put_nowait(request)
        self.busy = True
        self.submitted_at = time.monotonic()
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError("Map matcher process exited")
            if self.busy and time.monotonic() - self.submitted_at > 30:
                raise RuntimeError("Map matcher job timed out")
            return None
        self.busy = False
        if isinstance(result, dict):
            raise RuntimeError(result.get("error", "Unexpected matcher message"))
        return result

    def close(self):
        if self.process.pid is not None:
            try:
                self.requests.put_nowait(None)
            except queue.Full:
                pass
            # A large unconsumed pose can keep the child queue feeder alive.
            deadline = time.monotonic() + 2
            while self.process.is_alive() and time.monotonic() < deadline:
                try:
                    self.results.get(timeout=.05)
                except queue.Empty:
                    pass
                self.process.join(timeout=.01)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        for channel in (self.requests, self.results):
            channel.cancel_join_thread()
            channel.close()
