"""Reusable SIFT map index and global/local image-to-map matching."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .pose import estimate_pose, poses_agree, project


@dataclass(frozen=True)
class FeatureConfig:
    max_features: int = 1800
    ratio: float = .7
    candidates: int = 5


@dataclass
class ReferenceFeatures:
    index: int
    uv: np.ndarray
    xyz: np.ndarray
    descriptors: np.ndarray
    matcher: object = None

    def build_matcher(self):
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.matcher.add([np.ascontiguousarray(self.descriptors, np.float32)])
        self.matcher.train()


class FeatureDatabase:
    def __init__(self, references, config):
        self.references = references
        self.config = config
        self.sift = cv2.SIFT_create(nfeatures=config.max_features)
        for reference in references:
            reference.build_matcher()
        self.by_index = {r.index: r for r in references}

    def extract(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            return np.empty((0, 2)), np.empty((0, 128), np.float32)
        return np.array([k.pt for k in keypoints]), descriptors

    @classmethod
    def build(cls, reference_map, config, cache_dir=None):
        signature = json.dumps({"version": 1, "map": reference_map.digest,
                                "config": asdict(config), "opencv": cv2.__version__}, sort_keys=True)
        cache = None
        if cache_dir is not None:
            from hashlib import sha256
            cache = Path(cache_dir) / (sha256(signature.encode()).hexdigest() + ".npz")
            if cache.exists():
                try:
                    with np.load(cache, allow_pickle=False) as data:
                        if str(data["signature"]) != signature:
                            raise ValueError("Cache signature mismatch")
                        references = []
                        for j, index in enumerate(data["indices"]):
                            start, end = data["offsets"][j:j + 2]
                            references.append(ReferenceFeatures(int(index), data["uv"][start:end].copy(),
                                data["xyz"][start:end].copy(), data["descriptors"][start:end].copy()))
                    logging.info("Loaded feature cache: %s", cache)
                    return cls(references, config)
                except (ValueError, KeyError, OSError, EOFError):
                    logging.warning("Ignoring invalid feature cache: %s", cache)
        sift = cv2.SIFT_create(nfeatures=config.max_features)
        references = []
        for index, image in enumerate(reference_map.images):
            keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
            if descriptors is None:
                continue
            uv = np.array([k.pt for k in keypoints])
            xyz, valid = reference_map.lift(index, uv)
            valid &= reference_map.feature_depth_mask(index, uv)
            if valid.sum() >= 6:
                references.append(ReferenceFeatures(index, uv[valid], xyz[valid], descriptors[valid]))
        if not references:
            raise ValueError("Map has no usable SIFT landmarks")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            offsets = np.r_[0, np.cumsum([len(r.uv) for r in references])]
            with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    np.savez_compressed(stream, signature=signature,
                        indices=[r.index for r in references], offsets=offsets,
                        uv=np.concatenate([r.uv for r in references]),
                        xyz=np.concatenate([r.xyz for r in references]),
                        descriptors=np.concatenate([r.descriptors for r in references]))
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            temporary.replace(cache)
        logging.info("Built SIFT index: %d reference frames, %d landmarks", len(references),
                     sum(len(r.uv) for r in references))
        return cls(references, config)

    def localize(self, gray, K, pose_config, reference_indices=None):
        uv, descriptors = self.extract(gray)
        if len(uv) < pose_config.min_inliers:
            return None, "too_few_features"
        references = self.references if reference_indices is None else [
            self.by_index[i] for i in reference_indices if i in self.by_index]
        ranked = []
        for ref in references:
            pairs = ref.matcher.knnMatch(descriptors, k=2)
            matches = [a for pair in pairs if len(pair) == 2
                       for a, b in [pair] if a.distance < self.config.ratio * b.distance]
            # One-to-one within each reference: repeated textures must not inflate support.
            unique = {}
            for match in sorted(matches, key=lambda m: m.distance):
                unique.setdefault(match.trainIdx, match)
            ranked.append((ref, list(unique.values())))
        ranked.sort(key=lambda item: len(item[1]), reverse=True)
        candidates = []
        correspondences = []
        for ref, matches in ranked[:self.config.candidates]:
            if len(matches) < pose_config.min_inliers:
                continue
            qi = np.array([m.queryIdx for m in matches])
            ri = np.array([m.trainIdx for m in matches])
            xyz, pixels = ref.xyz[ri], uv[qi]
            refs = np.full(len(qi), ref.index)
            pose = estimate_pose(xyz, pixels, K, gray.shape, pose_config, qi, refs)
            correspondences.append((xyz, pixels, qi, refs))
            if pose is not None:
                candidates.append(pose)
        if not candidates:
            return None, "no_geometric_solution"
        candidates.sort(key=lambda p: (len(p.uv), -p.median_error), reverse=True)
        best = candidates[0]
        for other in candidates[1:]:
            if len(other.uv) >= .85 * len(best.uv) and not poses_agree(best, other):
                return None, "ambiguous_place"
        # Merge only correspondences supported by the selected geometry. A query
        # pixel gets one 3D association even if visible in several reference views.
        pooled = {}
        for xyz, pixels, qi, refs in correspondences:
            projected, depth = project(xyz, best.rvec, best.tvec, K)
            errors = np.linalg.norm(projected - pixels, axis=1)
            for j in np.flatnonzero((errors <= pose_config.ransac_px) & (depth > 0)):
                key = int(qi[j])
                if key not in pooled or errors[j] < pooled[key][0]:
                    pooled[key] = (errors[j], xyz[j], pixels[j], refs[j])
        if len(pooled) >= pose_config.min_inliers:
            keys = np.array(list(pooled))
            values = list(pooled.values())
            refined = estimate_pose(np.array([v[1] for v in values]),
                np.array([v[2] for v in values]), K, gray.shape, pose_config,
                keys, np.array([v[3] for v in values]))
            total_matches = len(set(int(q) for _, _, qs, _ in correspondences for q in qs))
            if (refined is not None and poses_agree(best, refined)
                    and len(refined.uv) >= len(best.uv)
                    and len(refined.uv) / total_matches >= pose_config.min_inlier_ratio):
                # Preserve pre-geometric matching count as a quality diagnostic.
                refined.matches = total_matches
                best = refined
        return best, "ok"

    def nearby(self, pose, radius=3):
        ids = set()
        for index in np.unique(pose.reference_ids):
            ids.update(range(max(0, int(index) - radius), int(index) + radius + 1))
        return sorted(ids.intersection(self.by_index))
