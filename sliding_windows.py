"""Match-guided overlapping windows in two image coordinate systems."""
from dataclasses import dataclass

import cv2
import numpy as np

from transform import is_similarity


def spread_indices(points, confidence, limit, min_distance=0., other=None):
    """Continuous farthest-point sampling; no bins or fixed image partitions."""
    if not len(points) or limit <= 0:
        return np.empty(0, int)
    chosen, distance = [], np.full(len(points), np.inf)
    index = int(np.argmax(confidence))
    for _ in range(min(limit, len(points))):
        chosen.append(index)
        d = np.linalg.norm(points - points[index], axis=1)
        if other is not None:
            d = np.minimum(d, np.linalg.norm(other - other[index], axis=1))
        distance = np.minimum(distance, d)
        distance[chosen] = -1
        index = int(np.argmax(distance * (.5 + .5 * confidence)))
        if distance[index] <= min_distance:
            break
    return np.array(chosen)


def neighborhoods(points, shape, limit=16):
    """Data-centered disjoint neighborhoods for held-out comparisons."""
    if not len(points):
        return np.empty(0, int)
    centers = spread_indices(points, np.ones(len(points)), limit, min(shape[:2]) * .18)
    return np.linalg.norm(points[:, None] - points[centers], axis=2).argmin(axis=1)


def density_weights(p0, p1, shape0, shape1):
    counts = []
    for p, shape in ((p0, shape0), (p1, shape1)):
        radius = max(4., min(shape[:2]) * .08)
        d2 = np.sum((p[:, None] - p[None]) ** 2, axis=2)
        counts.append(np.exp(-d2 / (2 * radius ** 2)).sum(axis=1))
    return 1 / np.maximum(counts[0], counts[1])


@dataclass
class Window:
    center0: np.ndarray
    center1: np.ndarray
    side0: float
    side1: float
    mapping: np.ndarray  # fixed -> moving in normalized image frames
    depth: int = 0


@dataclass
class SearchBudget:
    calls: int = 0
    pixels: int = 0
    seconds: float = 0.


def _inside_center(center, side, shape):
    size = np.array([shape[1], shape[0]], float)
    low = np.minimum(side / 2, size / 2)
    return np.clip(center, low, size - low)


def paired_window(center, side, mapping, shape0, shape1, depth=0):
    scale = float(np.hypot(mapping[0, 0], mapping[1, 0]))
    c0 = _inside_center(np.asarray(center, float), side, shape0)
    inv = cv2.invertAffineTransform(mapping)
    # Move both centers together. If both frames cannot fit, retain the mapping
    # and pad the remaining out-of-frame part instead of stretching the crop.
    for _ in range(3):
        c1 = mapping[:, :2] @ c0 + mapping[:, 2]
        target = _inside_center(c1, side * scale, shape1)
        c0 = _inside_center(inv[:, :2] @ target + inv[:, 2], side, shape0)
    c1 = mapping[:, :2] @ c0 + mapping[:, 2]
    return Window(c0, c1, float(side), side * scale, mapping, depth)


def inside_window(p0, p1, window, margin=0.):
    return ((np.abs(p0 - window.center0) <= window.side0 * (.5 - margin)).all(axis=1)
            & (np.abs(p1 - window.center1) <= window.side1 * (.5 - margin)).all(axis=1))


def overlap(a, b):
    scores = []
    for c1, s1, c2, s2 in ((a.center0,a.side0,b.center0,b.side0),
                           (a.center1,a.side1,b.center1,b.side1)):
        extent = np.maximum(0, np.minimum(c1+s1/2,c2+s2/2) - np.maximum(c1-s1/2,c2-s2/2))
        intersection = float(np.prod(extent))
        scores.append(intersection / max(s1*s1+s2*s2-intersection, 1e-6))
    return min(scores)


def propose_windows(p0, p1, confidence, shape0, shape1):
    base = max(40., min(shape0[:2]) * .22)
    # At most 16 seeds x 9 offsets, before any neural inference.
    seeds = spread_indices(p0, confidence, 16, base * .45,
                           other=p1 * (min(shape0[:2]) / min(shape1[:2])))
    candidates = []
    for seed in seeds:
        near = np.flatnonzero(np.linalg.norm(p0-p0[seed],axis=1) <= base)
        if len(near) < 4:
            continue
        mapping, inliers = cv2.estimateAffinePartial2D(p0[near],p1[near],
            method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=1000)
        if mapping is None or inliers is None or not is_similarity(mapping):
            continue
        selected = near[inliers.ravel().astype(bool)]
        if len(selected) < 4 or len(selected) < .6*len(near):
            continue
        if np.linalg.eigvalsh(np.cov(p0[selected].T))[0] < 4:
            continue
        scale = np.hypot(mapping[0,0],mapping[1,0])
        if not .2 <= scale <= 5:
            continue
        residual = np.linalg.norm(p0[selected] @ mapping[:,:2].T + mapping[:,2] - p1[selected],axis=1)
        side = base + 2 * min(float(np.percentile(residual,90))/scale, base*.2)
        for dy in (-.5,0,.5):
            for dx in (-.5,0,.5):
                window = paired_window(p0[seed]+np.array([dx,dy])*side,side,mapping,shape0,shape1)
                if np.sum(inside_window(p0,p1,window)) < 4:
                    continue
                if any(overlap(window,old) > .8 for old in candidates):
                    continue
                candidates.append(window)
    return candidates


def next_window(candidates, visited, p0, p1, confidence):
    if not candidates:
        return None
    covered = np.zeros(len(p0), bool)
    for old in visited:
        covered |= inside_window(p0,p1,old)
    def score(window):
        support = inside_window(p0,p1,window)
        gain = np.sum(confidence[support & ~covered])
        if visited:
            distance = min(np.linalg.norm(window.center0-old.center0)/window.side0 for old in visited)
        else:
            distance = 0
        # Raw match counts would favor the already easy/dense image center.
        return float(np.sqrt(gain)*(1+min(distance,3)) + .1*np.sqrt(np.sum(confidence[support])))
    best = max(range(len(candidates)),key=lambda i:score(candidates[i]))
    window = candidates.pop(best)
    candidates[:] = [c for c in candidates if overlap(c,window) < .8]
    return window


def crop_pair(fixed, moving, window, sf, sm, max_side, clip):
    sizes = [window.side0/sf, window.side1/sm]
    output = max(32, int(min(*sizes,max_side))//8*8)
    patches, transforms = [], []
    for image, center, side in zip((fixed,moving),(window.center0/sf,window.center1/sm),sizes):
        origin = center - side/2
        h,w = image.shape[:2]
        left,top = np.maximum(np.floor(origin).astype(int),0)
        right,bottom = np.minimum(np.ceil(origin+side+1).astype(int),[w,h])
        scale = output/side
        matrix = np.array([[scale,0,(left-origin[0])*scale],
                           [0,scale,(top-origin[1])*scale]],float)
        if right <= left or bottom <= top:
            patch = np.full((output,output),127,np.uint8)
        else:
            source = image[top:bottom,left:right]
            if source.ndim == 3:
                source = cv2.cvtColor(source,cv2.COLOR_RGB2GRAY)
            patch = cv2.warpAffine(source,matrix,(output,output),borderValue=127)
            patch = cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8)).apply(patch)
        patches.append(patch)
        transforms.append((scale,origin))
    return (*patches,*transforms)


def followup_window(window,p0,p1,shape0,shape1):
    if window.depth or len(p0) < 8:
        return None
    mapping,inliers = cv2.estimateAffinePartial2D(p0,p1,method=cv2.RANSAC,ransacReprojThreshold=3)
    if mapping is None or not is_similarity(mapping) or inliers.mean() < .75:
        return None
    center = np.median(p0[inliers.ravel().astype(bool)],axis=0)
    edge = np.max(np.abs(center-window.center0)) > window.side0*.25
    # Only one follow-up generation: slide to boundary support, otherwise zoom.
    return paired_window(center,window.side0*(1 if edge else .65),mapping,shape0,shape1,depth=1)
