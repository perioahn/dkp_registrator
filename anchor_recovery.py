"""On-demand reference-region suggestions and explicit similarity recovery.

Suggested pairs are observed correspondences, never projections of a fitted
transform. Region-local consensus deliberately precedes the final global fit.
"""
from collections import Counter

import cv2
import numpy as np

from register import _draw_matches, _fit_similarity_lstsq, _to_3x3


def _union(masks):
    if not masks:
        raise ValueError('기준 사진과 현재 사진에서 마스크를 선택하세요.')
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask
    if not union.any():
        raise ValueError('선택한 마스크가 비어 있습니다. 영역을 다시 선택하세요.')
    return union


def _spatial_labels(points, mask):
    """Coverage cells of the union bounding box, independent of click history.

    These cells measure coverage; they neither split the input image for matching
    nor claim to identify anatomical objects.
    """
    ys, xs = np.nonzero(mask)
    origin = np.array([xs.min(), ys.min()])
    span = np.array([np.ptp(xs)+1, np.ptp(ys)+1], dtype=float)
    xy = np.rint(points).astype(int)
    inside = ((xy >= 0) & (xy < [mask.shape[1], mask.shape[0]])).all(1)
    idx = np.flatnonzero(inside)
    inside[idx] &= mask[xy[idx, 1], xy[idx, 0]]
    cells = np.clip(((points-origin) / span * 4).astype(int), 0, 3)
    labels = np.where(inside, cells[:, 1]*4 + cells[:, 0], -1)
    occupied = np.unique(((ys-origin[1])*4//int(span[1]))*4 +
                         ((xs-origin[0])*4//int(span[0])))
    return labels, occupied, float(np.linalg.norm(span))


def select_suggestions(k0, k1, confidence, fixed_masks, moving_masks):
    """Up to 24 observed pairs spread over both unions, with local consensus.

    Never run one global RANSAC that could suppress a sparse, conflicting area.
    Confidence and residual thresholds are engineering filters, not clinical tolerances.
    """
    fixed_union, moving_union = _union(fixed_masks), _union(moving_masks)
    k0, k1 = np.asarray(k0, float).reshape(-1, 2), np.asarray(k1, float).reshape(-1, 2)
    cf = np.asarray(confidence, float)
    valid = np.isfinite(k0).all(1) & np.isfinite(k1).all(1) & np.isfinite(cf) & (cf >= .5)
    k0, k1, cf = k0[valid], k1[valid], cf[valid]
    # Dense duplicates must not count as independent support.
    keep, seen0, seen1 = [], set(), set()
    for i in np.argsort(-cf, kind='stable'):
        a, b = tuple(np.rint(k0[i]/3).astype(int)), tuple(np.rint(k1[i]/3).astype(int))
        if a not in seen0 and b not in seen1:
            keep.append(i); seen0.add(a); seen1.add(b)
    k0, k1, cf = k0[keep], k1[keep], cf[keep]
    fids, occupied_f, span0 = _spatial_labels(k0, fixed_union)
    mids, occupied_m, span1 = _spatial_labels(k1, moving_union)
    eligible = np.flatnonzero((fids >= 0) & (mids >= 0))
    supported = set()
    for fi in occupied_f:
        cell = eligible[fids[eligible] == fi]
        if not len(cell) or len(eligible) < 4:
            continue
        # Expand sparse cells into an overlapping local neighbourhood. Selection
        # boundaries must not become matching boundaries or erase sparse support.
        center = np.median(k0[cell], axis=0)
        nearest = eligible[np.argsort(np.linalg.norm(k0[eligible]-center, axis=1), kind='stable')[:12]]
        ix = np.unique(np.r_[cell, nearest])
        threshold = max(1.5, max(fixed_union.shape) * 3 / 640)
        matrix, inliers = cv2.estimateAffinePartial2D(k1[ix], k0[ix], method=cv2.RANSAC,
            ransacReprojThreshold=threshold, maxIters=1500, confidence=.995)
        if matrix is None or inliers is None or inliers.sum() < 4 or inliers.mean() < .6:
            continue
        accepted = ix[inliers.ravel().astype(bool)]
        supported.update(accepted[fids[accepted] == fi].tolist())
    ix = np.array(sorted(supported), dtype=int)
    suggestions = []
    chosen = []
    if len(ix):
        chosen = [ix[np.argmax(cf[ix])]]
        while len(chosen) < 24:
            distance0 = np.linalg.norm(k0[ix, None] - k0[chosen], axis=2).min(1) / span0
            distance1 = np.linalg.norm(k1[ix, None] - k1[chosen], axis=2).min(1) / span1
            distance = np.minimum(distance0, distance1)
            score = distance * (.5 + .5 * cf[ix])
            best = int(np.argmax(score))
            if distance[best] < .08:
                break
            chosen.append(ix[best])
        for i in chosen:
            suggestions.append({'fixed': k0[i].tolist(), 'moving': k1[i].tolist(),
                                'group': f'area-{fids[i]}', 'confidence': float(cf[i])})
    missing = [f'fixed:{i}' for i in occupied_f if i not in set(fids[chosen])]
    missing += [f'moving:{i}' for i in occupied_m if i not in set(mids[chosen])]
    return suggestions, missing


def _crop(image, masks):
    union = _union(masks)
    ys, xs = np.nonzero(union)
    x, y = max(0, xs.min()-16), max(0, ys.min()-16)
    right, bottom = min(image.shape[1], xs.max()+17), min(image.shape[0], ys.max()+17)
    gray = cv2.cvtColor(image[y:bottom, x:right], cv2.COLOR_RGB2GRAY)
    scale = min(1., 640 / max(gray.shape))
    w, h = max(32, int(gray.shape[1]*scale)//8*8), max(32, int(gray.shape[0]*scale)//8*8)
    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    resized = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(resized)
    return resized, np.array([gray.shape[1]/w, gray.shape[0]/h]), np.array([x, y])


def recommend_anchors(fixed, moving, fixed_masks, moving_masks):
    from matching import loftr_match
    if not fixed_masks or not moving_masks:
        raise ValueError('기준 사진과 현재 사진에서 구조물을 마스크로 선택하고 Z로 확정하세요.')
    fixed_masks, moving_masks = [_union(fixed_masks)], [_union(moving_masks)]
    m, sm, om = _crop(moving, moving_masks)

    def match(regions):
        f, sf, of = _crop(fixed, regions)
        k0, k1, cf = loftr_match(f, m, conf_threshold=.5)
        return (k0+.5)*sf-.5+of, (k1+.5)*sm-.5+om, cf

    k0, k1, cf = match(fixed_masks)
    pairs, missing = select_suggestions(k0, k1, cf, fixed_masks, moving_masks)
    # Retry at most four uncovered spatial areas, regardless of selection count.
    # Crops include a half-cell overlap; coverage cells are not hard crop edges.
    union = fixed_masks[0]
    ys, xs = np.nonzero(union)
    cell_w, cell_h = (np.ptp(xs)+1)/4, (np.ptp(ys)+1)/4
    for area in [v for v in missing if v.startswith('fixed:')][:4]:
        fi = int(area.split(':')[1])
        x, y = xs.min() + (fi%4)*cell_w, ys.min() + (fi//4)*cell_h
        local = np.zeros_like(union)
        left, top = max(0, int(x-cell_w/2)), max(0, int(y-cell_h/2))
        right, bottom = int(x+cell_w*1.5)+1, int(y+cell_h*1.5)+1
        local[top:bottom, left:right] = union[top:bottom, left:right]
        a, b, c = match([local])
        k0, k1, cf = np.concatenate((k0, a)), np.concatenate((k1, b)), np.concatenate((cf, c))
    return select_suggestions(k0, k1, cf, fixed_masks, moving_masks)


def register_anchors(fixed, moving, points, groups):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 4 or len(pts) < 2 or not np.isfinite(pts).all():
        raise ValueError('서로 떨어진 유효한 앵커를 2쌍 이상 지정하세요.')
    target, source = pts[:, :2], pts[:, 2:]
    for p in (source, target):
        if np.linalg.norm(np.ptp(p, axis=0)) < 2:
            raise ValueError('앵커가 한곳에 겹쳐 있습니다. 떨어진 위치를 지정하세요.')
    counts = Counter(groups)
    weights = np.array([1 / counts[g] for g in groups])
    matrix = _fit_similarity_lstsq(source, target, weights)
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if not np.isfinite(matrix).all() or scale < .05 or scale > 20:
        raise ValueError('앵커의 위치가 서로 맞지 않습니다. 점의 짝을 확인하세요.')
    error = np.linalg.norm(np.c_[source, np.ones(len(source))] @ matrix.T - target, axis=1)
    # Reference-pixel residual threshold scales with evaluation resolution.
    tolerance = max(1.5, max(fixed.shape[:2]) * 3 / 640)
    refs = [{'group': g, 'count': counts[g],
             'max_error': float(error[np.array(groups) == g].max())} for g in counts]
    conflict = bool(np.max(error) > tolerance)
    registered = cv2.warpAffine(moving, matrix, (fixed.shape[1], fixed.shape[0]))
    # A small diagnostic image avoids an extra full-resolution concatenation.
    sf, sm = 640/max(fixed.shape[:2]), 640/max(moving.shape[:2])
    f = cv2.resize(fixed, None, fx=sf, fy=sf)
    m = cv2.resize(moving, None, fx=sm, fy=sm)
    return {'status': 'warn', 'gate': 'anchor_similarity', 'label': '앵커 기준 정합',
            'reason': '선택한 앵커들이 서로 다른 정합을 요구합니다. 점의 짝과 결과를 확인하세요.' if conflict else '앵커 기준으로 맞췄습니다. 관심 부위를 직접 확인하세요.',
            'M_full': _to_3x3(matrix), 'registered_img': registered,
            'match_viz': _draw_matches(f, m, target*sf, source*sm),
            'metrics': {'n_inlier': len(pts), 'inlier_ratio': 1.,
                        'reproj_median': float(np.median(error)), 'reproj_p90': float(np.percentile(error, 90)),
                        'anchor_residuals': error.tolist(), 'reference_groups': refs,
                        'reference_conflict': conflict, 'validation': 'fit_only',
                        'scale': scale, 'rotation_deg': float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))}}
