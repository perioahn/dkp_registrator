"""Bounded, mask-free regional rematching. No user masks/anchors are created.

Candidates live in original image coordinates; fitting and validation use a
640-pixel long-side frame so thresholds do not depend on camera resolution.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from preprocess import resize_to_max
from transform import is_similarity, quality_gate_similarity
from sliding_windows import (SearchBudget, spread_indices, neighborhoods, density_weights,
                             propose_windows, next_window, crop_pair, followup_window, overlap,
                             inside_window)


def project(points, matrix):
    return np.asarray(points) @ matrix[:2, :2].T + matrix[:2, 2]


def collect_level(bank, k0, k1, confidence, sf, sm, prewarp,
                  rotation_f, offset_f, rotation_m, offset_m):
    """Undo prewarp, resize, crop and orientation before accumulating points."""
    if not len(k0):
        return
    p0 = project(np.asarray(k0) / sf + offset_f, np.linalg.inv(rotation_f))
    p1 = project(k1, np.linalg.inv(prewarp)) / sm + offset_m
    p1 = project(p1, np.linalg.inv(rotation_m))
    bank.append((p0, p1, np.asarray(confidence), min(sf, sm)))


def bounded_pool(p0, p1, confidence, shape0, shape1, limit=384):
    """Deduplicate endpoints, then spatially sample without fixed partitions."""
    valid = (np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
             & np.isfinite(confidence) & (confidence > 0)
             & (p0 >= 0).all(axis=1) & (p1 >= 0).all(axis=1)
             & (p0 < [shape0[1], shape0[0]]).all(axis=1)
             & (p1 < [shape1[1], shape1[0]]).all(axis=1))
    p0, p1, confidence = p0[valid], p1[valid], confidence[valid]
    keep, buckets0, buckets1 = [], {}, {}
    for i in np.argsort(-confidence, kind='stable'):
        duplicate = False
        keys = []
        # Hash buckets only accelerate endpoint distance checks; they do not
        # select windows, assign quotas, weight the fit or define validation.
        for points, buckets in ((p0,buckets0),(p1,buckets1)):
            bx,by = np.floor(points[i]/2).astype(int)
            keys.append((bx,by))
            nearby = [j for dx in (-1,0,1) for dy in (-1,0,1)
                      for j in buckets.get((bx+dx,by+dy),[])]
            if nearby and np.any(np.linalg.norm(points[nearby]-points[i],axis=1) <= 2):
                duplicate = True
                break
        if duplicate:
            continue
        keep.append(i)
        buckets0.setdefault(keys[0],[]).append(i)
        buckets1.setdefault(keys[1],[]).append(i)
    p0,p1,confidence = p0[keep],p1[keep],confidence[keep]
    if len(p0) > limit:
        chosen = spread_indices(p0,confidence,limit,
                                other=p1*(min(shape0[:2])/min(shape1[:2])))
        p0,p1,confidence = p0[chosen],p1[chosen],confidence[chosen]
    return p0,p1,confidence


def _fit(p0, p1, confidence, shape0, shape1, cfg):
    from register import _fit_similarity_lstsq
    if len(p0) < cfg.sim_gate.min_inlier_fail:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        p1, p0, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh,
        maxIters=2000, confidence=.99)
    if M is None or inliers is None:
        return None
    keep = inliers.ravel().astype(bool)
    if keep.sum() < cfg.sim_gate.min_inlier_fail:
        return None
    weights = density_weights(p0[keep],p1[keep],shape0,shape1)
    M = _fit_similarity_lstsq(p1[keep], p0[keep], confidence[keep] * weights)
    if M is None or not is_similarity(M):
        return None
    keep = np.linalg.norm(project(p1, M) - p0, axis=1) <= cfg.ransac_thresh
    status, metrics = quality_gate_similarity(p0, p1, M, keep,
                                               shape0[0] * shape0[1], cfg.sim_gate)
    if status == 'fail':
        return None
    return M, keep, status, metrics


def validate_candidate(candidate, baseline, p0, p1, shape, threshold, *, guard_only=False):
    """Shared held-out global evidence; no region may silently deteriorate."""
    if len(p0) < 12:
        return False, 'insufficient_validation'
    groups = neighborhoods(p0, shape)
    occupied = [g for g in np.unique(groups) if np.sum(groups == g) >= 2]
    if len(occupied) < 3:
        return False, 'insufficient_coverage'
    new = np.linalg.norm(project(p1, candidate) - p0, axis=1)
    if baseline is None:
        good = sum(np.median(new[groups == g]) <= threshold for g in occupied)
        return (good >= 3 and np.mean(new <= threshold) >= .8), 'recovery_validation'
    old = np.linalg.norm(project(p1, baseline) - p0, axis=1)
    before = np.array([np.median(old[groups == g]) for g in occupied])
    after = np.array([np.median(new[groups == g]) for g in occupied])
    # Global low-resolution matches have threshold-sized localization uncertainty.
    allowed = np.maximum(before + .5, threshold) if guard_only else before + .5
    if np.any(after > allowed):
        return False, 'region_regression'
    if guard_only:
        return True, 'global_guard_passed'
    if np.mean(new <= threshold) + .01 < np.mean(old <= threshold):
        return False, 'coverage_regression'
    # A count increase alone is insufficient to replace an already good fit.
    improved = (np.mean(after) < np.mean(before) - .05
                or np.sum(after <= threshold) > np.sum(before <= threshold))
    return bool(improved), 'improved' if improved else 'baseline_equivalent'


def refine_result(fixed, moving, baseline, bank, cfg, *, match_fn, budget=None):
    """Return an accepted regional result or the intact global result + audit."""
    started = time.perf_counter()
    budget = budget if budget is not None else SearchBudget()
    audit = {'enabled': True, 'attempted_crops': 0, 'accepted_crops': 0,
             'adopted': False, 'reason': 'no_seeds', 'method': 'adaptive_sliding', 'windows': []}

    def finish():
        elapsed = time.perf_counter()-started
        budget.seconds += elapsed
        audit.update(elapsed_seconds=round(elapsed,3),pair_crop_calls=budget.calls)

    def retained(reason):
        audit['reason'] = reason
        finish()
        return {**baseline, 'metrics': {**baseline.get('metrics', {}), 'regional': audit}}

    if not bank:
        return retained('no_seeds')
    sf, sm = min(1., 640 / max(fixed.shape[:2])), min(1., 640 / max(moving.shape[:2]))
    shape0 = np.array(fixed.shape[:2]) * sf
    shape1 = np.array(moving.shape[:2]) * sm
    p0, p1 = (np.concatenate([b[i] for b in bank]) for i in range(2))
    precision = np.array([b[3] if len(b) > 3 else 1. for b in bank])
    cf = np.concatenate([b[2] * (weight / precision.max()) for b, weight in zip(bank, precision)])
    p0, p1, cf = bounded_pool(p0 * sf, p1 * sm, cf, shape0, shape1)
    audit['global_points'] = len(p0)
    if len(p0) < 16:
        return retained('insufficient_seeds')
    windows = propose_windows(p0, p1, cf, shape0, shape1)
    if not windows:
        return retained('no_consistent_regions')

    # Hold out every fourth point within each region, before local rematching.
    held = np.zeros(len(p0), bool)
    groups = neighborhoods(p0,shape0)
    for group in np.unique(groups):
        held[np.flatnonzero(groups == group)[::4]] = True
    v0, v1 = p0[held], p1[held]
    # Confidence alone does not express localization precision. A full-image
    # downsample must not outweigh an original-resolution crop by sheer count.
    train0, train1, train_cf = p0[~held], p1[~held], cf[~held] * min(sf,sm)**2
    additions, validation, visited, followups = [], [], [], []
    while windows or followups:
        if (budget.calls >= cfg.regional_max_crops
                or budget.seconds + time.perf_counter()-started >= cfg.regional_budget_seconds):
            audit['budget_exhausted'] = True
            break
        covered = np.zeros(len(p0),bool)
        for previous in visited:
            covered |= inside_window(p0,p1,previous)
        # Further zoom is useful after supported areas are covered, not merely
        # because the first few easy windows have returned many matches.
        if followups and (not windows or (len(visited)>=3 and covered.mean()>=.65)):
            window = next_window(followups,visited,p0,p1,cf)
        else:
            window = next_window(windows or followups,visited,p0,p1,cf)
        visited.append(window)
        fpatch,mpatch,(fs,fo),(ms,mo) = crop_pair(
            fixed,moving,window,sf,sm,cfg.regional_max_side,cfg.clahe_clip)
        pixels = fpatch.size+mpatch.size
        if budget.pixels+pixels > cfg.regional_max_crops*2*cfg.regional_max_side**2:
            audit['budget_exhausted'] = True
            break
        budget.calls += 1
        budget.pixels += pixels
        audit['attempted_crops'] += 1
        audit['windows'].append({'fixed_center':(window.center0/sf).tolist(),
            'moving_center':(window.center1/sm).tolist(), 'fixed_side':window.side0/sf,
            'moving_side':window.side1/sm, 'depth':window.depth, 'input_side':fpatch.shape[0]})
        try:
            k0, k1, confidence = match_fn(fpatch, mpatch, conf_threshold=cfg.pyramid_conf)
        except (RuntimeError, cv2.error):
            # Optional refinement cannot discard an already completed baseline.
            return retained('local_inference_failed')
        inside = ((k0 >= 0).all(axis=1) & (k1 >= 0).all(axis=1)
                  & (k0 < fpatch.shape[0]).all(axis=1)
                  & (k1 < mpatch.shape[0]).all(axis=1)
                  & (confidence > cfg.pyramid_conf))
        k0, k1, confidence = k0[inside], k1[inside], confidence[inside]
        q0, q1 = (k0 / fs + fo) * sf, (k1 / ms + mo) * sm
        q0, q1, confidence = bounded_pool(q0, q1, confidence, shape0, shape1)
        if len(q0) < 8:
            continue
        local, inliers = cv2.estimateAffinePartial2D(
            q1, q0, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh)
        if local is None or not is_similarity(local) or int(inliers.sum()) < 8:
            continue
        keep = inliers.ravel().astype(bool)
        followup = followup_window(window,q0[keep],q1[keep],shape0,shape1)
        if followup is not None and all(overlap(followup,w) < .8 for w in visited+followups):
            followups.append(followup)
        # Do not reintroduce near-duplicates of validation points into fitting.
        keep &= (np.linalg.norm(q0[:, None] - v0, axis=2).min(axis=1) > 2)
        keep &= (np.linalg.norm(q1[:, None] - v1, axis=2).min(axis=1) > 2)
        if keep.sum() < 4:
            continue
        q0, q1, confidence = q0[keep], q1[keep], confidence[keep]
        local_held = np.zeros(len(q0), bool)
        groups = neighborhoods(q0,shape0)
        for group in np.unique(groups):
            local_held[np.flatnonzero(groups == group)[::4]] = True
        validation.append((q0[local_held], q1[local_held], confidence[local_held]))
        additions.append((q0[~local_held], q1[~local_held], confidence[~local_held]*min(fs,ms)**2))
        audit['accepted_crops'] += 1
    if not additions:
        return retained('no_additional_support')
    a0, a1, acf = (np.concatenate([a[i] for a in additions]) for i in range(3))
    a0, a1, acf = bounded_pool(a0, a1, acf, shape0, shape1)
    audit['regional_points'] = len(a0)
    local_v0, local_v1, local_vcf = (np.concatenate([a[i] for a in validation]) for i in range(3))
    local_v0, local_v1, _ = bounded_pool(local_v0, local_v1, local_vcf, shape0, shape1)
    train0, train1, train_cf = bounded_pool(
        np.concatenate([train0, a0]), np.concatenate([train1, a1]),
        np.concatenate([train_cf, acf]), shape0, shape1)
    # A held-out point from one crop must not leak back through an overlapping crop.
    separate = ((np.linalg.norm(train0[:, None] - local_v0, axis=2).min(axis=1) > 2)
                & (np.linalg.norm(train1[:, None] - local_v1, axis=2).min(axis=1) > 2))
    train0, train1, train_cf = train0[separate], train1[separate], train_cf[separate]
    base_M = baseline.get('M_full') if baseline.get('status') in ('pass', 'warn') else None
    normalized_base = (np.diag([sf, sf, 1]) @ base_M @ np.diag([1/sm, 1/sm, 1])
                       if base_M is not None else None)
    local_separate = ((np.linalg.norm(a0[:,None]-local_v0,axis=2).min(axis=1)>2)
                      & (np.linalg.norm(a1[:,None]-local_v1,axis=2).min(axis=1)>2))
    # The global bank can contain systematic coarse localization error. Compare
    # a merged fit with a precise-local fit; both must pass the same global guard.
    accepted_fits = []
    reason = 'candidate_gate_failed'
    for name,f0,f1,fc in (('merged',train0,train1,train_cf),
                          ('local',a0[local_separate],a1[local_separate],acf[local_separate])):
        fitted = _fit(f0,f1,fc,shape0,shape1,cfg)
        if fitted is None:
            continue
        M,keep,status,metrics = fitted
        accepted,reason = validate_candidate(M,normalized_base,v0,v1,shape0,
                                              cfg.ransac_thresh,guard_only=True)
        if accepted:
            accepted,reason = validate_candidate(M,normalized_base,local_v0,local_v1,
                                                  shape0,cfg.ransac_thresh)
        if accepted:
            errors = np.linalg.norm(project(local_v1,M)-local_v0,axis=1)
            labels = neighborhoods(local_v0,shape0)
            score = np.mean([np.median(errors[labels==g]) for g in np.unique(labels)])
            accepted_fits.append((score,name,f0,f1,fitted))
    if not accepted_fits:
        return retained(reason)
    _,fit_name,train0,train1,(M,keep,status,metrics) = min(accepted_fits,key=lambda f:f[0])
    audit['fit_source'] = fit_name
    reason = 'improved' if normalized_base is not None else 'recovery_validation'
    full = np.eye(3)
    full[:2] = M
    full = np.diag([1/sf, 1/sf, 1]) @ full @ np.diag([sm, sm, 1])
    if not is_similarity(full):
        return retained('invalid_similarity')
    from register import _draw_matches, false_color
    registered = cv2.warpAffine(moving, full[:2], (fixed.shape[1], fixed.shape[0]))
    audit.update(adopted=True, reason=reason, validation_points=len(v0)+len(local_v0))
    finish()
    metrics.update(regional=audit, validation='held_out_global_and_local')
    # Retain a warning from the baseline: additional fitting is not clinical validation.
    if baseline.get('status') == 'warn':
        status = 'warn'
    return {**baseline, 'status': status, 'gate': 'similarity', 'reason': None,
            'label': 'Pyramid + regional', 'n_matches': len(train0), 'metrics': metrics,
            'M_full': full, 'registered_img': registered,
            'false_color': false_color(fixed, registered),
            'match_viz': _draw_matches(resize_to_max(fixed, 640)[0],
                                       resize_to_max(moving, 640)[0], train0, train1, keep)}
