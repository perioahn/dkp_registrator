"""파이프라인 오케스트레이터 (Phase A~D 통합).

GUI 의존성 없음. tkinter import 금지.
"""

from __future__ import annotations

import numpy as np
import cv2

from matching import apply_soft_mask, filter_by_mask, loftr_match
from preprocess import apply_clahe, auto_orient_and_crop, resize_to_max
from transform import (
    compose_full_matrix,
    quality_gate_affine,
    quality_gate_similarity,
)

# 탐색 파라미터 레벨
CONF_LEVELS = (0.3, 0.2)
MAX_SIDES = (480, 640)
CLAHE_CLIPS = (2.0,)
MASK_SIGMAS = (5,)


def register_pair(fixed_img: np.ndarray,
                  moving_img: np.ndarray,
                  fixed_mask: np.ndarray,
                  moving_mask: np.ndarray,
                  refine: bool = False,
                  force_nocrop: bool = False,
                  hint: tuple | None = None) -> dict:
    """전체 정합 파이프라인.

    hint=(conf, max_side, clahe_clip, mask_sigma) 제공 시 최우선 시도.
    hint 미제공 시 기본 cascade: CLAHE=2.0, sigma=5, conf×max_side 순회.

    Args:
        fixed_img: 고정상 RGB 배열.
        moving_img: 이동상 RGB 배열.
        fixed_mask: 고정상 마스크 uint8.
        moving_mask: 이동상 마스크 uint8.
        refine: 미사용 (하위호환).
        force_nocrop: True이면 크롭 전처리 건너뛰기.
        hint: (conf, max_side[, clahe_clip, mask_sigma]) 튜플.

    Returns:
        정합 결과 딕셔너리.
    """
    debug = {}

    if np.sum(fixed_mask > 0) == 0 or np.sum(moving_mask > 0) == 0:
        return {
            'registered_img': None, 'M_full': None,
            'metrics': {'gate': 'none', 'status': 'fail', 'reason': 'empty_mask'},
            'path': 'failed', 'debug_images': debug,
        }

    # hint 정규화: 2-tuple → 4-tuple
    if hint is not None:
        hint = tuple(hint)
        if len(hint) == 2:
            hint = (hint[0], hint[1], 2.0, 5)

    # === Phase A: 크롭 + 회전 (CLAHE는 cascade에서) ===
    crop_ok = False
    if not force_nocrop:
        print("[Phase A] 자동 크롭+회전 전처리...")
        try:
            fixed_crop, fixed_mask_crop, M_rot_f, crop_off_f = \
                auto_orient_and_crop(fixed_img, fixed_mask)
            moving_crop, moving_mask_crop, M_rot_m, crop_off_m = \
                auto_orient_and_crop(moving_img, moving_mask)
            debug['fixed_crop'] = fixed_crop
            debug['moving_crop'] = moving_crop
            crop_ok = True
        except Exception as e:
            print(f"[WARN] 크롭 전처리 실패: {e}, no-crop 시도...")

    # === Phase B+C: cascade ===
    _MIN_FOR_ESTIMATE = 4
    result_path = None
    M_loftr = None
    final_metrics = None
    best_rejected = None
    use_crop = True
    scale_f = scale_m = None

    # --- 크롭 경로 cascade ---
    if crop_ok:
        _crop_cascade = [(c, m, 2.0, 5) for c in (0.3, 0.2, 0.15, 0.1)
                         for m in (640, 480)]
        if hint is not None:
            ht = hint
            if ht in _crop_cascade:
                _crop_cascade.remove(ht)
            _crop_cascade.insert(0, ht)
            print(f"[INFO] hint: conf={ht[0]}, ms={ht[1]}, "
                  f"clahe={ht[2]}, σ={ht[3]}")

        fixed_crop_gray = cv2.cvtColor(fixed_crop, cv2.COLOR_RGB2GRAY)
        moving_crop_gray = cv2.cvtColor(moving_crop, cv2.COLOR_RGB2GRAY)
        clahe_cache = {}

        for conf_thresh, ms, clip, sig in _crop_cascade:
            if clip not in clahe_cache:
                clahe_cache[clip] = (
                    apply_clahe(fixed_crop_gray, clip_limit=clip),
                    apply_clahe(moving_crop_gray, clip_limit=clip))
            fg, mg = clahe_cache[clip]
            debug.setdefault('fixed_clahe', fg)
            debug.setdefault('moving_clahe', mg)

            fixed_resized, sf = resize_to_max(fg, ms)
            moving_resized, sm = resize_to_max(mg, ms)

            fixed_mask_resized = cv2.resize(fixed_mask_crop,
                (fixed_resized.shape[1], fixed_resized.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            moving_mask_resized = cv2.resize(moving_mask_crop,
                (moving_resized.shape[1], moving_resized.shape[0]),
                interpolation=cv2.INTER_NEAREST)

            fixed_masked = apply_soft_mask(fixed_resized, fixed_mask_resized,
                                           sigma=sig)
            moving_masked = apply_soft_mask(moving_resized, moving_mask_resized,
                                            sigma=sig)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tooth_area = float(np.sum(cv2.erode(fixed_mask_resized, kernel) > 0))

            kpts0, kpts1, conf = loftr_match(
                fixed_masked, moving_masked, conf_threshold=conf_thresh)
            kpts0, kpts1, conf = filter_by_mask(
                kpts0, kpts1, conf, fixed_mask_resized, moving_mask_resized)
            n = len(kpts0)
            print(f"[Phase B] 크롭 (conf≥{conf_thresh}, ms={ms}, "
                  f"clahe={clip}, σ={sig}): {n}개")
            debug['n_filtered_matches'] = n

            if n < _MIN_FOR_ESTIMATE:
                continue

            # Similarity gate
            M_sim, inliers_sim = cv2.estimateAffinePartial2D(
                kpts1, kpts0, method=cv2.RANSAC,
                ransacReprojThreshold=3.0, confidence=0.99)
            if M_sim is not None:
                status, metrics = quality_gate_similarity(
                    kpts0, kpts1, M_sim, inliers_sim, tooth_area)
                metrics.update(gate='similarity', status=status,
                               crop_used=True, conf_threshold=conf_thresh,
                               max_side=ms, clahe_clip=clip, mask_sigma=sig)

                if status in ('pass', 'warn'):
                    M_loftr = M_sim
                    result_path = 'similarity'
                    final_metrics = metrics
                    scale_f, scale_m = sf, sm
                    print(f"[Phase C] Similarity: {status} "
                          f"(inlier={metrics['n_inlier']}/{metrics['n_total']}, "
                          f"rot={metrics.get('rotation_deg', 0):.1f}°, "
                          f"scale={metrics.get('scale', 1):.3f})")
                    break
                else:
                    print(f"[Phase C] Similarity: FAIL "
                          f"(inlier={metrics['n_inlier']}/{metrics['n_total']}, "
                          f"reproj={metrics.get('reproj_median', 0):.2f})")
                    if best_rejected is None or metrics['n_inlier'] > best_rejected.get('n_inlier', 0):
                        best_rejected = metrics

            # Affine gate
            if result_path is None:
                M_aff, inliers_aff = cv2.estimateAffine2D(
                    kpts1, kpts0, method=cv2.RANSAC,
                    ransacReprojThreshold=3.0, confidence=0.99)
                if M_aff is not None:
                    status, metrics = quality_gate_affine(
                        kpts0, kpts1, M_aff, inliers_aff, tooth_area)
                    metrics.update(gate='affine', status=status,
                                   crop_used=True, conf_threshold=conf_thresh,
                                   max_side=ms, clahe_clip=clip, mask_sigma=sig)

                    if status in ('pass', 'warn'):
                        M_loftr = M_aff
                        result_path = 'affine'
                        final_metrics = metrics
                        scale_f, scale_m = sf, sm
                        print(f"[Phase C] Affine: {status} "
                              f"(inlier={metrics['n_inlier']}/{metrics['n_total']})")
                        break
                    else:
                        print(f"[Phase C] Affine: FAIL "
                              f"(inlier={metrics['n_inlier']}/{metrics['n_total']}, "
                              f"reproj={metrics.get('reproj_median', 0):.2f})")
                        if best_rejected is None or metrics.get('n_inlier', 0) > best_rejected.get('n_inlier', 0):
                            best_rejected = metrics

    # --- No-crop 경로 cascade ---
    if result_path is None:
        print("[INFO] 크롭 경로 실패, no-crop 시도...")

        M_rot_f = np.eye(3); crop_off_f = (0, 0)
        M_rot_m = np.eye(3); crop_off_m = (0, 0)
        use_crop = False

        fixed_gray_full = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY)
        moving_gray_full = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)

        _nc_cascade = [(c, m, 2.0, 5) for c in (0.3, 0.2, 0.15, 0.1)
                       for m in (640, 480)]
        if hint is not None:
            ht = hint
            if ht in _nc_cascade:
                _nc_cascade.remove(ht)
            _nc_cascade.insert(0, ht)

        clahe_cache_nc = {}

        for conf_thresh, ms, clip, sig in _nc_cascade:
            if clip not in clahe_cache_nc:
                clahe_cache_nc[clip] = (
                    apply_clahe(fixed_gray_full, clip_limit=clip),
                    apply_clahe(moving_gray_full, clip_limit=clip))
            fg, mg = clahe_cache_nc[clip]

            fr_nc, sf = resize_to_max(fg, ms)
            mr_nc, sm = resize_to_max(mg, ms)

            fm_nc = cv2.resize(fixed_mask,
                (fr_nc.shape[1], fr_nc.shape[0]), interpolation=cv2.INTER_NEAREST)
            mm_nc = cv2.resize(moving_mask,
                (mr_nc.shape[1], mr_nc.shape[0]), interpolation=cv2.INTER_NEAREST)

            f_masked_nc = apply_soft_mask(fr_nc, fm_nc, sigma=sig)
            m_masked_nc = apply_soft_mask(mr_nc, mm_nc, sigma=sig)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tooth_area_nc = float(np.sum(cv2.erode(fm_nc, kernel) > 0))

            kpts0, kpts1, conf = loftr_match(
                f_masked_nc, m_masked_nc, conf_threshold=conf_thresh)
            kpts0, kpts1, conf = filter_by_mask(
                kpts0, kpts1, conf, fm_nc, mm_nc)
            n = len(kpts0)
            print(f"[Phase B] No-crop (conf≥{conf_thresh}, ms={ms}, "
                  f"clahe={clip}, σ={sig}): {n}개")

            if n < _MIN_FOR_ESTIMATE:
                continue

            # Similarity gate
            M_sim, inliers_sim = cv2.estimateAffinePartial2D(
                kpts1, kpts0, method=cv2.RANSAC,
                ransacReprojThreshold=3.0, confidence=0.99)
            if M_sim is not None:
                status, metrics = quality_gate_similarity(
                    kpts0, kpts1, M_sim, inliers_sim, tooth_area_nc)
                metrics.update(gate='similarity', status=status,
                               crop_used=False, conf_threshold=conf_thresh,
                               max_side=ms, clahe_clip=clip, mask_sigma=sig)

                if status in ('pass', 'warn'):
                    M_loftr = M_sim
                    result_path = 'similarity'
                    final_metrics = metrics
                    scale_f, scale_m = sf, sm
                    print(f"[Phase C] Similarity: {status} "
                          f"(inlier={metrics['n_inlier']}/{metrics['n_total']})")
                    break
                else:
                    print(f"[Phase C] Similarity: FAIL "
                          f"(inlier={metrics['n_inlier']}/{metrics['n_total']}, "
                          f"reproj={metrics.get('reproj_median', 0):.2f})")
                    if best_rejected is None or metrics['n_inlier'] > best_rejected.get('n_inlier', 0):
                        best_rejected = metrics

            # Affine gate
            if result_path is None:
                M_aff, inliers_aff = cv2.estimateAffine2D(
                    kpts1, kpts0, method=cv2.RANSAC,
                    ransacReprojThreshold=3.0, confidence=0.99)
                if M_aff is not None:
                    status, metrics = quality_gate_affine(
                        kpts0, kpts1, M_aff, inliers_aff, tooth_area_nc)
                    metrics.update(gate='affine', status=status,
                                   crop_used=False, conf_threshold=conf_thresh,
                                   max_side=ms, clahe_clip=clip, mask_sigma=sig)

                    if status in ('pass', 'warn'):
                        M_loftr = M_aff
                        result_path = 'affine'
                        final_metrics = metrics
                        scale_f, scale_m = sf, sm
                        print(f"[Phase C] Affine: {status} "
                              f"(inlier={metrics['n_inlier']}/{metrics['n_total']})")
                        break
                    else:
                        print(f"[Phase C] Affine: FAIL "
                              f"(inlier={metrics['n_inlier']}/{metrics['n_total']}, "
                              f"reproj={metrics.get('reproj_median', 0):.2f})")
                        if best_rejected is None or metrics.get('n_inlier', 0) > best_rejected.get('n_inlier', 0):
                            best_rejected = metrics

    # === 실패: 모든 cascade 소진 ===
    if result_path is None:
        fail_reason = 'insufficient_matches'
        if best_rejected is not None:
            fail_reason = (f"gate_fail: inlier={best_rejected.get('n_inlier', 0)}"
                          f"/{best_rejected.get('n_total', 0)}")

        print(f"[INFO] 정합 실패 — {fail_reason}")
        print("[INFO] 마스크를 더 넓게/정확하게 지정 후 재시도하세요.")

        if final_metrics is None:
            final_metrics = {'gate': 'none', 'status': 'fail'}
        final_metrics['reason'] = fail_reason

        return {
            'registered_img': None, 'M_full': None,
            'metrics': final_metrics, 'path': 'failed', 'debug_images': debug,
        }

    # === Phase D: 행렬 역산 + 원본 적용 ===
    print(f"[Phase D] path={result_path}, crop={'yes' if use_crop else 'no'}, "
          f"ms={final_metrics.get('max_side', '?')}")

    if result_path in ('similarity', 'affine'):
        try:
            M_full = compose_full_matrix(
                M_loftr,
                M_rot_f, crop_off_f, scale_f,
                M_rot_m, crop_off_m, scale_m
            )
        except np.linalg.LinAlgError:
            final_metrics['status'] = 'fail'
            return {
                'registered_img': None, 'M_full': None,
                'metrics': final_metrics, 'path': 'failed', 'debug_images': debug,
            }

        registered = cv2.warpAffine(
            moving_img, M_full[:2, :],
            (fixed_img.shape[1], fixed_img.shape[0])
        )

        debug['false_color'] = false_color(fixed_img, registered)

        return {
            'registered_img': registered,
            'M_full': M_full,
            'metrics': final_metrics,
            'path': result_path,
            'debug_images': debug,
        }

    return {
        'registered_img': None, 'M_full': None,
        'metrics': final_metrics, 'path': result_path, 'debug_images': debug,
    }


def _fit_similarity_lstsq(src: np.ndarray,
                          dst: np.ndarray) -> np.ndarray | None:
    """src→dst similarity transform (least squares)."""
    if len(src) < 2:
        return None
    x, y = src[:, 0].astype(np.float64), src[:, 1].astype(np.float64)
    xp, yp = dst[:, 0].astype(np.float64), dst[:, 1].astype(np.float64)
    n = len(src)
    A = np.zeros((2 * n, 4), dtype=np.float64)
    A[0::2, 0] = x;  A[0::2, 1] = -y; A[0::2, 2] = 1
    A[1::2, 0] = y;  A[1::2, 1] = x;  A[1::2, 3] = 1
    b = np.empty(2 * n, dtype=np.float64)
    b[0::2] = xp; b[1::2] = yp
    res = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[res[0], -res[1], res[2]],
                     [res[1],  res[0], res[3]]], dtype=np.float64)


def _fit_affine_lstsq(src: np.ndarray,
                       dst: np.ndarray) -> np.ndarray | None:
    """src→dst full affine transform (least squares)."""
    if len(src) < 3:
        return None
    x, y = src[:, 0].astype(np.float64), src[:, 1].astype(np.float64)
    xp, yp = dst[:, 0].astype(np.float64), dst[:, 1].astype(np.float64)
    n = len(src)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    A[0::2, 0] = x; A[0::2, 1] = y; A[0::2, 2] = 1
    A[1::2, 3] = x; A[1::2, 4] = y; A[1::2, 5] = 1
    b = np.empty(2 * n, dtype=np.float64)
    b[0::2] = xp; b[1::2] = yp
    res = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[res[0], res[1], res[2]],
                     [res[3], res[4], res[5]]], dtype=np.float64)


def register_test(fixed_img: np.ndarray, moving_img: np.ndarray,
                  fixed_mask: np.ndarray,
                  moving_mask: np.ndarray,
                  anchor_points: list[tuple] | None = None) -> list[dict]:
    """정합 테스트 (conf × CLAHE × max_side × sigma combos).

    LoFTR 캐싱. 각 조합에 gate check + warp 수행.

    Args:
        fixed_img: 고정상 RGB 배열.
        moving_img: 이동상 RGB 배열.
        fixed_mask: 고정상 마스크 uint8.
        moving_mask: 이동상 마스크 uint8.
        anchor_points: [(fx, fy, mx, my), ...] 강제 대응점.

    Returns:
        결과 딕셔너리 리스트.
    """
    _MIN_FOR_ESTIMATE = 4

    # Phase A: crop + orient
    crop_ok = False
    try:
        fc, fmc, M_rot_f, crop_off_f = auto_orient_and_crop(fixed_img, fixed_mask)
        mc, mmc, M_rot_m, crop_off_m = auto_orient_and_crop(moving_img, moving_mask)
        crop_ok = True
        print("[Register Test] Crop 전처리 완료")
    except Exception as e:
        print(f"[Register Test] Crop 실패: {e}, no-crop 사용")
        fc, mc = fixed_img, moving_img
        fmc, mmc = fixed_mask, moving_mask
        M_rot_f = np.eye(3); crop_off_f = (0, 0)
        M_rot_m = np.eye(3); crop_off_m = (0, 0)

    # 앵커 포인트: 원본 좌표 → crop/orient 좌표로 변환
    anchor_f_crop = []  # crop 좌표계 (resize 전)
    anchor_m_crop = []
    if anchor_points:
        for fx, fy, mx, my in anchor_points:
            # M_rot 적용 (3x3 affine)
            fp = M_rot_f[:2, :2] @ np.array([fx, fy]) + M_rot_f[:2, 2]
            mp = M_rot_m[:2, :2] @ np.array([mx, my]) + M_rot_m[:2, 2]
            # crop_offset 빼기
            fp[0] -= crop_off_f[0]
            fp[1] -= crop_off_f[1]
            mp[0] -= crop_off_m[0]
            mp[1] -= crop_off_m[1]
            anchor_f_crop.append(fp)
            anchor_m_crop.append(mp)
        print(f"[Register Test] Anchor points {len(anchor_points)}쌍 변환 완료")

    fc_gray = cv2.cvtColor(fc, cv2.COLOR_RGB2GRAY) if len(fc.shape) == 3 else fc
    mc_gray = cv2.cvtColor(mc, cv2.COLOR_RGB2GRAY) if len(mc.shape) == 3 else mc

    clahe_imgs = {}
    for clip in CLAHE_CLIPS:
        clahe_imgs[clip] = (
            apply_clahe(fc_gray, clip_limit=clip),
            apply_clahe(mc_gray, clip_limit=clip))

    # LoFTR: masked + unmasked 합산
    raw_cache = {}
    run_count = 0
    for clip in CLAHE_CLIPS:
        fg, mg = clahe_imgs[clip]
        for ms in MAX_SIDES:
            fr, sf = resize_to_max(fg, ms)
            mr, sm = resize_to_max(mg, ms)
            fm = cv2.resize(fmc, (fr.shape[1], fr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            mm = cv2.resize(mmc, (mr.shape[1], mr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tooth_area = float(np.sum(cv2.erode(fm, kernel) > 0))

            # Global
            nk0, nk1, ncf = loftr_match(fr, mr, conf_threshold=0.1)
            run_count += 1

            for sig in MASK_SIGMAS:
                # Masked
                f_masked = apply_soft_mask(fr, fm, sigma=sig)
                m_masked = apply_soft_mask(mr, mm, sigma=sig)
                mk0, mk1, mcf = loftr_match(
                    f_masked, m_masked, conf_threshold=0.1)
                mk0, mk1, mcf = filter_by_mask(mk0, mk1, mcf, fm, mm)
                run_count += 1

                # Combined
                k0 = np.concatenate([mk0, nk0]) if len(mk0) else nk0
                k1 = np.concatenate([mk1, nk1]) if len(mk1) else nk1
                cf = np.concatenate([mcf, ncf]) if len(mcf) else ncf

                raw_cache[(clip, ms, sig)] = {
                    'kpts0': k0, 'kpts1': k1, 'conf': cf,
                    'scale_f': sf, 'scale_m': sm,
                    'tooth_area': tooth_area,
                }
                print(f"[Register Test] LoFTR {run_count} "
                      f"(clahe={clip} ms={ms} σ={sig}): "
                      f"{len(k0)} matches (combined)")

    # Per combo: gate + warp
    results = []
    for conf_t in CONF_LEVELS:
        for clip in CLAHE_CLIPS:
            for ms in MAX_SIDES:
                for sig in MASK_SIGMAS:
                    raw = raw_cache[(clip, ms, sig)]
                    valid = raw['conf'] > conf_t
                    k0 = raw['kpts0'][valid]
                    k1 = raw['kpts1'][valid]
                    n = len(k0)

                    entry = {
                        'conf_threshold': conf_t, 'max_side': ms,
                        'clahe_clip': clip, 'mask_sigma': sig,
                        'n_matches': n, 'status': 'fail',
                        'gate': 'none', 'metrics': {}, 'false_color': None,
                    }

                    if n < _MIN_FOR_ESTIMATE:
                        entry['reason'] = f'{n} matches'
                        results.append(entry)
                        continue

                    sf = raw['scale_f']
                    sm = raw['scale_m']
                    ta = raw['tooth_area']
                    passed = False

                    # Similarity gate
                    M_sim, inliers_sim = cv2.estimateAffinePartial2D(
                        k1, k0, method=cv2.RANSAC,
                        ransacReprojThreshold=3.0, confidence=0.99)
                    if M_sim is not None:
                        status, metrics = quality_gate_similarity(
                            k0, k1, M_sim, inliers_sim, ta)
                        if status in ('pass', 'warn'):
                            M_use = M_sim
                            if anchor_f_crop:
                                inl = inliers_sim.ravel().astype(bool)
                                a_f = np.array(anchor_f_crop,
                                               dtype=np.float32) * sf
                                a_m = np.array(anchor_m_crop,
                                               dtype=np.float32) * sm
                                n_inl = int(np.sum(inl))
                                n_dup = max(n_inl // len(a_f), 5)
                                ak0 = np.concatenate(
                                    [k0[inl],
                                     np.repeat(a_f, n_dup, axis=0)])
                                ak1 = np.concatenate(
                                    [k1[inl],
                                     np.repeat(a_m, n_dup, axis=0)])
                                M_r = _fit_similarity_lstsq(ak1, ak0)
                                if M_r is not None:
                                    M_use = M_r
                                    print(f"[anchor] sim re-est: "
                                          f"{len(a_f)} anchors x{n_dup} "
                                          f"+ {n_inl} inliers")
                            try:
                                M_full = compose_full_matrix(
                                    M_use, M_rot_f, crop_off_f, sf,
                                    M_rot_m, crop_off_m, sm)
                                reg = cv2.warpAffine(
                                    moving_img, M_full[:2, :],
                                    (fixed_img.shape[1], fixed_img.shape[0]))
                                entry.update(
                                    status=status, gate='similarity',
                                    metrics=metrics, M_full=M_full,
                                    registered_img=reg,
                                    false_color=false_color(fixed_img, reg))
                                passed = True
                            except np.linalg.LinAlgError:
                                pass

                    # Affine gate
                    if not passed:
                        M_aff, inliers_aff = cv2.estimateAffine2D(
                            k1, k0, method=cv2.RANSAC,
                            ransacReprojThreshold=3.0, confidence=0.99)
                        if M_aff is not None:
                            status, metrics = quality_gate_affine(
                                k0, k1, M_aff, inliers_aff, ta)
                            if status in ('pass', 'warn'):
                                M_use = M_aff
                                if anchor_f_crop:
                                    inl = inliers_aff.ravel().astype(bool)
                                    a_f = np.array(anchor_f_crop,
                                                   dtype=np.float32) * sf
                                    a_m = np.array(anchor_m_crop,
                                                   dtype=np.float32) * sm
                                    n_inl = int(np.sum(inl))
                                    n_dup = max(n_inl // len(a_f), 5)
                                    ak0 = np.concatenate(
                                        [k0[inl],
                                         np.repeat(a_f, n_dup, axis=0)])
                                    ak1 = np.concatenate(
                                        [k1[inl],
                                         np.repeat(a_m, n_dup, axis=0)])
                                    M_r = _fit_affine_lstsq(ak1, ak0)
                                    if M_r is not None:
                                        M_use = M_r
                                        print(f"[anchor] aff re-est: "
                                              f"{len(a_f)} anchors x{n_dup}"
                                              f" + {n_inl} inliers")
                                try:
                                    M_full = compose_full_matrix(
                                        M_use, M_rot_f, crop_off_f, sf,
                                        M_rot_m, crop_off_m, sm)
                                    reg = cv2.warpAffine(
                                        moving_img, M_full[:2, :],
                                        (fixed_img.shape[1], fixed_img.shape[0]))
                                    entry.update(
                                        status=status, gate='affine',
                                        metrics=metrics, M_full=M_full,
                                        registered_img=reg,
                                        false_color=false_color(fixed_img, reg))
                                    passed = True
                                except np.linalg.LinAlgError:
                                    pass

                    if not passed:
                        entry['reason'] = 'gate_fail'

                    tag = entry['status'].upper() if passed else "FAIL"
                    print(f"[Register Test] conf>{conf_t} ms={ms} "
                          f"clahe={clip} σ={sig}: {n}개 → {tag}")
                    results.append(entry)

    return results



def false_color(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """정합 결과 false color 시각화.

    Args:
        img1: 기준 이미지 (RGB 또는 grayscale).
        img2: 정합된 이미지 (RGB 또는 grayscale).

    Returns:
        False color RGB 배열.
    """
    img1 = img1.copy()
    img2 = img2.copy()
    if img1.max() <= 1:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.max() <= 1:
        img2 = (img2 * 255).astype(np.uint8)

    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2

    result = img1.copy()
    result[:, :, 0] = gray2
    result[:, :, 2] = gray2
    return result
