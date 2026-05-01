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

# 피라미드 정합
PYRAMID_LEVELS = (320, 480, 640)
PYRAMID_CONF = 0.2


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


# ── 피라미드 헬퍼 ──

def _to_3x3(M_2x3: np.ndarray) -> np.ndarray:
    """2x3 affine → 3x3 homogeneous."""
    M = np.eye(3, dtype=np.float64)
    M[:2] = M_2x3
    return M


def _rescale_M(M_3x3: np.ndarray,
               sf_prev: float, sm_prev: float,
               sf_next: float, sm_next: float) -> np.ndarray:
    """피라미드 레벨 간 affine 좌표 변환.

    M은 moving→fixed 매핑.
    fixed 쪽은 sf_next/sf_prev 비율, moving 쪽은 sm_prev/sm_next 비율.
    """
    rf = sf_next / sf_prev
    rm = sm_prev / sm_next
    return np.diag([rf, rf, 1.0]) @ M_3x3 @ np.diag([rm, rm, 1.0])


def _make_fail_entry(reason: str) -> dict:
    """실패 결과 딕셔너리를 생성한다."""
    return {
        'conf_threshold': PYRAMID_CONF, 'max_side': PYRAMID_LEVELS[-1],
        'clahe_clip': 2.0, 'mask_sigma': 5, 'n_matches': 0,
        'status': 'fail', 'gate': 'none', 'metrics': {},
        'false_color': None, 'registered_img': None, 'reason': reason,
        'label': 'Pyramid',
    }


def _draw_matches(img_f: np.ndarray, img_m: np.ndarray,
                  kpts0: np.ndarray, kpts1: np.ndarray,
                  inliers: np.ndarray | None = None,
                  max_draw: int = 300) -> np.ndarray:
    """Fixed/Moving 좌우 배치 + 인라이어/아웃라이어 연결선 시각화."""
    f3 = cv2.cvtColor(img_f, cv2.COLOR_GRAY2RGB) if img_f.ndim == 2 else img_f.copy()
    m3 = cv2.cvtColor(img_m, cv2.COLOR_GRAY2RGB) if img_m.ndim == 2 else img_m.copy()
    h1, w1 = f3.shape[:2]
    h2, w2 = m3.shape[:2]
    h = max(h1, h2)
    out = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = f3
    out[:h2, w1:] = m3

    n = len(kpts0)
    if n == 0:
        return out

    inl = inliers.ravel().astype(bool) if inliers is not None else np.ones(n, dtype=bool)

    # Subsample: keep all inliers, subsample outliers
    idx = np.arange(n)
    if n > max_draw:
        inl_idx = idx[inl]
        out_idx = idx[~inl]
        keep = max_draw - len(inl_idx)
        if keep > 0 and len(out_idx) > keep:
            out_idx = np.random.choice(out_idx, keep, replace=False)
        elif keep <= 0:
            out_idx = np.array([], dtype=int)
        idx = np.concatenate([out_idx, inl_idx])

    # Outliers (red) first, then inliers (green) on top
    for i in idx:
        if inl[i]:
            continue
        p0 = (int(kpts0[i, 0]), int(kpts0[i, 1]))
        p1 = (int(kpts1[i, 0]) + w1, int(kpts1[i, 1]))
        cv2.line(out, p0, p1, (255, 50, 50), 1, cv2.LINE_AA)

    for i in idx:
        if not inl[i]:
            continue
        p0 = (int(kpts0[i, 0]), int(kpts0[i, 1]))
        p1 = (int(kpts1[i, 0]) + w1, int(kpts1[i, 1]))
        cv2.line(out, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(out, p0, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p1, 3, (0, 255, 0), -1, cv2.LINE_AA)

    return out


def _fit_similarity_lstsq(src: np.ndarray, dst: np.ndarray,
                          weights: np.ndarray | None = None,
                          ) -> np.ndarray | None:
    """src→dst similarity transform (weighted least squares)."""
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
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))
        W = np.repeat(w, 2)  # (2n,)
        A *= W[:, np.newaxis]
        b *= W
    res = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[res[0], -res[1], res[2]],
                     [res[1],  res[0], res[3]]], dtype=np.float64)


def _fit_affine_lstsq(src: np.ndarray, dst: np.ndarray,
                       weights: np.ndarray | None = None,
                       ) -> np.ndarray | None:
    """src→dst full affine transform (weighted least squares)."""
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
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))
        W = np.repeat(w, 2)
        A *= W[:, np.newaxis]
        b *= W
    res = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[res[0], res[1], res[2]],
                     [res[3], res[4], res[5]]], dtype=np.float64)


def _run_gate(k0, k1, conf, tooth_area):
    """Similarity → Affine 폴백으로 RANSAC + quality gate 수행."""
    # Similarity
    M_sim, inliers_sim = cv2.estimateAffinePartial2D(
        k1, k0, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, confidence=0.99)
    if M_sim is not None:
        status, met = quality_gate_similarity(
            k0, k1, M_sim, inliers_sim, tooth_area)
        if status in ('pass', 'warn'):
            return M_sim, inliers_sim, 'similarity', status, met, conf
    # Affine fallback
    M_aff, inliers_aff = cv2.estimateAffine2D(
        k1, k0, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, confidence=0.99)
    if M_aff is not None:
        status, met = quality_gate_affine(
            k0, k1, M_aff, inliers_aff, tooth_area)
        if status in ('pass', 'warn'):
            return M_aff, inliers_aff, 'affine', status, met, conf
    return None, None, 'none', 'fail', {}, conf


def _match_at_level(fixed_L, moving_L, fixed_mask_L, moving_mask_L,
                    conf_threshold):
    """Global + Masked LoFTR → 합산 매칭."""
    # Global
    nk0, nk1, ncf = loftr_match(fixed_L, moving_L, conf_threshold=0.1)
    # Masked
    f_masked = apply_soft_mask(fixed_L, fixed_mask_L, sigma=5)
    m_masked = apply_soft_mask(moving_L, moving_mask_L, sigma=5)
    mk0, mk1, mcf = loftr_match(f_masked, m_masked, conf_threshold=0.1)
    mk0, mk1, mcf = filter_by_mask(
        mk0, mk1, mcf, fixed_mask_L, moving_mask_L)
    # Combine
    if len(mk0) and len(nk0):
        k0 = np.concatenate([mk0, nk0])
        k1 = np.concatenate([mk1, nk1])
        cf = np.concatenate([mcf, ncf])
    elif len(mk0):
        k0, k1, cf = mk0, mk1, mcf
    else:
        k0, k1, cf = nk0, nk1, ncf
    # Confidence filter
    valid = cf > conf_threshold
    return k0[valid], k1[valid], cf[valid]


def _single_pass_fallback(fc_clahe, mc_clahe, fmc, mmc,
                          M_rot_f, crop_off_f, M_rot_m, crop_off_m,
                          fixed_img, moving_img,
                          anchor_f_crop, anchor_m_crop):
    """피라미드 L0 실패 시 640 단일 패스 폴백."""
    ms = 640
    fr, sf = resize_to_max(fc_clahe, ms)
    mr, sm = resize_to_max(mc_clahe, ms)
    fm = cv2.resize(fmc, (fr.shape[1], fr.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    mm = cv2.resize(mmc, (mr.shape[1], mr.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tooth_area = float(np.sum(cv2.erode(fm, kernel) > 0))

    k0, k1, conf = _match_at_level(fr, mr, fm, mm, PYRAMID_CONF)
    n = len(k0)
    print(f"[Fallback 640] {n} matches")

    if n < 4:
        print(f"[Fallback 640] 매치 부족 → FAIL")
        return _make_fail_entry(f'fallback_{n}_matches')

    M_est, inliers, gate, status, met, _ = _run_gate(
        k0, k1, conf, tooth_area)
    if M_est is None:
        print(f"[Fallback 640] gate FAIL")
        return _make_fail_entry('fallback_gate_fail')

    print(f"[Fallback 640] {gate} {status}"
          f"  inlier={met.get('n_inlier', 0)}"
          f"  ratio={met.get('inlier_ratio', 0):.2f}"
          f"  reproj={met.get('reproj_median', -1):.1f}"
          f"  rot={met.get('rotation_deg', 0):.1f}°"
          f"  scale={met.get('scale', 0):.3f}")

    M_use = M_est
    # Anchor refit
    if anchor_f_crop:
        inl = inliers.ravel().astype(bool)
        a_f = np.array(anchor_f_crop, dtype=np.float32) * sf
        a_m = np.array(anchor_m_crop, dtype=np.float32) * sm
        n_inl = int(np.sum(inl))
        n_dup = max(n_inl // len(a_f), 5)
        ak0 = np.concatenate([k0[inl], np.repeat(a_f, n_dup, axis=0)])
        ak1 = np.concatenate([k1[inl], np.repeat(a_m, n_dup, axis=0)])
        w = np.concatenate([conf[inl],
                            np.full(len(a_f) * n_dup, 1.0)])
        fit_fn = (_fit_similarity_lstsq if gate == 'similarity'
                  else _fit_affine_lstsq)
        M_r = fit_fn(ak1, ak0, weights=w)
        if M_r is not None:
            M_use = M_r

    try:
        M_full = compose_full_matrix(
            M_use, M_rot_f, crop_off_f, sf, M_rot_m, crop_off_m, sm)
        reg = cv2.warpAffine(
            moving_img, M_full[:2, :],
            (fixed_img.shape[1], fixed_img.shape[0]))
    except np.linalg.LinAlgError:
        return _make_fail_entry('fallback_singular')

    return {
        'conf_threshold': PYRAMID_CONF, 'max_side': ms,
        'clahe_clip': 2.0, 'mask_sigma': 5, 'n_matches': n,
        'status': status, 'gate': gate, 'metrics': met,
        'M_full': M_full, 'registered_img': reg,
        'false_color': false_color(fixed_img, reg),
        'pyramid_level': -1,  # fallback
        'label': 'Pyramid fallback',
        'match_viz': _draw_matches(fr, mr, k0, k1, inliers),
    }


def register_test(fixed_img: np.ndarray, moving_img: np.ndarray,
                  fixed_mask: np.ndarray,
                  moving_mask: np.ndarray,
                  anchor_points: list[tuple] | None = None) -> list[dict]:
    """3단계 피라미드 정합 (320→480→640).

    Args:
        fixed_img: 고정상 RGB 배열.
        moving_img: 이동상 RGB 배열.
        fixed_mask: 고정상 마스크 uint8.
        moving_mask: 이동상 마스크 uint8.
        anchor_points: [(fx, fy, mx, my), ...] 강제 대응점.

    Returns:
        결과 딕셔너리 리스트 (단일 항목).
    """
    _MIN_FOR_ESTIMATE = 4

    # ── Phase A: crop + orient ──
    try:
        fc, fmc, M_rot_f, crop_off_f = auto_orient_and_crop(
            fixed_img, fixed_mask)
        mc, mmc, M_rot_m, crop_off_m = auto_orient_and_crop(
            moving_img, moving_mask)
        print("[Pyramid] Crop 전처리 완료")
    except Exception as e:
        print(f"[Pyramid] Crop 실패: {e}, no-crop 사용")
        fc, mc = fixed_img, moving_img
        fmc, mmc = fixed_mask, moving_mask
        M_rot_f = np.eye(3); crop_off_f = (0, 0)
        M_rot_m = np.eye(3); crop_off_m = (0, 0)

    # ── 앵커 좌표 변환 ──
    anchor_f_crop, anchor_m_crop = [], []
    if anchor_points:
        for fx, fy, mx, my in anchor_points:
            fp = M_rot_f[:2, :2] @ np.array([fx, fy]) + M_rot_f[:2, 2]
            mp = M_rot_m[:2, :2] @ np.array([mx, my]) + M_rot_m[:2, 2]
            fp[0] -= crop_off_f[0]; fp[1] -= crop_off_f[1]
            mp[0] -= crop_off_m[0]; mp[1] -= crop_off_m[1]
            anchor_f_crop.append(fp)
            anchor_m_crop.append(mp)
        print(f"[Pyramid] Anchor {len(anchor_points)}쌍 변환")

    # ── Grayscale + CLAHE ──
    fc_gray = cv2.cvtColor(fc, cv2.COLOR_RGB2GRAY) if fc.ndim == 3 else fc
    mc_gray = cv2.cvtColor(mc, cv2.COLOR_RGB2GRAY) if mc.ndim == 3 else mc
    fc_clahe = apply_clahe(fc_gray, clip_limit=2.0)
    mc_clahe = apply_clahe(mc_gray, clip_limit=2.0)

    # ── 피라미드 체인 ──
    M_accum = None       # 누적 3x3 (moving_resized → fixed_resized)
    sf_prev = sm_prev = None
    final_sf = final_sm = None
    final_gate = final_status = None
    final_metrics = None
    last_kpts0 = last_kpts1 = last_conf = last_inliers = None
    last_M_scaled = np.eye(3)
    last_level = -1
    last_fixed_L = last_target_moving = None

    for li, max_side in enumerate(PYRAMID_LEVELS):
        print(f"\n[Pyramid L{li}] max_side={max_side}")

        # Resize to this level
        fixed_L, sf = resize_to_max(fc_clahe, max_side)
        moving_L, sm = resize_to_max(mc_clahe, max_side)
        fmask_L = cv2.resize(fmc, (fixed_L.shape[1], fixed_L.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        mmask_L = cv2.resize(mmc, (moving_L.shape[1], moving_L.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tooth_area = float(np.sum(cv2.erode(fmask_L, kernel) > 0))

        # Pre-align moving if we have previous M
        if M_accum is not None:
            M_scaled = _rescale_M(M_accum, sf_prev, sm_prev, sf, sm)
            h_f, w_f = fixed_L.shape[:2]
            target_moving = cv2.warpAffine(
                moving_L, M_scaled[:2, :], (w_f, h_f),
                borderMode=cv2.BORDER_CONSTANT, borderValue=127)
            # Pre-aligned → moving is in fixed coords → use fixed mask
            target_mmask = fmask_L
        else:
            target_moving = moving_L
            target_mmask = mmask_L
            M_scaled = np.eye(3)

        # LoFTR: global + masked 합산
        k0, k1, conf = _match_at_level(
            fixed_L, target_moving, fmask_L, target_mmask, PYRAMID_CONF)
        n = len(k0)
        print(f"[Pyramid L{li}] {n} matches")

        if n < _MIN_FOR_ESTIMATE:
            print(f"[Pyramid L{li}] 매치 부족 ({n})")
            if li == 0:
                print("[Pyramid] L0 실패 → 640 fallback")
                return [_single_pass_fallback(
                    fc_clahe, mc_clahe, fmc, mmc,
                    M_rot_f, crop_off_f, M_rot_m, crop_off_m,
                    fixed_img, moving_img,
                    anchor_f_crop, anchor_m_crop)]
            break  # 이전 레벨 결과 사용

        # RANSAC + quality gate
        M_est, inliers, gate, status, met, _ = _run_gate(
            k0, k1, conf, tooth_area)

        if M_est is None:
            print(f"[Pyramid L{li}] gate FAIL")
            if li == 0:
                print("[Pyramid] L0 gate 실패 → 640 fallback")
                return [_single_pass_fallback(
                    fc_clahe, mc_clahe, fmc, mmc,
                    M_rot_f, crop_off_f, M_rot_m, crop_off_m,
                    fixed_img, moving_img,
                    anchor_f_crop, anchor_m_crop)]
            break

        # Compose: M_level = M_delta @ M_scaled
        M_delta_3x3 = _to_3x3(M_est)
        M_accum = M_delta_3x3 @ M_scaled

        sf_prev, sm_prev = sf, sm
        final_sf, final_sm = sf, sm
        final_gate, final_status = gate, status
        final_metrics = met
        last_kpts0, last_kpts1, last_conf = k0, k1, conf
        last_inliers = inliers
        last_M_scaled = M_scaled
        last_level = li
        last_fixed_L = fixed_L
        last_target_moving = target_moving

        print(f"[Pyramid L{li}] {gate} {status}"
              f"  inlier={met.get('n_inlier', 0)}"
              f"  ratio={met.get('inlier_ratio', 0):.2f}"
              f"  reproj={met.get('reproj_median', -1):.1f}"
              f"  rot={met.get('rotation_deg', 0):.1f}°"
              f"  scale={met.get('scale', 0):.3f}")

    # ── 피라미드 완료 확인 ──
    if M_accum is None or final_metrics is None:
        return [_make_fail_entry('pyramid_all_levels_failed')]

    # ── Anchor refit (최종 레벨, confidence 가중) ──
    M_use = M_accum[:2, :]  # 2x3
    if anchor_f_crop and final_sf is not None and last_inliers is not None:
        inl = last_inliers.ravel().astype(bool)
        inlier_k0 = last_kpts0[inl]
        inlier_k1_warped = last_kpts1[inl]  # warped 좌표

        # warped 좌표 → 원본 moving 좌표로 역변환
        if last_level > 0:
            M_inv = np.linalg.inv(last_M_scaled)
            pts = np.hstack([inlier_k1_warped,
                             np.ones((len(inlier_k1_warped), 1))])
            inlier_k1_orig = (M_inv @ pts.T).T[:, :2]
        else:
            inlier_k1_orig = inlier_k1_warped

        a_f = np.array(anchor_f_crop, dtype=np.float32) * final_sf
        a_m = np.array(anchor_m_crop, dtype=np.float32) * final_sm
        n_inl = int(np.sum(inl))
        n_dup = max(n_inl // len(a_f), 5)

        ak0 = np.concatenate([inlier_k0, np.repeat(a_f, n_dup, axis=0)])
        ak1 = np.concatenate([inlier_k1_orig.astype(np.float32),
                               np.repeat(a_m, n_dup, axis=0)])
        w = np.concatenate([last_conf[inl],
                            np.full(len(a_f) * n_dup, 1.0)])

        fit_fn = (_fit_similarity_lstsq if final_gate == 'similarity'
                  else _fit_affine_lstsq)
        M_r = fit_fn(ak1, ak0, weights=w)
        if M_r is not None:
            M_use = M_r
            print(f"[Pyramid] Anchor refit: {len(a_f)} anchors x{n_dup} "
                  f"+ {n_inl} inliers (weighted)")

    # ── Compose full matrix + warp ──
    try:
        M_full = compose_full_matrix(
            M_use, M_rot_f, crop_off_f, final_sf,
            M_rot_m, crop_off_m, final_sm)
        reg = cv2.warpAffine(
            moving_img, M_full[:2, :],
            (fixed_img.shape[1], fixed_img.shape[0]))
    except np.linalg.LinAlgError:
        return [_make_fail_entry('matrix_singular')]

    mviz = _draw_matches(last_fixed_L, last_target_moving,
                         last_kpts0, last_kpts1, last_inliers)
    entry = {
        'conf_threshold': PYRAMID_CONF,
        'max_side': PYRAMID_LEVELS[last_level],
        'clahe_clip': 2.0, 'mask_sigma': 5,
        'n_matches': len(last_kpts0),
        'status': final_status, 'gate': final_gate,
        'metrics': final_metrics, 'M_full': M_full,
        'registered_img': reg,
        'false_color': false_color(fixed_img, reg),
        'pyramid_level': last_level,
        'label': f"Pyramid L{last_level}",
        'match_viz': mviz,
    }

    bm = final_metrics or {}
    print(f"\n{'='*50}")
    print(f"[Result] Pyramid L{last_level} — {final_status}"
          f"  inlier={bm.get('n_inlier', 0)}"
          f"  ratio={bm.get('inlier_ratio', 0):.2f}"
          f"  reproj={bm.get('reproj_median', -1):.1f}")
    print(f"{'='*50}\n")

    return [entry]


def _apply_orientation(img: np.ndarray, flip: bool, k: int) -> np.ndarray:
    """이미지에 flip(좌우반전) + k×90° CCW 회전 적용."""
    if flip:
        img = cv2.flip(img, 1)
    if k:
        img = np.ascontiguousarray(np.rot90(img, k))
    return img


def _transform_anchors_orient(
        anchors: list[tuple] | None,
        w: int, h: int,
        flip: bool, k: int) -> list[tuple] | None:
    """앵커의 moving 좌표만 flip + k×90° CCW 변환.

    Args:
        anchors: [(fx, fy, mx, my), ...].
        w, h: 변환 전 moving 이미지 너비/높이.
        flip: 좌우반전 여부.
        k: 90° CCW 회전 횟수 (0~3).

    Returns:
        변환된 앵커 리스트.
    """
    if not anchors:
        return None
    out = []
    for fx, fy, mx, my in anchors:
        x, y = mx, my
        cw, ch = w, h
        if flip:
            x = cw - 1 - x
        for _ in range(k):
            # 90° CCW: (x, y) → (y, cw-1-x), 새 크기 (ch=cw_old, cw=ch_old) -- 아니다
            # 다시: 이미지 (h, w) → np.rot90 → (w, h)
            # 점 (x, y) → (y, w-1-x)
            new_x = y
            new_y = cw - 1 - x
            x, y = new_x, new_y
            cw, ch = ch, cw  # 회전 후 너비/높이 swap
        out.append((fx, fy, x, y))
    return out


def register_test_lazy(fixed_img: np.ndarray, moving_img: np.ndarray,
                       fixed_mask: np.ndarray,
                       moving_mask: np.ndarray,
                       anchor_points: list[tuple] | None = None,
                       progress_callback=None) -> list[dict]:
    """Lazy 모드: moving 이미지 8가지 회전/flip 조합 시도 후 최적 결과 반환.

    원본 + 90/180/270° 회전 × flip 여부 = 8가지.
    각 조합에 대해 register_test를 호출하고 (status_rank, n_inlier) 점수로 선택.

    Args:
        fixed_img: 고정상 RGB 배열.
        moving_img: 이동상 RGB 배열.
        fixed_mask: 고정상 마스크 uint8.
        moving_mask: 이동상 마스크 uint8.
        anchor_points: [(fx, fy, mx, my), ...] 강제 대응점.
        progress_callback: ``fn(current, total, label)`` 형태의 진행 콜백.

    Returns:
        결과 리스트 (단일 entry — 최적 orientation 결과).
    """
    h, w = moving_img.shape[:2]
    rank = {'pass': 2, 'warn': 1, 'fail': 0}

    best_entry = None
    best_score = (-1, -1)
    best_label = None
    attempts = []
    total = 8
    cur = 0

    for flip in (False, True):
        for k in range(4):
            cur += 1
            label = f"{'F' if flip else ''}R{k * 90}"
            if progress_callback is not None:
                try:
                    progress_callback(cur, total, label)
                except Exception:
                    pass
            print(f"\n{'#'*50}")
            print(f"[Lazy {cur}/{total}] orientation: flip={flip} "
                  f"rot={k * 90}° ({label})")
            print(f"{'#'*50}")

            m_t = _apply_orientation(moving_img, flip, k)
            mmask_t = _apply_orientation(moving_mask, flip, k)
            anchors_t = _transform_anchors_orient(
                anchor_points, w, h, flip, k)

            try:
                results = register_test(
                    fixed_img, m_t, fixed_mask, mmask_t,
                    anchor_points=anchors_t)
            except Exception as e:
                print(f"[Lazy] {label} 실패: {e}")
                continue

            if not results:
                continue
            r = results[0]
            r['lazy_orientation'] = (flip, k)
            r['lazy_label'] = label
            score = (rank.get(r['status'], 0),
                     r.get('metrics', {}).get('n_inlier', 0))
            attempts.append((label, r['status'], score[1]))
            if score > best_score:
                best_score = score
                best_entry = r
                best_label = label

    print(f"\n{'='*60}")
    print(f"[Lazy] Tried {len(attempts)} orientations:")
    for lbl, st, ni in attempts:
        marker = " ★" if lbl == best_label else ""
        print(f"  {lbl}: {st.upper()}  inlier={ni}{marker}")
    print(f"[Lazy] Best: {best_label}")
    print(f"{'='*60}\n")

    if best_entry is None:
        return [_make_fail_entry('lazy_all_failed')]

    # 라벨에 orientation 추가
    orig_label = best_entry.get('label', '?')
    best_entry['label'] = f"{orig_label} [{best_label}]"
    return [best_entry]


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
