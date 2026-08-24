"""파이프라인 오케스트레이터 (Phase A~D 통합).

GUI 의존성 없음. tkinter import 금지.
"""

from __future__ import annotations

import numpy as np
import cv2

from config import DEFAULT, PipelineConfig
from matching import apply_soft_mask, filter_by_mask, loftr_match
from preprocess import apply_clahe, auto_orient_and_crop, resize_to_max
from transform import (
    compose_full_matrix,
    quality_gate_affine,
    quality_gate_similarity,
)

# 하위호환 별칭 (설정의 단일 출처는 config.PipelineConfig)
PYRAMID_LEVELS = DEFAULT.pyramid_levels
PYRAMID_CONF = DEFAULT.pyramid_conf


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


def _run_gate(k0, k1, conf, tooth_area, cfg: PipelineConfig = DEFAULT):
    """Similarity → Affine 폴백으로 RANSAC + quality gate 수행."""
    # Similarity
    M_sim, inliers_sim = cv2.estimateAffinePartial2D(
        k1, k0, method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_thresh, confidence=0.99)
    if M_sim is not None:
        status, met = quality_gate_similarity(
            k0, k1, M_sim, inliers_sim, tooth_area, cfg=cfg.sim_gate)
        if status in ('pass', 'warn'):
            return M_sim, inliers_sim, 'similarity', status, met, conf
    # Affine fallback (allow_affine=False면 similarity만 — 비율 보존)
    if not cfg.allow_affine:
        return None, None, 'none', 'fail', {}, conf
    M_aff, inliers_aff = cv2.estimateAffine2D(
        k1, k0, method=cv2.RANSAC,
        ransacReprojThreshold=cfg.ransac_thresh, confidence=0.99)
    if M_aff is not None:
        status, met = quality_gate_affine(
            k0, k1, M_aff, inliers_aff, tooth_area, cfg=cfg.aff_gate)
        if status in ('pass', 'warn'):
            return M_aff, inliers_aff, 'affine', status, met, conf
    return None, None, 'none', 'fail', {}, conf


def _match_at_level(fixed_L, moving_L, fixed_mask_L, moving_mask_L,
                    conf_threshold, cfg: PipelineConfig = DEFAULT):
    """Global + Masked LoFTR → 합산 매칭."""
    # Global
    nk0, nk1, ncf = loftr_match(fixed_L, moving_L, conf_threshold=0.1)
    # Masked
    f_masked = apply_soft_mask(fixed_L, fixed_mask_L, sigma=cfg.mask_sigma)
    m_masked = apply_soft_mask(moving_L, moving_mask_L, sigma=cfg.mask_sigma)
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
                          anchor_f_crop, anchor_m_crop,
                          cfg: PipelineConfig = DEFAULT):
    """피라미드 L0 실패 시 최고해상 단일 패스 폴백."""
    ms = cfg.pyramid_levels[-1]
    fr, sf = resize_to_max(fc_clahe, ms)
    mr, sm = resize_to_max(mc_clahe, ms)
    fm = cv2.resize(fmc, (fr.shape[1], fr.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    mm = cv2.resize(mmc, (mr.shape[1], mr.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tooth_area = float(np.sum(cv2.erode(fm, kernel) > 0))

    k0, k1, conf = _match_at_level(fr, mr, fm, mm, cfg.pyramid_conf, cfg=cfg)
    n = len(k0)
    print(f"[Fallback {ms}] {n} matches")

    if n < cfg.min_matches:
        print(f"[Fallback {ms}] 매치 부족 → FAIL")
        return _make_fail_entry(f'fallback_{n}_matches')

    M_est, inliers, gate, status, met, _ = _run_gate(
        k0, k1, conf, tooth_area, cfg=cfg)
    if M_est is None:
        print(f"[Fallback {ms}] gate FAIL")
        return _make_fail_entry('fallback_gate_fail')

    print(f"[Fallback {ms}] {gate} {status}"
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
        'conf_threshold': cfg.pyramid_conf, 'max_side': ms,
        'clahe_clip': cfg.clahe_clip, 'mask_sigma': cfg.mask_sigma,
        'n_matches': n,
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
                  anchor_points: list[tuple] | None = None,
                  cfg: PipelineConfig = DEFAULT) -> list[dict]:
    """다단계 피라미드 정합 (기본 320→480→640).

    Args:
        fixed_img: 고정상 RGB 배열.
        moving_img: 이동상 RGB 배열.
        fixed_mask: 고정상 마스크 uint8.
        moving_mask: 이동상 마스크 uint8.
        anchor_points: [(fx, fy, mx, my), ...] 강제 대응점.
        cfg: 파이프라인 설정 (config.PROFILES 참고).

    Returns:
        결과 딕셔너리 리스트 (단일 항목).
    """
    _MIN_FOR_ESTIMATE = cfg.min_matches

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
    fc_clahe = apply_clahe(fc_gray, clip_limit=cfg.clahe_clip)
    mc_clahe = apply_clahe(mc_gray, clip_limit=cfg.clahe_clip)

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

    for li, max_side in enumerate(cfg.pyramid_levels):
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
            fixed_L, target_moving, fmask_L, target_mmask,
            cfg.pyramid_conf, cfg=cfg)
        n = len(k0)
        print(f"[Pyramid L{li}] {n} matches")

        if n < _MIN_FOR_ESTIMATE:
            print(f"[Pyramid L{li}] 매치 부족 ({n})")
            if li == 0:
                print("[Pyramid] L0 실패 → 단일 패스 fallback")
                return [_single_pass_fallback(
                    fc_clahe, mc_clahe, fmc, mmc,
                    M_rot_f, crop_off_f, M_rot_m, crop_off_m,
                    fixed_img, moving_img,
                    anchor_f_crop, anchor_m_crop, cfg=cfg)]
            break  # 이전 레벨 결과 사용

        # RANSAC + quality gate
        M_est, inliers, gate, status, met, _ = _run_gate(
            k0, k1, conf, tooth_area, cfg=cfg)

        if M_est is None:
            print(f"[Pyramid L{li}] gate FAIL")
            if li == 0:
                print("[Pyramid] L0 gate 실패 → 단일 패스 fallback")
                return [_single_pass_fallback(
                    fc_clahe, mc_clahe, fmc, mmc,
                    M_rot_f, crop_off_f, M_rot_m, crop_off_m,
                    fixed_img, moving_img,
                    anchor_f_crop, anchor_m_crop, cfg=cfg)]
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
        'conf_threshold': cfg.pyramid_conf,
        'max_side': cfg.pyramid_levels[last_level],
        'clahe_clip': cfg.clahe_clip, 'mask_sigma': cfg.mask_sigma,
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


def _prescreen_orientations(fixed_img, moving_img, fixed_mask, moving_mask,
                            cfg: PipelineConfig = DEFAULT) -> list[tuple]:
    """8가지 orientation을 저해상 단일 패스로 점수화한다.

    크롭 전처리 없이 prescreen_side 해상도에서 LoFTR+게이트만 수행 —
    풀 파이프라인 대비 수십 배 저렴한 근사 순위. 반환은 점수 내림차순
    [(score, flip, k, label), ...], score=(status_rank, n_inlier).
    """
    side = cfg.lazy_prescreen_side
    rank = {'pass': 2, 'warn': 1, 'fail': 0}

    f_gray = cv2.cvtColor(fixed_img, cv2.COLOR_RGB2GRAY) \
        if fixed_img.ndim == 3 else fixed_img
    m_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY) \
        if moving_img.ndim == 3 else moving_img
    f_small, _ = resize_to_max(apply_clahe(f_gray, clip_limit=cfg.clahe_clip), side)
    m_small0, _ = resize_to_max(apply_clahe(m_gray, clip_limit=cfg.clahe_clip), side)
    fm_small = cv2.resize(fixed_mask, (f_small.shape[1], f_small.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    mm_small0 = cv2.resize(moving_mask, (m_small0.shape[1], m_small0.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tooth_area = float(np.sum(cv2.erode(fm_small, kernel) > 0))

    scored = []
    for flip in (False, True):
        for k in range(4):
            label = f"{'F' if flip else ''}R{k * 90}"
            m_s = _apply_orientation(m_small0, flip, k)
            mm_s = _apply_orientation(mm_small0, flip, k)
            try:
                k0, k1, conf = _match_at_level(
                    f_small, m_s, fm_small, mm_s, cfg.pyramid_conf, cfg=cfg)
                if len(k0) < cfg.min_matches:
                    score = (0, 0)
                else:
                    _, _, _, status, met, _ = _run_gate(
                        k0, k1, conf, tooth_area, cfg=cfg)
                    score = (rank.get(status, 0), met.get('n_inlier', 0))
            except Exception as e:
                print(f"[Lazy prescreen] {label} 오류: {e}")
                score = (0, 0)
            scored.append((score, flip, k, label))
            print(f"[Lazy prescreen] {label}: rank={score[0]} inlier={score[1]}")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def register_test_lazy(fixed_img: np.ndarray, moving_img: np.ndarray,
                       fixed_mask: np.ndarray,
                       moving_mask: np.ndarray,
                       anchor_points: list[tuple] | None = None,
                       progress_callback=None,
                       cfg: PipelineConfig = DEFAULT) -> list[dict]:
    """Lazy 모드: 8가지 회전/flip 조합 중 최적 orientation으로 정합.

    저해상 프리스크리닝으로 8조합을 점수화한 뒤, 상위 후보부터 풀 파이프라인
    실행. pass가 나오면 즉시 종료 — 올바른 orientation이 통상 1순위라 8조합
    전수 풀 실행 대비 4~8배 빠르다. pass가 안 나오면 8조합을 모두 시도하고
    최고점을 반환한다(구버전과 동일한 커버리지).

    Args:
        fixed_img: 고정상 RGB 배열.
        moving_img: 이동상 RGB 배열.
        fixed_mask: 고정상 마스크 uint8.
        moving_mask: 이동상 마스크 uint8.
        anchor_points: [(fx, fy, mx, my), ...] 강제 대응점.
        progress_callback: ``fn(current, total, label)`` 형태의 진행 콜백.
        cfg: 파이프라인 설정.

    Returns:
        결과 리스트 (단일 entry — 최적 orientation 결과).
    """
    h, w = moving_img.shape[:2]
    rank = {'pass': 2, 'warn': 1, 'fail': 0}
    total = 8

    if progress_callback is not None:
        try:
            progress_callback(0, total, "프리스크린")
        except Exception:
            pass

    print(f"\n[Lazy] 저해상({cfg.lazy_prescreen_side}px) 프리스크리닝 8조합...")
    ranked = _prescreen_orientations(
        fixed_img, moving_img, fixed_mask, moving_mask, cfg=cfg)

    best_entry = None
    best_score = (-1, -1)
    best_label = None
    attempts = []

    for cur, (pre_score, flip, k, label) in enumerate(ranked, start=1):
        if progress_callback is not None:
            try:
                progress_callback(cur, total, label)
            except Exception:
                pass
        print(f"\n{'#'*50}")
        print(f"[Lazy {cur}/{total}] full run: flip={flip} rot={k * 90}° "
              f"({label}, prescreen rank={pre_score[0]} inlier={pre_score[1]})")
        print(f"{'#'*50}")

        m_t = _apply_orientation(moving_img, flip, k)
        mmask_t = _apply_orientation(moving_mask, flip, k)
        anchors_t = _transform_anchors_orient(anchor_points, w, h, flip, k)

        try:
            results = register_test(
                fixed_img, m_t, fixed_mask, mmask_t,
                anchor_points=anchors_t, cfg=cfg)
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

        # 조기 종료는 pass에서만 — warn은 뒤 orientation에서 pass가 나올 수
        # 있으므로 전수 계속 (프리스크리닝은 실행 순서만 결정, 커버리지 동일)
        if best_score[0] == 2:
            print(f"[Lazy] {label} PASS → 조기 종료")
            break

    print(f"\n{'='*60}")
    print(f"[Lazy] Tried {len(attempts)}/{total} orientations (프리스크린 순):")
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


def register_pair(fixed_img: np.ndarray,
                  moving_img: np.ndarray,
                  fixed_mask: np.ndarray,
                  moving_mask: np.ndarray,
                  refine: bool = False,
                  force_nocrop: bool = False,
                  hint: tuple | None = None,
                  cfg: PipelineConfig = DEFAULT) -> dict:
    """(하위호환 wrapper) 구 단일패스 cascade API — 피라미드 엔진으로 위임.

    refine/force_nocrop/hint는 구 API 호환을 위해 받되 무시된다
    (피라미드 정합이 해당 케이스를 포괄).

    Returns:
        {'registered_img', 'M_full', 'metrics', 'path', 'debug_images'}
    """
    entry = register_test(fixed_img, moving_img, fixed_mask, moving_mask,
                          cfg=cfg)[0]
    metrics = dict(entry.get('metrics') or {})
    metrics.setdefault('gate', entry.get('gate', 'none'))
    metrics.setdefault('status', entry.get('status', 'fail'))
    if entry.get('reason'):
        metrics.setdefault('reason', entry['reason'])
    failed = entry.get('registered_img') is None
    return {
        'registered_img': entry.get('registered_img'),
        'M_full': entry.get('M_full'),
        'metrics': metrics,
        'path': 'failed' if failed else entry.get('gate', 'none'),
        'debug_images': {'false_color': entry.get('false_color')},
    }


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
