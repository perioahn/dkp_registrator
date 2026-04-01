"""
register.py — 파이프라인 오케스트레이터 (Phase A~D 통합)

GUI 의존성 없음. tkinter import 금지.
"""

import numpy as np
import cv2

from preprocess import apply_clahe, auto_orient_and_crop, resize_to_max
from matching import loftr_match, filter_by_mask
from transform import compose_full_matrix, quality_gate_similarity, quality_gate_affine


def register_pair(fixed_img: np.ndarray,
                  moving_img: np.ndarray,
                  fixed_mask: np.ndarray,
                  moving_mask: np.ndarray,
                  max_side: int = 640,
                  allow_legacy_fallback: bool = True) -> dict:
    """
    전체 정합 파이프라인.

    Args:
        fixed_img, moving_img: RGB uint8 원본 해상도
        fixed_mask, moving_mask: binary mask (uint8, 0 or 255)
        max_side: LoFTR 입력 장변 최대
        allow_legacy_fallback: legacy OF 루프 허용 여부

    Returns:
        dict: registered_img, M_full, metrics, path, debug_images
    """
    debug = {}

    # === Phase A: 자동 전처리 ===

    # A1~A3: 회전 + 크롭
    fixed_crop, fixed_mask_crop, M_rot_f, crop_off_f = \
        auto_orient_and_crop(fixed_img, fixed_mask)
    moving_crop, moving_mask_crop, M_rot_m, crop_off_m = \
        auto_orient_and_crop(moving_img, moving_mask)

    debug['fixed_crop'] = fixed_crop
    debug['moving_crop'] = moving_crop

    # A4: grayscale
    fixed_gray = cv2.cvtColor(fixed_crop, cv2.COLOR_RGB2GRAY)
    moving_gray = cv2.cvtColor(moving_crop, cv2.COLOR_RGB2GRAY)

    # A5: CLAHE
    fixed_clahe = apply_clahe(fixed_gray)
    moving_clahe = apply_clahe(moving_gray)

    debug['fixed_clahe'] = fixed_clahe
    debug['moving_clahe'] = moving_clahe

    # A6: 리사이즈 (장변 640px)
    fixed_resized, scale_f = resize_to_max(fixed_clahe, max_side)
    moving_resized, scale_m = resize_to_max(moving_clahe, max_side)

    # 마스크도 동일 리사이즈
    fixed_mask_resized = cv2.resize(fixed_mask_crop,
        (fixed_resized.shape[1], fixed_resized.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    moving_mask_resized = cv2.resize(moving_mask_crop,
        (moving_resized.shape[1], moving_resized.shape[0]),
        interpolation=cv2.INTER_NEAREST)

    # tooth_mask_area (품질 게이트용)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded_mask = cv2.erode(fixed_mask_resized, kernel)
    tooth_mask_area = float(np.sum(eroded_mask > 0))

    # === Phase B: LoFTR 매칭 ===

    kpts0, kpts1, conf = loftr_match(fixed_resized, moving_resized)

    debug['n_raw_matches'] = len(kpts0)

    # 마스크 필터링
    kpts0, kpts1, conf = filter_by_mask(
        kpts0, kpts1, conf,
        fixed_mask_resized, moving_mask_resized
    )

    debug['n_filtered_matches'] = len(kpts0)

    # === Phase C: 변환 추정 + 품질 판정 ===

    result_path = None
    M_loftr = None
    final_metrics = None

    # C1: Similarity (1차)
    if len(kpts0) >= 12:
        M_sim, inliers_sim = cv2.estimateAffinePartial2D(
            kpts1, kpts0,  # moving → fixed
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99
        )

        if M_sim is not None:
            status, metrics = quality_gate_similarity(
                kpts0, kpts1, M_sim, inliers_sim, tooth_mask_area
            )
            metrics['gate'] = 'similarity'

            if status in ('pass', 'warn'):
                M_loftr = M_sim
                result_path = 'similarity'
                final_metrics = metrics
                if status == 'warn':
                    print(f"[WARN] Similarity 정합 경고: {metrics}")

    # C2: Affine (2차 fallback)
    if result_path is None and len(kpts0) >= 12:
        M_aff, inliers_aff = cv2.estimateAffine2D(
            kpts1, kpts0,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.99
        )

        if M_aff is not None:
            status, metrics = quality_gate_affine(
                kpts0, kpts1, M_aff, inliers_aff, tooth_mask_area
            )
            metrics['gate'] = 'affine'

            if status in ('pass', 'warn'):
                M_loftr = M_aff
                result_path = 'affine'
                final_metrics = metrics

    # C3: Legacy OF 루프 (최후방)
    if result_path is None and allow_legacy_fallback:
        print("[INFO] LoFTR 실패 → Legacy OF 루프로 전환")
        result_path = 'legacy'
        final_metrics = {'gate': 'legacy', 'reason': 'loftr_fallthrough'}
        # TODO: core_crop_250902.py reg_body_single 래핑

    if result_path is None:
        return {
            'registered_img': None,
            'M_full': None,
            'metrics': final_metrics or {'gate': 'none', 'reason': 'all_failed'},
            'path': 'failed',
            'debug_images': debug,
        }

    # === Phase D: 행렬 역산 + 원본 적용 ===

    if result_path in ('similarity', 'affine'):
        M_full = compose_full_matrix(
            M_loftr,
            M_rot_f, crop_off_f, scale_f,
            M_rot_m, crop_off_m, scale_m
        )

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

    # legacy path
    return {
        'registered_img': None,
        'M_full': None,
        'metrics': final_metrics,
        'path': result_path,
        'debug_images': debug,
    }


def false_color(img1, img2):
    """정합 결과 시각화 — 기존 false_color() 재활용."""
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
