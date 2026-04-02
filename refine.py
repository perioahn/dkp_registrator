"""
refine.py — Phase E: Lightweight Refinement

LoFTR+RANSAC 결과(M_full)를 초기값으로, SimpleITK MI 기반으로 미세 보정.
기존 reg_body_single()의 마지막 register_img_s(..., 100, 100)과 동일 원리.
LoFTR이 좋은 초기값을 주므로 20~30 iter면 수렴. 추가 시간: 1~2초.
"""

import numpy as np
import cv2

from preprocess import apply_clahe
import legacy_of

# ROI가 이 크기 이하면 refinement 건너뜀
_MIN_ROI_SIDE = 30


def refine_similarity_delta(fixed_img, moving_img, fixed_mask,
                             M_full, n_iter=25):
    """Phase E: LoFTR+RANSAC 결과를 SimpleITK MI로 미세 보정.

    Args:
        fixed_img: RGB uint8 원본 해상도
        moving_img: RGB uint8 원본 해상도
        fixed_mask: binary mask (uint8, 0 or 255)
        M_full: 3x3 변환 행렬 (Phase D 결과)
        n_iter: SimpleITK iterations (기본 25)

    Returns:
        M_refined (3x3) 또는 None (SimpleITK 없거나 실패)
    """
    if not legacy_of.LEGACY_AVAILABLE:
        return None

    h, w = fixed_img.shape[:2]

    # 1. warp moving → fixed 좌표계
    M_2x3 = M_full[:2, :]
    warped = cv2.warpAffine(moving_img, M_2x3, (w, h))

    # 2. tooth ROI 크롭 (마스크 bounding box + 5% padding)
    x, y, bw, bh = cv2.boundingRect(
        (fixed_mask > 0).astype(np.uint8))

    if bw < _MIN_ROI_SIDE or bh < _MIN_ROI_SIDE:
        return None

    pad = int(max(bw, bh) * 0.05)
    y1 = max(0, y - pad)
    y2 = min(h, y + bh + pad)
    x1 = max(0, x - pad)
    x2 = min(w, x + bw + pad)

    # 3. grayscale + CLAHE
    fixed_crop = apply_clahe(
        cv2.cvtColor(fixed_img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY))
    warped_crop = apply_clahe(
        cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY))

    # 4. SimpleITK similarity refinement
    try:
        _, M_delta = legacy_of.register_img_s(fixed_crop, warped_crop,
                                               numberOfIterations=n_iter,
                                               convergenceWindowSize=10)
    except Exception as e:
        print(f"[WARN] SimpleITK refinement failed: {e}")
        return None

    if M_delta is None:
        return None

    # 5. crop offset 보정: M_delta는 crop 좌표계 → 원본 좌표계로 변환
    M3 = np.eye(3)
    M3[:2, :2] = M_delta[:2, :2]
    M3[0, 2] = M_delta[0, 2]
    M3[1, 2] = M_delta[1, 2]

    T_to_crop = np.eye(3)
    T_to_crop[0, 2] = -x1
    T_to_crop[1, 2] = -y1

    T_from_crop = np.eye(3)
    T_from_crop[0, 2] = x1
    T_from_crop[1, 2] = y1

    M_delta_orig = T_from_crop @ M3 @ T_to_crop

    # 6. 최종 합성
    M_full_3x3 = np.eye(3)
    M_full_3x3[:2, :] = M_full[:2, :]
    if M_full.shape[0] >= 3:
        M_full_3x3 = M_full.copy()

    M_final = M_delta_orig @ M_full_3x3
    return M_final
