"""
transform.py — Phase C: 변환 행렬 합성 + 품질 게이트

GUI 의존성 없음. numpy, cv2만 사용.
"""

import numpy as np
import cv2


def compose_full_matrix(M_loftr_2x3: np.ndarray,
                         M_rot_f: np.ndarray,
                         crop_offset_f: tuple,
                         resize_scale_f: float,
                         M_rot_m: np.ndarray,
                         crop_offset_m: tuple,
                         resize_scale_m: float) -> np.ndarray:
    """
    크롭/리사이즈 좌표계 → 원본 좌표계 역산.

    M_full = A_f⁻¹ @ M_loftr_3x3 @ A_m
    여기서 A = S @ C @ M_rot (각 이미지의 전처리 체인)

    Returns:
        M_full: 3×3 변환 행렬 (moving 원본 → fixed 원본)
    """
    def to_3x3(M_2x3):
        M = np.eye(3)
        M[:2, :] = M_2x3
        return M

    def translation_matrix(tx, ty):
        M = np.eye(3)
        M[0, 2] = tx
        M[1, 2] = ty
        return M

    def scale_matrix(s):
        M = np.eye(3)
        M[0, 0] = s
        M[1, 1] = s
        return M

    # Fixed 체인: 원본 → 회전 → 크롭 → 리사이즈
    C_f = translation_matrix(-crop_offset_f[0], -crop_offset_f[1])
    S_f = scale_matrix(resize_scale_f)
    A_f = S_f @ C_f @ M_rot_f

    # Moving 체인
    C_m = translation_matrix(-crop_offset_m[0], -crop_offset_m[1])
    S_m = scale_matrix(resize_scale_m)
    A_m = S_m @ C_m @ M_rot_m

    # 합성
    M_full = np.linalg.inv(A_f) @ to_3x3(M_loftr_2x3) @ A_m

    return M_full


def quality_gate_similarity(kpts_f: np.ndarray,
                             kpts_m: np.ndarray,
                             M_2x3: np.ndarray,
                             inliers: np.ndarray,
                             tooth_mask_area: float):
    """
    Similarity 변환 품질 판정 (pass/warn/fail).

    Returns:
        status ('pass'/'warn'/'fail'), metrics dict
    """
    n_total = len(kpts_f)
    inlier_mask = inliers.flatten().astype(bool)
    n_inlier = int(np.sum(inlier_mask))

    # reprojection error
    kpts_m_h = np.hstack([kpts_m, np.ones((n_total, 1))])
    M_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    projected = (M_3x3 @ kpts_m_h.T).T[:, :2]
    errors = np.linalg.norm(projected - kpts_f, axis=1)
    inlier_errors = errors[inlier_mask]

    reproj_median = float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 999.0
    reproj_p90 = float(np.percentile(inlier_errors, 90)) if len(inlier_errors) > 0 else 999.0

    # hull coverage
    inlier_pts = kpts_f[inlier_mask].astype(np.float32)
    hull_area = 0.0
    if len(inlier_pts) >= 3:
        hull = cv2.convexHull(inlier_pts)
        hull_area = float(cv2.contourArea(hull))
    coverage = hull_area / tooth_mask_area if tooth_mask_area > 0 else 0.0

    # 변환 파라미터 분해
    det = M_2x3[0, 0] * M_2x3[1, 1] - M_2x3[0, 1] * M_2x3[1, 0]
    scale = float(np.sqrt(abs(det)))
    rotation = float(np.degrees(np.arctan2(M_2x3[1, 0], M_2x3[0, 0])))

    metrics = {
        'n_total': n_total,
        'n_inlier': n_inlier,
        'inlier_ratio': n_inlier / n_total if n_total > 0 else 0.0,
        'reproj_median': reproj_median,
        'reproj_p90': reproj_p90,
        'hull_area': hull_area,
        'hull_coverage': coverage,
        'det': float(det),
        'scale': scale,
        'rotation_deg': rotation,
    }

    # Hard fail
    if n_inlier < 12:
        return 'fail', metrics

    hard_fail = (
        det <= 0 or
        reproj_median >= 5.0 or
        reproj_p90 >= 12.0 or
        scale < 0.7 or scale > 1.4 or
        abs(rotation) > 20
    )
    if hard_fail:
        return 'fail', metrics

    # Warn
    warn = (
        n_inlier < 30 or
        reproj_median >= 3.0 or
        coverage < 0.2 or
        scale < 0.8 or scale > 1.2 or
        abs(rotation) > 15
    )
    if warn:
        return 'warn', metrics

    return 'pass', metrics


def quality_gate_affine(kpts_f: np.ndarray,
                         kpts_m: np.ndarray,
                         M_2x3: np.ndarray,
                         inliers: np.ndarray,
                         tooth_mask_area: float):
    """
    Affine 변환 품질 판정. scale/rotation 체크 없음.

    Returns:
        status ('pass'/'warn'/'fail'), metrics dict
    """
    n_total = len(kpts_f)
    inlier_mask = inliers.flatten().astype(bool)
    n_inlier = int(np.sum(inlier_mask))

    kpts_m_h = np.hstack([kpts_m, np.ones((n_total, 1))])
    M_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    projected = (M_3x3 @ kpts_m_h.T).T[:, :2]
    errors = np.linalg.norm(projected - kpts_f, axis=1)
    inlier_errors = errors[inlier_mask]

    reproj_median = float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 999.0

    det = M_2x3[0, 0] * M_2x3[1, 1] - M_2x3[0, 1] * M_2x3[1, 0]

    inlier_pts = kpts_f[inlier_mask].astype(np.float32)
    hull_area = 0.0
    if len(inlier_pts) >= 3:
        hull = cv2.convexHull(inlier_pts)
        hull_area = float(cv2.contourArea(hull))
    coverage = hull_area / tooth_mask_area if tooth_mask_area > 0 else 0.0

    metrics = {
        'n_total': n_total,
        'n_inlier': n_inlier,
        'inlier_ratio': n_inlier / n_total if n_total > 0 else 0.0,
        'reproj_median': reproj_median,
        'hull_coverage': coverage,
        'det': float(det),
    }

    if n_inlier < 12 or det <= 0 or reproj_median >= 5.0:
        return 'fail', metrics
    if coverage < 0.15:
        return 'warn', metrics
    return 'pass', metrics
