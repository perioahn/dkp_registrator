"""Phase A: 이미지 전처리 (CLAHE, 회전, 크롭, 리사이즈).

GUI 의존성 없음. numpy, cv2만 사용.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(img_gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """CLAHE 적용. equalize_adapthist 대체."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)


def rotate_with_matrix(image: np.ndarray,
                        angle_deg: float,
                        center: tuple | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    잘림 없는 회전 + 3×3 변환 행렬 반환.

    Args:
        image: 입력 이미지
        angle_deg: 회전 각도 (도, 양수=반시계)
        center: 회전 중심 (None이면 이미지 중심)

    Returns:
        rotated_img, M_rot_3x3
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    # OpenCV getRotationMatrix2D: 양수 angle = 반시계
    M_2x3 = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos_a = abs(M_2x3[0, 0])
    sin_a = abs(M_2x3[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # 중심점 보정 (확장된 캔버스)
    M_2x3[0, 2] += new_w / 2 - center[0]
    M_2x3[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(image, M_2x3, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    M_3x3 = np.eye(3)
    M_3x3[:2, :] = M_2x3

    return rotated, M_3x3


def auto_orient_and_crop(
        image: np.ndarray, mask: np.ndarray,
        padding_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    SAM2 마스크 기반 크롭. (회전 비활성화 — EXIF 적용 후 사용자 방향 신뢰)

    이전 버전에서는 마스크 minAreaRect 주축으로 자동 회전을 적용했으나,
    fixed/moving 마스크 형태가 크게 다른 경우 (예: 치아 1개 vs 치아 3개 배열)
    각각 다른 방향으로 회전돼 정합 실패 원인이 됨. 이제 EXIF Orientation이
    로드 시 픽셀에 적용되므로 사용자 방향을 신뢰하고 회전은 생략한다.
    회전이 필요하면 Lazy Mode 사용.

    Returns:
        cropped_img, cropped_mask, M_rot_3x3 (항등), crop_offset
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("마스크에서 contour를 찾을 수 없음")

    M_rot = np.eye(3)
    h, w = image.shape[:2]

    # 크롭 (패딩 포함)
    x, y, bw, bh = cv2.boundingRect(mask)
    pad = int(max(bw, bh) * padding_ratio)

    y1 = max(0, y - pad)
    y2 = min(h, y + bh + pad)
    x1 = max(0, x - pad)
    x2 = min(w, x + bw + pad)

    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    crop_offset = (x1, y1)

    return cropped_img, cropped_mask, M_rot, crop_offset


def resize_to_max(img: np.ndarray,
                  max_side: int = 640) -> tuple[np.ndarray, float]:
    """장변 기준 리사이즈.

    Args:
        img: 입력 이미지.
        max_side: 최대 변 길이.

    Returns:
        (리사이즈된 이미지, 스케일 팩터) 튜플.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)

    if long_side <= max_side:
        return img.copy(), 1.0

    scale_factor = max_side / long_side
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, scale_factor
