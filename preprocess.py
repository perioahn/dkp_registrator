"""
preprocess.py — Phase A: 이미지 전처리 (CLAHE, 회전, 크롭, 리사이즈)

GUI 의존성 없음. numpy, cv2만 사용.
"""

import numpy as np
import cv2


def apply_clahe(img_gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """CLAHE 적용. equalize_adapthist 대체."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)


def rotate_with_matrix(image: np.ndarray,
                        angle_deg: float,
                        center: tuple = None):
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


def auto_orient_and_crop(image: np.ndarray,
                          mask: np.ndarray,
                          padding_ratio: float = 0.1):
    """
    SAM2 마스크 기반 자동 회전보정 + 크롭.

    Returns:
        cropped_img, cropped_mask, M_rot_3x3, crop_offset
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("마스크에서 contour를 찾을 수 없음")

    all_pts = np.vstack(contours)
    rect = cv2.minAreaRect(all_pts)
    center, (rect_w, rect_h), angle = rect

    # angle 정규화: width > height 보장
    if rect_w < rect_h:
        angle += 90
        rect_w, rect_h = rect_h, rect_w

    # 180도 모호성 제거: [-90, 90) 범위로 정규화
    # 직사각형은 180도 회전 대칭이므로 동일 형상이 0도 또는 180도로 반환될 수 있음
    # 제한: 실제 치아 배열이 정확히 ±90도에 걸칠 경우 불안정할 수 있으나,
    # 임상 사진에서 이 각도는 극히 드묾. aspect_ratio < 1.2 가드가 보완.
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180

    # 정사각형 배열이면 회전 불안정 → 스킵
    aspect_ratio = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-6)
    if aspect_ratio < 1.2:
        angle = 0

    # 회전
    rotated_img, M_rot = rotate_with_matrix(image, angle)

    # mask 회전 (INTER_NEAREST)
    M_2x3 = M_rot[:2, :]
    new_h, new_w = rotated_img.shape[:2]
    rotated_mask = cv2.warpAffine(mask, M_2x3, (new_w, new_h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    rotated_mask = (rotated_mask > 127).astype(np.uint8) * 255

    # 크롭 (패딩 포함)
    x, y, bw, bh = cv2.boundingRect(rotated_mask)
    pad = int(max(bw, bh) * padding_ratio)

    y1 = max(0, y - pad)
    y2 = min(new_h, y + bh + pad)
    x1 = max(0, x - pad)
    x2 = min(new_w, x + bw + pad)

    cropped_img = rotated_img[y1:y2, x1:x2]
    cropped_mask = rotated_mask[y1:y2, x1:x2]
    crop_offset = (x1, y1)

    return cropped_img, cropped_mask, M_rot, crop_offset


def resize_to_max(img: np.ndarray, max_side: int = 640):
    """
    장변 기준 리사이즈. LoFTR 입력 해상도 일관성 보장.

    Returns:
        resized_img, scale_factor
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
