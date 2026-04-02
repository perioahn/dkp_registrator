"""Phase B: LoFTR 매칭 + 마스크 기반 필터링.

모델 싱글턴 로딩. GUI 의존성 없음.
"""

from __future__ import annotations

import cv2
import kornia.feature as KF
import numpy as np
import torch

# 모듈 레벨 싱글턴
_loftr_model = None


def _get_loftr_model(pretrained: str = 'indoor_new') -> KF.LoFTR:
    """LoFTR 모델 싱글턴 로딩."""
    global _loftr_model
    if _loftr_model is None:
        try:
            _loftr_model = KF.LoFTR(pretrained=pretrained)
        except Exception:
            _loftr_model = KF.LoFTR(pretrained='indoor')
        _loftr_model.eval()
        if torch.cuda.is_available():
            _loftr_model = _loftr_model.cuda()
    return _loftr_model


def loftr_match(
        img1_gray: np.ndarray, img2_gray: np.ndarray,
        conf_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LoFTR dense 매칭.

    Args:
        img1_gray: fixed grayscale (uint8)
        img2_gray: moving grayscale (uint8)
        conf_threshold: confidence 최소 임계값

    Returns:
        kpts0 (N×2), kpts1 (N×2), confidence (N,)
    """
    model = _get_loftr_model()
    device = next(model.parameters()).device

    input1 = torch.from_numpy(img1_gray).float()[None, None] / 255.0
    input2 = torch.from_numpy(img2_gray).float()[None, None] / 255.0

    try:
        input1 = input1.to(device)
        input2 = input2.to(device)
        with torch.no_grad():
            correspondences = model({"image0": input1, "image1": input2})
    except torch.cuda.OutOfMemoryError:
        print("[WARN] CUDA OOM - CPU fallback")
        torch.cuda.empty_cache()
        model_cpu = model.cpu()
        with torch.no_grad():
            correspondences = model_cpu({"image0": input1.cpu(), "image1": input2.cpu()})
        try:
            model.to(device)  # 원복
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            pass  # GPU 복귀 실패 — 다음 호출 시 CPU로 계속

    kpts0 = correspondences['keypoints0'].cpu().numpy()
    kpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()

    valid = conf > conf_threshold
    return kpts0[valid], kpts1[valid], conf[valid]


def apply_soft_mask(img_gray: np.ndarray,
                    mask: np.ndarray,
                    sigma: int = 5,
                    fill: int = 127) -> np.ndarray:
    """
    LoFTR 입력 전 소프트 마스크 적용. 배경을 균일 gray로 채워
    LoFTR attention이 치아 영역에 집중하도록 유도.

    Args:
        img_gray: grayscale uint8
        mask: binary mask uint8 (0 or 255)
        sigma: Gaussian blur sigma
        fill: 배경 채움값

    Returns:
        masked grayscale uint8
    """
    ksize = int(2 * np.ceil(3 * sigma) + 1) | 1
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                              (ksize, ksize), sigma)
    result = img_gray.astype(np.float32) * soft + fill * (1.0 - soft)
    return result.astype(np.uint8)


def filter_by_mask(
        kpts0: np.ndarray, kpts1: np.ndarray, conf: np.ndarray,
        mask0: np.ndarray, mask1: np.ndarray,
        sigma: int = 7, threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft mask 기반 치아 영역 대응점 필터링.

    Args:
        kpts0: Fixed 키포인트 (N×2).
        kpts1: Moving 키포인트 (N×2).
        conf: Confidence 배열 (N,).
        mask0: Fixed 마스크.
        mask1: Moving 마스크.
        sigma: Gaussian blur sigma.
        threshold: 마스크 임계값.

    Returns:
        (필터링된 kpts0, kpts1, conf) 튜플.
    """
    ksize = sigma * 2 + 1
    soft0 = cv2.GaussianBlur(mask0.astype(np.float32) / 255.0,
                              (ksize, ksize), sigma)
    soft1 = cv2.GaussianBlur(mask1.astype(np.float32) / 255.0,
                              (ksize, ksize), sigma)

    h0, w0 = soft0.shape[:2]
    h1, w1 = soft1.shape[:2]

    mask_scores = np.zeros(len(kpts0))
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(kpts0, kpts1)):
        iy0 = int(np.clip(y0, 0, h0 - 1))
        ix0 = int(np.clip(x0, 0, w0 - 1))
        iy1 = int(np.clip(y1, 0, h1 - 1))
        ix1 = int(np.clip(x1, 0, w1 - 1))
        mask_scores[i] = min(soft0[iy0, ix0], soft1[iy1, ix1])

    valid = mask_scores > threshold
    return kpts0[valid], kpts1[valid], conf[valid]
