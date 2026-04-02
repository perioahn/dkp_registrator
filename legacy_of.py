"""Wraps the iterative OF + SimpleITK registration loop from legacy core_crop_250902.py.

Functions are extracted from the legacy module to avoid importing its heavy
top-level dependencies (torch, transformers, rembg, TkAgg, etc.).
Source: C:/Users/User/Dropbox/UTIL/REG/core_crop_250902.py
"""

import numpy as np
import cv2

LEGACY_AVAILABLE = True
try:
    import SimpleITK as sitk
    from skimage import registration, exposure, color, transform
except ImportError:
    LEGACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper functions extracted from core_crop_250902.py (lines 78-280)
# ---------------------------------------------------------------------------

def _uint8_img(img):
    if img.max() <= 1:
        img = exposure.rescale_intensity(img, in_range='image', out_range=(0, 255))
    if img.dtype != 'uint8':
        img = img.astype(np.uint8)
    return img


def _overlay_img(img1, img2, w1=0.5, w2=0.5):
    img1 = _uint8_img(img1)
    img2 = _uint8_img(img2)
    if len(img1.shape) == 2 and len(img2.shape) == 3:
        img1 = np.stack((img1, img1, img1), axis=-1)
    if len(img1.shape) == 3 and len(img2.shape) == 2:
        img2 = np.stack((img2, img2, img2), axis=-1)
    ov_image = cv2.addWeighted(img1, w1, img2, w2, 0)
    return ov_image


def _template_fab(fixed, moving):
    fixed_o = _uint8_img(fixed)
    moving_o = _uint8_img(moving)
    fixed_height, fixed_width = fixed_o.shape[:2]
    moving_height, moving_width = moving_o.shape[:2]

    min_width = min(fixed_width, moving_width)
    min_height = min(fixed_height, moving_height)

    f_trimmed_x = (fixed_width - min_width) // 2
    f_trimmed_y = (fixed_height - min_height) // 2
    m_trimmed_x = (moving_width - min_width) // 2
    m_trimmed_y = (moving_height - min_height) // 2

    fixed_trimmed = fixed_o[f_trimmed_y:f_trimmed_y + min_height,
                            f_trimmed_x:f_trimmed_x + min_width]
    moving_trimmed = moving_o[m_trimmed_y:m_trimmed_y + min_height,
                              m_trimmed_x:m_trimmed_x + min_width]

    return fixed_trimmed, moving_trimmed, f_trimmed_x, f_trimmed_y, m_trimmed_x, m_trimmed_y


def _optical_flow(img1, img2, fraction=1, attachment=50, tightness=50,
                  num_warp=3, num_iter=3):
    fixed = _uint8_img(img1)
    moving = _uint8_img(img2)
    if len(fixed.shape) == 3:
        fixed = color.rgb2gray(fixed)
    if len(moving.shape) == 3:
        moving = color.rgb2gray(moving)
    flow = registration.optical_flow_tvl1(
        fixed, moving,
        attachment=attachment, tightness=tightness,
        num_warp=num_warp, num_iter=num_iter,
        tol=0.000001, prefilter=False)
    v, u = flow
    nr, nc = fixed.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    moving_warped = transform.warp(
        moving,
        np.array([row_coords + v * fraction, col_coords + u * fraction]),
        mode='edge')
    moving_warped = exposure.rescale_intensity(moving_warped, in_range='image',
                                               out_range=(0, 255))
    moving_warped = moving_warped.astype(np.uint8)
    return moving_warped


def _mask_app(img, mask):
    if len(img.shape) == 3:
        mask = np.stack((mask, mask, mask), axis=-1)
    img_masked = np.multiply(img, mask)
    img_masked = _uint8_img(img_masked)
    return img_masked


def _radial_mask_app(image):
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    y, x = np.indices((h, w))
    distance_map = np.hypot(abs(x - center_x) * h / w, abs(y - center_y))
    max_distance = max(center_x, center_y)
    gradient_mask = np.maximum(1 - (distance_map / max_distance) * 2, 0)
    distance_map2 = np.maximum(abs(x - center_x) * h / w, abs(y - center_y))
    max_distance2 = np.maximum(abs(center_x) * h / w, abs(center_y))
    gradient_mask2 = 1 - (distance_map2 / max_distance2) ** 2
    gradient_mask = np.maximum(gradient_mask, gradient_mask2)
    # NOTE: 아래 블록은 legacy 원본 그대로 보존. gr_mask가 (h,w)로 할당되어
    # 슬라이스가 [0:h, 0:w] → 실질적으로 gradient_mask3 = gradient_mask.
    # 원본에서 다른 크기 캔버스를 위한 코드로 보이나 현재는 no-op.
    gr_mask = np.zeros((h, w))
    gr_hs = int(gr_mask.shape[0] // 2) - int(h // 2)
    gr_he = gr_hs + h
    gr_ws = int(gr_mask.shape[1] // 2) - int(w // 2)
    gr_we = gr_ws + w
    gr_mask[gr_hs:gr_he, gr_ws:gr_we] = gradient_mask
    gradient_mask3 = cv2.resize(gr_mask, (w, h), interpolation=cv2.INTER_AREA)
    gradient_mask255 = gradient_mask3 * 255
    fl_mask = (gradient_mask255 / 255).astype(np.float64)
    image_rm = _mask_app(image, fl_mask)
    return image_rm


# NOTE: _register_img_a와 _register_img_s는 ~90% 중복이지만,
# legacy 원본(core_crop_250902.py:207-280)의 verbatim 복사본.
# 차이점: _a는 AffineTransform + resample 단계 포함, _s는 Similarity2DTransform.
# 원본 추적성을 위해 별도 함수로 유지.
def _register_img_a(fixed, moving, numberOfIterations=10, convergenceWindowSize=10):
    fixed_image_sitk = sitk.GetImageFromArray(fixed)
    moving_image_sitk = sitk.GetImageFromArray(moving)

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image_sitk)
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkLinear)
    moving_image_sitk = resample.Execute(moving_image_sitk)

    fixed_image_sitk = sitk.Cast(fixed_image_sitk, sitk.sitkFloat32)
    moving_image_sitk = sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(1.0)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(
        learningRate=0.0001, numberOfIterations=numberOfIterations,
        convergenceMinimumValue=1e-9, convergenceWindowSize=convergenceWindowSize)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(sitk.AffineTransform(fixed_image_sitk.GetDimension()))
    R.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    final_transform = R.Execute(fixed_image_sitk, moving_image_sitk)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    registered_image_sitk = resampler.Execute(moving_image_sitk)
    registered_image = sitk.GetArrayFromImage(registered_image_sitk)
    final_transform_inv = final_transform.GetInverse()
    matrix = np.eye(3)
    matrix[:2, :2] = np.array(final_transform_inv.GetMatrix()).reshape((2, 2))
    translation = final_transform_inv.GetTranslation()
    matrix[0, 2] = translation[0]
    matrix[1, 2] = translation[1]
    return registered_image, matrix


def _register_img_s(fixed, moving, numberOfIterations=10, convergenceWindowSize=10):
    fixed_image_sitk = sitk.GetImageFromArray(fixed)
    moving_image_sitk = sitk.GetImageFromArray(moving)
    fixed_image_sitk = sitk.Cast(fixed_image_sitk, sitk.sitkFloat32)
    moving_image_sitk = sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetMetricSamplingPercentage(1.0)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(
        learningRate=0.0001, numberOfIterations=numberOfIterations,
        convergenceMinimumValue=1e-9, convergenceWindowSize=convergenceWindowSize)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(sitk.Similarity2DTransform())
    R.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    final_transform = R.Execute(fixed_image_sitk, moving_image_sitk)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    registered_image_sitk = resampler.Execute(moving_image_sitk)
    registered_image = sitk.GetArrayFromImage(registered_image_sitk)
    final_transform_inv = final_transform.GetInverse()
    matrix = np.eye(3)
    matrix[:2, :2] = np.array(final_transform_inv.GetMatrix()).reshape((2, 2))
    translation = final_transform_inv.GetTranslation()
    matrix[0, 2] = translation[0]
    matrix[1, 2] = translation[1]
    return registered_image, matrix


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_img_s(fixed, moving, numberOfIterations=10,
                    convergenceWindowSize=10):
    """SimpleITK Similarity2D registration (public API).

    Phase E refinement에서 사용. LEGACY_AVAILABLE=False면 (None, eye(3)) 반환.
    """
    if not LEGACY_AVAILABLE:
        return None, np.eye(3)
    return _register_img_s(fixed, moving, numberOfIterations,
                            convergenceWindowSize)


def run_of_loop(fixed_gray: np.ndarray,
                moving_gray: np.ndarray,
                allow_affine: bool = False,
                max_iter: int = 50,
                auto_stop: bool = True) -> dict:
    """Iterative OF + SimpleITK registration loop (legacy fallback).

    Args:
        fixed_gray:  Preprocessed grayscale uint8 image (reference).
        moving_gray: Preprocessed grayscale uint8 image (to be registered).
        allow_affine: Use Affine (6 DoF) instead of Similarity2D (4 DoF).
        max_iter:    Maximum number of OF iterations.
        auto_stop:   Stop early when cosine similarity of translation
                     vectors reverses direction (oscillation detection).

    Returns:
        dict with keys: M_2x3, n_iterations, converged, final_cosine
    """
    null_result = {
        'M_2x3': None,
        'n_iterations': 0,
        'converged': False,
        'final_cosine': None,
    }

    if not LEGACY_AVAILABLE:
        return null_result

    try:
        return _run_of_loop_inner(fixed_gray, moving_gray,
                                  allow_affine, max_iter, auto_stop)
    except Exception as e:
        print(f"[WARN] Legacy OF loop failed: {type(e).__name__}: {e}")
        return null_result


def _run_of_loop_inner(fixed_gray, moving_gray,
                       allow_affine, max_iter, auto_stop):
    """Core loop, separated so the outer function can catch exceptions."""

    register_fn = _register_img_a if allow_affine else _register_img_s

    # Step 1: template_fab — center-crop to same size
    (fixed_t, moving_t,
     f_trimmed_x, f_trimmed_y,
     m_trimmed_x, m_trimmed_y) = _template_fab(fixed_gray, moving_gray)

    # Step 2: radial mask on fixed template
    fixed_t_masked = _radial_mask_app(fixed_t)

    # Step 3-5: init
    pseudo_registered = moving_t
    matrix_i = np.eye(3)
    prev_matrix_i = np.eye(3)
    prev_translation_vec = np.array([0.0, 0.0])

    converged = False
    final_cosine = None
    n_iterations = max_iter

    # Step 6: main iteration loop
    for i in range(1, max_iter + 1):
        # a. mask pseudo_registered
        pseudo_registered_masked = _radial_mask_app(pseudo_registered)

        # b. optical flow
        of_registered = _optical_flow(fixed_t_masked, pseudo_registered_masked,
                                      fraction=0.5)

        # c+d. blend then register
        blend = _overlay_img(of_registered, fixed_t_masked, 0.5, 0.5)
        _, matrix = register_fn(blend, pseudo_registered_masked, 10, 10)

        # e. accumulate transform
        matrix_i = matrix @ matrix_i

        # f. convergence check via cosine similarity of translation delta
        current_vec = matrix_i[:2, 2] - prev_matrix_i[:2, 2]
        if i > 1:
            norm_prod = (np.linalg.norm(current_vec)
                         * np.linalg.norm(prev_translation_vec) + 1e-12)
            cos_sim = np.dot(current_vec, prev_translation_vec) / norm_prod
            final_cosine = float(cos_sim)
            if cos_sim < -0.8 and i > 8 and auto_stop:
                matrix_i = prev_matrix_i
                n_iterations = i
                converged = True
                break

        # g. update previous state
        prev_translation_vec = current_vec
        prev_matrix_i = matrix_i.copy()

        # h. warp moving with accumulated transform
        pseudo_registered = cv2.warpAffine(
            moving_t, matrix_i[:2, :],
            (fixed_t.shape[1], fixed_t.shape[0]))
    # Step 7: final tight pass
    pseudo_registered_masked = _radial_mask_app(pseudo_registered)
    _, matrix = register_fn(fixed_t_masked, pseudo_registered_masked, 100, 100)
    matrix_i = matrix @ matrix_i

    # Step 8: trim offset correction (template space -> input image space)
    # Correct chain: p_fixed = M @ (p_moving - m_offset) + f_offset
    #              = T_f @ M @ T_{-m}  (not just t += f - m)
    T_f = np.eye(3)
    T_f[0, 2] = f_trimmed_x
    T_f[1, 2] = f_trimmed_y
    T_m_inv = np.eye(3)
    T_m_inv[0, 2] = -m_trimmed_x
    T_m_inv[1, 2] = -m_trimmed_y
    matrix_i = T_f @ matrix_i @ T_m_inv

    return {
        'M_2x3': matrix_i[:2, :],
        'n_iterations': n_iterations,
        'converged': converged,
        'final_cosine': final_cosine,
    }
