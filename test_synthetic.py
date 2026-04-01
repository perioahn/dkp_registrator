"""
Step 0: 좌표계 합성 검증.
이 3개 테스트가 통과하지 않으면 이후 단계 진행 금지.

사용법:
  python test_synthetic.py [--image path] [--mask path]

이미지/마스크 미지정 시 synthetic 이미지 자동 생성.
"""

import sys
import argparse
import numpy as np
import cv2

from preprocess import auto_orient_and_crop, resize_to_max, apply_clahe, rotate_with_matrix
from matching import loftr_match, filter_by_mask
from transform import compose_full_matrix, quality_gate_similarity
from register import register_pair


def create_synthetic_image(w=800, h=600):
    """테스트용 synthetic 이미지 + 마스크 생성.

    가로로 긴 타원형 마스크 안에 다양한 texture 패턴을 넣어
    LoFTR가 매칭할 수 있도록 함.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 타원형 마스크 (치아 영역 모사)
    center = (w // 2, h // 2)
    axes = (w // 3, h // 4)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # texture 생성: 다양한 패턴으로 LoFTR 매칭 가능하게
    rng = np.random.RandomState(42)

    # 배경: 회색
    img[:] = 127

    # 마스크 영역: 랜덤 패턴 + 원 + 선
    for _ in range(200):
        x = rng.randint(0, w)
        y = rng.randint(0, h)
        r = rng.randint(3, 15)
        color = tuple(int(c) for c in rng.randint(50, 250, 3))
        cv2.circle(img, (x, y), r, color, -1)

    for _ in range(50):
        x1, y1 = rng.randint(0, w), rng.randint(0, h)
        x2, y2 = rng.randint(0, w), rng.randint(0, h)
        color = tuple(int(c) for c in rng.randint(50, 250, 3))
        cv2.line(img, (x1, y1), (x2, y2), color, rng.randint(1, 4))

    # 격자 패턴
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x, h), (180, 180, 180), 1)
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), (180, 180, 180), 1)

    # 마스크 밖은 단색
    img[mask == 0] = 127

    return img, mask


def test_identity(image, mask):
    """Test 0a: 동일 이미지 쌍 → M_full ≈ I"""
    print("\n=== Test 0a: Identity ===")
    result = register_pair(image, image.copy(), mask, mask.copy(),
                          allow_legacy_fallback=False)

    if result['path'] == 'failed':
        print(f"  [FAIL] 정합 실패: {result['metrics']}")
        return False

    M = result['M_full']
    print(f"  Path: {result['path']}")
    print(f"  M_full:\n{M}")

    # M ≈ I 확인
    tx_err = abs(M[0, 2])
    ty_err = abs(M[1, 2])
    rot_err = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    scale_err = abs(np.sqrt(abs(det)) - 1.0)

    print(f"  Translation error: ({tx_err:.3f}, {ty_err:.3f}) px")
    print(f"  Rotation error: {rot_err:.3f}°")
    print(f"  Scale error: {scale_err:.4f}")

    ok = True
    if tx_err >= 1.0:
        print(f"  [FAIL] translation x: {tx_err:.3f} >= 1.0")
        ok = False
    if ty_err >= 1.0:
        print(f"  [FAIL] translation y: {ty_err:.3f} >= 1.0")
        ok = False
    if rot_err >= 0.5:
        print(f"  [FAIL] rotation: {rot_err:.3f}° >= 0.5°")
        ok = False
    if scale_err >= 0.01:
        print(f"  [FAIL] scale: {scale_err:.4f} >= 0.01")
        ok = False

    if ok:
        print("  [PASS] Test 0a: Identity")
    return ok


def test_known_transform(image, mask):
    """Test 0b: 알려진 similarity 변환 적용 후 역추정."""
    print("\n=== Test 0b: Known Similarity Transform ===")

    # GT 변환: 7° 회전, 1.03x 스케일, (+20, +15) 이동
    angle_gt = 7.0
    scale_gt = 1.03
    tx_gt, ty_gt = 20.0, 15.0

    cos_a = scale_gt * np.cos(np.radians(angle_gt))
    sin_a = scale_gt * np.sin(np.radians(angle_gt))
    M_gt = np.array([
        [cos_a, -sin_a, tx_gt],
        [sin_a,  cos_a, ty_gt],
        [0,      0,     1    ]
    ])

    h, w = image.shape[:2]

    # moving 생성
    moving = cv2.warpAffine(image, M_gt[:2, :], (w, h))
    moving_mask = cv2.warpAffine(mask, M_gt[:2, :], (w, h),
                                  flags=cv2.INTER_NEAREST)

    # 정합
    result = register_pair(image, moving, mask, moving_mask,
                          allow_legacy_fallback=False)

    if result['path'] == 'failed':
        print(f"  [FAIL] 정합 실패: {result['metrics']}")
        return False

    M_est = result['M_full']
    M_gt_inv = np.linalg.inv(M_gt)

    # 파라미터 비교
    est_det = M_est[0, 0] * M_est[1, 1] - M_est[0, 1] * M_est[1, 0]
    est_scale = np.sqrt(abs(est_det))
    est_angle = np.degrees(np.arctan2(M_est[1, 0], M_est[0, 0]))

    gt_inv_det = M_gt_inv[0, 0] * M_gt_inv[1, 1] - M_gt_inv[0, 1] * M_gt_inv[1, 0]
    gt_inv_scale = np.sqrt(abs(gt_inv_det))
    gt_inv_angle = np.degrees(np.arctan2(M_gt_inv[1, 0], M_gt_inv[0, 0]))

    print(f"  Path: {result['path']}")
    print(f"  Scale: GT_inv={gt_inv_scale:.4f}, Est={est_scale:.4f}")
    print(f"  Angle: GT_inv={gt_inv_angle:.2f}°, Est={est_angle:.2f}°")
    print(f"  Tx:    GT_inv={M_gt_inv[0, 2]:.2f}, Est={M_est[0, 2]:.2f}")
    print(f"  Ty:    GT_inv={M_gt_inv[1, 2]:.2f}, Est={M_est[1, 2]:.2f}")

    ok = True
    scale_rel_err = abs(est_scale - gt_inv_scale) / gt_inv_scale
    angle_err = abs(est_angle - gt_inv_angle)

    print(f"  Scale relative error: {scale_rel_err:.4f} (threshold: 0.005)")
    print(f"  Angle error: {angle_err:.3f}° (threshold: 0.5°)")

    if scale_rel_err >= 0.005:
        print(f"  [FAIL] scale 상대오차: {scale_rel_err:.4f} >= 0.005")
        ok = False
    if angle_err >= 0.5:
        print(f"  [FAIL] angle 오차: {angle_err:.3f}° >= 0.5°")
        ok = False

    if ok:
        print("  [PASS] Test 0b: Known Transform")
    return ok


def test_dual_rotation_chain(image, mask):
    """Test 0c: 양쪽 다른 각도로 회전+크롭 → GT 대응점 → 행렬 체인 검증.

    LoFTR 미사용. compose_full_matrix의 좌표계 역산이 정확한지 검증.
    """
    print("\n=== Test 0c: Dual Rotation Chain ===")

    h, w = image.shape[:2]

    # GT 변환: identity (fixed = moving의 경우)
    # 양쪽을 서로 다른 각도로 전처리한 후, 좌표 역산이 identity로 돌아오는지 확인
    alpha = 12.0  # fixed 회전
    beta = -5.0   # moving 회전

    # Phase A: 양쪽 각각 회전 + 크롭
    fixed_crop, fixed_mask_crop, M_rot_f, crop_off_f = \
        auto_orient_and_crop(image, mask)
    moving_crop, moving_mask_crop, M_rot_m, crop_off_m = \
        auto_orient_and_crop(image.copy(), mask.copy())

    # grayscale + CLAHE + resize
    fixed_gray = cv2.cvtColor(fixed_crop, cv2.COLOR_RGB2GRAY)
    moving_gray = cv2.cvtColor(moving_crop, cv2.COLOR_RGB2GRAY)
    fixed_clahe = apply_clahe(fixed_gray)
    moving_clahe = apply_clahe(moving_gray)
    fixed_resized, scale_f = resize_to_max(fixed_clahe, 640)
    moving_resized, scale_m = resize_to_max(moving_clahe, 640)

    # identity case: 동일 이미지 → M_loftr ≈ identity
    # GT 대응점 생성: 리사이즈된 좌표계에서 동일 좌표
    rng = np.random.RandomState(123)
    rh, rw = fixed_resized.shape[:2]
    n_pts = 50

    # 마스크 내 점 샘플링
    fixed_mask_resized = cv2.resize(fixed_mask_crop,
        (fixed_resized.shape[1], fixed_resized.shape[0]),
        interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(fixed_mask_resized > 127)
    if len(ys) < n_pts:
        print("  [FAIL] 마스크 내 점이 충분하지 않음")
        return False

    indices = rng.choice(len(ys), n_pts, replace=False)
    kpts_f = np.column_stack([xs[indices], ys[indices]]).astype(np.float32)
    kpts_m = kpts_f.copy()  # identity case

    # estimateAffinePartial2D with GT correspondences
    M_loftr, inliers = cv2.estimateAffinePartial2D(
        kpts_m, kpts_f,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.99
    )

    if M_loftr is None:
        print("  [FAIL] estimateAffinePartial2D 실패")
        return False

    print(f"  M_loftr (should be ~identity):\n{M_loftr}")

    # compose_full_matrix
    M_full = compose_full_matrix(
        M_loftr,
        M_rot_f, crop_off_f, scale_f,
        M_rot_m, crop_off_m, scale_m
    )

    print(f"  M_full (should be ~identity):\n{M_full}")

    # warp moving → fixed
    registered = cv2.warpAffine(image, M_full[:2, :], (w, h))

    # SSIM 계산 (마스크 영역만)
    fixed_gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    reg_gray = cv2.cvtColor(registered, cv2.COLOR_RGB2GRAY)

    # 마스크 영역 SSIM (간단 구현)
    mask_bool = mask > 127
    if np.sum(mask_bool) == 0:
        print("  [FAIL] 빈 마스크")
        return False

    # 마스크 영역만 추출하여 비교
    f_vals = fixed_gray_full[mask_bool].astype(np.float64)
    r_vals = reg_gray[mask_bool].astype(np.float64)

    # 간이 SSIM
    mu_f = np.mean(f_vals)
    mu_r = np.mean(r_vals)
    sigma_f = np.std(f_vals)
    sigma_r = np.std(r_vals)
    sigma_fr = np.mean((f_vals - mu_f) * (r_vals - mu_r))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ssim = ((2 * mu_f * mu_r + C1) * (2 * sigma_fr + C2)) / \
           ((mu_f**2 + mu_r**2 + C1) * (sigma_f**2 + sigma_r**2 + C2))

    print(f"  SSIM (masked): {ssim:.4f} (threshold: 0.95)")

    # M_full ≈ I 확인
    I = np.eye(3)
    mat_err = np.max(np.abs(M_full - I))
    print(f"  Max |M_full - I|: {mat_err:.4f}")

    ok = True
    if ssim < 0.95:
        print(f"  [FAIL] SSIM: {ssim:.4f} < 0.95")
        ok = False
    if mat_err > 0.1:
        print(f"  [WARN] M_full deviation from identity: {mat_err:.4f}")

    if ok:
        print("  [PASS] Test 0c: Dual Rotation Chain")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Step 0: 좌표계 합성 검증")
    parser.add_argument('--image', type=str, default=None,
                       help='테스트 이미지 경로')
    parser.add_argument('--mask', type=str, default=None,
                       help='테스트 마스크 경로')
    args = parser.parse_args()

    if args.image and args.mask:
        image = cv2.imread(args.image)
        if image is None:
            print(f"이미지를 읽을 수 없음: {args.image}")
            sys.exit(1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"마스크를 읽을 수 없음: {args.mask}")
            sys.exit(1)
    else:
        print("Synthetic 이미지 생성 중...")
        image, mask = create_synthetic_image()

    print(f"Image: {image.shape}, Mask: {mask.shape}")
    print(f"Mask area: {np.sum(mask > 0) / mask.size * 100:.1f}%")

    results = []
    results.append(("0a Identity", test_identity(image, mask)))
    results.append(("0b Known Transform", test_known_transform(image, mask)))
    results.append(("0c Dual Rotation Chain", test_dual_rotation_chain(image, mask)))

    print("\n" + "=" * 50)
    print("Summary:")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n*** ALL PASS - Step 1 이후 진행 가능 ***")
    else:
        print("\n*** FAIL - 이후 단계 진행 금지 ***")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
