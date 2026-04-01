# Technical Design Document (TDD)
# 임상사진 자동 정합 파이프라인 v2

**문서 버전**: 1.0
**작성일**: 2026-04-01
**기반 문서**: `pipeline_v2_plan.md`, `SDD.md`

---

## 1. 구현 세부 명세

### 1.1 파일 구조

```
C:\Users\User\Desktop\TEST\
├── preprocess.py          # Phase A: 전처리 함수군
├── matching.py            # Phase B: LoFTR 매칭 + 필터링
├── transform.py           # Phase C: 행렬 합성 + 품질 게이트
├── register.py            # Phase A~D 통합 오케스트레이터
├── test_synthetic.py      # Step 0: 좌표계 합성 검증
├── test_offline.py        # Step 3: pair1~4 오프라인 검증
├── core_crop_250902.py    # legacy (읽기 전용, fallback 전용)
├── SDD.md                 # Software Design Document
├── TDD.md                 # Technical Design Document (본 문서)
└── pipeline_v2_plan.md    # 구현 계획서
```

---

## 2. preprocess.py 구현 명세

### 2.1 apply_clahe

```
함수: apply_clahe(img_gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray
```

**알고리즘**:
1. `cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))`
2. `clahe.apply(img_gray)` → uint8 반환

**전제 조건**: `img_gray`는 uint8 단채널. 컬러 이미지 입력 시 예외 없이 잘못된 결과 발생 (호출자 책임).

**기존 코드와의 차이**: `equalize_adapthist` (skimage, clip_limit=0.1) 대체. 실험 결과 pair3에서 13→481 매칭 (37배 개선).

---

### 2.2 rotate_with_matrix

```
함수: rotate_with_matrix(image: np.ndarray, angle_deg: float, center: tuple = None)
      -> (rotated_img: np.ndarray, M_rot_3x3: np.ndarray)
```

**알고리즘**:
1. `center` 기본값 = `(w/2, h/2)`
2. `cv2.getRotationMatrix2D(center, -angle_deg, 1.0)` → M_2x3
   - OpenCV 관례: 양수 angle = 반시계. 우리는 `angle_deg` 양수 = 반시계로 통일하므로 `-angle_deg` 전달
3. 캔버스 확장 계산:
   ```
   cos_a = abs(M_2x3[0, 0])
   sin_a = abs(M_2x3[0, 1])
   new_w = int(h * sin_a + w * cos_a)
   new_h = int(h * cos_a + w * sin_a)
   ```
4. 중심점 보정 (확장된 캔버스에 맞춤):
   ```
   M_2x3[0, 2] += new_w / 2 - center[0]
   M_2x3[1, 2] += new_h / 2 - center[1]
   ```
5. `cv2.warpAffine(image, M_2x3, (new_w, new_h), borderMode=BORDER_CONSTANT, borderValue=0)`
6. 3×3 homogeneous 행렬 구성: `M_3x3[:2, :] = M_2x3`

**핵심 반환값**: `M_3x3`은 원본 좌표 → 회전+확장 좌표 변환. `compose_full_matrix()`에서 필수.

**주의사항**:
- mask 회전 시에는 이 함수를 직접 쓰지 않고, 반환된 `M_2x3`으로 `cv2.warpAffine(..., flags=INTER_NEAREST)` 별도 호출
- borderValue=0: 검은색 패딩 (이미지용). mask용은 호출자에서 0으로 설정

---

### 2.3 auto_orient_and_crop

```
함수: auto_orient_and_crop(image: np.ndarray, mask: np.ndarray, padding_ratio: float = 0.1)
      -> (cropped_img, cropped_mask, M_rot_3x3, crop_offset)
```

**알고리즘 상세**:

```
Step 1: minAreaRect 추출
  contours = cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
  all_pts = np.vstack(contours)
  rect = cv2.minAreaRect(all_pts)
  center, (rect_w, rect_h), angle = rect

Step 2: Angle 정규화
  if rect_w < rect_h:
      angle += 90
      swap(rect_w, rect_h)
  # → width > height 보장 (가로가 긴 방향)

Step 3: 정사각형 배열 감지
  aspect_ratio = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-6)
  if aspect_ratio < 1.2:
      angle = 0  # 회전 불안정 → 스킵

Step 4: 회전
  rotated_img, M_rot = rotate_with_matrix(image, angle)

Step 5: mask 회전 (INTER_NEAREST)
  M_2x3 = M_rot[:2, :]
  rotated_mask = cv2.warpAffine(mask, M_2x3, (new_w, new_h),
                                 flags=INTER_NEAREST, borderValue=0)
  rotated_mask = (rotated_mask > 127).astype(uint8) * 255

Step 6: 크롭 (패딩 포함)
  x, y, bw, bh = cv2.boundingRect(rotated_mask)
  pad = int(max(bw, bh) * padding_ratio)
  crop_offset = (max(0, x - pad), max(0, y - pad))
  cropped = rotated[y1:y2, x1:x2]
```

**minAreaRect angle 정규화 근거**:
- `cv2.minAreaRect`는 angle을 -90~0 또는 0~90으로 반환 (OpenCV 버전에 따라 다름)
- width/height 정의도 불안정 (어떤 변이 width인지 OpenCV 내부 규칙에 따름)
- `width > height` 정규화로 항상 "가로 > 세로" 보장 → 회전 방향 일관성 확보

**crop_offset 좌표계**: 회전된 이미지 좌표계에서의 (x1, y1). 원본 좌표계 아님.

---

### 2.4 resize_to_max

```
함수: resize_to_max(img: np.ndarray, max_side: int = 640)
      -> (resized_img: np.ndarray, scale_factor: float)
```

**알고리즘**:
1. `long_side = max(h, w)`
2. `long_side <= max_side` → 복사본 반환, scale=1.0
3. `scale_factor = max_side / long_side`
4. `cv2.resize(img, (new_w, new_h), interpolation=INTER_AREA)` (축소 시 INTER_AREA 최적)

**640px 선택 근거**: Kornia LoFTR 공식 권고 — indoor 모델은 640×480 이하. 기존 `find_scaling_factor()`는 면적 기반이라 pair마다 입력 해상도가 불안정했음.

---

## 3. matching.py 구현 명세

### 3.1 모델 싱글턴 로딩

```python
_loftr_model = None  # 모듈 레벨 전역

def _get_loftr_model(pretrained='indoor_new') -> KF.LoFTR:
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
```

**Pretrained 선택 근거**:
- `indoor_new` (1순위): ScanNet 학습, 최신 weight. 구강 사진은 ~30cm 촬영 거리, 좁은 시야 → indoor 적합
- `indoor` (2순위 fallback): 구 버전 weight. `indoor_new` 미지원 환경용
- `outdoor`: MegaDepth (건물/풍경). 비교 실험용으로만 사용

**기존 코드 대비 개선**: `get_depth_map()`이 매 호출마다 모델 재로딩하던 버그 방지

---

### 3.2 loftr_match

```
함수: loftr_match(img1_gray: uint8, img2_gray: uint8, conf_threshold: float = 0.5)
      -> (kpts0: N×2, kpts1: N×2, conf: N)
```

**LoFTR 입력 텐서 규격**:
```
input = torch.from_numpy(img_gray).float()[None, None] / 255.0
# shape: (1, 1, H, W), dtype: float32, range: [0, 1]
```

**출력 좌표계**: LoFTR은 입력 이미지 크기 기준 (리사이즈된 크롭 좌표계)의 (x, y) 반환.

**conf_threshold = 0.5 근거**: 실험에서 0.5 이상이면 고품질 매칭. 0.3까지 내리면 노이즈 증가, 0.7은 너무 보수적 (pair3에서 매칭 부족).

**mask0/mask1 API 미사용 근거**:
- kornia LoFTR의 `mask0`/`mask1`은 원래 padding/invalid region용
- tooth mask로 직접 넣으면 LoFTR 내부 attention 패턴이 왜곡될 수 있음
- 기본 경로: 마스크 없이 매칭 → 출력 후처리에서 `filter_by_mask()` 적용

---

### 3.3 filter_by_mask

```
함수: filter_by_mask(kpts0, kpts1, conf, mask0, mask1, sigma=7, threshold=0.3)
      -> (filtered_kpts0, filtered_kpts1, filtered_conf)
```

**Soft mask 생성**:
```
soft = cv2.GaussianBlur(mask.astype(float32) / 255.0, (sigma*2+1, sigma*2+1), sigma)
```

**필터링 기준**: `min(soft0[y0, x0], soft1[y1, x1]) > threshold`
- 양쪽 모두 마스크 영역 안이어야 통과
- `min` 연산으로 한쪽이라도 배경이면 제거

**sigma=7, threshold=0.3 근거**: 마스크 경계에서 5~10px 범위의 soft transition. Hard binary (threshold=0.5)보다 경계 근처 유효 대응점 보존.

---

## 4. transform.py 구현 명세

### 4.1 compose_full_matrix — 좌표계 역산 수학

**변환 체인 (각 이미지)**:
```
원본 (p_orig)
  → rotate (M_rot)           : p_rot = M_rot @ p_orig
  → crop (translate)          : p_crop = p_rot - [cx, cy, 0]
  → resize (scale)            : p_resize = s * p_crop
```

행렬 표현:
```
A = S @ C @ M_rot

여기서:
  S = scale_matrix(resize_scale)
  C = translation_matrix(-crop_offset_x, -crop_offset_y)
  M_rot = rotate_with_matrix()의 3×3 출력
```

**LoFTR 매칭 좌표계**: `p_resize` 좌표계. 즉:
```
p_resize_fixed = M_loftr @ p_resize_moving
```

**원본 좌표계 역산**:
```
A_f @ p_orig_fixed = M_loftr @ A_m @ p_orig_moving
p_orig_fixed = A_f⁻¹ @ M_loftr @ A_m @ p_orig_moving

∴ M_full = A_f⁻¹ @ M_loftr_3x3 @ A_m
```

**A_f⁻¹ 누락 방지**: `M_full = M_loftr @ A_m`만 하면 "크롭 좌표에서는 맞는데 원본에서 틀어지는" 버그 발생. 디버깅 극히 어려움 (미세한 오프셋+스케일 오차).

---

### 4.2 quality_gate_similarity — 판정 기준 상세

#### Hard Fail 조건 (즉시 다음 fallback)

| 조건 | 임계값 | 근거 |
|---|---|---|
| n_inlier | < 12 | 치아 1~2개 노출 시 12~20개 정도 inlier 가능. 12 미만은 불안정 |
| det(M) | ≤ 0 | Reflection 발생 (물리적으로 불가능한 변환) |
| reproj_median | ≥ 5px | 640px 이미지에서 5px 중앙값은 심각한 오차 |
| reproj_p90 | ≥ 12px | 상위 10% 오차가 12px 이상이면 outlier가 RANSAC 통과 |
| scale | < 0.7 또는 > 1.4 | 동일 프로토콜 촬영 → 40% 이상 스케일 차이 불가 |
| \|rotation\| | > 20° | 임상사진 프로토콜상 20° 이상 회전 차이 비현실적 |

#### Warn 조건 (결과 사용하되 경고)

| 조건 | 임계값 | 근거 |
|---|---|---|
| n_inlier | < 30 | 안정적 추정에는 30+ 선호 |
| reproj_median | ≥ 3px | 3px 이상이면 정밀도 의심 |
| hull_coverage | < 0.2 | inlier가 좁은 영역에 집중 → 외삽 위험 |
| scale | 0.8~1.2 벗어남 | 정상 범위 밖이지만 가능한 범위 |
| \|rotation\| | > 15° | 주의 필요하나 가능한 범위 |

#### Reproj error 계산

```python
kpts_m_h = np.hstack([kpts_m, np.ones((n, 1))])  # homogeneous
projected = (M_3x3 @ kpts_m_h.T).T[:, :2]
errors = np.linalg.norm(projected - kpts_f, axis=1)
inlier_errors = errors[inlier_mask]
reproj_median = np.median(inlier_errors)
reproj_p90 = np.percentile(inlier_errors, 90)
```

**mean 대신 median + p90 사용 근거**: RANSAC 후에도 marginal inlier (inlier threshold 근처)가 mean을 왜곡할 수 있음. Median은 robust, p90은 worst-case 감시.

#### Hull coverage 분모

```python
kernel = cv2.getStructuringElement(MORPH_ELLIPSE, (5, 5))
eroded_mask = cv2.erode(mask_resized, kernel)
tooth_mask_area = np.sum(eroded_mask > 0)
coverage = hull_area / tooth_mask_area
```

**tooth_mask_area 사용 근거**: `crop_area` (전체 크롭 영역)를 분모로 쓰면 배경 부분에는 대응점이 없으므로 coverage가 인위적으로 낮아짐. 필터 기준(soft mask)과 평가 기준(coverage 분모)을 동일 영역으로 맞춤.

**erosion 근거**: 마스크 경계에 노이즈가 있을 수 있음. 3~5px erosion으로 경계 제거.

---

### 4.3 quality_gate_affine — Similarity와의 차이

| 항목 | Similarity 게이트 | Affine 게이트 |
|---|---|---|
| scale/rotation 체크 | O (det→scale, atan2→rotation) | X |
| shear 체크 | 해당 없음 | 불가 (SVD 분해 필요하나 해석 복잡) |
| det sign 체크 | O | O |
| reproj + inlier | O | O |
| hull coverage | O (0.2) | O (0.15, 더 관대) |

**Affine에서 scale/rotation 체크 제거 근거**: Affine 행렬 `[a b tx; c d ty]`에서 `det→scale`, `atan2(c,a)→rotation` 분해는 shear 성분이 있으면 무의미. SVD 분해 `(U, S, Vt) = svd(A[:2,:2])`로 rotation/scale/shear 분리 가능하나, 임상적 해석이 어려워 threshold 설정 불가.

---

## 5. register.py 구현 명세

### 5.1 register_pair 실행 흐름

```
Phase A: 전처리
  ├─ auto_orient_and_crop(fixed) → (crop_f, mask_crop_f, M_rot_f, off_f)
  ├─ auto_orient_and_crop(moving) → (crop_m, mask_crop_m, M_rot_m, off_m)
  ├─ cvtColor → grayscale
  ├─ apply_clahe → CLAHE 적용
  ├─ resize_to_max(640) → (resized_f, scale_f), (resized_m, scale_m)
  └─ mask도 동일 resize (INTER_NEAREST)

Phase B: LoFTR 매칭
  ├─ loftr_match(resized_f, resized_m, conf=0.5) → (kpts0, kpts1, conf)
  └─ filter_by_mask(kpts0, kpts1, conf, mask0, mask1) → filtered

Phase C: 변환 추정 + 품질 판정
  ├─ [1차] estimateAffinePartial2D(kpts1→kpts0, RANSAC, reproj=3.0, conf=0.99)
  │   └─ quality_gate_similarity → pass/warn → 채택
  ├─ [2차] estimateAffine2D(kpts1→kpts0, RANSAC, reproj=3.0, conf=0.99)
  │   └─ quality_gate_affine → pass/warn → 채택
  └─ [3차] legacy OF 루프 (allow_legacy_fallback=True일 때만)

Phase D: 행렬 역산 + 원본 적용
  ├─ compose_full_matrix(M_loftr, M_rot_f, off_f, scale_f, M_rot_m, off_m, scale_m)
  └─ cv2.warpAffine(moving_img, M_full[:2,:], (fixed_w, fixed_h))
```

### 5.2 estimateAffinePartial2D vs estimateAffine2D

| 항목 | estimateAffinePartial2D | estimateAffine2D |
|---|---|---|
| DoF | 4 (rotation + uniform scale + translation) | 6 (rotation + non-uniform scale + shear + translation) |
| 행렬 형태 | `[s·cosθ, -s·sinθ, tx; s·sinθ, s·cosθ, ty]` | 일반 `[a, b, tx; c, d, ty]` |
| 적합 대상 | rigid body (치아) | 비강체 변형 포함 시 |
| RANSAC reproj | 3.0px | 3.0px |

**방향 주의**: `estimateAffinePartial2D(src=kpts1, dst=kpts0)` → moving→fixed 방향. `src`에 moving, `dst`에 fixed.

### 5.3 Legacy OF Fallback 래핑

기존 `core_crop_250902.py`의 `reg_body_single()` OF 반복 루프를 호출하되:
- `equalize_adapthist` → `apply_clahe`로 교체
- `chan_vese + u2net + depth_map` → SAM2 마스크로 교체
- 나머지 (`radial_mask_app`, `template_fab`, OF 반복, cosine similarity 수렴 감지`) 유지

**구현 방식**: `core_crop_250902.py`는 수정하지 않음. 필요한 함수만 import하여 래핑.

---

## 6. test_synthetic.py 구현 명세

### 6.1 Test 0a: Identity

```
입력: 동일 이미지 2장, 동일 마스크
기대: M_full ≈ I (3×3 identity)
검증:
  - |M[0,2]| < 1.0  (translation x < 1px)
  - |M[1,2]| < 1.0  (translation y < 1px)
  - |atan2(M[1,0], M[0,0])| < 0.5°  (rotation)
  - |sqrt(det) - 1.0| < 0.01  (scale)
```

### 6.2 Test 0b: Known Similarity Transform

```
GT 변환: 7° 회전, 1.03x 스케일, (+20, +15) 이동
moving = warpAffine(fixed, M_gt)
moving_mask = warpAffine(fixed_mask, M_gt, INTER_NEAREST)

register_pair(fixed, moving, mask, moving_mask) → M_est

검증: M_est ≈ M_gt⁻¹ (moving→fixed 방향)
  - |est_scale - gt_inv_scale| / gt_inv_scale < 0.005  (0.5% 상대오차)
  - |est_angle - gt_inv_angle| < 0.5°
```

**LoFTR 사용 여부**: 실제 LoFTR 호출. synthetic이지만 warpAffine된 이미지에서도 LoFTR이 잘 동작하는지 같이 검증.

### 6.3 Test 0c: Dual Rotation Chain

```
fixed를 α=12° 회전+크롭
moving을 β=-5° 회전+크롭
LoFTR 대신 GT 대응점 직접 생성:
  - fixed에서 N개 점 샘플
  - M_gt로 변환하여 moving 대응점 생성
  - 각각 auto_orient_and_crop 변환 적용

compose_full_matrix() 출력으로 원본 moving → 원본 fixed warp
warp 결과 vs 원본 fixed의 SSIM > 0.95
```

**이 테스트의 핵심**: LoFTR 성능이 아닌, 좌표계 변환 체인(`M_rot`, `crop_offset`, `resize_scale`)이 정확히 역산되는지 검증. GT 대응점을 사용하여 매칭 불확실성을 제거.

---

## 7. test_offline.py 구현 명세

### 7.1 입력 디렉토리 구조

```
TEST/
├── pair1/
│   ├── fixed.jpg
│   ├── moving.jpg
│   ├── fixed_mask.png    # SAM2로 미리 생성, uint8, 0 or 255
│   └── moving_mask.png
├── pair2/ ...
├── pair3/ ...  (극단 배경 차이 — 핵심 테스트)
└── pair4/ ...
```

### 7.2 출력

```
TEST/results/
├── pair1_registered.jpg
├── pair1_false_color.jpg
├── pair1_debug/
│   ├── fixed_crop.jpg
│   ├── moving_crop.jpg
│   ├── fixed_clahe.jpg
│   └── moving_clahe.jpg
├── pair2_registered.jpg
├── ...
└── metrics.csv
```

### 7.3 metrics.csv 스키마

| 컬럼 | 타입 | 설명 |
|---|---|---|
| pair | str | pair1, pair2, ... |
| path | str | similarity / affine / legacy / failed |
| n_raw | int | 전체 LoFTR 매칭 수 |
| n_filtered | int | 마스크 필터링 후 매칭 수 |
| n_inlier | int | RANSAC inlier 수 |
| inlier_ratio | float | n_inlier / n_filtered |
| reproj_median | float | reprojection error 중앙값 (px) |
| reproj_p90 | float | reprojection error 90th percentile (px) |
| hull_coverage | float | convex hull / tooth_mask_area |
| scale | float | 추정 스케일 |
| rotation_deg | float | 추정 회전각 (도) |
| elapsed_sec | float | 처리 시간 (초) |
| status | str | pass / warn / fail |

### 7.4 성공 기준

| 기준 | 설명 |
|---|---|
| pair1~4 중 최소 3개 | similarity 경로 pass |
| 나머지 | warn 이하 (fail 없음) |
| pair3 | 정합 성공 (pass 또는 warn) |
| 처리 시간 | pair당 < 10초 (GPU 기준) |

---

## 8. 에러 처리 전략

### 8.1 예외 처리 범위

| 위치 | 예외 상황 | 처리 |
|---|---|---|
| `auto_orient_and_crop` | contour 없음 | `ValueError` raise |
| `loftr_match` | CUDA OOM | torch.cuda.empty_cache() 후 CPU fallback |
| `compose_full_matrix` | A_f 특이행렬 | `np.linalg.LinAlgError` → fail 반환 |
| `register_pair` | 매칭 0개 | fallback 체인 진입 |
| 모델 로딩 | `indoor_new` 미지원 | `indoor` fallback |

### 8.2 예외를 처리하지 않는 경우

- `img_gray`에 컬러 이미지 전달: 호출자 책임 (내부 체크 없음)
- 마스크 크기 ≠ 이미지 크기: 호출자 책임
- 파일 I/O: test_offline.py에서만 처리

---

## 9. 의존성 및 환경

### 9.1 필수 패키지

```
numpy
opencv-python          # cv2
torch                  # LoFTR, SAM2
kornia                 # LoFTR (kornia.feature.LoFTR)
scikit-image           # legacy fallback (skimage.registration 등)
SimpleITK              # legacy fallback (register_img_s/a)
```

### 9.2 선택 패키지

```
segment-anything-2     # SAM2 (Step 4 이후). Windows 네이티브 호환성 리스크
matplotlib             # 시각화 (test_offline 디버그 이미지)
```

### 9.3 제거 가능 패키지

```
rembg                  # seg_img_u2net → SAM2 대체
transformers           # DPT-Large depth map → 불필요
```

### 9.4 환경 제약

| 항목 | 요구사항 |
|---|---|
| OS | Windows 11 (네이티브) |
| Python | 3.10+ |
| CUDA | 권장 (CPU에서도 동작하나 속도 저하) |
| VRAM | 4GB+ (LoFTR ~1GB + SAM2 ~2GB 동시 로딩) |
| SAM2 | Windows 네이티브 호환성 확인 필요 (공식: WSL 권장) |

---

## 10. 기존 코드 재활용 매핑

### 10.1 재활용 (신규 모듈로 이동)

| 기존 함수 | 이동 위치 | 수정 |
|---|---|---|
| `uint8_img()` | preprocess.py | 수정 없음 |
| `false_color()` | register.py | 수정 없음 |
| `overlay_img()` | register.py | 수정 없음 |
| `rotate_image_with_padding()` 로직 | preprocess.py `rotate_with_matrix()` | 3×3 행렬 반환 추가 |

### 10.2 Legacy Fallback 전용 유지

| 기존 함수 | 용도 |
|---|---|
| `register_img_s()` | OF 루프 내 similarity 정합 |
| `register_img_a()` | OF 루프 내 affine 정합 |
| `optical_flow()` | tvl1 OF |
| `template_fab()` | 크기 맞춤 |
| `radial_mask_app()` | 가장자리 감쇠 |

### 10.3 제거 (신규 파이프라인에서 불필요)

| 기존 함수/모듈 | 대체 |
|---|---|
| `manual_alignment_window()` + 선분 관련 | SAM2 + minAreaRect |
| `open_roi_selection()` + 사각형 드래그 | 자동 크롭 |
| `seg_img()` (chan_vese) | SAM2 |
| `seg_img_u2net()` (rembg) | SAM2 |
| `get_depth_map()` (DPT-Large) | 불필요 |
| `equalize_two_images()` | CLAHE |
| `find_scaling_factor()` | `resize_to_max(640)` |
