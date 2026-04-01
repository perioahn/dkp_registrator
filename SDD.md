# Software Design Document (SDD)
# 임상사진 자동 정합 파이프라인 v2

**문서 버전**: 1.0
**작성일**: 2026-04-01
**기반 문서**: `pipeline_v2_plan.md`

---

## 1. 개요

### 1.1 시스템 목적

구강 수술 임상사진(수술 전/중/후)을 자동으로 정합(registration)하는 파이프라인.
기존 `core_crop_250902.py` (2175줄, GUI+로직 결합)를 모듈화·자동화하여 대체.

### 1.2 범위

- SAM2 기반 인터랙티브 마스크 생성
- LoFTR 기반 dense 대응점 매칭
- Similarity/Affine 변환 추정 + 품질 게이트
- 원본 해상도 변환 적용
- Legacy OF 루프 fallback 유지

### 1.3 용어 정의

| 용어 | 설명 |
|---|---|
| Fixed image | 기준 이미지 (정합 목표) |
| Moving image | 이동 이미지 (fixed에 맞춰 변환됨) |
| LoFTR | Local Feature Transformer. Detector-free dense matcher |
| SAM2 | Segment Anything Model 2. 인터랙티브 세그멘테이션 |
| Similarity transform | 4-DoF: 회전 + 균일 스케일 + 이동 |
| Affine transform | 6-DoF: 회전 + 비균일 스케일 + shear + 이동 |
| minAreaRect | cv2.minAreaRect. 최소 면적 회전 직사각형 |
| CLAHE | Contrast Limited Adaptive Histogram Equalization |

---

## 2. 아키텍처 개요

### 2.1 시스템 구조

```
┌─────────────────────────────────────────────────────┐
│                    main_gui.py                      │
│              (tkinter GUI, Step 5~6)                │
├─────────────────────────────────────────────────────┤
│                    register.py                      │
│       register_pair() — 파이프라인 오케스트레이터    │
├──────────┬──────────────┬───────────────────────────┤
│preprocess│  matching.py │     transform.py          │
│   .py    │              │                           │
│          │  LoFTR 매칭  │  행렬 합성 + 품질 게이트  │
│ CLAHE    │  마스크 필터  │  compose_full_matrix()    │
│ 회전/크롭│              │  quality_gate_*()         │
│ 리사이즈 │              │                           │
├──────────┴──────────────┴───────────────────────────┤
│              core_crop_250902.py                    │
│         (legacy fallback — 수정 안 함)              │
└─────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름 (Phase A → D)

```
 [입력]                    [Phase A]              [Phase B]          [Phase C]           [Phase D]
 fixed_img  ──┐        ┌─ auto_orient_and_crop ─┐
 fixed_mask ──┤        │  → rotate_with_matrix  │
              ├───────►│  → crop + padding      ├──► CLAHE ──► resize ──┐
 moving_img ──┤        │  → M_rot, crop_offset  │                      │
 moving_mask──┘        └─────────────────────────┘                      │
                                                                        ▼
                                                              loftr_match()
                                                                        │
                                                              filter_by_mask()
                                                                        │
                                                                        ▼
                                                         estimateAffinePartial2D
                                                         quality_gate_similarity
                                                                  │ fail?
                                                         estimateAffine2D
                                                         quality_gate_affine
                                                                  │ fail?
                                                         legacy OF 루프
                                                                        │
                                                                        ▼
                                                          compose_full_matrix()
                                                          M_full = A_f⁻¹ @ M @ A_m
                                                                        │
                                                                        ▼
                                                          warpAffine(moving, M_full)
                                                                        │
                                                                   [출력]
                                                            registered_img
                                                            M_full, metrics
```

### 2.3 설계 원칙

1. **GUI 분리**: `register.py` 이하 모듈은 tkinter 등 GUI 라이브러리 import 금지
2. **모델 싱글턴**: LoFTR, SAM2 등 모델은 모듈 레벨 1회 로드
3. **Fallback 체인**: Similarity → Affine → Legacy OF. 각 단계에 정량적 품질 게이트
4. **좌표계 추적**: 모든 변환 단계의 행렬을 보존하여 원본 해상도 역산 가능
5. **디버그 투명성**: 모든 중간 결과를 `debug_images` dict에 저장

---

## 3. 모듈 설계

### 3.1 preprocess.py

**책임**: 이미지 전처리 (CLAHE, 회전, 크롭, 리사이즈)

| 함수 | 입력 | 출력 | 설명 |
|---|---|---|---|
| `apply_clahe(img_gray, clip_limit, tile_size)` | uint8 gray | uint8 gray | CLAHE 적용 |
| `rotate_with_matrix(image, angle_deg, center)` | image, 각도 | (rotated_img, M_3x3) | 잘림없는 회전 + 3×3 행렬 반환 |
| `auto_orient_and_crop(image, mask, padding_ratio)` | RGB image, binary mask | (cropped_img, cropped_mask, M_rot_3x3, crop_offset) | minAreaRect 기반 자동 회전보정 + 크롭 |
| `resize_to_max(img, max_side)` | image, max_side | (resized_img, scale_factor) | 장변 기준 리사이즈 |

**의존성**: numpy, cv2

### 3.2 matching.py

**책임**: LoFTR 매칭 및 대응점 후처리

| 함수 | 입력 | 출력 | 설명 |
|---|---|---|---|
| `loftr_match(img1_gray, img2_gray, conf_threshold)` | uint8 gray 2장 | (kpts0, kpts1, conf) | LoFTR dense 매칭 |
| `filter_by_mask(kpts0, kpts1, conf, mask0, mask1, sigma, threshold)` | 대응점 + 마스크 | (filtered kpts0, kpts1, conf) | soft mask 기반 필터링 |

**의존성**: numpy, cv2, torch, kornia

**내부 상태**: `_loftr_model` (모듈 레벨 싱글턴)

### 3.3 transform.py

**책임**: 변환 행렬 합성 및 품질 판정

| 함수 | 입력 | 출력 | 설명 |
|---|---|---|---|
| `compose_full_matrix(M_loftr, M_rot_f, crop_off_f, scale_f, M_rot_m, crop_off_m, scale_m)` | 각 단계 행렬/오프셋 | M_full (3×3) | 크롭 좌표계 → 원본 좌표계 역산 |
| `quality_gate_similarity(kpts_f, kpts_m, M_2x3, inliers, tooth_mask_area)` | 대응점 + 추정 행렬 | (status, metrics) | 3단계 품질 판정 (pass/warn/fail) |
| `quality_gate_affine(kpts_f, kpts_m, M_2x3, inliers, tooth_mask_area)` | 대응점 + 추정 행렬 | (status, metrics) | Affine용 품질 판정 |

**의존성**: numpy, cv2

### 3.4 register.py

**책임**: 파이프라인 오케스트레이션

| 함수 | 입력 | 출력 | 설명 |
|---|---|---|---|
| `register_pair(fixed_img, moving_img, fixed_mask, moving_mask, max_side, allow_legacy_fallback)` | RGB images + masks | dict: {registered_img, M_full, metrics, path, debug_images} | 전체 파이프라인 실행 |
| `false_color(img1, img2)` | 두 이미지 | false color overlay | 정합 결과 시각화 |

**의존성**: preprocess, matching, transform, (optional) core_crop_250902

### 3.5 test_synthetic.py

**책임**: 좌표계 합성 검증 (Step 0)

| 테스트 | 검증 내용 | 통과 기준 |
|---|---|---|
| `test_identity` | 동일 이미지 → M ≈ I | translation < 1px, rotation < 0.5°, scale 오차 < 0.01 |
| `test_known_transform` | 7° 회전, 1.03x 스케일, (+20,+15) 이동 | 각 파라미터 상대오차 < 0.5% |
| `test_dual_rotation_chain` | 양쪽 다른 각도 회전+크롭 → GT 대응점 → 역산 | SSIM > 0.95 |

### 3.6 test_offline.py

**책임**: pair1~4 실제 이미지 검증 (Step 3)

**출력**: `results/` 디렉토리에 registered 이미지, false_color, debug 이미지, `metrics.csv`

---

## 4. 인터페이스 정의

### 4.1 register_pair() 출력 dict

```python
{
    'registered_img': np.ndarray | None,    # 정합된 moving (원본 해상도, RGB uint8)
    'M_full': np.ndarray | None,            # 3×3 변환 행렬 (moving→fixed)
    'metrics': {
        'gate': str,              # 'similarity' | 'affine' | 'legacy' | 'none'
        'n_total': int,           # 전체 대응점 수
        'n_inlier': int,          # RANSAC inlier 수
        'inlier_ratio': float,    # inlier 비율
        'reproj_median': float,   # reprojection error 중앙값 (px)
        'reproj_p90': float,      # reprojection error 90th percentile (px)
        'hull_area': float,       # inlier convex hull 면적 (px²)
        'hull_coverage': float,   # hull / tooth_mask_area 비율
        'det': float,             # 행렬 determinant
        'scale': float,           # 추정 스케일
        'rotation_deg': float,    # 추정 회전각 (도)
    },
    'path': str,                  # 'similarity' | 'affine' | 'legacy' | 'failed'
    'debug_images': {
        'fixed_crop': np.ndarray,
        'moving_crop': np.ndarray,
        'fixed_clahe': np.ndarray,
        'moving_clahe': np.ndarray,
        'n_raw_matches': int,
        'n_filtered_matches': int,
        'false_color': np.ndarray,
    },
}
```

### 4.2 auto_orient_and_crop() 좌표계 변환 체인

```
원본 좌표 p_orig
    │
    ▼ M_rot_3x3 (rotate_with_matrix)
회전 좌표 p_rot = M_rot @ p_orig
    │
    ▼ crop_offset (x1, y1)
크롭 좌표 p_crop = p_rot - [x1, y1, 0]
    │
    ▼ scale_factor (resize_to_max)
리사이즈 좌표 p_resized = p_crop * scale_factor
```

역산: `p_orig = M_rot⁻¹ @ (p_crop + [x1, y1, 0])`

### 4.3 모듈 간 의존성

```
main_gui.py ──► register.py ──► preprocess.py
                     │
                     ├────────► matching.py
                     │
                     ├────────► transform.py
                     │
                     └────────► core_crop_250902.py (legacy fallback only)
```

순환 의존성 없음. 단방향 의존 그래프.

---

## 5. 비기능 요구사항

### 5.1 성능

| 항목 | 목표 |
|---|---|
| pair당 처리 시간 | < 10초 (GPU 기준) |
| GPU VRAM | LoFTR ~1GB + SAM2 ~2GB. 동시 로딩 시 4GB+ |
| 모델 로딩 | 모듈 레벨 1회. 매 호출 재로딩 금지 |

### 5.2 품질

| 단계 | 기준 |
|---|---|
| Step 0 | synthetic test 3개 모두 통과 |
| Step 3 | pair1~4 중 최소 3개 similarity pass, 나머지 warn 이하 |
| Step 3 | pair3 (극단 배경 차이) 정합 성공 |

### 5.3 호환성

- OS: Windows 11 (네이티브). SAM2는 WSL 리스크 존재
- Python: 3.10+
- GPU: CUDA 지원 (CPU fallback 가능하나 속도 저하)

---

## 6. Fallback 전략

```
┌──────────────────────────────┐
│ 1차: Similarity (4-DoF)      │
│   estimateAffinePartial2D    │
│   quality_gate_similarity    │
│   pass/warn → 채택           │
├──────────────────────────────┤
│ 2차: Affine (6-DoF)          │  ← Similarity fail 시
│   estimateAffine2D           │
│   quality_gate_affine        │
│   pass/warn → 채택           │
├──────────────────────────────┤
│ 3차: Legacy OF 루프           │  ← Affine fail + allow_legacy=True
│   CLAHE + SAM2 mask 적용 버전│
│   기존 reg_body_single 래핑  │
├──────────────────────────────┤
│ 실패: 모두 fail               │  ← path='failed', registered_img=None
└──────────────────────────────┘
```

---

## 7. 개발 단계

| Step | 내용 | 선행 조건 | 산출물 |
|---|---|---|---|
| 0 | 좌표계 synthetic 검증 | - | test_synthetic.py (3 tests pass) |
| 1 | preprocess.py 구현 | - | preprocess.py + 단위 테스트 |
| 2 | matching.py + transform.py + register.py | Step 0, 1 | 3개 모듈 + register_pair() |
| 3 | pair1~4 오프라인 검증 | Step 2 | test_offline.py, results/, metrics.csv |
| 4 | SAM2 UI 연결 | Step 3 | SAM2 → register_pair() 통합 |
| 5 | GUI 통합 + legacy 옵션 | Step 4 | main_gui.py |

**불변 규칙**: Step 0 synthetic test 미통과 시 이후 단계 진행 금지.
