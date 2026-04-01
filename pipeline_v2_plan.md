# 임상사진 자동 정합 파이프라인 v2 — 구현 계획서

**작성일**: 2026-04-01  
**목적**: Claude Code에게 넘겨 단계별 구현 진행용  
**기존 코드**: `core_crop_250902.py` (2175줄, tkinter GUI + 정합 로직 결합)

---

## 1. 프로젝트 배경

구강 수술 임상사진(수술 전/중/후)을 자동으로 정합하는 파이프라인.  
기존 파이프라인은 수동 조작이 많고(선분 각도 보정, ROI 사각형 드래그), Optical Flow 반복 루프가 느리며, 배경이 극단적으로 다른 쌍(pair3: 술전 vs 판막거상+임플란트 식립 직후)에서 실패함.

### 1.1 임상사진 도메인 특성 (설계 제약)

- 치은/점막: texture 거의 없음 → SIFT/SuperPoint 등 keypoint detector 사용 불가
- 법랑질/침/미러: specular 반사 심함
- 조직 부종/퇴축: 부분 비강체 변형 존재
- 장당 노출 치아: **최대 5개** (수술사진 기준)
- 치아 자체는 rigid body → similarity 변환(4-DoF: 회전+균일스케일+이동)으로 충분
- 촬영 각도 편차 있음 (동일 프로토콜 아님)

### 1.2 기존 파이프라인 (core_crop_250902.py) 구조

```
GUI 파일 선택 → Preview (flip/rotate) → Manual Alignment (선분 2개 그려서 각도 보정)
→ ROI Selection (수동 사각형 2개 드래그) → reg_body_single():
    equalize_adapthist + match_histograms
    → 멀티채널 합성 (chan_vese + u2net/rembg + depth_map/DPT-Large)
    → template_fab (크기 맞춤) → radial_mask
    → [Optical Flow (tvl1) → overlay → Rigid/Similarity 정합 (SimpleITK)] × 최대 50회 반복
    → cosine similarity로 수렴 감지
    → 최종 matrix → scaling_factor 역산 → 원본 해상도 warpAffine
```

**문제점**:
- 수동 조작 5단계 (파일선택, flip/rotate, 선분정렬, ROI드래그, 대기)
- OF 반복 루프: 30초~수 분
- pair3 (배경 극단 차이)에서 실패
- 전역 변수 30개+, import 중복, GUI/로직 결합, get_depth_map()이 매 호출마다 모델 재로딩
- save_and_close_roi() 중복 정의

### 1.3 새 인사이트 (실험으로 검증됨)

| 발견 | 상세 |
|---|---|
| **SAM2 + minAreaRect** | 클릭으로 마스크 생성 → minAreaRect로 자동 회전/크롭 → 수동 선분정렬 + ROI드래그 완전 대체 |
| **CLAHE** | pair3에서 ORIGINAL 13→481 매칭 (37배). equalize_adapthist 대체 |
| **LoFTR** | detector-free matcher. low-texture 영역에서 dense 대응점 추출 가능. kornia 라이브러리 |
| **CROP G+MASK 1368개** | SAM2 마스크 기반 크롭 + CLAHE 조합이 최고 성능 |
| **마스크 입력 전략** | 일반 케이스: 마스크는 출력 후처리에만. 배경 극단 차이 쌍: 입력에도 마스크 유효 |

### 1.4 핵심 설계 변경

기존의 "OF로 뭉개서 끌어온 후 rigid 정합 반복" 대신,  
**LoFTR 대응점 → cv2.estimateAffinePartial2D로 1회에 similarity matrix 직접 추정**.  
치아는 rigid body이므로 이론적으로 맞고, 속도가 수십 배 빨라짐.

---

## 2. 새 파이프라인 아키텍처

```
[사용자] 고정상/이동상 파일 선택 + SAM2 클릭으로 치아 마스크 생성
    ↓
[Phase A: 자동 전처리]
  A1. minAreaRect → 회전 각도 + 크롭 영역 자동 결정 (양쪽 각각)
  A2. rotate_with_matrix() → 잘림 없는 회전 + 3×3 행렬 반환
  A3. 크롭 (패딩 포함)
  A4. CLAHE 적용
  A5. grayscale 변환
  A6. 장변 640px 리사이즈 (LoFTR 입력 규격)
    ↓
[Phase B: LoFTR 매칭]
  B1. kornia LoFTR (indoor_new 우선, grayscale 1×1×H×W)
  B2. confidence threshold 필터링
  B3. SAM2 soft mask 기반 치아 영역 대응점만 필터링
    ↓
[Phase C: 변환 행렬 추정 + 품질 판정]
  C1. cv2.estimateAffinePartial2D (similarity, RANSAC)
  C2. quality_gate_similarity (pass/warn/fail)
  C3. 실패 시 → estimateAffine2D + quality_gate_affine
  C4. 그래도 실패 시 → legacy OF 루프 (CLAHE+SAM2 mask 적용 버전)
    ↓
[Phase D: 행렬 역산 + 원본 적용]
  D1. compose_full_matrix(): 크롭 좌표계 → 원본 좌표계 변환
      M_orig = A_f⁻¹ @ M_loftr @ A_m
  D2. cv2.warpAffine(원본 moving, M_orig, 원본 fixed 크기)
    ↓
[출력] registered_img, M_full, metrics, debug_images
```

---

## 3. 파일 구조

```
project/
├── preprocess.py        # CLAHE, rotate_with_matrix, auto_orient_and_crop, resize
├── matching.py          # LoFTR wrapper + 대응점 필터링
├── transform.py         # 행렬 합성, 역산, 품질 게이트
├── register.py          # register_pair() 메인 함수 (Phase A~D 통합)
├── test_synthetic.py    # Step 0: synthetic 변환 검증
├── test_offline.py      # Step 2: pair1~4 오프라인 테스트
├── core_crop_250902.py  # legacy (수정 안 함, fallback 전용)
└── main_gui.py          # Step 5~6에서 작성 (최종 GUI)
```

**원칙**: GUI 의존성 0인 순수 백엔드 먼저 완성. tkinter import 금지 (test_synthetic, test_offline, register.py, 등).

---

## 4. 개발 순서

### Step 0: 좌표계 검증 (synthetic test) — 최우선

**목적**: compose_full_matrix()의 행렬 체인이 수학적으로 정확한지, 실제 이미지/LoFTR 투입 전에 검증.

**파일**: `test_synthetic.py`

```
Test 0a: Identity
  - 같은 이미지 2장, 같은 마스크
  - 전체 파이프라인 통과 → M_full ≈ I (identity)
  - 허용 오차: translation < 1px, rotation < 0.5°, scale 오차 < 0.01

Test 0b: Known Similarity Transform
  - 원본 이미지에 알려진 변환 적용: 7° 회전, 1.03x 스케일, (+20, +15) 이동
  - cv2.warpAffine으로 moving 생성
  - LoFTR 대신 ground truth 대응점 직접 투입 (변환 전후 좌표)
  - register_pair() 실행 → 추정된 M vs GT M 비교
  - 허용 오차: 각 파라미터 < 0.5% 상대 오차

Test 0c: 양쪽 회전/크롭 체인 검증
  - fixed를 α=12° 회전+크롭, moving을 β=-5° 회전+크롭
  - LoFTR 대신 GT 대응점 투입
  - compose_full_matrix() 출력으로 원본 moving → 원본 fixed warp
  - warp 결과 vs 원본 fixed의 SSIM > 0.95

※ 이 3개 테스트가 통과하지 않으면 이후 단계 진행 금지
```

### Step 1: preprocess.py 구현

### Step 2: matching.py + transform.py + register.py 구현

### Step 3: test_offline.py로 pair1~4 검증

### Step 4: SAM2 UI 연결 (Step 3 통과 후에만)

### Step 5: GUI 통합 + legacy 옵션

상세는 아래 섹션 5~10에서 함수 단위로 기술.

---

## 5. preprocess.py 상세

### 5.1 apply_clahe

```python
def apply_clahe(img_gray: np.ndarray, 
                clip_limit: float = 2.0, 
                tile_size: int = 8) -> np.ndarray:
    """
    OpenCV CLAHE 적용.
    
    기존 코드의 equalize_adapthist (skimage, clip_limit=0.1) 대체.
    pair3 실험에서 ORIGINAL 13→481 매칭 (37배 개선) 검증됨.
    
    Args:
        img_gray: uint8 grayscale 이미지
        clip_limit: 대비 제한 (기본 2.0)
        tile_size: 타일 크기 (기본 8)
    
    Returns:
        CLAHE 적용된 uint8 grayscale 이미지
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                             tileGridSize=(tile_size, tile_size))
    return clahe.apply(img_gray)
```

### 5.2 rotate_with_matrix

```python
def rotate_with_matrix(image: np.ndarray, 
                        angle_deg: float, 
                        center: tuple = None) -> tuple:
    """
    이미지를 회전하되, 잘림 없이 캔버스를 확장하고 3×3 변환 행렬을 같이 반환.
    
    기존 코드의 rotate_image_with_padding() 기반이지만,
    변환 행렬을 반환하는 점이 핵심 차이.
    compose_full_matrix()에서 이 행렬이 필수.
    
    Args:
        image: 입력 이미지 (RGB 또는 grayscale)
        angle_deg: 회전 각도 (도, 양수=반시계)
        center: 회전 중심 (None이면 이미지 중심)
    
    Returns:
        rotated_img: 회전된 이미지 (확장된 캔버스)
        M_rot_3x3: 3×3 homogeneous 변환 행렬 (원본좌표 → 회전좌표)
    
    주의:
        - mask 회전 시에는 interpolation=INTER_NEAREST 사용할 것
        - 이 함수 자체는 INTER_LINEAR 사용
        - borderValue=0 (검은색) — mask가 아닌 이미지용
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    
    M_2x3 = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    # OpenCV getRotationMatrix2D: 양수 angle = 반시계 회전
    # 우리는 angle_deg 양수 = 반시계로 통일
    # 따라서 -angle_deg를 넘겨서 시계방향 보정 (OpenCV 관례)
    
    cos_a = abs(M_2x3[0, 0])
    sin_a = abs(M_2x3[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # 중심점 보정 (확장된 캔버스에 맞춤)
    M_2x3[0, 2] += new_w / 2 - center[0]
    M_2x3[1, 2] += new_h / 2 - center[1]
    
    rotated = cv2.warpAffine(image, M_2x3, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    
    M_3x3 = np.eye(3)
    M_3x3[:2, :] = M_2x3
    
    return rotated, M_3x3
```

### 5.3 auto_orient_and_crop

```python
def auto_orient_and_crop(image: np.ndarray, 
                          mask: np.ndarray, 
                          padding_ratio: float = 0.1) -> tuple:
    """
    SAM2 마스크 기반으로 자동 회전 보정 + 크롭.
    
    기존의 manual_alignment_window() (선분 각도 보정) +
    open_roi_selection() (수동 사각형 드래그)을 완전 대체.
    
    처리 흐름:
      1. mask에서 minAreaRect 추출 → 회전 각도 결정
      2. rotate_with_matrix()로 이미지 + 마스크 회전
      3. 회전된 마스크의 bounding box + padding으로 크롭
    
    Args:
        image: 원본 이미지 (RGB)
        mask: binary mask (uint8, 0 or 255)
        padding_ratio: 크롭 영역 대비 패딩 비율 (기본 10%)
    
    Returns:
        cropped_img: 크롭된 이미지
        cropped_mask: 크롭된 마스크
        M_rot_3x3: 회전 변환 행렬 (원본→회전)
        crop_offset: (x, y) 크롭 시작점 (회전된 좌표계에서)
    
    주의사항:
        - minAreaRect angle 정규화: width > height 보장
        - aspect ratio ≈ 1 (정사각형 배열)이면 회전 스킵
        - mask warp: INTER_NEAREST + > 127 threshold로 binary 유지
        - 기존 rotate_image_with_padding()의 잘림 방지 로직 계승
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("마스크에서 contour를 찾을 수 없음")
    
    all_pts = np.vstack(contours)
    rect = cv2.minAreaRect(all_pts)
    center, (rect_w, rect_h), angle = rect
    
    # === angle 정규화 ===
    # minAreaRect: angle은 x축과 첫 번째 변 사이 각도, -90~0 또는 0~90
    # width > height를 보장하도록 정규화
    if rect_w < rect_h:
        angle += 90
        rect_w, rect_h = rect_h, rect_w
    
    # 정사각형 배열 (aspect ratio ~1)이면 회전이 불안정 → 스킵
    aspect_ratio = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-6)
    if aspect_ratio < 1.2:
        angle = 0  # 회전하지 않음
    
    # === 회전 ===
    rotated_img, M_rot = rotate_with_matrix(image, angle)
    
    # mask 회전 (INTER_NEAREST로)
    h, w = mask.shape[:2]
    M_2x3 = M_rot[:2, :]
    new_h, new_w = rotated_img.shape[:2]
    rotated_mask = cv2.warpAffine(mask, M_2x3, (new_w, new_h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
    rotated_mask = (rotated_mask > 127).astype(np.uint8) * 255
    
    # === 크롭 ===
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
```

### 5.4 resize_to_max

```python
def resize_to_max(img: np.ndarray, max_side: int = 640) -> tuple:
    """
    장변 기준으로 리사이즈. LoFTR 입력 해상도 일관성 보장.
    
    기존 find_scaling_factor()는 목표 면적 기반이라 pair마다 입력이 흔들렸음.
    Kornia 튜토리얼 권고: indoor 모델은 640×480 이하.
    
    Args:
        img: 입력 이미지
        max_side: 장변 최대 픽셀 수 (기본 640)
    
    Returns:
        resized_img: 리사이즈된 이미지
        scale_factor: 리사이즈 비율 (< 1.0이면 축소됨)
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
```

---

## 6. matching.py 상세

### 6.1 LoFTR 매칭

```python
import torch
import kornia.feature as KF

# 모듈 레벨에서 1회 로드 (매 호출 재로딩 방지)
_loftr_model = None

def _get_loftr_model(pretrained: str = 'indoor_new') -> KF.LoFTR:
    """
    LoFTR 모델 싱글턴 로딩.
    
    pretrained 옵션:
      - 'indoor_new': ScanNet 학습, 최신 버전 (1순위)
      - 'indoor': ScanNet 학습, 기존 버전 (2순위)
      - 'outdoor': MegaDepth 학습 (비교 실험용)
    
    구강 사진은 촬영 거리 ~30cm, 좁은 시야, 작은 baseline
    → indoor 계열이 적합. outdoor(건물/풍경)는 부적합.
    
    주의: LightGlue (SuperPoint+LightGlue) 검토했으나 부적합.
    SuperPoint 등 keypoint detector는 low-texture 치은에서 실패.
    LoFTR은 detector-free이므로 이 도메인에 적합.
    """
    global _loftr_model
    if _loftr_model is None:
        try:
            _loftr_model = KF.LoFTR(pretrained=pretrained)
        except Exception:
            # indoor_new 미지원 시 indoor로 fallback
            _loftr_model = KF.LoFTR(pretrained='indoor')
        _loftr_model.eval()
        if torch.cuda.is_available():
            _loftr_model = _loftr_model.cuda()
    return _loftr_model


def loftr_match(img1_gray: np.ndarray, 
                img2_gray: np.ndarray, 
                conf_threshold: float = 0.5) -> tuple:
    """
    LoFTR로 두 grayscale 이미지 사이 대응점 추출.
    
    입력 규격:
      - grayscale, uint8
      - 장변 640px 이하로 리사이즈된 상태 (resize_to_max 거친 후)
      - LoFTR 내부적으로 (N, 1, H, W) float tensor로 변환
    
    Args:
        img1_gray: 고정상 grayscale (uint8)
        img2_gray: 이동상 grayscale (uint8)
        conf_threshold: confidence 최소 임계값 (기본 0.5)
    
    Returns:
        kpts0: 고정상 대응점 (N×2, float32)
        kpts1: 이동상 대응점 (N×2, float32)
        confidence: 각 대응점 confidence (N,)
    
    주의:
        - kornia의 mask0/mask1 API는 본래 padding/invalid region용.
          tooth mask를 여기에 넣는 건 실험 옵션이지 기본 경로 아님.
        - 기본 경로: mask 없이 매칭 → 출력 후처리에서 mask 필터링
    """
    model = _get_loftr_model()
    device = next(model.parameters()).device
    
    input1 = torch.from_numpy(img1_gray).float()[None, None] / 255.0
    input2 = torch.from_numpy(img2_gray).float()[None, None] / 255.0
    input1 = input1.to(device)
    input2 = input2.to(device)
    
    with torch.no_grad():
        correspondences = model({"image0": input1, "image1": input2})
    
    kpts0 = correspondences['keypoints0'].cpu().numpy()
    kpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()
    
    # confidence 필터
    valid = conf > conf_threshold
    kpts0 = kpts0[valid]
    kpts1 = kpts1[valid]
    conf = conf[valid]
    
    return kpts0, kpts1, conf
```

### 6.2 마스크 기반 대응점 필터링

```python
def filter_by_mask(kpts0: np.ndarray, 
                   kpts1: np.ndarray, 
                   conf: np.ndarray,
                   mask0: np.ndarray, 
                   mask1: np.ndarray, 
                   sigma: int = 7, 
                   threshold: float = 0.3) -> tuple:
    """
    SAM2 마스크 기반으로 치아 영역 대응점만 필터링.
    
    soft mask (Gaussian blur)를 써서 마스크 경계 근처도 부드럽게 포함.
    hard binary mask를 쓰면 경계에서 유효한 대응점을 잃을 수 있음.
    
    실험 결과:
      - 일반 케이스: 마스크 후처리만으로 충분
      - 배경 극단 차이 쌍 (pair3): 입력에도 마스크 적용이 유효할 수 있음
        → 이 함수와 별개로, loftr_match 전에 이미지에 직접 마스크를 씌우는
          2차 시도 로직은 register.py의 fallback에서 처리
    
    Args:
        kpts0, kpts1: 대응점 쌍 (N×2)
        conf: confidence 배열 (N,)
        mask0, mask1: binary mask (uint8, 0 or 255), 크롭+리사이즈 좌표계
        sigma: Gaussian blur sigma
        threshold: soft mask 최소 임계값 (양쪽 모두 넘어야 통과)
    
    Returns:
        filtered kpts0, kpts1, conf
    """
    soft0 = cv2.GaussianBlur(mask0.astype(np.float32) / 255.0, 
                              (sigma*2+1, sigma*2+1), sigma)
    soft1 = cv2.GaussianBlur(mask1.astype(np.float32) / 255.0, 
                              (sigma*2+1, sigma*2+1), sigma)
    
    h0, w0 = soft0.shape[:2]
    h1, w1 = soft1.shape[:2]
    
    mask_scores = np.zeros(len(kpts0))
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(kpts0, kpts1)):
        iy0, ix0 = int(np.clip(y0, 0, h0-1)), int(np.clip(x0, 0, w0-1))
        iy1, ix1 = int(np.clip(y1, 0, h1-1)), int(np.clip(x1, 0, w1-1))
        mask_scores[i] = min(soft0[iy0, ix0], soft1[iy1, ix1])
    
    valid = mask_scores > threshold
    return kpts0[valid], kpts1[valid], conf[valid]
```

---

## 7. transform.py 상세

### 7.1 행렬 합성

```python
def compose_full_matrix(M_loftr_2x3: np.ndarray,
                         M_rot_f: np.ndarray, 
                         crop_offset_f: tuple, 
                         resize_scale_f: float,
                         M_rot_m: np.ndarray, 
                         crop_offset_m: tuple, 
                         resize_scale_m: float) -> np.ndarray:
    """
    크롭/리사이즈 좌표계에서의 LoFTR 변환 행렬을
    원본 좌표계로 되돌리는 합성 함수.
    
    === 핵심 수학 ===
    
    각 이미지의 변환 체인:
      원본 → rotate(M_rot) → crop(translate) → resize(scale)
      
    이를 A로 표현하면:
      A_f = S_f @ C_f @ M_rot_f  (fixed 쪽 체인)
      A_m = S_m @ C_m @ M_rot_m  (moving 쪽 체인)
    
    LoFTR은 이 좌표계에서 동작하므로:
      p_crop_fixed = M_loftr @ p_crop_moving
    
    원본 좌표계로 되돌리면:
      p_orig_fixed = A_f⁻¹ @ M_loftr @ A_m @ p_orig_moving
    
    따라서:
      M_full = A_f⁻¹ @ M_loftr @ A_m
    
    === 이전 계획의 버그 (수정됨) ===
    초기 계획에서는 fixed 쪽 역변환(A_f⁻¹)을 누락해서
    "크롭에서는 맞는데 원본에 적용하면 살짝 틀어지는" 버그가 발생할 뻔했음.
    
    === padding_offset 관련 ===
    rotate_with_matrix()가 캔버스 확장 + 중심 보정을 M_rot에 포함하므로,
    별도의 padding_offset 파라미터는 불필요. M_rot만으로 충분.
    
    Args:
        M_loftr_2x3: LoFTR 좌표계에서의 변환 (2×3)
        M_rot_f: fixed 회전 행렬 (3×3, rotate_with_matrix 출력)
        crop_offset_f: fixed 크롭 시작점 (x, y)
        resize_scale_f: fixed 리사이즈 비율
        M_rot_m: moving 회전 행렬 (3×3)
        crop_offset_m: moving 크롭 시작점 (x, y)
        resize_scale_m: moving 리사이즈 비율
    
    Returns:
        M_full: 원본 좌표계 변환 행렬 (3×3)
                moving 원본 → fixed 원본
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
    
    # Moving 체인: 동일
    C_m = translation_matrix(-crop_offset_m[0], -crop_offset_m[1])
    S_m = scale_matrix(resize_scale_m)
    A_m = S_m @ C_m @ M_rot_m
    
    # 합성: M_full = A_f⁻¹ @ M_loftr @ A_m
    M_full = np.linalg.inv(A_f) @ to_3x3(M_loftr_2x3) @ A_m
    
    return M_full
```

### 7.2 품질 게이트 (similarity용)

```python
def quality_gate_similarity(kpts_f: np.ndarray, 
                             kpts_m: np.ndarray, 
                             M_2x3: np.ndarray, 
                             inliers: np.ndarray, 
                             tooth_mask_area: float) -> tuple:
    """
    Similarity 변환 추정 결과의 품질 판정.
    
    === 판정 구조: 3단계 (fail / warn / pass) ===
    
    Hard fail 조건 (즉시 다음 fallback으로):
      - inlier 수 < 12
      - det(M) ≤ 0 (reflection 발생)
      - reproj error median ≥ 5px
      - reproj error 90th percentile ≥ 12px
      - scale < 0.7 or > 1.4
      - |rotation| > 20°
    
    Warn 조건 (결과는 쓰되 경고 기록):
      - inlier 수 < 30
      - reproj error median ≥ 3px
      - hull coverage < 0.2
      - scale 범위 0.8~1.2 벗어남
      - |rotation| > 15°
    
    Pass: 위 조건 모두 해당 없음
    
    === 설계 결정 근거 ===
    
    1. n_inlier 12 vs 30:
       치아 1~2개만 노출된 쌍에서 12~20개 좋은 inlier로도 정합 가능.
       30을 hard fail로 두면 실제 좋은 정합도 fallback으로 빠짐.
       → 12 = hard minimum, 30 = preferred
    
    2. reproj error: mean 대신 median + 90th percentile
       RANSAC 후에도 marginal inlier가 mean을 왜곡할 수 있음.
    
    3. hull coverage 분모: crop_area가 아닌 tooth_mask_area
       minAreaRect + padding 크롭이면 배경이 많이 포함됨.
       배경에는 대응점이 없으므로 crop_area 기준이면
       실제로 잘 된 정합도 coverage가 낮게 나옴.
       필터 기준(soft mask)과 평가 기준(coverage 분모)을 동일 영역으로 맞춤.
       tooth_mask_area = np.sum(eroded_mask > 0)
       erosion: 3~5px kernel로 경계 노이즈 제거
    
    4. scale/rotation 해석:
       similarity matrix [s·cosθ, -s·sinθ; s·sinθ, s·cosθ] 형태이므로
       det → scale = sqrt(det), atan2 → rotation 분해가 정확함.
       (affine에서는 이 분해가 의미 없어짐 → 별도 게이트)
    
    Args:
        kpts_f: fixed 대응점 (N×2)
        kpts_m: moving 대응점 (N×2)  
        M_2x3: estimateAffinePartial2D 출력 (2×3)
        inliers: RANSAC inlier 마스크 (N×1, uint8)
        tooth_mask_area: eroded tooth mask의 nonzero 픽셀 수
    
    Returns:
        status: 'pass', 'warn', 'fail' 중 하나
        metrics: dict (모든 수치 기록)
    """
    n_total = len(kpts_f)
    inlier_mask = inliers.flatten().astype(bool)
    n_inlier = int(np.sum(inlier_mask))
    
    # --- reprojection error ---
    kpts_m_h = np.hstack([kpts_m, np.ones((n_total, 1))])
    M_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    projected = (M_3x3 @ kpts_m_h.T).T[:, :2]
    errors = np.linalg.norm(projected - kpts_f, axis=1)
    inlier_errors = errors[inlier_mask]
    
    reproj_median = float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 999.0
    reproj_p90 = float(np.percentile(inlier_errors, 90)) if len(inlier_errors) > 0 else 999.0
    
    # --- hull coverage ---
    inlier_pts = kpts_f[inlier_mask].astype(np.float32)
    hull_area = 0.0
    if len(inlier_pts) >= 3:
        hull = cv2.convexHull(inlier_pts)
        hull_area = float(cv2.contourArea(hull))
    coverage = hull_area / tooth_mask_area if tooth_mask_area > 0 else 0.0
    
    # --- 변환 파라미터 분해 ---
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
    
    # --- 판정 ---
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
```

### 7.3 품질 게이트 (affine용)

```python
def quality_gate_affine(kpts_f: np.ndarray, 
                         kpts_m: np.ndarray, 
                         M_2x3: np.ndarray, 
                         inliers: np.ndarray, 
                         tooth_mask_area: float) -> tuple:
    """
    Affine 변환 추정 결과의 품질 판정.
    
    Similarity 게이트와 차이점:
      - scale/rotation 상식 체크 없음
      - affine은 shear가 포함되므로 det→scale, atan2→rotation 분해가
        의미를 잃음. SVD 분해는 가능하지만 해석이 복잡.
      - det sign + reproj + inlier ratio + hull coverage만 체크
    
    OpenCV도 estimateAffinePartial2D (제한 affine)와
    estimateAffine2D (일반 affine)를 분리 제공하며,
    각각 다른 해석이 필요.
    
    Args/Returns: similarity 게이트와 동일 구조
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
```

---

## 8. register.py 상세

### 8.1 register_pair (메인 함수)

```python
def register_pair(fixed_img: np.ndarray, 
                  moving_img: np.ndarray, 
                  fixed_mask: np.ndarray, 
                  moving_mask: np.ndarray,
                  max_side: int = 640,
                  allow_legacy_fallback: bool = True) -> dict:
    """
    전체 정합 파이프라인 메인 함수.
    
    === 입력 ===
    - fixed_img, moving_img: RGB uint8 원본 해상도
    - fixed_mask, moving_mask: binary mask (uint8, 0 or 255)
      ※ 개발 단계에서는 미리 저장된 PNG. 이후 SAM2 UI에서 생성.
    
    === 출력 (dict) ===
    - 'registered_img': 정합된 moving 이미지 (원본 해상도)
    - 'M_full': 최종 변환 행렬 (3×3)
    - 'metrics': 품질 지표 dict
    - 'path': 'similarity' / 'affine' / 'legacy'
    - 'debug_images': dict of intermediate images (시각화용)
    
    === Fallback 체인 ===
    1. estimateAffinePartial2D (similarity, 4-DoF)
       → quality_gate_similarity → pass/warn이면 채택
    2. 실패 시: estimateAffine2D (affine, 6-DoF)
       → quality_gate_affine → pass/warn이면 채택
    3. 그래도 실패 시: legacy OF 루프 (CLAHE+mask 적용 버전)
       → allow_legacy_fallback=True일 때만
    
    === 설계 결정 ===
    - GUI 의존성 0. tkinter import 없음.
    - 모든 중간 결과를 debug_images에 저장 (시각화는 호출자 책임)
    - legacy path는 삭제하지 않고 checkbox 뒤에 숨김
    """
    debug = {}
    
    # === Phase A: 자동 전처리 ===
    
    # A1~A3: 양쪽 각각 회전 + 크롭
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
    # erosion으로 경계 노이즈 제거 후 면적 계산
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
    
    # --- C1: Similarity (1차) ---
    if len(kpts0) >= 12:
        M_sim, inliers_sim = cv2.estimateAffinePartial2D(
            kpts1, kpts0,  # moving → fixed 방향
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
    
    # --- C2: Affine (2차 fallback) ---
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
    
    # --- C3: Legacy OF 루프 (최후방) ---
    if result_path is None and allow_legacy_fallback:
        print("[INFO] LoFTR 실패 → Legacy OF 루프로 전환")
        # TODO: 기존 reg_body_single의 OF 루프를 
        #       CLAHE + SAM2 mask 적용 버전으로 호출
        #       이 부분은 기존 코드에서 추출하여 래핑
        result_path = 'legacy'
        final_metrics = {'gate': 'legacy', 'reason': 'loftr_fallthrough'}
        # legacy 결과는 별도 처리...
        # (기존 코드의 reg_body_single 호출)
    
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
    
    # legacy path의 경우
    # ... (기존 코드 래핑)


def false_color(img1, img2):
    """기존 코드의 false_color() 그대로 재활용"""
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
```

---

## 9. test_synthetic.py 상세

```python
"""
Step 0: 좌표계 합성 검증.
이 테스트가 통과하지 않으면 이후 단계 진행 금지.

사용법:
  python test_synthetic.py --image test_image.jpg --mask test_mask.png

테스트 3개:
  0a. Identity: 같은 이미지 → M ≈ I
  0b. Known similarity: 7° 회전, 1.03x 스케일, (+20,+15) 이동
  0c. 양쪽 다른 각도로 회전/크롭 → GT 대응점으로 행렬 체인 검증
"""

def test_identity(image, mask):
    """동일 이미지 쌍 → identity 행렬 반환 확인"""
    result = register_pair(image, image.copy(), mask, mask.copy())
    M = result['M_full']
    
    # M ≈ I 확인
    I = np.eye(3)
    error = np.abs(M - I)
    
    assert error[0, 2] < 1.0, f"translation x 오차: {error[0,2]:.3f}"
    assert error[1, 2] < 1.0, f"translation y 오차: {error[1,2]:.3f}"
    assert abs(np.degrees(np.arctan2(M[1,0], M[0,0]))) < 0.5, "rotation 오차"
    
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    assert abs(np.sqrt(abs(det)) - 1.0) < 0.01, f"scale 오차: {np.sqrt(abs(det)):.4f}"
    
    print("[PASS] Test 0a: Identity")


def test_known_transform(image, mask):
    """알려진 변환 적용 후 역추정 확인"""
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
    
    # moving 생성
    moving = cv2.warpAffine(image, M_gt[:2,:], (image.shape[1], image.shape[0]))
    moving_mask = cv2.warpAffine(mask, M_gt[:2,:], (mask.shape[1], mask.shape[0]),
                                  flags=cv2.INTER_NEAREST)
    
    # 정합
    result = register_pair(image, moving, mask, moving_mask)
    M_est = result['M_full']
    
    # M_est ≈ M_gt⁻¹ (moving→fixed 방향)
    # 또는: M_est @ [moving point] ≈ [fixed point]
    # 검증: GT 변환의 역행렬과 비교
    M_gt_inv = np.linalg.inv(M_gt)
    
    # 파라미터 비교
    est_scale = np.sqrt(abs(M_est[0,0]*M_est[1,1] - M_est[0,1]*M_est[1,0]))
    est_angle = np.degrees(np.arctan2(M_est[1,0], M_est[0,0]))
    gt_inv_scale = np.sqrt(abs(M_gt_inv[0,0]*M_gt_inv[1,1] - M_gt_inv[0,1]*M_gt_inv[1,0]))
    gt_inv_angle = np.degrees(np.arctan2(M_gt_inv[1,0], M_gt_inv[0,0]))
    
    print(f"  Scale: GT={gt_inv_scale:.4f}, Est={est_scale:.4f}")
    print(f"  Angle: GT={gt_inv_angle:.2f}°, Est={est_angle:.2f}°")
    print(f"  Tx:    GT={M_gt_inv[0,2]:.2f}, Est={M_est[0,2]:.2f}")
    print(f"  Ty:    GT={M_gt_inv[1,2]:.2f}, Est={M_est[1,2]:.2f}")
    
    assert abs(est_scale - gt_inv_scale) / gt_inv_scale < 0.005, "scale 상대오차 > 0.5%"
    assert abs(est_angle - gt_inv_angle) < 0.5, "angle 오차 > 0.5°"
    
    print("[PASS] Test 0b: Known Transform")


def test_dual_rotation_chain(image, mask):
    """양쪽을 서로 다른 각도로 회전+크롭 후, GT 대응점으로 행렬 체인 검증"""
    # fixed: 12° 회전, moving: -5° 회전
    # 이후 각각 auto_orient_and_crop 통과
    # LoFTR 대신 GT 대응점 직접 투입
    # compose_full_matrix 출력 검증
    
    # ... (구현: GT 대응점 생성 → estimateAffinePartial2D → compose → warp → SSIM)
    
    # warp 결과 vs 원본 fixed의 SSIM > 0.95
    
    print("[PASS] Test 0c: Dual Rotation Chain")
```

---

## 10. test_offline.py 상세

```python
"""
Step 3: pair1~4 오프라인 테스트.

사용법:
  python test_offline.py --data_dir C:\Users\User\Desktop\TEST\

입력 구조:
  TEST/
  ├── pair1/
  │   ├── fixed.jpg
  │   ├── moving.jpg
  │   ├── fixed_mask.png
  │   └── moving_mask.png
  ├── pair2/ ...
  ├── pair3/ ...
  └── pair4/ ...

출력:
  TEST/results/
  ├── pair1_registered.jpg
  ├── pair1_false_color.jpg
  ├── pair1_debug/           # 중간 과정 이미지
  ├── ...
  └── metrics.csv            # 전체 pair 품질 지표
"""

# metrics.csv 컬럼:
# pair, path, n_raw, n_filtered, n_inlier, inlier_ratio,
# reproj_median, reproj_p90, hull_coverage, scale, rotation_deg,
# elapsed_sec, status

# 각 pair마다:
# 1. 저장된 마스크 PNG 로드
# 2. register_pair() 호출
# 3. 결과 저장 + metrics 기록
# 4. false_color 오버레이 저장
```

---

## 11. 기존 코드에서 재활용 / 제거 목록

### 재활용 (수정 없이 또는 경미한 수정)

| 함수 | 위치 | 용도 |
|---|---|---|
| `uint8_img()` | preprocess.py로 이동 | 공통 유틸리티 |
| `false_color()` | register.py로 이동 | 시각화 |
| `overlay_img()` | register.py로 이동 | 시각화 |
| `register_img_s()` | legacy fallback 내부 | OF 루프 내 similarity 정합 |
| `register_img_a()` | legacy fallback 내부 | OF 루프 내 affine 정합 |
| `optical_flow()` | legacy fallback 내부 | tvl1 OF |
| `template_fab()` | legacy fallback 내부 | 크기 맞춤 |
| `radial_mask_app()` | legacy fallback 내부 | 가장자리 감쇠 |
| `rotate_image_with_padding()` 로직 | preprocess.py rotate_with_matrix()로 발전 | 잘림 방지 회전 |

### 제거 (새 파이프라인에서 불필요)

| 함수/모듈 | 이유 |
|---|---|
| `manual_alignment_window()` + 선분 관련 함수 전체 | SAM2 + minAreaRect로 대체 |
| `open_roi_selection()` + 사각형 드래그 관련 | SAM2 마스크 기반 자동 크롭으로 대체 |
| `seg_img()` (chan_vese) | SAM2 마스크로 대체 |
| `seg_img_u2net()` (rembg) | SAM2 마스크로 대체 |
| `get_depth_map()` (DPT-Large) | 주 경로에서 불필요, legacy에서도 SAM2 mask가 대체 |
| `equalize_two_images()` | CLAHE로 대체 |
| `find_scaling_factor()` | resize_to_max(640)로 대체 |

### legacy fallback에만 유지 (삭제 안 함)

```
reg_body_single()의 OF 반복 루프 부분:
  optical_flow() → overlay → register_img_s/a() → matrix 누적
  → cosine similarity 수렴 감지

수정점:
  - equalize_adapthist → CLAHE
  - chan_vese + u2net + depth_map → SAM2 마스크
  - 나머지 (radial_mask, template_fab, 반복 구조) 유지
```

---

## 12. 의존성

```
# 필수
numpy
opencv-python (cv2)
torch
kornia              # LoFTR
scikit-image        # legacy fallback에서 사용 (optical_flow 등)
SimpleITK           # legacy fallback에서 사용 (register_img_s/a)

# SAM2 (Step 4 이후)
# segment-anything-2  
# ※ 공식 SAM2 저장소는 Windows에서 WSL 사용을 강하게 권장
# ※ Windows 네이티브 호환성 확인 필요 (환경 리스크)

# 제거 가능 (새 파이프라인에서 미사용)
# rembg             # seg_img_u2net용이었음
# transformers      # DPT-Large depth map용이었음
```

---

## 13. 환경 주의사항

- **Windows 경로**: 기존 코드의 파일 경로가 `C:\Users\User\Desktop\TEST\` 등 Windows 형식
- **SAM2 + Windows**: 공식 저장소가 WSL 권장. Step 4 전에 환경 리스크 확인 필요
- **GPU 메모리**: LoFTR ~1GB, SAM2 ~2GB. 동시 로딩 시 4GB+ VRAM 필요
- **모델 싱글턴 로딩**: `_get_loftr_model()`처럼 모듈 레벨 1회 로드. 기존 `get_depth_map()`의 매 호출 재로딩 실수 반복 금지

---

## 14. 핵심 경고 (구현 시 주의)

1. **compose_full_matrix()에서 fixed 쪽 역변환 누락 금지**  
   `M_full = A_f⁻¹ @ M_loftr @ A_m` — A_f 역변환이 빠지면 "크롭에서는 맞는데 원본에서 틀어지는" 디버깅 난이도 극상 버그 발생

2. **minAreaRect angle 불안정**  
   정사각형 배열(aspect ratio ~1)이면 angle이 흔들림. 회전 스킵 조건 필수

3. **LoFTR pretrained weight**  
   `indoor_new` 또는 `indoor` 사용. `outdoor`는 비교 실험용. 구강 사진은 실내 소규모 baseline

4. **mask0/mask1 API**  
   kornia LoFTR의 mask0/mask1은 원래 padding/invalid region용. tooth mask에 바로 넣는 건 실험 옵션이지 기본 경로 아님

5. **hull coverage 분모**  
   crop_area가 아닌 tooth_mask_area (eroded). 필터 기준과 평가 기준의 영역 정의 일치시킬 것

6. **affine 게이트에 similarity 해석식 사용 금지**  
   shear가 있으면 det→scale, atan2→rotation 분해 무의미

7. **synthetic test 먼저**  
   pair 실험 전에 identity + known transform + dual chain 통과 필수. 좌표계 버그는 실제 pair로는 원인 구분 불가

---

## 15. 성공 기준

| 단계 | 기준 |
|---|---|
| Step 0 | synthetic test 3개 모두 통과 |
| Step 1~2 | preprocess/matching/transform/register 단위 테스트 통과 |
| Step 3 | pair1~4 중 최소 3개 similarity 경로 pass, 나머지 warn 이하 |
| Step 3 | pair3 (극단 배경 차이) 포함하여 정합 성공 |
| Step 3 | 전체 처리 시간: pair당 10초 이내 (GPU 기준) |
| Step 4 | SAM2 클릭 → 마스크 → register_pair() 연결 동작 |
| Step 5 | 기존 GUI에 새 파이프라인 + legacy 옵션 통합 |
