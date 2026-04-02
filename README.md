# DKPregistrator

치과 임상 사진 정합(Registration) 도구. LoFTR 특징 매칭 + SAM2 마스크 기반 파이프라인.

## 사용 방법

### 1. 이미지 선택
- **Fixed**: 기준 이미지 (Browse로 선택)
- **Moving**: 정합할 이미지 (Browse로 선택)

### 2. SAM2 마스크 선택
- **Select Masks (SAM2)** 버튼 클릭
- Fixed/Moving 이미지가 나란히 표시됨
- 각 이미지에서 치아 영역을 클릭하여 마스크 지정
  - **클릭**: 포인트 추가 (좌클릭 = 포함, 우클릭 = 제외)
  - **Z**: 현재 쪽 확정
  - **X**: 현재 쪽 리셋
  - **C**: 양쪽 모두 확정 후 종료
  - **Q**: 취소

### 3. Register (정합 실행)
- **Register** 버튼 클릭
- 8가지 조합(confidence × max_side)으로 자동 정합 수행
- 2×4 그리드로 결과 비교 표시
  - 행: max_side (480, 640)
  - 열: confidence threshold (0.3, 0.2, 0.15, 0.1)
- ★ 표시가 최적 결과 (자동 선택)
- 그리드 클릭으로 다른 결과 선택 가능

### 4. 결과 저장
- 그리드 창의 **Save Selected** 또는 메인 화면의 **Save Result** 클릭
- 기본 파일명: `Fixed이름_R_Moving이름.jpg`

### Match Check (선택사항)
- 정합 전 매칭 품질을 미리 확인하는 기능
- 각 조합별 매칭 포인트 수와 분포를 시각화

## 요구 환경

- Python 3.10+
- PyTorch, torchvision
- kornia (LoFTR)
- SAM2 모델 파일 (`sam2_hiera_large.pt`)
- OpenCV, NumPy, Pillow, matplotlib, scikit-image

```bash
pip install torch torchvision kornia opencv-python numpy Pillow matplotlib scikit-image
```

## 실행

```bash
python main_gui.py
```

또는 빌드된 실행 파일(Windows/macOS)을 사용.
