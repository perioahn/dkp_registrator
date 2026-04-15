# DKPregistrator

치과 임상 사진 정합(Registration) 도구. LoFTR 특징 매칭 + SAM2 마스크 기반 파이프라인.

## 사용 방법

### 1. 이미지 선택
- **Fixed**: 기준 이미지 (Browse로 선택)
- **Moving1~11**: 정합할 이미지 (Browse로 선택)
  - **+ Moving**: 이동상 슬롯 추가 (최대 11개)
  - **- Moving**: 마지막 슬롯 제거

### 2. SAM2 마스크 선택
- **Select Masks (SAM2)** 버튼 클릭
- Fixed + 모든 Moving 이미지가 그리드로 표시됨
- 각 이미지에서 치아 영역을 클릭하여 마스크 지정
  - **좌클릭**: 포함 영역 선택
  - **우클릭**: 제외 영역 선택
  - **Z**: 현재 개체 확정 → 다음 개체
  - **X**: 현재 이미지 리셋
  - **C**: 전체 완료
  - **Q**: 취소
  - **A**: 앵커 포인트 설정 (Fixed 1점 + 각 Moving 1점씩 = 1세트)
  - **D**: 앵커 초기화

### 3. Register (정합 실행)
- **Register** 버튼 클릭
- 3단계 피라미드 정합 (320→480→640) 자동 실행
- 저해상도에서 coarse 정합 → 고해상도로 전파하여 정밀도 향상

### 4. 결과 확인 및 저장
- 정합 완료 시 결과 창 자동 표시 (정합 이미지 + false color)
- **Save**: 결과 창 다시 열기 (개별 Save / Save All)
- **Matches**: 키포인트 매칭 시각화 (녹색=inlier, 빨간=outlier)
- 기본 파일명: `Fixed이름_R_Moving이름.jpg`

## 요구 환경

- Python 3.10+
- PyTorch, torchvision
- kornia (LoFTR)
- SAM2 (`sam2-hiera-tiny`, HuggingFace에서 자동 다운로드)
- OpenCV, NumPy, Pillow, matplotlib

```bash
pip install torch torchvision kornia opencv-python numpy Pillow matplotlib
```

## 실행

```bash
python main_gui.py
```

또는 [Releases](https://github.com/perioahn/dkp_registrator/releases)에서 빌드된 실행 파일(Windows/macOS)을 다운로드.
