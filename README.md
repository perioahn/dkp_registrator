# DKP Registrator

치과 임상 사진 정합(Registration) 도구. LoFTR 특징 매칭 + SAM2 마스크 기반 파이프라인.

## 다운로드

[Releases](https://github.com/perioahn/dkp_registrator/releases) 페이지에서 Windows / macOS 빌드를 다운로드하세요.

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

### 3. Register (정합 실행)
- **Register** 버튼 클릭
- Moving 이미지별로 16가지 조합(confidence × CLAHE × max_side × mask_sigma)으로 자동 정합
- 탭별 결과 표시 (Moving 이미지당 1개 탭)
- ★ 표시가 최적 결과 (자동 선택)
- 그리드 클릭으로 다른 결과 선택 가능

### 4. 결과 저장
- **Save Selected**: 현재 탭의 선택 결과 저장
- **Save All Selected**: 모든 Moving의 선택 결과를 폴더에 일괄 저장 (2개 이상 정합 시)
- 기본 파일명: `Fixed이름_R_Moving이름.jpg`

## 소스에서 실행

```bash
pip install torch torchvision kornia opencv-python numpy Pillow matplotlib sam2 huggingface_hub
python main_gui.py
```
