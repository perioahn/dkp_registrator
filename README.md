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
- **Lazy Mode** 체크박스 (선택): 좌우반전 + 0/90/180/270° 회전 8가지 조합 중
  최적을 자동 선택. 저해상 프리스크리닝으로 8조합을 순위화한 뒤 상위 1~2개만
  풀 정합을 실행하므로 일반 모드 대비 약간만 느림. 진행 바로 현재 시도 표시.

### 4. 결과 확인 및 저장
- 정합 완료 시 결과 창 자동 표시 (정합 이미지 + false color)
- **Save**: 결과 창 다시 열기 (개별 Save / Save All)
- **Matches**: 키포인트 매칭 시각화 (녹색=inlier, 빨간=outlier)
- 기본 파일명: `Fixed이름_R_Moving이름.jpg`

## 웹 UI (신규)

브라우저 기반 UI — tkinter GUI와 동일한 엔진을 사용합니다.

```bash
python webapp/server.py        # http://127.0.0.1:8790 자동 오픈
```

1. 기준(Fixed) 1장 + Moving 여러 장 업로드
2. 각 이미지에서 치아 **좌클릭=포함 / 우클릭=제외**, [개체 확정(Z)]으로 다음 개체
3. ▶ Register (Lazy 토글, normal/strict/relaxed 프로필 선택 가능)
4. 결과: 품질 배지(PASS/WARN/FAIL+사유) + 와이프 슬라이더 / False color /
   플리커 / 나란히 / 매칭점 보기, 휠 줌·드래그 팬, 💾 저장

## 요구 환경

- Python 3.10+
- PyTorch, torchvision
- kornia (LoFTR)
- SAM2 (`sam2-hiera-tiny`, HuggingFace에서 자동 다운로드)
- OpenCV, NumPy, Pillow, matplotlib

```bash
pip install torch torchvision kornia opencv-python numpy Pillow matplotlib
```

## GPU(CUDA) 지원

**CUDA가 없어도 동작합니다.** 실행 시 GPU를 자동 감지하며, 없으면 CPU로 전환됩니다:

- SAM2 마스크: `cuda` 사용 가능 여부를 자동 감지 (`sam2_mask.py`)
- LoFTR 매칭: GPU가 있을 때만 GPU 사용, **GPU 메모리 부족(OOM) 시 CPU로 자동 폴백** 후 다음 작업에서 GPU 복귀 (`matching.py`)

| 환경 | 동작 | 체감 |
|---|---|---|
| NVIDIA GPU + CUDA torch | GPU 가속 | 마스크·매칭 빠름 |
| GPU 없음 / CPU torch | 자동 CPU 모드 | 동일 결과, 수 배 느림 (tiny 모델이라 사용 가능한 수준) |
| GPU 메모리 부족 | 해당 작업만 CPU 폴백 | 중단 없음 |

GPU 가속을 쓰려면 CUDA 버전에 맞는 PyTorch를 설치하세요 (예: CUDA 12.x):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

기본 `pip install torch`(CPU 전용 빌드)로도 기능상 문제는 없습니다.
Releases의 빌드 실행파일은 CPU 빌드 기준이라 어느 PC에서든 돌아갑니다.

## 실행

```bash
python main_gui.py
```

또는 [Releases](https://github.com/perioahn/dkp_registrator/releases)에서 빌드된 실행 파일(Windows/macOS)을 다운로드.
