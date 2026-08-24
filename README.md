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

## 설치 A — 실행파일 (Python 불필요)

[Releases](../../releases)에서 받기:
- Windows: `DKPregistrator-Windows.zip` 압축 해제 → `DKPregistrator.exe` 실행
- macOS: `DKPregistrator-macOS.zip` 압축 해제 → 앱 실행 (서명 안 된 앱이라 첫 실행은 우클릭 → 열기)

## 설치 B — 소스 실행 (Python 3.10+ 설치된 PC)

cmd(명령 프롬프트)에서 순서대로:

```bat
git clone https://github.com/perioahn/dkp_registrator
cd dkp_registrator
pip install -r requirements.txt
python launcher.py
```

git 없으면 초록 **Code → Download ZIP** 받아 압축 해제 후 그 폴더에서 `pip install ...`부터.
`requirements.txt` 내용물 (개별 설치 시):

```bat
pip install torch torchvision kornia opencv-python-headless numpy Pillow matplotlib scikit-image sam2 huggingface_hub fastapi uvicorn python-multipart
```

- SAM2 모델 가중치(`sam2-hiera-tiny`)는 첫 실행 때 HuggingFace에서 자동 다운로드
- 실행하면 브라우저에 웹 UI가 열림 (`python launcher.py --tk` = 구형 tkinter GUI)

## 정합 모델 — 비율 보존

같은 악궁을 다른 날 찍은 사진의 실제 관계는 **similarity**(회전 + 균일 배율 + 평행이동)입니다.
그래서 `normal`·`strict` 프로필은 similarity만 사용해 **가로세로 비율이 절대 왜곡되지 않습니다.**
매칭이 부족하면 넓은 inlier 집합으로 similarity를 재적합해 구제하고, 그래도 안 되면 FAIL로 알립니다.

`relaxed` 프로필만 최후 수단으로 affine(전단·비등방 배율)을 허용하며, 이 경우 결과 화면에
**⚠ affine (비율 왜곡 가능)** 배지가 표시됩니다.

## GPU 가속

**GPU가 없어도 동작합니다.** GPU가 있으면 큰 폭으로 빨라집니다 (실측, RTX 4080 기준):

| 항목 | 속도 향상 |
|---|---|
| LoFTR 매칭 | **×16** |
| SAM2 마스크 클릭 반응 | **×3.7** |
| 정합 전체 체감 | **약 10배** |

### Windows 실행파일

실행파일에는 CPU용 PyTorch가 들어 있어 **어떤 PC에서도 받자마자 바로 실행**됩니다.
NVIDIA GPU가 있으면 왼쪽 아래 **[⚡ GPU 가속 켜기]** 버튼이 나타나며, 누르면
GPU용 PyTorch를 내려받아 설치합니다.

- 다운로드 약 2.5GB (최초 1회 · 인터넷 필요) — 설치 중에도 앱은 계속 사용 가능
- 설치가 끝나고 **앱을 다시 시작하면** GPU 가속이 적용됩니다
- 설치에 실패해도 앱은 CPU로 그대로 동작합니다 (로그: `%LOCALAPPDATA%\DKPRegistrator\gpu_setup.log`)
- CUDA 툴킷 설치는 불필요 — **NVIDIA 드라이버만 최신이면** 동작 (cu124 기준 525 이상)
- GPU 메모리 부족(OOM) 시 해당 작업만 CPU로 자동 폴백 — 중단 없음
- 되돌리려면 `%LOCALAPPDATA%\DKPRegistrator\cuda` 폴더를 삭제하면 CPU로 복귀

### macOS

Apple Silicon이면 **MPS(Metal) 가속이 자동 적용**됩니다 — 추가 설치 없음.
(일부 미지원 연산은 CPU로 자동 폴백)

### 소스 실행

CUDA 빌드 PyTorch가 설치돼 있으면 자동으로 GPU를 사용합니다:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

기본 `pip install torch`(CPU 전용 빌드)로도 기능상 문제는 없습니다.
다른 CUDA 계열 휠은 [pytorch.org](https://pytorch.org/get-started/locally/) 안내를 따르세요.

## 실행

```bash
python main_gui.py
```

또는 [Releases](https://github.com/perioahn/dkp_registrator/releases)에서 빌드된 실행 파일(Windows/macOS)을 다운로드.
