# DKP Registrator

치과 임상사진을 맞추고 비교하는 로컬 작업대입니다. 정합은 이동·회전·같은 비율 확대·축소만 사용하므로 사진의 형태와 가로세로 비율이 바뀌지 않으며, 원본 사진은 유지됩니다. LoFTR 정합과 선택적인 SAM2 마스크를 사용합니다.

## 1. 받아서 실행하기

[최신 실행 파일](https://github.com/perioahn/dkp_registrator/releases/latest)에서 운영체제에 맞는 파일을 받습니다.

- **Windows x64:** `DKPregistrator-Windows-x64.zip`(약 270MB)을 모두 압축 해제하고 `DKPregistrator.exe`를 실행합니다. `_internal` 폴더도 함께 있어야 합니다.
- 서명되지 않은 앱이라 처음 실행할 때 **Windows의 PC 보호** 창이 뜰 수 있습니다. 이때 **추가 정보 → 실행**을 누릅니다. 뜨지 않는 PC도 있습니다.
- **macOS Apple Silicon:** `DKPregistrator-macOS-arm64.zip`을 풀고 `.app`을 실행합니다. 개발자 인증·공증을 받은 배포판은 아닙니다. macOS가 차단하면 시스템 설정의 개인정보 보호 및 보안에서 해당 앱의 열기를 허용해야 할 수 있습니다.

실행하면 기본 브라우저에 작업대가 열리고, 화면 하단에 **로컬 사진 처리**가 표시됩니다. Python을 별도로 설치할 필요가 없습니다. 정합·마스크 모델은 처음 사용할 때 내려받으므로 첫 사용에는 인터넷이 필요합니다.

## 2. 사진 추가 → 정합 → 저장

1. **사진 추가**에서 JPEG/PNG를 선택하거나 끌어 놓습니다. 처음 불러온 사진이 기준이 됩니다.
2. 다른 사진을 클릭해 기준과 나란히 놓고 **현재 정합**을 누릅니다. 마스크 없이 시작할 수 있습니다. 여러 장은 **선택 N장 정합**, **전체 정합**을 씁니다.
3. 결과에 **정합 완료 / 검토 필요 / 이번 실패** 상태가 표시됩니다.
4. **나란히 / 와이프 / 색상 겹침 / 깜빡임 / 매칭점**으로 눈으로 확인합니다. 매칭점에서 초록은 남은 짝, 빨강은 버린 짝입니다.
5. **정보·저장**에서 결과를 저장합니다. 파일 이름은 `기준이름_R_사진이름.jpg`입니다.

기준을 바꾸려면 사진을 선택하고 **이 사진을 기준으로 지정**을 누른 뒤 다시 정합합니다. 확대는 휠, 이동은 Space+드래그, 화면 맞춤은 0입니다. 그 밖의 단축키는 앱 화면에 안내됩니다.

**앱을 종료하거나 다시 시작하기 전에 필요한 결과를 저장하세요.** 작업 목록과 미저장 결과는 재시작 후 복구되지 않습니다.

## 3. 잘 맞지 않을 때: 마스크로 치관 지정

화면 탭은 **비교 / 마스크 / 미세조정**입니다. 정합 안내에는 **마스크 선택 / 앵커 자동 추천 / 앵커 직접 찍기 / 미세조정**이 있습니다.

1. 마스크 대상(기준 사진 / 현재 사진)을 고릅니다.
2. 치관을 클릭하면 노랗게 미리 보입니다. 클릭을 더하면 넓어지고, 우클릭은 그 부분을 뺍니다.
3. **Z**로 개체를 확정하면 보라색이 됩니다. 다음 치관도 같은 방식으로 반복합니다. 붙은 치관은 한 개체로 묶어도 됩니다.
4. 다른 사진에서도 같은 치관을 지정한 뒤 **현재 정합**을 누릅니다.

**X**는 마스크 초기화이고, Ctrl+Z로 되돌립니다. 회전·반전된 사진은 **정합 설정 → 사진 방향 자동탐색**을 켜고 다시 시도할 수 있습니다. 자동 지표와 시각적 검토를 함께 사용하세요.

<details>
<summary>기준 사진 편집</summary>

[Photo Editor](https://github.com/perioahn/photo-editor)의 로컬 편집 기능을 통합했습니다. 90도/미세 회전, 좌우·상하 반전, 수평선, 밝기·대비, 크롭과 격자, 원본 비교를 제공합니다. 편집을 적용하면 기준 사진에 반영되며 원본은 유지됩니다. 편집한 뒤에는 다시 정합하세요. 재정합에 실패하면 이전 성공 결과와 **이번 실패**를 구분해 표시합니다.

</details>

<details>
<summary>선택 사항: GPU 가속 (Windows, NVIDIA)</summary>

**GPU 없이도 CPU로 사용할 수 있습니다.** Apple Silicon은 지원되는 연산에 Metal(MPS)을 사용합니다.

1. **정합 설정 → GPU 사용 조건 ⓘ**에서 NVIDIA GPU 감지 여부를 확인합니다.
2. **GPU 가속 설치**를 누르면 확인창이 뜹니다(약 2.5GB, 설치 후 앱 다시 시작). 압축을 풀면 약 4.6GB이므로 다운로드 파일과 함께 저장할 디스크 여유가 필요합니다. 예: i7-13700K·RTX 4080 PC에서 설치는 약 1분, 같은 사진 쌍 재정합은 CPU 10.9초 → GPU 1.1초였습니다(1회 측정).
3. 필요한 결과를 저장한 뒤 **GPU 적용하고 다시 시작**을 누릅니다. 재시작 후 정합 설정에서 **NVIDIA GPU 사용**을 확인합니다.

자동 설치는 CUDA 12.4용 PyTorch(`cu124`)를 사용하며 CUDA Toolkit을 따로 설치할 필요는 없습니다. GPU 이름이 감지되어도 실제 연산 호환성이 보장되지는 않습니다. 해당 GPU를 지원하는 NVIDIA 드라이버와 PyTorch가 필요하며, Intel·AMD 그래픽만 있는 Windows PC는 CPU로 사용합니다. 기본 설치 위치는 `%LOCALAPPDATA%\DKPRegistrator\cuda`, 오류 기록은 `%LOCALAPPDATA%\DKPRegistrator\gpu_setup.log`입니다. 근거: [NVIDIA CUDA 하위 호환성](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [CUDA 12.4 릴리스·드라이버 정보](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html).

</details>

<details>
<summary>소스 실행·개발과 검증 (개발자용)</summary>

실행 파일 사용자는 아래 작업이 필요 없습니다. 별도 Python 환경에서 의존성을 설치하고 실행합니다.
```powershell
python -m pip install -r requirements.txt
python launcher.py
```
검증 환경은 Windows Python 3.13이며, 배포 빌드는 Python 3.11을 사용합니다. 소스 실행에서 GPU를 사용하려면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 운영체제·GPU에 맞는 PyTorch를 설치하세요. 앱의 자동 GPU 설치 버튼은 Windows 배포 실행 파일용입니다.
```powershell
python -m pytest tests/test_engine.py tests/test_workspace_api.py tests/test_similarity_only.py -q
cd webapp/frontend
npm ci
npm run verify
npx playwright install chromium
npm run test:e2e
```
브라우저 통합 테스트는 별도 서버와 합성 사진/결정적 추론 경계를 사용합니다. 실제 LoFTR 검증은 Python 엔진 테스트에서 실행합니다. 임상사진 정확도 검증을 대신하지 않습니다. 태그 `v*` 푸시 또는 **Build App** 수동 실행은 두 운영체제에서 프런트엔드를 다시 빌드하고 패키지를 확인한 뒤 ZIP을 생성합니다. 태그 빌드는 Releases에 첨부됩니다. 편집 기능 출처: [Photo Editor 통합 기록](docs/photo-editor-integration.md).

</details>
