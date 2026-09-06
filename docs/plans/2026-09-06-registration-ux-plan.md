# DKP Registrator 사용자 편의 개선 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 비율 보존을 필수로 하는 정합기의 UI 전반을 개선하고, 마스크·앵커·Fixed 편집·정합·저장을 예측 가능하게 만든다. Photo Editor에는 폴더 열기와 사진 한 장 열기를 분리한다.

**Architecture:** 기존 Vue 3/FastAPI와 정합 엔진을 유지한다. Photo Editor의 편집 캔버스와 조정값 모델을 통합하고, 서버가 편집 리비전·작업 이력·정합 작업 대상을 관리한다. 원본 사진, 편집된 기준 사진, 해당 기준으로 계산한 결과를 구분한다.

**Tech Stack:** Vue 3, TypeScript, Vite, FastAPI, NumPy/OpenCV, 기존 SAM2/LoFTR 엔진. 회귀 검증에 pytest, 프런트 상태 검증에 Vitest, 브라우저 흐름에 Playwright를 추가한다.

**최신 UI 개정:** [사진 비교·편집 도구 조사 및 작업대 UX 개정안](2026-09-06-photo-tool-ux-addendum.md)을 함께 적용한다. 이 개정안이 아래 초기 배치보다 우선한다. 통합 사진 추가 → 첫 사진 기본 Fixed → 언제든 기준 변경, 인라인 작업대, 확대 연결·원본 해상도 부분 확대, 연속 검토를 포함한다.

---

## 1. 조사 범위와 결정 상태

- 작성일: 2026-09-06.
- 정합기 기준: 로컬 `ed2f8f8fdfd25543aa9aab10e6926ade0e6561bd`.
- Photo Editor 기준: GitHub `perioahn/photo-editor`, `ae420188736791bb9b9683e031aa3d854425e762` (2026-07-06).
- 이 문서는 조사와 구현 계획이다. 앱 소스 변경, 모델 실행, 실사진 정합 검증은 아직 하지 않았다.
- 사용자 확정: **C 실행취소는 제외. Ctrl+Z/Cmd+Z로 실행취소. C는 새 기능을 배정하지 않는다.**
- 사용자 확정: **relaxed 제거. 모든 출력은 비율 보존. 전단·비등방 배율을 허용하는 우회 옵션도 두지 않는다.**
- 사용자 확정: 필요한 기능을 유지·추가하면서 정합기 UI 전반을 재구성한다.
- 사용자 추가 요청: **독립 Photo Editor에도 폴더 열기 / 사진 한 장 열기를 분리한다. 단일 파일 입력은 multiple 없이 한 장만 선택한다.**
- 사용자 추가 요청: **정합기는 Fixed/Moving을 따로 불러오지 않는다. 여러 사진을 한번에 추가하고 첫 사진을 기본 Fixed로 하며 중간 기준 변경도 허용한다.**
- D는 **찍는 중인 앵커 취소 / 선택된 앵커쌍 제거**로 제안한다. 기존 모든 쌍 삭제는 별도 명시적 버튼으로 제공한다.
- 정식 웹 표준(WCAG)과 제품의 공식 UX 관행(Adobe/Slicer)을 구분했다. 치과 정합기의 A/Z/X 같은 특정 키 배치에 대한 단일 업계 표준이 있다는 주장은 하지 않는다.

## 2. 현재 코드에서 확인한 문제

| 확인 위치 | 현재 동작 | 개선 이유 |
|---|---|---|
| `webapp/frontend/src/App.vue:176-187` | 완료 시 전체 목록에서 결과가 있는 첫 Moving을 고름. refresh 완료 전 기존 목록으로 선택 | 선택 정합 결과 대신 관계없는 맨 위 사진을 표시할 수 있음 |
| `App.vue:183` | 성공한 체크 항목을 자동 해제 | 정합 직후 선택 저장을 다시 선택해야 함 |
| `components/MaskEditor.vue:59` | window keydown에서 Z/X만 처리, modifier 검사가 없음 | Ctrl+Z도 Z=마스크 확정으로 처리될 수 있음. C/앵커 키 미지원 |
| `webapp/server.py:400` 부근 | undo는 현재 점 또는 마지막 확정 마스크를 pop | Z 확정 직전 상태와 X 초기화를 온전히 복원하지 못함 |
| `webapp/server.py:75` | Session 설명에는 앵커가 있지만 실제 저장 필드 없음 | 웹 앵커 UI/API와 엔진 연결 필요 |
| `webapp/server.py:434` | 정합 함수에 anchor_points를 전달하지 않음 | 기존 앵커 엔진이 웹에서 사용되지 않음 |
| `register.py:258,447` | 앵커 재적합 후 예전 품질 지표 유지 | 수정된 결과의 오차와 배지가 일치하지 않을 수 있음 |
| `ResultViewer.vue` | 컴포넌트 전환 시 줌·팬 재설정, 조정 모드 이탈 시 미적용 변화 삭제 | 연속 검토 및 실수 복구가 불편함 |
| `webapp/server.py:79` | 서버 세션 생성 시 이전 session 디렉터리 삭제 | 서버를 재시작하면 작업 복원 불가. 후속 개선 후보 |

Photo Editor 역시 그대로 복사할 대상은 아니다. 최신 소스는 원본 비교 키가 `/`인데 README는 `\`로 남아 있다. 밝기 슬라이더는 input 이벤트마다 undo를 적재하므로, 통합 시 **드래그 1회 = 이력 1개**로 수정한다.

## 3. 조사 근거와 적용할 원칙

| 공식 근거 | 확인한 관행 | 이 프로젝트에 적용하는 설계 판단 |
|---|---|---|
| [Lightroom Classic Reference View](https://helpx.adobe.com/lightroom-classic/desktop/process-and-develop-photos/develop-module-tools.html) | Reference와 Active를 분리하고 기준 사진을 고정하여 비교 | Fixed는 항상 고정 표시, 현재 보는 Moving과 일괄 처리 체크를 분리 |
| [Lightroom 조정 실행취소](https://helpx.adobe.com/uk/lightroom-classic/help/develop-module-tools.html#undo_image_adjustments) | 작업 이력과 여러 단계 취소, 개별 값 초기화 | 초기화도 취소 가능, 변경 단위로 이력 기록, 다음 취소 내용 표시 |
| [Photoshop 비파괴 편집](https://helpx.adobe.com/photoshop/using/nondestructive-editing.html) | 원본 데이터를 보존하고 조정값을 분리 | Fixed 원본과 편집 레시피 보존, 편집본으로 실제 정합, 반복 JPEG 저장 방지 |
| [3D Slicer Markups](https://slicer.readthedocs.io/en/latest/user_guide/modules/markups.html) | 명시적 점 배치 모드, 점 이동·삭제, 목록·라벨 | 번호 대응점, 현재 입력 단계 표시, 드래그 수정, 쌍 단위 제거 |
| [3D Slicer Landmark Registration 예제](https://www.slicer.org/wiki/Documentation:Nightly:Registration:RegistrationLibrary:RegLib_C48) | 양쪽에서 대응하는 해부학적 위치를 같은 순서로 지정 | Fixed/Moving 점을 독립 목록 인덱스로 느슨하게 연결하지 않고 안정적인 pair ID 사용. 3D의 최소 점 개수는 2D에 그대로 적용하지 않음 |
| [WCAG 2.2: Character Key Shortcuts](https://www.w3.org/WAI/WCAG22/Understanding/character-key-shortcuts) | 문자 단축키는 비활성화·재매핑·해당 컴포넌트 포커스 조건 중 하나 필요 | 작업 캔버스에 포커스가 있을 때만 A/D/Z/X 작동. 모든 동작에 버튼 제공 |
| [WCAG 2.2: Status Messages](https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html) | 상태 변화를 포커스를 이동시키지 않고 알릴 수 있어야 함 | 진행·완료 메시지를 상태 영역에 표시하고, 사용자가 다른 사진을 보고 있으면 자동 이동하지 않음 |

## 4. 접근 방식 비교

| 안 | 범위와 장단점 | 판단 |
|---|---|---|
| 기존 앱에 편집 구성요소 통합 | Photo Editor의 캔버스·기하·톤 조정 재사용. 단축키·이력·이미지 revision은 정합기에 맞게 통합. 같은 창에서 완료 | **권장** |
| 외부 Photo Editor를 열고 편집본 재업로드 | 구현은 상대적으로 작지만 저장·찾기·재업로드, 마스크/앵커 연결 재설정이 필요 | 사용자가 원하는 편의 개선이 작음 |
| 두 앱을 하나의 대형 프로젝트로 전면 재작성 | 장기 공유는 쉽지만 폴더 편집기 기능까지 범위가 커지고 기존 정합 회귀 위험 증가 | 이번 범위에서 제외 |

Photo Editor의 단일 사진 입력 개선은 해당 저장소의 별도 변경으로 진행한다. 정합기는 해당 커밋의 편집 기능을 가져오고 원본 파일/커밋과 변경 이유를 `docs/photo-editor-integration.md`에 기록한다. GitHub Pages iframe이나 인터넷 서비스 호출 없이 로컬 앱에서 동작하도록 번들링한다.

## 5. 초기 화면 구성과 개정 적용

아래 배치는 초기 초안 기록이다. 최종 구현은 [작업대 UX 개정안의 화면 구성](2026-09-06-photo-tool-ux-addendum.md#3-작업대-배치)을 따른다. 고정 카드/역할별 업로드 중심을 통합 사진 목록·기준 지정 방식으로 바꾸고, 도구와 비교 방식을 분리한다.

```text
┌ 기준 사진 [편집] ──────────── 현재: DSC_0328 · 5/12 ─ [되돌리기 ▾] [도움말] ┐
│ 사진 목록       │ 작업: [마스크] [앵커] [결과]                          │
│ Fixed 고정 카드 │ ┌ Fixed / 기준 ─────┐ ┌ Moving / 현재 사진 ─────┐   │
│                │ │                   │ │                          │   │
│ □ DSC_0324     │ │    사진 캔버스     │ │      사진 캔버스         │   │
│ ☑ DSC_0328 ◀   │ │                   │ │                          │   │
│ ☑ DSC_0330     │ └───────────────────┘ └──────────────────────────┘   │
│ 필터: 전체/확인 │ 도구별 상태: 'Fixed의 2번 대응점을 찍으세요'             │
│ 필요/미정합    │                                                        │
├────────────────┴────────────────────────────────────────────────────────┤
│ 체크 2장 [현재 정합] [선택 2장 정합] [전체 ▾]  [현재 저장] [선택 저장]    │
│ Z 확정 · X 초기화 · Ctrl+Z 취소 · A 앵커 · D 취소    | 작업 상태        │
└─────────────────────────────────────────────────────────────────────────┘
```

- 앵커 탭은 Fixed/Moving 2분할. Fixed 한 장+Moving 전체를 작은 격자로 강제 표시하지 않는다.
- 마스크 탭은 현재 사진을 크게 표시하고 Fixed 참고 패널을 접고 펼칠 수 있게 한다. 앵커와 달리 2분할 강제는 하지 않는다.
- Fixed 편집은 중앙 작업 영역에 인라인으로 열고 오른쪽에 밝기·대비·회전·크롭 도구를 배치한다. 완료 후 원래 선택한 Moving으로 복귀한다.
- 결과 탭은 현재 기능인 와이프·겹쳐보기·깜빡임·나란히·매칭점·미세조정을 유지한다. 원시 수치는 '자세히'에 모은다.
- `PASS/WARN/FAIL`은 `정합 완료/확인 필요/정합 실패`로 표시하고 색과 글자를 함께 쓴다. '정합 완료'는 알고리즘 평가이지 임상적 정확도 보증을 뜻하지 않는다.
- 프로필은 `기본/엄격`만 유지하고 고급 설정으로 접는다. relaxed 선택·재시도 안내·비율 변형 허용 배지를 제거한다.
- 사진을 클릭한 활성 상태와 체크박스의 일괄 처리 대상을 시각적으로 구분한다. 체크는 완료 후 유지한다.
- 기본 버튼 높이 36px 이상, 주요 버튼 40px. WCAG 최소 타깃 크기를 제품 디자인 목표와 혼동하지 않는다. 1366×768과 1920×1080, 125/150% 배율에서 확인한다.
- 좁은 화면에서는 오른쪽 속성 패널을 접고 2분할을 위아래로 전환한다. 기능을 아이콘만으로 숨기지 않는다.
- 비율을 바꾸는 늘이기/자유변형 도구를 두지 않는다. 크롭은 영역만 자르고 사진 속 형태는 늘이거나 눌리지 않는다.
- 사진 추가 드래그앤드롭, 전체 선택/해제, 파일명 검색, 미정합/확인 필요/실패 필터, 이전/다음 사진을 추가한다.
- 정합 대기열은 사진별 대기/처리/완료/실패를 표시한다. `남은 작업 중지`는 현재 사진 처리를 마친 뒤 중지하고 완료 결과를 유지한다. 실행 중 GPU 연산을 강제 종료하는 방식은 쓰지 않는다.
- 단일 사진 정합·일괄 정합·저장 버튼의 우선순위를 명확히 하며, 비활성 버튼에 이유를 표시한다.
- 사진 교체/삭제와 Fixed 편집을 구분한다. 다른 원본 사진으로 교체하면 이전 원본의 앵커를 새 사진에 재사용하지 않는다.
- 원본 보기, 마스크 표시/숨기기와 투명도, 앵커 표시/숨기기를 제공한다. 선택한 점에는 번호·윤곽을 표시하여 색에만 의존하지 않는다.

## 6. 단축키와 실행취소 계약

| 키 | 동작 | 범위/예외 |
|---|---|---|
| Z | 현재 마스크 개체 확정 | 마스크 캔버스에서만. 마스크가 없으면 변화 없음 |
| X | 현재 사진의 마스크 전체 초기화 | 앵커·다른 사진은 유지. 전체 초기화를 하나의 취소 가능한 작업으로 기록 |
| C | 배정 없음 | 실행취소로 사용하지 않음 |
| Ctrl+Z / Cmd+Z | 최근 편집 취소 | modifier부터 검사해서 Z 확정과 충돌 방지 |
| Ctrl+Shift+Z / Cmd+Shift+Z | 다시 실행 | undo 후 새로운 작업이면 redo 분기 제거 |
| A | 앵커 입력 시작 | 선택 Moving 대상의 앵커 탭으로 전환. Fixed → Moving 순서 |
| D | 입력 중인 불완전 쌍 취소. 입력 중이 아니면 선택한 완성 쌍 제거 | 선택 쌍이 없으면 마지막 쌍을 명시적으로 선택·표시한 경우에만 제거. 전체 초기화는 별도 버튼 |
| Esc | 진행 중 도구/드래그 취소 | 저장된 다른 앵커·마스크는 유지 |
| R | Fixed 편집에서 크롭 도구 | 정합기 전역 마스크 키와 충돌시키지 않음 |
| / | Fixed 편집의 보정 전/후 비교 | 회전·크롭 위치 유지, 밝기·대비만 비교. 별도로 '원본 전체 보기' 버튼 |
| 0 | 화면에 맞춤 | 원본/편집 데이터는 바꾸지 않음 |
| Space+드래그 | 화면 이동 | 마스크/앵커 추가 방지. 휠 줌은 포인터 위치 기준 |
| ← / → | 이전/다음 Moving | 입력/슬라이더 포커스일 때는 원래 조작을 유지 |

키 처리 우선순위: 텍스트 입력/IME 조합/모달 → Ctrl/Cmd 조합 → 활성 도구 전용 키 → 일반 탐색. `input`, `textarea`, `select`, `contenteditable`, `isComposing`, `repeat`를 고려한다. 캔버스 포커스에서 `KeyboardEvent.code`를 사용해 한글 자판 상태의 물리 키도 처리하되 IME 입력 중에는 개입하지 않는다. Ctrl+A/Ctrl+D/Ctrl+X 등 브라우저·편집 기본 조합을 가로채지 않는다.

실행취소 단위:

1. 마스크 포함/제외 점 1개, Z 확정 1회, X 초기화 1회.
2. 앵커쌍 추가/삭제 1회, 점 드래그 1회. 불완전 쌍 취소는 로컬 도구 동작이다.
3. Fixed 편집 패널 내부의 슬라이더 드래그/크롭/회전 1회. 패널의 적용은 서버 이력에 1개 복합 작업으로 기록한다.
4. 결과 미세조정 적용 1회. 자동 정합 결과의 교체도 이전 결과 참조로 복원 가능하게 한다.
5. 사진 탐색, 줌/팬, 내보낸 파일 생성은 편집 이력 대상이 아니다. 파일 삭제를 undo 효과로 실행하지 않는다.

서버 반영 편집은 세션의 시간순 이력이다. 다른 사진에 대한 최근 작업을 취소하면 그 사진으로 이동하고 `DSC_0328: 마스크 확정 취소`처럼 이유를 보여 준다. 적용 전 Fixed 편집/미세조정의 로컬 draft가 있으면 그 draft 이력이 우선한다. 버튼에는 `되돌리기: 마스크 초기화`처럼 다음 동작을 표시한다.

SAM2 실행 중에는 동일 상태 변경을 직렬화하고 중복 키 입력을 받지 않는다. 이력은 성공적으로 적용된 명령만 적재한다. 마스크 배열은 매번 RGB 원본까지 복사하지 않고 압축 마스크 또는 공유 불변 객체 참조로 저장한다. 초깃값은 최근 50개 편집 단계, 이미지 버전은 로컬 캐시에 보관하며 메모리/디스크 사용량을 측정해 상한을 정한다.

## 7. 선택 정합 완료 후 무엇을 보여 줄 것인가

상태를 `activeImageId`, `checkedIds`, `job.targetIds`, `job.preferredResultId`, `navigationRevision`으로 분리한다.

1. 현재 정합: 현재 Moving 1장을 대상으로 고정한다.
2. 선택 정합: 클릭 순간의 체크 목록을 화면 순서로 snapshot한다. 보고 있던 Moving이 포함되면 그 사진을 우선 결과로 설정한다.
3. 포함되지 않으면 이번 대상 중 화면상 첫 항목을 우선 결과로 설정한다. 기존 결과가 있는 전체 목록의 첫 항목을 찾지 않는다.
4. 개별 완료 시 해당 행의 상태만 갱신한다. 처리마다 화면을 강제로 바꾸지 않는다.
5. 최종 완료 시 상태 갱신을 await한다. 작업 시작 후 사용자가 사진/탭을 변경하지 않았을 때만 우선 결과로 전환한다.
6. 사용자가 다른 사진을 보고 있으면 현재 위치를 유지하고 `선택 2장 완료 [결과 보기]`를 표시한다.
7. 선택한 사진이 실패했다면 그 사진의 실패 이유와 수정 버튼을 표시한다. 다른 성공 사진으로 대체하지 않는다.
8. 재정합 실패로 기존 결과를 보존했으면 `이번 정합 실패 · 이전 결과 표시`를 함께 보여 준다. 성공으로 집계하지 않는다.
9. 체크는 완료 후 그대로 둔다. `실패 항목만 선택`, `선택 해제`, `이번 작업 결과 저장`을 별도 제공한다.
10. 단일 오류가 나더라도 나머지 사진을 처리하고 `완료/확인 필요/실패/이전 결과 유지`를 집계한다.

job ID와 session revision을 이벤트에 포함한다. 이전 작업의 늦은 SSE 이벤트나 오래된 refresh 응답이 새로운 선택·진행 상태를 덮어쓰지 않게 한다. 선택/체크는 클라이언트별이고 서버 작업은 공유이므로, 작업을 시작하지 않은 다른 탭은 완료 시 자동 이동하지 않는다.

## 8. 앵커 입력과 엔진 연결

- A → 왼쪽 Fixed에서 1점 → 오른쪽 선택 Moving에서 대응점 → 1번 쌍 완성.
- 완성 후 방금 만든 쌍을 선택한 일반 선택 모드로 돌아간다. 따라서 바로 D를 누르면 방금 만든 쌍이 취소된다. 여러 점은 A 반복 또는 명시적 '연속 입력' 토글로 지정한다.
- 1, 2, 3 번호를 양쪽에 동일하게 표시하고 목록에서 쌍 선택·삭제·활성화/비활성화한다.
- 점 드래그 시 확대 상태에서도 정확히 원본 좌표로 저장한다. 좌표값 자체는 일반 UI에 노출하지 않는다.
- Moving을 바꾸면 해당 사진의 쌍만 표시한다. Fixed 점을 공유하고 싶으면 '이 기준점 재사용'을 선택하는 흐름을 후속으로 추가할 수 있다. 모든 Moving 클릭을 강제하지 않는다.
- 마스크와 독립적으로 쓸 수 있게 한다. 마스크 없는 전체영역 정합에도 완성된 활성 쌍을 전달한다.
- 원본 좌표와 현재 표시 좌표를 구분하고 이미지 변환 함수에만 좌표 변환 책임을 둔다.
- 일반/예비 단일 패스/Lazy 경로에서 동일한 앵커 정책을 적용한다.
- 기존 높은 가중치 재적합은 우선 재사용한다. 명칭은 '대응점 보조'이며 '정확히 고정'이라고 안내하지 않는다.
- 앵커 적용 이후 자동 대응점의 재투영 오차 및 앵커 잔차를 다시 계산한다. 기존 품질 게이트 함수가 M으로 오차를 계산하므로 변경 후 M을 명시적으로 전달한다. 결과 표시는 재계산된 값과 일치해야 한다.
- 1쌍은 자동 정합의 보조 입력으로 허용한다. 자동 매칭이 완전히 실패한 사진을 앵커만으로 정합하는 기능은 이번 기본 연결과 분리한다. 필요 시 별도 모드로 계획하며, 2D similarity에서는 서로 다른 2쌍 이상과 점 간 거리/퇴화 검사를 요구한다.
- 모든 정합·앵커·미세조정·저장 경로에 비율 보존 조건을 적용한다. normal/strict 사이에는 품질 판정의 엄격도 차이만 둔다.

### 비율 보존을 강제하는 변경

1. `config.py`의 relaxed 프로필 및 `allow_affine=True`를 통한 출력 허용 경로를 제거한다. 이전 클라이언트가 relaxed를 전송하면 서버는 422와 '지원하지 않는 프로필'을 반환한다. 화면에 남은 오래된 선택은 기본 프로필로 교정한다.
2. `register.py`의 full affine 반환과 앵커의 affine 재적합을 제거한다. 일반/피라미드 폴백/Lazy/구형 Tk 실행에서 동일하게 적용한다.
3. 현재 코드는 affine RANSAC으로 후보 대응점만 찾고 similarity로 다시 맞추는 구제 경로가 있다. 이 중간 후보 탐색은 최종 결과의 비율을 깨지 않으므로 유지할 수 있지만, 그 affine 행렬은 반환·렌더·저장에 절대 사용하지 않는다.
4. 최종 2×2 선형 부분 A에 `A.T @ A ≈ s² I` 조건을 검사한다. 수동 반전은 허용하므로 det 부호만으로 비율 파괴를 판정하지 않는다. 비등방 배율/전단/비유한 값/0 이하 배율은 실패로 처리한다.
5. 결과 미세조정의 배율은 단일 값만 제공하고 내부·API에서도 축별 배율 입력을 받지 않는다.
6. 예전에 만들어진 affine 결과가 남아 있으면 '비율 보존 조건 미충족 — 재정합 필요'로 구분한다. 최신의 유효 결과로 저장하는 경로에는 포함시키지 않는다.
7. 기존 `relaxed로 재시도` 안내를 `대응점 추가`, `마스크 수정`, `방향 자동탐색`, `수동 미세조정`으로 교체한다.

## 9. Fixed에 Photo Editor 통합

재사용 근거: [Photo Editor App.vue](https://github.com/perioahn/photo-editor/blob/ae420188736791bb9b9683e031aa3d854425e762/src/App.vue), [EditorCanvas.vue](https://github.com/perioahn/photo-editor/blob/ae420188736791bb9b9683e031aa3d854425e762/src/components/EditorCanvas.vue), [edits.ts](https://github.com/perioahn/photo-editor/blob/ae420188736791bb9b9683e031aa3d854425e762/src/edits.ts).

### 포함 기능

- 90° 회전, 좌우/상하 반전, 선으로 수평 맞추기, 미세 회전.
- 밝기·대비, 수치 입력과 개별 리셋, 보정 전/후 비교.
- 자유/1:1/4:3/3:2/16:10/16:9/사용자 지정 크롭 비율과 격자.
- 줌·팬·화면 맞춤, 적용·취소·다시 편집·원본으로 복원.
- `이 기준으로 적용`하면 내보내기/재업로드 없이 바로 정합기에 반영한다.

Photo Editor의 폴더 탐색·NEF 로딩·저장 후 다음 기능은 Fixed 한 장 편집에 필요하지 않아 이번 통합 범위에 넣지 않는다. 기존 정합기의 JPEG/PNG 입력을 기준으로 한다. NEF 지원은 별도 요구 시 공통 입력 계층에서 검토한다.

### 원본, 미리보기, 실제 정합

1. 업로드 원본을 불변으로 유지한다. 브라우저에는 `/api/image/{id}/source`로 원본을 제공하되 실제 원본 파일 경로를 외부에 노출하지 않는다.
2. 편집은 원본으로부터 조정값을 적용해 미리보기한다. 기존 캔버스의 캐시와 requestAnimationFrame을 활용한다.
3. 적용 시 원본에서 풀해상도 편집본을 한 번 렌더하고 **PNG**로 로컬 서버에 전달한다. 기존 `renderFinal()`의 JPEG/EXIF 내보내기를 그대로 중간 처리에 사용하지 않는다.
4. 레시피, 원본→편집본 기하변환 G, 출력 크기, base revision을 함께 전달하고 서버가 동일한 편집 레시피와 크기에 맞는지 검증한다.
5. 서버는 새 fixed revision을 원자적으로 생성한다. get_full/get_work/SAM2 캐시/미니맵을 새 버전으로 교체한다. 화면만 바꾸고 정합에 옛 픽셀을 쓰는 상황을 막는다.
6. 실패하면 기존 기준·마스크·앵커·결과를 유지한다. 새 파일과 새 revision의 준비가 완료되기 전 현 상태를 지우지 않는다.
7. 원본은 EXIF 방향을 한 번만 정규화한다. 브라우저 렌더와 서버/Pillow 기준 방향의 일치 여부를 EXIF 회전 샘플로 검증한다.

### 기존 마스크·앵커·결과 처리

| 변화 | 마스크·앵커 | 기존 정합 결과 |
|---|---|---|
| 밝기·대비만 | 위치와 확정 마스크 보존, SAM2 임베딩은 갱신 | 변환을 임시 비교용으로 유지하되 '기준 편집 후 확인 필요' 표시. 재정합 전 새 기준의 평가 결과로 간주하지 않음 |
| 회전·반전·크롭 | 원본 좌표 앵커를 G로 다시 표시. 확정 마스크는 원래 생성 revision과 변환을 보존해 새 캔버스에 투영 | 계산 당시 Fixed revision으로 기존 결과를 계속 볼 수 있게 보존하고 '다시 정합 필요' 표시 |
| 크롭 밖으로 나간 앵커 | 삭제하지 않고 '영역 밖'으로 비활성화, 크롭 복원 시 다시 표시 | 해당 비활성 쌍은 엔진에서 제외 |
| 편집 적용 취소 | 이전 revision·마스크·앵커·결과의 참조를 한 번에 복원 | 일관된 이전 상태로 복원 |

기하 좌표 규약은 `p_edit = G @ p_source`, `p_source = inverse(G) @ p_edit`로 고정한다. 마스크의 warp는 최근 마스크를 계속 재회전시키지 않고 보관된 생성 revision에서 직접 수행한다. 화면 픽셀→work→full 변환은 가로/세로 배율을 각각 계산하여 정수 리사이즈의 반올림 오차를 피한다.

결과는 계산 당시의 Fixed revision을 참조한다. 현재 Fixed 픽셀 위에 이전 revision 결과를 덮어 보여 주는 동작을 금지한다. 미세조정도 결과가 참조하는 Fixed 크기/Moving 방향으로 수행한다. 재정합이 실패하면 이전 결과의 revision과 '이전 결과' 표시는 유지한다.

## 10. 최소 상태/API 설계

### 상태

```text
Session: fixedId, images, pairAnchors, results, history
Image: id, originalPath, currentRevisionId (role은 Session.fixedId와 비교하여 계산)
ImageRevision: id, sourceId, edits, G, width, height, renderedPath
AnchorPair: id, fixedId, movingId, fixedSourcePoint, movingSourcePoint, enabled
MaskObject: id, imageId, sourceRevisionId, sourceWorkSize, maskAsset, prompts
Result: id, fixedId, movingId, fixedRevisionId, movingRevisionId,
        inputRevision, matrix, lazyOrientation, metrics, freshness, reviewStatus
Job: id, targetIds, inputSnapshot, items[], running
HistoryEntry: id, label, imageId, beforeRefs, afterRefs
```

- 입력 버전에는 이미지 편집뿐 아니라 마스크·앵커 변경을 포함한다.
- UI에는 revision/matrix 대신 '이전 결과', '다시 정합 필요'로 표시한다.

### API 변경안

| API | 역할 |
|---|---|
| GET `/api/state` | 이미지 revision, 앵커 수, 결과의 최신 여부, job 요약, undo/redo 라벨 제공 |
| POST `/api/upload` | 기본은 역할 구분 없는 사진 배치 추가. 첫 성공 사진을 초기 Fixed로 지정 |
| POST `/api/fixed` | 기존 사진을 새 기준으로 지정. 이전 기준은 Moving, 기준별 결과·앵커 보존 |
| POST `/api/history/undo`, `/redo` | 최근 서버 반영 편집 복원. 새 session revision과 영향 이미지 반환 |
| GET/PUT `/api/anchors/{moving_id}` | pair ID 기반 원본 좌표 쌍 목록. base revision 검증, 한 번의 PUT을 이력 1개로 기록 |
| GET `/api/image/{id}/source` | 편집용 원본 바이너리 제공 |
| POST `/api/image/{id}/edit` | multipart PNG+레시피+G+base revision. 적용 성공 시 관련 상태를 한번에 교체 |
| POST `/api/register` | 기존 인자를 유지하며 응답에 job_id/target_ids 추가, 입력 snapshot 생성 |
| SSE register | 모든 이벤트에 job_id, 개별 상태에 moving_id, 결과 보존 사유 및 최신 여부 포함 |
| 기존 save API | 이번 job/현재/체크 선택의 저장 대상 명확화. 이전 결과 저장 시 UI에서 표시 |

기존 함수형 서버를 전면 재작성하지 않는다. 순수 상태/이력만 작은 모듈로 분리하고 엔드포인트는 현 server.py에서 점진적으로 연결한다. 테스트를 위해 세션 저장 위치를 주입 가능하게 하여 import나 테스트 실행으로 사용 중 세션을 지우지 않는다.

## 11. 구현 순서와 파일

각 작업은 실패 시나리오를 먼저 고정하고, 최소 구현 후 지정 검증을 통과시키며 별도 커밋으로 나눈다. 테스트 도입은 단축키 문자열 자체보다 실제 상태 회복과 비동기 선택 문제를 검증하는 데 집중한다.

### 작업 0 — 통합 사진 집합·기준 변경·작업대 뼈대

**수정:** `App.vue`, `webapp/server.py`, `webapp/session_state.py`, `src/viewstate.ts`.
**추가:** `Workspace.vue`, `PhotoBrowser.vue`, `ComparisonViewport.vue`, `ContextTools.vue`, 기준 변경 회귀 시나리오.

1. `사진 추가` 하나로 여러 장을 받으며 기존 별도 Fixed/Moving 입력 흐름을 교체한다.
2. 초기 첫 성공 사진을 기준으로 지정한다. 추가 업로드 시 기준은 유지한다.
3. fixedId를 바꾸는 한 명령으로 역할·기준별 앵커·결과 조회를 갱신한다. undo 상태에는 이전 기준을 포함한다.
4. 한 작업대에서 현재 사진·도구·비교 방식을 독립적으로 관리한다. 기준 변경 후 잘못된 이전 결과를 겹치지 않는다.
5. 상세 동작과 검증은 UX 개정안 2·7·8절을 따른다.

**통과 기준:** 여러 장 1회 입력 → 기본 기준 지정. 현재 사진을 기준으로 변경 → 이전 기준이 Moving. 원본별 편집·마스크 및 기준별 결과·앵커 손실 없음.

### 작업 1 — 선택 정합 결과·체크·저장 흐름

**수정:** `webapp/frontend/src/App.vue`, `webapp/server.py`.
**추가:** `webapp/frontend/src/registrationSelection.ts`, `webapp/frontend/src/registrationSelection.test.ts`.

1. 현재 목록 1번에 기존 결과, 5번만 정합하는 회귀 시나리오를 작성한다.
2. target snapshot, preferred result, navigation revision, 요청 시점의 선택을 분리한다.
3. done 처리에서 refresh를 await하고 실행한 job에 속하는 대상만 선택한다.
4. 완료 후 체크 유지, 실패만 선택, 이번 결과 저장을 연결한다.
5. 순서가 뒤바뀐 SSE/refresh 응답, 완료 전 수동 탐색을 검증한다.

**통과 기준:** 5번 정합 → 5번 결과. 진행 중 8번으로 이동 → 8번 유지. 체크 3장 → 완료 후 그대로 3장 저장 가능.

### 작업 1B — relaxed 제거·비율 보존 강제

**수정:** `config.py`, `register.py`, `webapp/server.py`, `ResultViewer.vue`, `README.md`, `tests/test_engine.py`.
**추가:** `tests/test_similarity_only.py`.

1. shear/축별 다른 배율을 가진 행렬이 거절되는 테스트를 작성한다.
2. relaxed 및 full affine 결과 반환 경로를 제거하고 모든 앵커 재적합을 similarity로 고정한다.
3. API에서 허용된 normal/strict만 받고, UI의 오류 안내와 기존 배지를 갱신한다.
4. 일반/Lazy/폴백/앵커/미세조정 결과의 비율 보존 불변식을 검사한다.

**통과 기준:** 어떤 옵션·실패 폴백을 타도 비등방 배율/전단 결과가 저장되지 않는다. 정상 회전·이동·균일 확대·반전은 허용한다.

### 작업 2 — 편집 상태·리비전·이력 기반

**수정:** `webapp/server.py`.
**추가:** `webapp/session_state.py`, `webapp/history.py`, `tests/test_session_history.py`.

1. 테스트 세션 위치 주입 및 임시 디렉터리 fixture를 만든다.
2. mask 점/확정/reset의 before/after를 복원하는 명령을 정의한다.
3. 마스크 배열 불변 참조와 압축 저장 정책을 적용한다.
4. undo/redo API, 상태 라벨 및 revision 검사를 붙인다.
5. Z 직전 상태와 X 이전 전체 상태가 동일하게 복원되는지 검증한다.
6. 정합은 입력 snapshot을 사용한다. SAM2 및 같은 세션 변경은 서버에서도 직렬화한다.

**통과 기준:** 점 3개→Z→X→undo→undo로 확정 마스크와 그 직전 3개 프롬프트를 순서대로 복원. 중간 API 실패 시 이력·상태에 반쪽 변경 없음.

### 작업 3 — 공통 단축키와 도구 상태

**수정:** `App.vue`, `components/MaskEditor.vue`, `components/ResultViewer.vue`.
**추가:** `src/composables/useShortcuts.ts`, `src/composables/useShortcuts.test.ts`, `src/components/HistoryControls.vue`.

1. Ctrl+Z가 Z 확정을 호출하는 회귀 시나리오부터 추가한다.
2. 한 곳에서 도구/포커스/수정키를 판정하고 개별 window 리스너를 정리한다.
3. Z/X/Ctrl+Z 및 redo, 버튼의 동작 경로를 공통 명령으로 통일한다. C는 처리하지 않는다.
4. 텍스트 입력/한글 조합/키 반복/숫자 슬라이더에서의 예외를 검증한다.

**통과 기준:** Ctrl+Z는 SAM2 confirm을 절대로 호출하지 않음. C는 상태를 변경하지 않음. X는 현재 사진 마스크만 초기화. 단축키와 버튼 결과가 동일.

### 작업 4 — 점 앵커 UI와 기존 엔진 연결

**수정:** `App.vue`, `webapp/server.py`, `register.py`.
**추가:** `src/components/AnchorEditor.vue`, `src/imageCoordinates.ts`, `tests/test_anchor_registration.py`.

1. 2분할 사진, 번호 라벨, Fixed→Moving 입력 상태를 구현한다.
2. A/D/Esc, pair ID 기반 CRUD, 드래그 이동, 쌍 삭제 undo를 연결한다.
3. 화면·work·full 좌표 변환과 마스크 없는 경로의 전달을 검증한다.
4. register_test/register_test_lazy에 완성된 활성 쌍을 전달한다.
5. 앵커 재적합 후 게이트/잔차 갱신을 일반/단일 패스 경로에 모두 적용한다.

**통과 기준:** 이미지 크기/줌/반전/90° 회전 조합에서 앵커가 같은 위치를 가리킴. D로 2번 쌍 제거 후 Ctrl+Z로 2번만 복원. 잔차는 최종 M으로 계산.

### 작업 5 — Photo Editor 편집 캔버스 통합

**수정:** `App.vue`, `webapp/server.py`, `frontend/package.json`.
**추가:** `src/components/FixedEditor.vue`, `src/components/PhotoEditorCanvas.vue`, `src/photoEdits.ts`, `docs/photo-editor-integration.md`.

1. Photo Editor의 Edits/렌더/캔버스를 위에서 고정한 커밋으로 가져오고 파일 입출력 의존을 어댑터로 분리한다.
2. 기존 캐시·LUT를 재사용하고 슬라이더 이력을 포인터 down~up 1회로 묶는다.
3. 조정값과 G를 같은 기하 연산에서 산출하고 PNG 렌더 경로를 추가한다.
4. Fixed 카드의 편집→적용→원래 Moving 복귀를 연결한다.
5. 폴더/NEF/EXIF JPEG 내보내기 코드를 끌고 오지 않는다. 필요한 의존성만 추가한다.

**통과 기준:** 회전·반전·수평·크롭·밝기·대비 조작이 기존 에디터와 대응하며 재업로드 없이 정합기 픽셀에 반영됨. 30회 슬라이더 이벤트가 1회의 undo로 복원됨.

### 작업 6 — 편집 후 좌표·캐시·결과 일관성

**수정:** `webapp/session_state.py`, `webapp/server.py`, `src/imageCoordinates.ts`, `ResultViewer.vue`.
**추가:** `tests/test_fixed_revisions.py`, `src/photoEdits.test.ts`.

1. 밝기 편집 시 마스크·앵커 좌표 보존, 기하 편집 시 G에 따른 표시 변경을 구현한다.
2. source revision 마스크로부터 새 마스크를 투영하며 crop 밖의 점은 보존하되 제외한다.
3. old result의 Fixed revision을 pin하고 새 결과와 구분한다.
4. 편집 commit/undo 때 썸네일·원본 캐시·SAM2 임베딩을 같이 갱신한다.
5. 원본/결과 저장은 반복 중간 JPEG 재인코딩 없이 수행한다.

**통과 기준:** 90° 회전 및 크롭 후 표식 이미지·앵커·마스크가 동일한 위치. 편집 적용 undo가 기준·주석·기존 결과를 함께 복원. 오래된 결과를 새 기준에 겹치지 않음.

### 작업 7 — 전체 화면 및 연속 검토

**수정:** `App.vue`, `src/style.css`, `src/viewstate.ts`, `MaskEditor.vue`, `ResultViewer.vue`.

1. Fixed 고정 카드, 도구별 툴바, 항상 보이는 상태/단축키 줄을 정리한다.
2. 현재/체크/이번 job 작업 대상을 버튼 라벨에 명확히 표시한다. 검색·필터·드래그앤드롭·전체 선택/해제를 추가한다.
3. 줌·팬을 이미지/도구별로 보존한다. 정합된 결과끼리 동기 줌은 같은 Fixed revision일 때만 공유한다.
4. 미적용 미세조정 draft를 사진별로 보관하고 돌아오면 복원한다.
5. 화면 진입 안내를 '마스크 없이도 정합 가능' 정책에 맞추고 오류별 다음 행동을 제시한다.
6. 상태 메시지 role, 키보드 포커스, 버튼 라벨, 좁은 화면 접기 동작을 검증한다.
7. 서버 job에 cancel_requested를 두고 '남은 작업 중지'를 사진 처리 경계에서 적용한다. 완료/중지 이벤트를 구분하고 처리된 결과를 보존한다.

**통과 기준:** 1366×768에서 작업 캔버스와 현재/선택 정합 버튼이 사용 가능하며 125/150% 배율에서 핵심 조작이 잘리지 않음.

### 작업 8 — 통합 검증과 문서·번들

**수정:** `README.md`, `webapp/frontend/package.json`, `webapp/frontend/dist/*`(빌드 결과), `tests/`.
**추가:** `webapp/frontend/e2e/registration-workflow.spec.ts`, 필요한 Vitest/Playwright 설정.

1. 합성 표식 사진과 가짜 SAM2/정합 응답으로 UI/API 회귀 테스트를 먼저 수행한다.
2. 기존 엔진 테스트와 별도로 한두 개 승인된 로컬 실제 페어를 써서 마스크/앵커/Fixed 편집 연결을 확인한다. 사진은 외부 서비스에 업로드하지 않는다.
3. 키 안내·README·툴팁을 단일 단축키 정의에 맞춘다.
4. 프런트 빌드 후 실제 launcher가 제공하는 dist에서 최종 흐름을 확인한다.

명령(구현 시 도입한 테스트 설정과 함께 실행):

```powershell
# dkp_registrator에서: 상태/좌표/API 회귀, 사용자 세션과 분리된 fixture 사용
py -3.13 -m pytest tests/test_session_history.py tests/test_anchor_registration.py tests/test_fixed_revisions.py tests/test_similarity_only.py -q
py -3.13 -m pytest tests/test_engine.py -q

# webapp/frontend에서
npm run test -- --run
npm run build
npx playwright test
```

예상 결과는 전부 PASS 및 빌드 종료 코드 0이다. 엔진 테스트의 LoFTR 가중치 다운로드/장치 의존은 순수 상태 테스트와 구분하여 보고한다. 이 계획 작성 단계에서 위 테스트를 실행한 것은 아니다.

## 12. 반드시 확인할 사용자 시나리오

| 시나리오 | 기대 결과 |
|---|---|
| 여러 장을 한 번에 추가 | 첫 성공 사진이 기본 Fixed. 나머지 즉시 정합 대상 |
| 중간에 다른 사진을 Fixed로 지정 | 이전 Fixed는 Moving. 기준별 앵커/결과 구분, 편집·마스크 유지 |
| 기존 결과가 있는 1번을 두고 5번만 선택 정합 | 5번 결과 표시 |
| 5번 정합 도중 8번 마스크를 보러 이동 | 완료되어도 8번 유지, 완료 링크만 표시 |
| 3장 정합 후 선택 저장 | 같은 3장이 선택된 상태로 저장 가능 |
| 마스크 클릭→Z→X→Ctrl+Z→Ctrl+Z | 초기화 전, 확정 전 상태 순서대로 복원 |
| 한글 파일명 입력 중 A/X/Z | 입력만 수행, 도구 전환·초기화 없음 |
| 마스크 화면에서 C | 상태 변경 없음 |
| relaxed를 API에 직접 전송 | 지원하지 않는 프로필로 거절, 비율 변형 결과 생성 안 함 |
| 매칭이 나빠 full affine이 더 잘 맞는 사진 | similarity로 유효하게 맞추거나 실패. 형태를 찌그러뜨려 성공 처리하지 않음 |
| 확대·팬 상태에서 앵커 입력 | 화면 표시점과 엔진 입력점 일치 |
| 3쌍 중 2번만 D로 취소→undo | 나머지 쌍 유지, 2번 복원 |
| 마스크 없이 앵커 1쌍으로 정합 | 자동 정합 보조로 전달, 앵커만의 성공을 보장하지 않음 |
| Fixed 밝기 편집 | 마스크/앵커 위치 유지, 실제 정합 픽셀 갱신 |
| Fixed 회전+크롭 | 마스크·앵커 좌표 일치, 이전 결과 명확히 구분 |
| Fixed 적용 중 실패 | 이전 전체 상태 유지 |
| Fixed 적용→undo | 기준 이미지와 종속 상태 함께 복원 |
| 재정합 실패·이전 성공 결과 존재 | '이번 실패/이전 결과 표시', 최신 성공으로 집계 안 함 |
| 같은 작업을 보는 두 브라우저 탭 | 늦은 이벤트 무시, 시작하지 않은 탭의 활성 사진 유지 |
| 미세조정 후 다른 사진→복귀 | draft·줌·팬 유지 |
| Windows EXIF 회전 JPEG / 한글 경로 | 편집·정합·저장 방향 일치 |
| 10장 중 3번째 처리 중 남은 작업 중지 | 현재 3번째까지 완료, 나머지는 미처리 표시, 완료 결과 유지 |

## 13. 독립 Photo Editor — 폴더/사진 한 장 열기 분리

### 현재 근거

`src/App.vue`에는 폴더용 `webkitdirectory multiple` input과 `onFallbackFiles()`만 연결되어 있다. `src/files.ts`의 `toPhoto()`는 단일 File 처리/NEF 미리보기 추출을 이미 지원하지만 외부로 export되지 않았다. 같은 변환 함수를 폴더와 단일 파일 진입점에서 공유하면 된다.

### 사용자 동작

| 진입 | 동작 | 편집/저장 |
|---|---|---|
| 폴더 열기 | 기존 폴더 선택. 해당 폴더의 지원 사진 목록을 생성 | 필름스트립·이전/다음·저장 후 다음 유지 |
| 사진 한 장 열기 | 일반 파일 선택 대화상자. JPEG/PNG/NEF 중 한 장만 선택 | 즉시 편집. 사진이 한 장이면 이전/다음·저장 후 다음은 숨기거나 비활성 이유 표시 |
| 선택 취소 | 현재 열려 있는 사진과 편집값을 유지 | 저장 폴더와 이력을 지우지 않음 |

- 상단과 시작 화면 모두 `폴더 열기`, `사진 한 장 열기` 두 버튼을 표시한다.
- 단일 파일 input에는 `webkitdirectory`와 `multiple`을 붙이지 않는다. 처리 함수에서도 파일 1개로 제한한다.
- 단일 파일을 성공적으로 디코드한 다음 새 작업으로 전환한다. 손상 사진/선택 취소 시 이전 편집 상태를 보존한다.
- 새 작업 시작 시 미저장 편집이 있으면 기존 보호 흐름을 적용한다. 도구 전환·이력·선택은 새 세션과 함께 초기화한다.
- 원본은 덮어쓰지 않고 기존 `이름_e.jpg` 저장 규칙을 유지한다. 단일 파일을 열었다고 원본 폴더에 쓸 권한이 생겼다고 가정하지 않는다. 저장 폴더 지정 또는 다운로드 흐름을 유지한다.
- 같은 파일명으로 다른 사진을 열어도 기존 `editsMap`/undo stack이 섞이지 않도록 세션 초기화 또는 안정적인 photo ID를 쓴다.
- 사용하지 않는 object URL은 성공적인 전환 후 해제한다.

### 작업 9 — Photo Editor의 별도 변경

**저장소:** `perioahn/photo-editor` (정합기 폴더 안의 임시 조사본을 배포 작업 디렉터리로 사용하지 않는다. 구현 시 정식 작업 체크아웃에서 진행).
**수정:** `src/App.vue`, `src/files.ts`, `src/style.css`, `README.md`.
**검증:** 파일 입력과 세션 전환의 브라우저 흐름. 저영향 버튼 분리 자체를 위한 형식적 단위 테스트는 만들지 않는다.

1. 단일 File→Photo 변환을 공통 함수로 노출한다.
2. 폴더용 input과 단일 사진 input을 분리하고 전용 핸들러를 연결한다.
3. 성공적인 로드 후 상태 전환을 묶고 기존 미저장 작업 보호를 연결한다.
4. 단일 사진 모드의 버튼 표시·저장 문구를 다듬는다.
5. JPEG 한 장, PNG 한 장, NEF 한 장, 폴더, 선택 취소, 손상 파일, 같은 이름의 다른 사진을 검증한다.
6. `npm run build` 통과 후 preview에서 확인한다. GitHub Pages 배포는 계획 단계에 포함하지 않으며 실제 배포 시 별도로 수행·검증한다.

**통과 기준:** 사진 한 장 열기로 폴더를 고를 필요 없이 즉시 편집할 수 있고, 폴더 연속 편집 기능은 유지된다.

## 14. 범위 경계와 후속 후보

- 이번 핵심 범위: 선택 결과 문제, 체크 유지, 단축키/실행취소, 점 앵커, Fixed 편집, 좌표·이력 일관성, UI 전반 개선, relaxed 제거·비율 보존 강제, Photo Editor 단일 사진 열기와 검증.
- 별도 후속 후보: 서버 재시작 후 자동 복구, 여러 환자/세션 관리, Moving 전체에 사진 보정 확대, NEF 입력, 앵커만으로 초기 정합, 공유 Fixed 점 다중 Moving 입력.
- 서버 재시작 복구는 현재 세션 삭제 구조를 바꿔야 하므로 화면 상태 유지와 같은 기능으로 뭉뚱그리지 않는다. 새로고침 시 서버가 살아 있는 동안의 상태 복원은 이번 state API에서 지원한다.
- 정확도/성능 개선을 확인하기 전 속도 배수·임상적 정확도 같은 수치를 UI에 새로 약속하지 않는다.
- 구현 시작 전에 달라진 HEAD와 사용자 답변을 다시 확인하고, 이 문서의 가정을 업데이트한다.
