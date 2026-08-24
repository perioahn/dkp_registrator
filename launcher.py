"""DKP Registrator 실행 진입점 (exe/app 빌드용).

기본 = 웹 UI (브라우저 자동 오픈). `--tk` 옵션으로 기존 tkinter GUI 실행.
"""

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # macOS: 미지원 MPS 연산 per-op CPU 폴백

# PyInstaller windowed 빌드는 stdout/stderr가 None → uvicorn 로깅의 isatty() 크래시 방지
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import gpu_setup  # noqa: E402


def main() -> None:
    gpu_setup.activate()  # GPU 가속을 켜뒀으면 관리 폴더 torch 우선 (아니면 번들 CPU)
    if "--tk" in sys.argv:
        sys.argv.remove("--tk")
        import main_gui
        main_gui.main()
    else:
        from webapp.server import main as web_main
        web_main()


if __name__ == "__main__":
    main()
