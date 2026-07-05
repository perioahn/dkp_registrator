"""DKP Registrator 실행 진입점 (exe/app 빌드용).

기본 = 웹 UI (브라우저 자동 오픈). `--tk` 옵션으로 기존 tkinter GUI 실행.
"""

import sys


def main() -> None:
    if "--tk" in sys.argv:
        sys.argv.remove("--tk")
        import main_gui
        main_gui.main()
    else:
        from webapp.server import main as web_main
        web_main()


if __name__ == "__main__":
    main()
