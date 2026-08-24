"""GPU(CUDA) 가속 선택 설치 — 배포판 exe에서 앱 내 버튼으로 켜고 끈다.

배포 exe에는 CPU용 torch가 들어 있어 **언제나 바로 실행**된다. NVIDIA GPU가 있으면
사용자가 앱에서 [GPU 가속 켜기]를 눌러 CUDA용 torch 휠을 관리 폴더에 내려받고,
다음 실행부터 그쪽을 우선 import 한다(실패해도 앱은 CPU로 계속 동작).

pip을 쓰지 않는다 — frozen(exe) 환경에서 pip 실행이 불안정해 휠(zip)을 직접 받아 푼다.
의존성(numpy·filelock·sympy 등)은 이미 배포판에 포함돼 있어 추가 설치가 필요 없다.

관리 폴더: %LOCALAPPDATA%\\DKPRegistrator\\cuda  (테스트 재지정: DKP_TORCH_DIR)
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

_PKGS = ("torch", "torchvision")
_ROOTS = ("torch", "torchvision", "torchgen", "functorch")  # 휠에 함께 들어오는 최상위 패키지
_INDEX = "https://download.pytorch.org/whl/cu124/{pkg}/"
_UA = {"User-Agent": "DKPregistrator (+https://github.com/perioahn/dkp_registrator)"}


def base_dir() -> str:
    override = os.environ.get("DKP_TORCH_DIR")
    if override:
        return override
    base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    return os.path.join(base, "DKPRegistrator", "cuda")


def _marker() -> str:
    return os.path.join(base_dir(), "ok.json")


def _log_path() -> str:
    d = os.path.dirname(base_dir())
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return os.path.join(os.path.expanduser("~"), "dkp_gpu.log")
    return os.path.join(d, "gpu_setup.log")


def _log(msg: str) -> None:
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def installed() -> bool:
    return os.path.isfile(_marker())


def gpu_name() -> str | None:
    """nvidia-smi로 GPU 이름 조회 (없으면 None)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        r = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    lines = r.stdout.decode(errors="replace").strip().splitlines()
    return lines[0].strip() if lines and lines[0].strip() else None


class _ManagedFinder:
    """관리 폴더의 torch를 번들 torch보다 우선 import 시키는 meta_path 파인더.

    PyInstaller의 FrozenImporter가 sys.path보다 앞서므로 sys.path.insert만으로는
    번들 CPU torch가 이긴다 — 최상위 패키지 이름만 가로채 관리 폴더로 돌린다.
    """

    def __init__(self, path: str) -> None:
        self._path = [path]

    def find_spec(self, name, path=None, target=None):
        if name not in _ROOTS:
            return None  # 하위 모듈은 부모 __path__(관리 폴더)를 따라감
        return importlib.machinery.PathFinder.find_spec(name, self._path, target)


def activate() -> bool:
    """설치돼 있으면 관리 폴더 torch를 우선하도록 등록. 반환 = 활성화 여부.

    launcher가 엔진 import 전에 호출한다. 실패해도 예외를 던지지 않는다(CPU로 계속).
    """
    try:
        if not installed():
            return False
        d = base_dir()
        if not os.path.isdir(os.path.join(d, "torch")):
            _log("마커는 있으나 torch 폴더 없음 — 무시")
            return False
        sys.meta_path.insert(0, _ManagedFinder(d))
        sys.path.insert(0, d)
        os.environ.setdefault("DKP_CUDA_ACTIVE", "1")
        _log(f"CUDA 관리 폴더 활성화: {d}")
        return True
    except Exception as e:  # 어떤 경우에도 앱 기동을 막지 않는다
        _log(f"activate 실패(무시): {e}")
        return False


def _wheel_url(pkg: str) -> str:
    """현재 파이썬 버전에 맞는 win_amd64 휠 URL (인덱스 페이지 파싱)."""
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    html = urllib.request.urlopen(
        urllib.request.Request(_INDEX.format(pkg=pkg), headers=_UA), timeout=60
    ).read().decode(errors="replace")
    hits = re.findall(rf'href="([^"]*{pkg}-[^"]*{tag}-{tag}-win_amd64\.whl[^"]*)"', html)
    if not hits:
        raise RuntimeError(f"{pkg}: {tag} win_amd64 휠을 찾지 못했습니다")
    u = hits[-1].split("#")[0]
    return u if u.startswith("http") else "https://download.pytorch.org" + u


def _download(url: str, dest: str, on_progress) -> None:
    """진행률 콜백과 함께 다운로드 (.part → rename)."""
    part = dest + ".part"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if on_progress and (now - last > 0.5 or done == total):
                last = now
                on_progress(done, total)
    os.replace(part, dest)


def install_cuda(on_status=None) -> None:
    """CUDA용 torch/torchvision 휠을 내려받아 관리 폴더에 설치. 실패 시 예외.

    on_status(dict): {"phase": ..., "pkg": ..., "done": bytes, "total": bytes} 진행 보고.
    """
    def say(**kw):
        if on_status:
            on_status(kw)

    d = base_dir()
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    _log(f"CUDA 설치 시작 → {d}")
    try:
        for pkg in _PKGS:
            say(phase="url", pkg=pkg)
            url = _wheel_url(pkg)
            whl = os.path.join(tmp, url.split("/")[-1].split("?")[0])
            _log(f"{pkg} 다운로드: {url}")
            say(phase="download", pkg=pkg, done=0, total=0)
            _download(url, whl,
                      lambda done, total, p=pkg: say(phase="download", pkg=p,
                                                     done=done, total=total))
            say(phase="extract", pkg=pkg)
            _log(f"{pkg} 압축 해제")
            with zipfile.ZipFile(whl) as z:
                z.extractall(tmp)
            os.remove(whl)
        say(phase="finalize")
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)
        with open(_marker(), "w", encoding="utf-8") as f:
            json.dump({"variant": "cu124",
                       "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
                       "installed": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        _log("CUDA 설치 완료")
        say(phase="done")
    except BaseException as e:
        import traceback
        _log(f"설치 실패: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def remove_cuda() -> None:
    """관리 폴더 삭제 — 다음 실행부터 번들 CPU torch 사용."""
    shutil.rmtree(base_dir(), ignore_errors=True)
    _log("CUDA 관리 폴더 삭제")
